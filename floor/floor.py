"""Pharmacy Floor: reception agent + five specialist tools. Decider 2B makes every routing and safety decision."""
from __future__ import annotations

import os
import re
import time

from strands import Agent

from recipes.care_router import CareRouterIntervention
from recipes.care_router.decider import DEFAULT_CHECKPOINT, CareRouter
from recipes.care_router.handler import ROUTE_INSTRUCTIONS
from recipes.turn_router import AGENTS, SAFETY_ALL, TurnRouter, TurnRouterIntervention

from . import confidence as C
from .decision_log import TurnLog
from .steps import REPLY_CHECK, TOOL_CHECK, ask
from .final_gate import FIXED, final_gate, specialist_gate
from .llm import make_model, mode, model_label
from .specialists import Specialists

RECEPTION_PROMPT = ("You are the reception desk of a drugstore floor. You never answer yourself: you call exactly one "
                    "specialist tool, the one named in the [ROUTE] note, and return its answer unchanged.")
# Deterministic crisis words, checked before Decider (a hard rule never waits for a probability).
# Added after Decider scored an indirect overdose message at P(self_harm)=0.15-0.38 in our eval (case 40).
CRISIS_WORDS = re.compile(r"kill myself|end(ing)? (it|my life)|don'?t want to (live|wake up)|so i (don'?t|won'?t) wake up|"
                          r"whole (box|bottle|pack)|overdose|suicid", re.I)
LABELS = {
    "agent": "Who should answer?", "topic_check": "Second look: what is the user asking for?",
    "emergency_now": "Emergency sign?", "self_harm": "Self-harm risk?", "special_population": "Child, pregnant or very old?",
    "manipulation": "Trying to change the rules?", "mentions_symptom": "Mentions a body problem?",
    "route": "Medicine here, or see a doctor?", "genital_area": "Genital area?", "anal_area": "Anal area?",
    "bleeding": "Any blood or bleeding?", "strong_pain": "Strong pain?", "long_duration": "Lasting a long time?",
    "on_topic": "Reply answers what was asked?", "dose_or_diagnosis": "Reply gives a dose or names an illness?",
    "crisis words": "Crisis words in the message?", "care_rule": "Hard safety rule", "grounding check": "Facts and numbers check",
    "reply check": "Safety wording check",
}
NEVER_ASK_LLM = {"emergency_now", "self_harm"}  # middle band on these = yes, never a Bedrock tie-break
SAFER_IF_HESITANT = {"emergency_now", "self_harm", "special_population", "mentions_symptom", "manipulation"}


class _Cached:
    """Lets the Care Router InterventionHandler reuse this turn's decision instead of calling Decider twice."""
    def __init__(self):
        self.decision = None

    def decide(self, text):
        return self.decision


_ENGINE = None


def load_engine():
    """One Decider engine per process. bf16 on CPU by default (~3.7 GB peak instead of ~9 GB in fp32)."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    import torch
    from strands_decider import infer
    torch.set_num_threads(int(os.environ.get("DECIDER_THREADS", min(6, os.cpu_count() or 1))))
    if os.environ.get("DECIDER_DTYPE", "bf16") == "bf16":
        # strands-decider 0.2.0 upcasts the torso to fp32 on CPU for speed; skip that to halve memory.
        infer.SystemOneEngine._upcast_torso_for_cpu = lambda self: None
    _ENGINE = infer.load_engine(os.environ.get("DECIDER_CHECKPOINT", DEFAULT_CHECKPOINT), device=os.environ.get("DECIDER_DEVICE", "cpu"))
    return _ENGINE


class PharmacyFloor:
    def __init__(self, engine=None):
        self.engine = engine or load_engine()
        self.turn_router = TurnRouter(self.engine)
        self.added = self.turn_router.added
        from strands_decider.schema import NoulQuestion
        from recipes.turn_router.questions import load_added
        self.added_symptom = load_added("symptom")
        self.care_router = CareRouter(self.engine, extra={k: NoulQuestion(instructions=a["question"], criteria={
            "true": a["true"], "false": a["false"]}) for k, a in self.added_symptom.items()},
            extra_danger=tuple(k for k, a in self.added_symptom.items() if a["on_yes"] == "danger_sign"))
        self.cached = _Cached()
        self.specialists = Specialists(CareRouterIntervention(self.cached))

    # ---- middle band: an unsure answer gets a second look from Bedrock (or one step safer offline)
    def _judge(self, key: str, text: str, log: TurnLog, td) -> None:
        # Emergency and self-harm never go to an LLM: unsure = yes (fixed emergency / crisis-support message).
        if key in NEVER_ASK_LLM or self.added.get(key, {}).get("on_yes") in NEVER_ASK_LLM:
            td.flags[key] = 1.0
            log.add("second_look", key, "yes (unsure -> safe side, no LLM)", None, "python-rule",
                    label=f"Unsure: {LABELS.get(key, key)} -> treat as yes", outcome=C.STOP)
            return
        if mode() == "bedrock":
            if key == "agent":
                q = "Which ONE staff member should answer? Options: " + ", ".join(AGENTS) + ". Reply with the option only."
            else:
                q = (SAFETY_ALL[key][0] if key in SAFETY_ALL else self.added[key]["question"]) + " Reply with yes or no only."
            t0 = time.perf_counter()
            try:
                jm = make_model(max_tokens=20); jm.label = f"hesitation:{key}"
                out = str(Agent(model=jm, system_prompt="You are a strict classifier.",
                                callback_handler=None)(f"Message: {text}\n{q}")).strip().lower()
            except Exception as e:  # budget, max tokens, access: fall through to the safer default
                out = f"error:{type(e).__name__}"
            ms = (time.perf_counter() - t0) * 1000
            if key == "agent":
                pick = next((a for a in AGENTS if a in out), None)
                if pick:
                    td.agent = pick
                log.add("second_look", key, pick or f"unparsed:{out[:20]}", None, "bedrock", ms,
                        label="Bedrock tie-break: who should answer?", outcome=C.SECOND_LOOK)
                return
            if out.startswith(("yes", "no")):
                td.flags[key] = 1.0 if out.startswith("yes") else 0.0
                log.add("second_look", key, out[:3], None, "bedrock", ms,
                        label=f"Bedrock tie-break: {LABELS.get(key, key)}", outcome=C.SECOND_LOOK)
                return
        # offline / unparseable: one step safer
        if key == "agent":
            if td.flags.get("mentions_symptom", 0) >= 0.3:
                td.agent = "symptoms"
            log.add("second_look", key, td.agent, td.agent_confidence, "stub-safer-default",
                    label="Unsure who should answer: safer default", outcome=C.SAFER)
        elif key in SAFER_IF_HESITANT or key in self.added:
            td.flags[key] = 1.0
            log.add("second_look", key, "yes (safer)", None, "stub-safer-default",
                    label=f"Unsure: {LABELS.get(key, key)} -> assume yes", outcome=C.SAFER)

    def _handoff(self, log, why, t_start, care=None):
        log.handoff = why
        log.add("handoff", "pharmacist", why, None, "python-rule", label="Hand off to the pharmacist", outcome=C.HANDOFF)
        text = C.PHARMACIST_HANDOFF
        if care is not None and care.route == "see_doctor":  # a handoff never drops the see-a-doctor advice
            text += " " + FIXED["see_doctor"]
        return self._finish(log, "pharmacist", text, t_start)

    def handle(self, text: str) -> TurnLog:
        t_start = time.perf_counter()
        log = TurnLog(user=text)
        L = lambda k: LABELS.get(k, self.added.get(k, {}).get("label", k))  # noqa: E731
        if os.environ.get("FLOOR_CRISIS_WORDS", "on") != "off" and CRISIS_WORDS.search(text):
            log.add("safety", "crisis words", "found", None, "python-rule", label=L("crisis words"), outcome=C.STOP)
            return self._finish(log, "emergency", FIXED["self_harm"], t_start)

        # Step 1 - who should answer? (one Decider call also carries every safety question)
        td = self.turn_router.decide(text)
        ab = C.band(td.agent_confidence, C.AGENT_PROCEED, C.AGENT_HANDOFF)
        log.add("who", "agent", td.agent, td.agent_confidence, "decider-2b", td.ms, label=L("agent"),
                outcome={"proceed": C.PROCEED, "middle": C.SECOND_LOOK, "handoff": C.HANDOFF}[ab])

        # Step 2 - safety checks, every turn
        for k, p in list(td.flags.items()):
            fb = C.flag_band(p) if k not in self.added else ("yes" if p >= self.added[k]["yes_at"] else "no")
            stop_like = k in ("emergency_now", "self_harm") or k in self.added
            out = {"yes": C.STOP if stop_like else C.SAFER, "unsure": C.SECOND_LOOK, "no": C.PROCEED}[fb]
            log.add("safety", k, f"{fb} (P={p:.2f})", p, "decider-2b", label=L(k), outcome=out)
        for k in td.hesitant_on:
            if k != "agent":
                self._judge(k, text, log, td)
        for k, a in self.added.items():  # an added question maps onto an existing safety outcome
            if td.flags.get(k, 0.0) >= a["yes_at"] and not td.yes(a["on_yes"]):
                td.flags[a["on_yes"]] = 1.0
                log.add("safety", "added_question", f"{k} -> {a['on_yes']}", None, "python-rule",
                        label=f"Added question '{k}' says yes: treat as {a['on_yes']}", outcome=C.STOP)
        for k in ("self_harm", "emergency_now"):
            if td.yes(k):
                fixed = "self_harm" if k == "self_harm" else "emergency"
                log.add("safety", "fixed_reply", f"fixed {fixed} reply, no LLM", None, "python-rule",
                        label="Send the fixed safety message", outcome=C.STOP)
                return self._finish(log, "emergency", FIXED[fixed], t_start)
        agent = td.agent
        if td.yes("manipulation"):
            agent = "small_talk"
            log.add("safety", "manipulation_rule", "small_talk (no rule changes)", None, "python-rule",
                    label="Rule change attempt: small talk only", outcome=C.SAFER)

        # Step 3 - before the helper (tool) runs: a second, independent Decider look at the request.
        # Agree -> proceed (even a low-confidence pick is confirmed). Disagree -> Bedrock tie-break,
        # or the pharmacist when the first pick was already below AGENT_HANDOFF.
        ans, ms = ask(self.engine, text, TOOL_CHECK)
        chk = ans["topic_check"]
        agree = chk.choice == agent
        log.add("before_tool", "topic_check", chk.choice + (" (agrees)" if agree else f" (differs from {agent})"),
                chk.confidence, "decider-2b", ms, label=L("topic_check"), outcome=C.PROCEED if agree else C.SECOND_LOOK)
        if not agree and agent != "small_talk":
            if ab == "handoff":
                return self._handoff(log, f"unsure who should answer ({td.agent_confidence:.2f}, second look disagreed)", t_start)
            td.agent = agent
            self._judge("agent", text, log, td)
            agent = td.agent

        # Step 4 - symptom check: Care Router whenever a symptom is mentioned, whatever the helper
        care = None
        symptom_turn = agent == "symptoms" or td.yes("mentions_symptom")
        self.specialists.care_note = None
        if symptom_turn:
            care = self.care_router.decide(text)
            rule_decided = any(not r.startswith(("low confidence", "Decider:")) for r in care.reasons)
            rb = C.band(care.route_confidence, C.ROUTE_PROCEED, C.ROUTE_HANDOFF)
            log.add("symptom", "route", care.route + ("" if care.route == care.decider_route else f" (Decider said {care.decider_route})"),
                    care.route_confidence, "decider-2b", care.latency_ms,
                    label=L("route"), outcome=C.PROCEED if rb == "proceed" or rule_decided else
                    (C.SAFER if rb == "middle" else C.HANDOFF))
            for k, p in care.flags.items():
                log.add("symptom", k, f"{C.flag_band(p)} (P={p:.2f})", p, "decider-2b", label=L(k),
                        outcome=C.SAFER if p >= 0.5 else C.PROCEED)
            for r in care.reasons:
                log.add("symptom", "care_rule", r, None, "python-rule", label=L("care_rule"),
                        outcome=C.SAFER if ("see_doctor" in r or "->" in r or "doctor" in r) else C.PROCEED)
            if td.yes("special_population") and care.route == "otc_ok":
                care.route = "see_doctor"
                log.add("symptom", "special_population", "otc_ok -> see_doctor (pharmacist/doctor in person)", None,
                        "python-rule", label="Child, pregnant or very old: see a doctor", outcome=C.SAFER)
            if care.route == "emergency":
                return self._finish(log, "emergency", FIXED["emergency"], t_start)
            if rb == "handoff" and not rule_decided:
                return self._handoff(log, f"unsure about the symptom route (confidence {care.route_confidence:.2f})", t_start)
            self.cached.decision = care
            if agent != "symptoms":
                self.specialists.care_note = f"[CARE ROUTE - fixed by policy: {care.route}] {ROUTE_INSTRUCTIONS[care.route]}"

        # Step 5 - writing: the chosen helper (a Strands tool) writes with Bedrock; Decider's pick is forced by the handler.
        guard = TurnRouterIntervention(lambda _t: agent)
        reception = Agent(model=make_model(), system_prompt=RECEPTION_PROMPT, tools=self.specialists.tools(),
                          interventions=[guard], callback_handler=None)
        self.specialists.last_output = None
        t0 = time.perf_counter()
        if os.environ.get("FLOOR_RECEPTION", "direct") == "llm":
            reception_text = str(reception(text)).strip()
            how = "reception LLM (2 calls) + helper"
        else:
            # Decider already chose the tool, so no LLM call is spent on tool selection.
            # The call still goes through the tool executor, so before_tool_call on the guard still runs.
            guard.chosen = agent
            res = getattr(reception.tool, agent)(message=text, record_direct_tool_call=False)
            reception_text = " ".join(c.get("text", "") for c in res.get("content", []))
            how = "helper writes the reply"
        log.add("write", f"{agent} helper", how, None, "bedrock" if mode() == "bedrock" else "stub",
                (time.perf_counter() - t0) * 1000, label=f"Write the reply ({agent.replace('_', ' ')})", outcome=C.INFO)
        reply = self.specialists.last_output or re.sub(r"<thinking>.*?</thinking>", "", reception_text, flags=re.S).strip()

        # Step 6 - before the reply goes out: Python gates + typed Decider checks on the written reply
        reply, sv = specialist_gate(reply, agent=agent, store_facts=self.specialists.store_facts)
        log.add("before_reply", "grounding check", sv, None, "python-rule", label=L("grounding check"),
                outcome=C.PROCEED if sv == "pass" else C.SAFER)
        self.specialists.store_facts = None
        if agent != "small_talk":
            ans, ms = ask(self.engine, f"User message: {text}\nReply: {reply}", REPLY_CHECK)
            pt = ans["on_topic"].noul
            ob = C.band(pt, C.REPLY_ON_TOPIC_PROCEED, C.REPLY_ON_TOPIC_HANDOFF)
            log.add("before_reply", "on_topic", f"{'yes' if pt >= 0.5 else 'no'} (P={pt:.2f})", pt, "decider-2b", ms,
                    label=L("on_topic"), outcome={"proceed": C.PROCEED, "middle": C.SAFER, "handoff": C.HANDOFF}[ob])
            if ob == "handoff":
                return self._handoff(log, f"reply did not match the question (P={pt:.2f})", t_start, care)
            if agent in ("symptoms", "medicine_question", "feelings"):
                pd = ans["dose_or_diagnosis"].noul
                bad = pd >= C.REPLY_DOSE_DX_HANDOFF
                log.add("before_reply", "dose_or_diagnosis", f"{'yes' if bad else 'no'} (P={pd:.2f})", pd, "decider-2b",
                        label=L("dose_or_diagnosis"), outcome=C.HANDOFF if bad else C.PROCEED)
                if bad:
                    return self._handoff(log, f"reply may state a dose or an illness (P={pd:.2f})", t_start, care)
            if ob == "middle" and "pharmacist" not in reply.lower():
                reply = reply.rstrip() + " " + C.PHARMACIST_NOTE
        reply, verdict = final_gate(reply, care_route=care.route if care else None, symptom_turn=symptom_turn)
        log.add("before_reply", "reply check", verdict, None, "python-rule", label=L("reply check"),
                outcome=C.PROCEED if verdict == "pass" else C.SAFER)
        return self._finish(log, agent, reply, t_start)

    def _finish(self, log, agent, reply, t_start):
        log.final_agent, log.reply = agent, reply
        log.total_ms = (time.perf_counter() - t_start) * 1000
        return log

    @staticmethod
    def llm_label():
        return model_label()
