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

from .decision_log import TurnLog
from .final_gate import FIXED, final_gate
from .llm import make_model, mode, model_label
from .specialists import Specialists

RECEPTION_PROMPT = ("You are the reception desk of a drugstore floor. You never answer yourself: you call exactly one "
                    "specialist tool, the one named in the [ROUTE] note, and return its answer unchanged.")
# Deterministic crisis words, checked before Decider (a hard rule never waits for a probability).
# Added after Decider scored an indirect overdose message at P(self_harm)=0.15-0.38 in our eval (case 40).
CRISIS_WORDS = re.compile(r"kill myself|end(ing)? (it|my life)|don'?t want to (live|wake up)|so i (don'?t|won'?t) wake up|"
                          r"whole (box|bottle|pack)|overdose|suicid", re.I)
SAFER_IF_HESITANT = {"emergency_now", "self_harm", "special_population", "mentions_symptom", "manipulation"}


class _Cached:
    """Lets the Care Router InterventionHandler reuse this turn's decision instead of calling Decider twice."""
    def __init__(self):
        self.decision = None

    def decide(self, text):
        return self.decision


def load_engine():
    from strands_decider.infer import load_engine as _le
    return _le(os.environ.get("DECIDER_CHECKPOINT", DEFAULT_CHECKPOINT), device=os.environ.get("DECIDER_DEVICE", "cpu"))


class PharmacyFloor:
    def __init__(self, engine=None):
        self.engine = engine or load_engine()
        self.turn_router = TurnRouter(self.engine)
        self.care_router = CareRouter(self.engine)
        self.cached = _Cached()
        self.specialists = Specialists(CareRouterIntervention(self.cached))

    # ---- hesitation band: unsure Decider answers go to Bedrock (or a safer default offline)
    def _judge(self, key: str, text: str, log: TurnLog, td) -> None:
        if mode() == "bedrock":
            if key == "agent":
                q = "Which ONE staff member should answer? Options: " + ", ".join(AGENTS) + ". Reply with the option only."
            else:
                q = SAFETY_ALL[key][0] + " Reply with yes or no only."
            t0 = time.perf_counter()
            out = str(Agent(model=make_model(max_tokens=10), system_prompt="You are a strict classifier.",
                            callback_handler=None)(f"Message: {text}\n{q}")).strip().lower()
            ms = (time.perf_counter() - t0) * 1000
            if key == "agent":
                pick = next((a for a in AGENTS if a in out), None)
                if pick:
                    td.agent = pick
                log.add("hesitation", key, pick or f"unparsed:{out[:20]}", None, "bedrock", ms)
                return
            if out.startswith(("yes", "no")):
                td.flags[key] = 1.0 if out.startswith("yes") else 0.0
                log.add("hesitation", key, out[:3], None, "bedrock", ms)
                return
        # offline / unparseable: one step safer
        if key == "agent":
            if td.flags.get("mentions_symptom", 0) >= 0.3:
                td.agent = "symptoms"
            log.add("hesitation", key, td.agent, td.agent_confidence, "stub-safer-default")
        elif key in SAFER_IF_HESITANT:
            td.flags[key] = 1.0
            log.add("hesitation", key, "yes (safer)", None, "stub-safer-default")

    def handle(self, text: str) -> TurnLog:
        t_start = time.perf_counter()
        log = TurnLog(user=text)
        if CRISIS_WORDS.search(text):
            log.add("pre_gate", "crisis words", "self_harm", None, "python-rule")
            return self._finish(log, "emergency", FIXED["self_harm"], t_start)
        td = self.turn_router.decide(text)
        log.add("turn_router", "agent", td.agent, td.agent_confidence, "decider-2b", td.ms)
        for k, p in td.flags.items():
            log.add("turn_router", k, f"P(yes)={p:.2f}", p, "decider-2b")
        for k in td.hesitant_on:
            self._judge(k, text, log, td)

        # Safety layer 1: emergency / self-harm -> fixed text, no LLM.
        for k in ("self_harm", "emergency_now"):
            if td.yes(k):
                fixed = "self_harm" if k == "self_harm" else "emergency"
                log.add("policy", k, f"fixed {fixed} reply", None, "python-rule")
                return self._finish(log, "emergency", FIXED[fixed], t_start)

        agent = td.agent
        if td.yes("manipulation"):
            agent = "small_talk"
            log.add("policy", "manipulation", "small_talk (no rule changes)", None, "python-rule")

        # Safety layer 2: Care Router whenever a symptom is mentioned, whatever the specialist.
        care = None
        symptom_turn = agent == "symptoms" or td.yes("mentions_symptom")
        self.specialists.care_note = None
        if symptom_turn:
            care = self.care_router.decide(text)
            log.add("care_router", "route", care.route, care.route_confidence, "decider-2b", care.latency_ms)
            for k, p in care.flags.items():
                log.add("care_router", k, f"P(yes)={p:.2f}", p, "decider-2b")
            for r in care.reasons:
                log.add("policy", "care_rule", r, None, "python-rule")
            if td.yes("special_population") and care.route == "otc_ok":
                care.route = "see_doctor"
                log.add("policy", "special_population", "otc_ok -> see_doctor (pharmacist/doctor in person)", None, "python-rule")
            if care.route == "emergency":
                return self._finish(log, "emergency", FIXED["emergency"], t_start)
            self.cached.decision = care
            if agent != "symptoms":
                self.specialists.care_note = f"[CARE ROUTE — fixed by policy: {care.route}] {ROUTE_INSTRUCTIONS[care.route]}"

        # Tool selection: reception agent with five specialist tools; Decider's pick is forced by the handler.
        guard = TurnRouterIntervention(lambda _t: agent)
        reception = Agent(model=make_model(), system_prompt=RECEPTION_PROMPT, tools=self.specialists.tools(),
                          interventions=[guard], callback_handler=None)
        reply = str(reception(text)).strip()
        reply = re.sub(r"\s*\[ROUTE[^\]]*\]", "", reply)

        reply, verdict = final_gate(reply, care_route=care.route if care else None, symptom_turn=symptom_turn)
        log.add("final_gate", "reply check", verdict, None, "python-rule")
        return self._finish(log, agent, reply, t_start)

    def _finish(self, log, agent, reply, t_start):
        log.final_agent, log.reply = agent, reply
        log.total_ms = (time.perf_counter() - t_start) * 1000
        return log

    @staticmethod
    def llm_label():
        return model_label()
