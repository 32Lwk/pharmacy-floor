"""Thin wrapper around the local Strands Decider 2B engine.

Real API (strands-decider 0.2.0): `strands_decider.infer.load_engine(checkpoint, device=...)`
returns a `SystemOneEngine`; `engine.ask(state, questions)` returns a `SystemOneResponse`
whose `.answers[name]` is a `ChoiceAnswer` (.choice, .probabilities, .confidence)
or a `NoulAnswer` (.noul = P(true)).
"""

from __future__ import annotations

import time

from .policy import CareDecision, apply_policy
from .questions import BLEEDING, FLAG_KEYS, QUESTIONS

DEFAULT_CHECKPOINT = "StrandsAgents/strands-decider-2B-hobson-v19"


class CareRouter:
    def __init__(self, engine=None, *, checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cpu",
                 confidence_threshold: float | None = None, extra: dict | None = None,
                 extra_danger: tuple = ("bleeding",)):
        if engine is None:
            from strands_decider.infer import load_engine

            engine = load_engine(checkpoint, device=device)
        self.engine = engine
        self.kw = {} if confidence_threshold is None else {"confidence_threshold": confidence_threshold}
        # extra yes/no questions (e.g. from rules/added_questions.json): key -> NoulQuestion
        self.questions = {k: v for k, v in QUESTIONS.items() if k in ("route",) + FLAG_KEYS}
        self.questions.update({"bleeding": BLEEDING} if extra is None else extra)
        self.extra_danger = tuple(extra_danger)

    def raw(self, text: str):
        t0 = time.perf_counter()
        resp = self.engine.ask(text, self.questions)
        return resp, (time.perf_counter() - t0) * 1000

    def decide(self, text: str) -> CareDecision:
        resp, ms = self.raw(text)
        r = resp.answers["route"]
        flags = {k: a.noul for k, a in resp.answers.items() if a.type == "noul"}
        d = apply_policy(r.choice, r.confidence, dict(r.probabilities), flags, extra_danger=self.extra_danger, **self.kw)
        d.latency_ms = ms
        return d
