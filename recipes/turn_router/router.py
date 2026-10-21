"""TurnRouter: one Decider call per turn -> which specialist, plus safety flags, plus a hesitation verdict."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .questions import AGENTS, build_questions, load_added

# Hand-picked, not tuned on the eval set.
AGENT_CONFIDENCE_FLOOR = 0.5           # below -> hesitant about the agent choice
SAFETY_BAND = (0.4, 0.6)               # P(true) inside -> hesitant. Hand-picked: unrelated yes/no questions sat at 0.1-0.36 in our runs
SAFETY_YES = 0.6                       # at/above -> flag is yes


@dataclass
class TurnDecision:
    agent: str
    agent_confidence: float
    agent_probabilities: dict[str, float]
    flags: dict[str, float]
    hesitant_on: list[str] = field(default_factory=list)  # question keys inside the band
    ms: float = 0.0

    def yes(self, key: str) -> bool:
        return self.flags.get(key, 0.0) >= SAFETY_YES


class TurnRouter:
    def __init__(self, engine, *, full: bool = False, added: dict | None = None):
        self.engine = engine
        self.added = load_added() if added is None else added
        self.questions = build_questions(full, self.added)

    def decide(self, text: str) -> TurnDecision:
        t0 = time.perf_counter()
        ans = self.engine.ask(text, self.questions).answers
        ms = (time.perf_counter() - t0) * 1000
        a = ans["agent"]
        flags = {k: v.noul for k, v in ans.items() if v.type == "noul"}
        d = TurnDecision(a.choice, a.confidence, dict(a.probabilities), flags, ms=ms)
        if a.confidence < AGENT_CONFIDENCE_FLOOR:
            d.hesitant_on.append("agent")
        if "agent_check" in ans and ans["agent_check"].choice != a.choice:
            d.hesitant_on.append("agent")
        d.hesitant_on += [k for k, p in flags.items() if SAFETY_BAND[0] <= p < SAFETY_BAND[1]]
        return d


__all__ = ["TurnRouter", "TurnDecision", "AGENTS"]
