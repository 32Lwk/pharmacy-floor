"""TurnRouter: one Decider call per turn -> which specialist, plus safety flags, plus a hesitation verdict."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .questions import AGENT_HINTS, AGENTS, build_questions, load_added

# Thresholds live in ONE place: floor/confidence.py. A yes/no with FLAG_NO <= P < FLAG_YES is "unsure"
# (hesitant); P >= FLAG_YES is yes. The agent choice is not marked hesitant here: the floor compares its
# confidence with AGENT_PROCEED / AGENT_HANDOFF.
from floor.confidence import FLAG_NO, FLAG_YES


@dataclass
class TurnDecision:
    agent: str
    agent_confidence: float
    agent_probabilities: dict[str, float]
    flags: dict[str, float]
    hesitant_on: list[str] = field(default_factory=list)  # question keys inside the band
    ms: float = 0.0

    def yes(self, key: str) -> bool:
        return self.flags.get(key, 0.0) >= FLAG_YES


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
        if "agent_check" in ans and ans["agent_check"].choice != a.choice:
            d.hesitant_on.append("agent")
        d.hesitant_on += [k for k, p in flags.items() if FLAG_NO <= p < FLAG_YES and k not in AGENT_HINTS]
        return d


__all__ = ["TurnRouter", "TurnDecision", "AGENTS"]
