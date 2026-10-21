"""Per-turn decision log shown in the demo panel."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Entry:
    step: str          # e.g. "turn_router", "care_router", "hesitation", "policy"
    question: str
    answer: str
    confidence: float | None
    decided_by: str    # "decider-2b" | "bedrock" | "python-rule" | "stub-safer-default"
    ms: float | None = None


@dataclass
class TurnLog:
    user: str
    entries: list[Entry] = field(default_factory=list)
    final_agent: str = ""
    reply: str = ""
    total_ms: float = 0.0

    def add(self, *a, **k):
        self.entries.append(Entry(*a, **k))

    def to_dict(self):
        return asdict(self)
