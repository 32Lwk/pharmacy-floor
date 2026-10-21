"""Fixed policy, enforced in plain Python on top of Decider's answers.

A hard rule never depends on a probability. Decider classifies messy words;
this code decides what is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .questions import DANGER_KEYS, ROUTES

DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # hand-picked; measure on your own traffic
# P(true) above which a yes/no flag counts as "yes". Hand-picked, asymmetric on purpose:
# missing a genital-area symptom is the costly error, so that flag trips earlier.
FLAG_THRESHOLDS = {"genital_area": 0.4}
DEFAULT_FLAG_THRESHOLD = 0.5


@dataclass
class CareDecision:
    route: str  # final route after policy
    decider_route: str  # what Decider picked
    route_confidence: float
    route_probabilities: dict[str, float]
    flags: dict[str, float]  # P(true) for each yes/no question
    reasons: list[str] = field(default_factory=list)
    latency_ms: float | None = None


def _safer(route: str) -> str:
    i = ROUTES.index(route)
    return ROUTES[min(i + 1, len(ROUTES) - 1)]


def apply_policy(decider_route, route_confidence, route_probabilities, flags, *,
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> CareDecision:
    d = CareDecision(decider_route, decider_route, route_confidence, route_probabilities, flags)
    yes = {k: v >= FLAG_THRESHOLDS.get(k, DEFAULT_FLAG_THRESHOLD) for k, v in flags.items()}
    danger = [k for k in DANGER_KEYS if yes.get(k)]
    low_conf = route_confidence < confidence_threshold

    if d.route == "emergency":
        d.reasons.append("Decider: emergency (never downgraded)")
    # Rule 1: genital area -> see_doctor, always. No OTC, no confidence override.
    elif yes.get("genital_area"):
        d.route = "see_doctor"
        d.reasons.append("genital area: always see a doctor (no OTC)")
        return d
    # Rule 2: hemorrhoids / anal itch -> OTC allowed unless bleeding, strong pain, long duration.
    elif yes.get("anal_area"):
        if danger:
            d.route = "see_doctor"
            d.reasons.append(f"anal area + danger sign {danger}: see a doctor")
        else:
            d.route = "otc_ok"
            d.reasons.append("anal area, no danger sign: OTC allowed (with see-a-doctor signs)")
            low_conf = False  # the rule, not the model, made this call
    elif danger and d.route == "otc_ok":
        d.route = "see_doctor"
        d.reasons.append(f"danger sign {danger}: OTC not offered")

    # Rule 3: low confidence -> one step toward the safer side.
    if low_conf:
        before = d.route
        d.route = _safer(d.route)
        if d.route != before:
            d.reasons.append(f"low confidence {route_confidence:.2f} < {confidence_threshold}: {before} -> {d.route}")
    return d
