"""Wiring: a Strands `InterventionHandler` that runs Care Router before the agent acts.

Real API (strands-agents 1.61.1): `strands.interventions.InterventionHandler`,
hook method `before_invocation(event)` (fires before any model call or tool),
returning `Proceed` / `Deny` / `Guide` / `Transform`. We return `Transform` to append the
fixed route to the user's turn so the LLM writes text *inside* the route and cannot pick it.
"""

from __future__ import annotations

from strands.interventions import InterventionHandler, Proceed, Transform

from .decider import CareRouter
from .policy import CareDecision

ROUTE_INSTRUCTIONS = {
    "otc_ok": "An over-the-counter medicine may be an option. Give general guidance and list signs that mean they should see a doctor.",
    "see_doctor": "Do NOT suggest any medicine. Kindly recommend visiting a clinic and say which kind of clinic may fit.",
    "emergency": "Tell them to get emergency care now (call 119 in Japan). Keep it very short. Do NOT suggest any medicine.",
}


def _last_user_text(messages) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return " ".join(b["text"] for b in m.get("content", []) if "text" in b)
    return ""


class CareRouterIntervention(InterventionHandler):
    name = "care-router"

    @property
    def on_error(self):
        return "throw"  # fail closed: no decision, no reply

    def __init__(self, router: CareRouter):
        self.router = router
        self.last_decision: CareDecision | None = None

    def before_invocation(self, event, **kwargs):
        text = _last_user_text(event.messages)
        if not text:
            return Proceed(reason="no user text")
        d = self.router.decide(text)
        self.last_decision = d
        note = (
            f"[CARE ROUTE — fixed by policy, you may not change it: {d.route}] "
            f"{ROUTE_INSTRUCTIONS[d.route]} Say this is guidance, not a diagnosis."
        )

        def apply(ev):
            for m in reversed(ev.messages):
                if m.get("role") == "user":
                    m["content"] = list(m["content"]) + [{"text": note}]
                    break

        return Transform(apply=apply, reason=f"route={d.route}")
