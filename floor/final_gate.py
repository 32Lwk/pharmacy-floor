"""Final check before any reply leaves the floor (plain Python, no model)."""
from __future__ import annotations

import re

OTC_PUSH = re.compile(r"over-the-counter( \w+)? (may|might|can|could|will) help|you (can|could|may) (take|try|buy|use) |i (would )?(recommend|suggest) (taking|trying|using|buying)", re.I)
FIXED = {
    "emergency": "Please get emergency care now. In Japan, call 119 (ambulance). If someone is with you, ask them to help. I can't help with medicine for this.",
    "self_harm": "I'm really sorry you're feeling this way. You don't have to go through it alone. If you might act on these thoughts, call 119 now. You can also talk to someone at Japan's Inochi no Denwa: 0570-783-556.",
    "see_doctor": "For this, please see a doctor rather than choosing a medicine here. Our pharmacist can tell you which kind of clinic may fit. This is guidance, not a diagnosis.",
}
DISCLAIMER = "This is guidance, not a diagnosis."


def final_gate(reply: str, *, care_route: str | None, symptom_turn: bool) -> tuple[str, str]:
    """Returns (reply, verdict). Never lets an OTC push through on a see_doctor/emergency turn."""
    if care_route in ("see_doctor", "emergency") and OTC_PUSH.search(reply):
        return FIXED[care_route], f"replaced: OTC wording on a {care_route} turn"
    if care_route == "see_doctor" and not re.search(r"doctor|clinic", reply, re.I):
        return reply.rstrip() + " " + FIXED["see_doctor"], "appended see-a-doctor (reply did not say it)"
    if symptom_turn and "not a diagnosis" not in reply.lower():
        return reply.rstrip() + " " + DISCLAIMER, "appended disclaimer"
    return reply, "pass"
