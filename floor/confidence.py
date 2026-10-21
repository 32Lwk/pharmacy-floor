"""Confidence-driven behavior. Every Decider answer comes with a calibrated probability; these
named thresholds turn it into one of three moves:

  PROCEED  - confident enough: act on the answer.
  MIDDLE   - unsure: take one step safer, or get a second look (a second Decider question, then Bedrock).
  HANDOFF  - too unsure to act: hand the customer to the pharmacist at the counter.

All values are hand-picked from the probabilities we saw on the dev/demo messages, not tuned on the
held-out set. Change them here; tests in tests/test_confidence.py pin the behavior.
"""
from __future__ import annotations

# 1. Who should answer? (choice over 5 helpers; uniform would be 0.20)
AGENT_PROCEED = 0.60      # >= : use the helper Decider picked
AGENT_HANDOFF = 0.35      # <  : nobody fits well enough -> pharmacist. In between: second look before the tool call.

# 2. Safety yes/no flags (P(yes))
FLAG_YES = 0.60           # >= : treat as yes
FLAG_NO = 0.40            # <  : treat as no. In between: ask Bedrock once, or assume yes (safer) offline.

# 3. Symptom check: Care Router route confidence (choice over otc_ok / see_doctor / emergency; uniform 0.33)
ROUTE_PROCEED = 0.50      # >= : use the route (Care Router's own rule steps safer below this)
ROUTE_HANDOFF = 0.40      # <  : and no hard rule decided it -> pharmacist

# 4. Before the reply goes out (P(yes) of the reply checks)
REPLY_ON_TOPIC_PROCEED = 0.60   # >= : reply answers what the user asked
REPLY_ON_TOPIC_HANDOFF = 0.40   # <  : reply drifted to another problem/product -> pharmacist; between: add a pharmacist note
REPLY_DOSE_DX_HANDOFF = 0.60    # >= : reply states a dose or names a likely illness -> pharmacist (regex gates run too)

PROCEED, SAFER, SECOND_LOOK, HANDOFF, STOP, INFO = "proceed", "safer", "second_look", "handoff", "stop", "info"


def band(p: float, proceed: float, handoff: float) -> str:
    """'proceed' | 'middle' | 'handoff' for a confidence where higher is better."""
    if p >= proceed:
        return "proceed"
    return "handoff" if p < handoff else "middle"


def flag_band(p: float) -> str:
    """'yes' | 'unsure' | 'no' for a yes/no probability."""
    return "yes" if p >= FLAG_YES else ("no" if p < FLAG_NO else "unsure")


PHARMACIST_HANDOFF = ("I'd like our pharmacist to help you with this one, so you get the right answer. "
                      "Please show this message at the counter, or press 'Ask the pharmacist'.")
PHARMACIST_NOTE = "If anything is unclear, our pharmacist at the counter is happy to check it with you."
