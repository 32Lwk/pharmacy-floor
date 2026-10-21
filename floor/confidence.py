"""Confidence-driven behavior. Every Decider answer comes with a calibrated probability; these
named thresholds turn it into one of three moves:

  PROCEED  - confident enough: act on the answer.
  MIDDLE   - unsure: take one step safer, or get a second look (a second Decider question, then Bedrock).
  HANDOFF  - too unsure to act: hand the customer to the pharmacist at the counter.

This file is the ONE place for thresholds: recipes/turn_router and recipes/care_router import from it.

Band convention (used everywhere, code and docs), for a probability p and a pair (low, high):
  p >= high        -> the "high" side (proceed / yes)
  low <= p < high  -> the middle (unsure)
  p < low          -> the "low" side (handoff / no)

All values are hand-picked from the probabilities we saw on the dev/demo messages, not tuned on the
held-out set. Change them here; tests in tests/test_confidence.py pin the behavior.
"""
from __future__ import annotations

# Step 1 - who should answer? (choice over 5 helpers; uniform would be 0.20)
AGENT_PROCEED = 0.60      # p >= 0.60: use the pick, no second look
AGENT_HANDOFF = 0.35      # p <  0.35: second look; if it disagrees -> pharmacist.  0.35 <= p < 0.60: second look; disagree -> Bedrock tie-break

# Step 2 - safety yes/no flags, P(yes)
FLAG_YES = 0.60           # p >= 0.60: yes
FLAG_NO = 0.40            # p <  0.40: no.  0.40 <= p < 0.60: unsure -> emergency/self-harm: yes; others: Bedrock tie-break (offline: yes)
SYMPTOM_LEAN = 0.30       # offline tie-break for "who answers": mentions_symptom >= 0.30 -> symptoms helper

# Step 4 - symptom check (Care Router). Route = choice over otc_ok / see_doctor / emergency; uniform 0.33
ROUTE_PROCEED = 0.50      # p >= 0.50: use the route.  0.40 <= p < 0.50: one step safer
ROUTE_HANDOFF = 0.40      # p <  0.40 and no hard rule decided it: pharmacist (else one step safer)
CARE_FLAG_YES = 0.50      # Care Router yes/no flags (anal_area, bleeding, strong_pain, long_duration, added questions): p >= 0.50 is yes
CARE_FLAG_YES_GENITAL = 0.40  # genital_area trips earlier on purpose: missing it is the costly error

# Step 6 - before the reply goes out, P(yes) of the reply checks
REPLY_ON_TOPIC_PROCEED = 0.60   # p >= 0.60: reply answers what was asked
REPLY_ON_TOPIC_HANDOFF = 0.40   # p <  0.40: pharmacist.  0.40 <= p < 0.60: add a pharmacist note
REPLY_DOSE_DX_HANDOFF = 0.60    # p >= 0.60: reply states a dose or names a likely illness -> pharmacist (regex gates run too)

STEPS = {1: "Who should answer?", 2: "Safety checks", 3: "Before the helper runs", 4: "Symptom check",
         5: "Writing the reply", 6: "Before the reply goes out"}

PROCEED, SAFER, SECOND_LOOK, HANDOFF, STOP, INFO = "proceed", "safer", "second_look", "handoff", "stop", "info"


def band(p: float, proceed: float, handoff: float) -> str:
    """'proceed' | 'middle' | 'handoff' for a confidence where higher is better."""
    if p >= proceed:
        return "proceed"
    return "handoff" if p < handoff else "middle"


def flag_band(p: float, yes: float = FLAG_YES, no: float = FLAG_NO) -> str:
    """'yes' (p >= yes) | 'unsure' (no <= p < yes) | 'no' (p < no)."""
    return "yes" if p >= yes else ("no" if p < no else "unsure")


PHARMACIST_HANDOFF = ("I'd like our pharmacist to help you with this one, so you get the right answer. "
                      "Please show this message at the counter, or press 'Ask the pharmacist'.")
PHARMACIST_NOTE = "If anything is unclear, our pharmacist at the counter is happy to check it with you."
