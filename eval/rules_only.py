"""Baseline: keyword rules only (no model). Same fixed policy, flags from regexes."""

import re

from patterns.care_router.policy import apply_policy

EMERG = r"chest pain|can'?t breathe|cannot breathe|unconscious|passed out|seizure|face (is )?droop|slurred|vomiting blood|anaphyla|throat (is )?swelling"
DOCTOR = r"fever.*(\d+ days|week)|weeks?|month|worse|spreading|lump|pregnan|infant|baby"
GENITAL = r"genital|penis|testic|vagina|vulva|groin"
ANAL = r"anus|anal|hemorrhoid|haemorrhoid|piles|bottom"
BLEED = r"blood|bleed"
PAIN = r"severe|strong pain|terrible pain|unbearable|intense"
LONG = r"weeks?|months?"


def rules_decide(text: str):
    t = text.lower()
    route = "emergency" if re.search(EMERG, t) else "see_doctor" if re.search(DOCTOR, t) else "otc_ok"
    flags = {k: float(bool(re.search(p, t))) for k, p in
             (("genital_area", GENITAL), ("anal_area", ANAL), ("bleeding", BLEED), ("strong_pain", PAIN), ("long_duration", LONG))}
    return apply_policy(route, 1.0, {route: 1.0}, flags)
