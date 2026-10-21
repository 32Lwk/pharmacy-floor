"""Typed questions and option sets for the Care Router decision pattern.

Uses the real `strands_decider.schema` types (strands-decider 0.2.0):
`ChoiceQuestion` (pick one of N named options) and `NoulQuestion` (yes/no -> P(true)).
"""

from __future__ import annotations

from strands_decider.schema import ChoiceQuestion, NoulQuestion

ROUTES = ("otc_ok", "see_doctor", "emergency")  # ordered from least to most care

QUESTIONS = {
    "route": ChoiceQuestion(
        instructions="Where should this person go first for this health concern?",
        criteria={
            "otc_ok": "a mild, common, short-lived symptom that a drugstore (over-the-counter) medicine may help",
            "see_doctor": "the symptom needs a clinic visit: it is unusual, lasting, worsening, or involves a sensitive area",
            "emergency": "a possibly life-threatening sign that needs emergency care right now",
        },
    ),
    "genital_area": NoulQuestion(
        instructions="Is the symptom located on the sex organs (penis, testicles, vagina, vulva)? The anus, buttocks and bottom are NOT sex organs.",
        criteria={
            "true": "the symptom is on the penis, testicles, vagina or vulva",
            "false": "the symptom is elsewhere, including the anus or bottom",
        },
    ),
    "anal_area": NoulQuestion(
        instructions="Does the message describe hemorrhoids or itching or discomfort around the anus?",
        criteria={
            "true": "the user mentions hemorrhoids, the anus, or itching or pain around the bottom",
            "false": "nothing about hemorrhoids or the anal area is mentioned",
        },
    ),
    # Danger signs are asked as three atomic questions: one combined question
    # ("bleeding, strong pain, or long duration?") missed "blood on the toilet paper" in testing.
    "bleeding": NoulQuestion(
        instructions="Does the user report any blood or bleeding?",
        criteria={"true": "blood or bleeding is mentioned", "false": "no blood or bleeding is mentioned"},
    ),
    "strong_pain": NoulQuestion(
        instructions="Does the user describe the pain as strong, intense or severe?",
        criteria={
            "true": "the pain is described as strong, intense, severe or unbearable",
            "false": "no strong pain is described",
        },
    ),
    "long_duration": NoulQuestion(
        instructions="Has the symptom lasted more than one week?",
        criteria={
            "true": "the user says it has lasted more than a week (weeks or months)",
            "false": "it started within the last week, or no duration is given",
        },
    ),
}

FLAG_KEYS = ("genital_area", "anal_area", "bleeding", "strong_pain", "long_duration")
DANGER_KEYS = ("bleeding", "strong_pain", "long_duration")
