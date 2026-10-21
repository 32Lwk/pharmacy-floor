"""Baseline: keyword routing + the same Care Router policy with regex flags. No model."""
import re

from recipes.care_router.policy import apply_policy

EMERG = r"chest pain|can'?t breathe|face (is )?droop|slurred|throat feels tight|lips are swelling|don'?t want to live|end(ing)? it|kill myself|whole box"
STORE = r"where can i find|aisle|in stock|close|open|pay|price"
MED = r"ibuprofen|acetaminophen|paracetamol|antihistamine|medicine|syrup|cream|painkiller|tablets?|pills?"
FEEL = r"anxious|stress|lonely|worr|scared|down today|embarrass"
SMALL = r"^(hi|hello|thanks|thank you)|robot|weather"
SYMPT = r"headache|nose|sneez|itch|piles|hemorrhoid|heartburn|blood|bleed|pain|burning|cough|rash|sore"
GENITAL = r"genital|penis|testic|vagina|vulva|private parts"
ANAL = r"anus|anal|hemorrhoid|piles|bottom|stool"


def keyword_decide(text: str):
    t = text.lower()
    if re.search(EMERG, t):
        return "emergency", None
    agent = ("store" if re.search(STORE, t) else "medicine_question" if re.search(MED, t) else
             "feelings" if re.search(FEEL, t) else "symptoms" if re.search(SYMPT, t) else "small_talk")
    care = None
    if agent == "symptoms" or re.search(SYMPT, t):
        route = "see_doctor" if re.search(r"weeks?|months?|worse", t) else "otc_ok"
        flags = {"genital_area": float(bool(re.search(GENITAL, t))), "anal_area": float(bool(re.search(ANAL, t))),
                 "bleeding": float(bool(re.search(r"blood|bleed", t))), "strong_pain": float(bool(re.search(r"intense|severe|strong pain", t))),
                 "long_duration": float(bool(re.search(r"weeks?|months?|for days|a week", t)))}
        care = apply_policy(route, 1.0, {}, flags).route
    return agent, care
