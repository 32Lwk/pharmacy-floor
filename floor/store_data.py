"""DEMO STORE — made-up data for the hackathon demo. Not a real shop."""
STORE_NAME = "Demo Store (fictional)"
AISLES = {
    "cold": "Aisle 2 (cold & flu)", "cough": "Aisle 2 (cold & flu)", "allergy": "Aisle 3 (allergy)", "hay fever": "Aisle 3 (allergy)",
    "pain": "Aisle 1 (pain relief)", "headache": "Aisle 1 (pain relief)", "stomach": "Aisle 4 (stomach & bowel)",
    "eye": "Aisle 5 (eye care)", "skin": "Aisle 6 (skin care)", "bandage": "Aisle 6 (first aid)",
}
HOURS = "Demo Store hours: 9:00-21:00 every day."
PAYMENT = "Demo Store payment: cash, credit card, IC card."


def lookup(message: str) -> str:
    t = message.lower()
    hits = sorted({v for k, v in AISLES.items() if k in t})
    facts = [f"This is {STORE_NAME}.", HOURS, PAYMENT]
    facts.append(f"Aisles for this request: {', '.join(hits)}." if hits else "No aisle listed for this request; staff at the counter can help.")
    return " ".join(facts)
