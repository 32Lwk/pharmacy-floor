"""Made-up demo shop. Not real store data."""
AISLES = {
    "cold": "Aisle 2 (cold & flu)", "cough": "Aisle 2 (cold & flu)", "allergy": "Aisle 3 (allergy)", "hay fever": "Aisle 3 (allergy)",
    "pain": "Aisle 1 (pain relief)", "headache": "Aisle 1 (pain relief)", "stomach": "Aisle 4 (stomach & bowel)",
    "eye": "Aisle 5 (eye care)", "skin": "Aisle 6 (skin care)", "bandage": "Aisle 6 (first aid)",
}
HOURS = "The shop is open 9:00-21:00 every day (demo data)."


def lookup(message: str) -> str:
    t = message.lower()
    hits = sorted({v for k, v in AISLES.items() if k in t})
    if hits:
        return f"You can find that in {', '.join(hits)}. {HOURS}"
    return f"{HOURS} A staff member at the counter can help you find items."
