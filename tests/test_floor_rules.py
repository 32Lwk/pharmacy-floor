"""No-model tests for the final gate and crisis words."""
from floor.final_gate import final_gate
from floor.floor import CRISIS_WORDS


def test_otc_push_replaced_on_doctor_turn():
    r, v = final_gate("An over-the-counter cream may help.", care_route="see_doctor", symptom_turn=True)
    assert "see a doctor" in r and v.startswith("replaced")


def test_doctor_turn_must_mention_doctor():
    r, _ = final_gate("Please read the package insert.", care_route="see_doctor", symptom_turn=True)
    assert "doctor" in r


def test_disclaimer_added_on_symptom_turn():
    r, _ = final_gate("Rest and drink water. A pharmacist can help.", care_route="otc_ok", symptom_turn=True)
    assert r.endswith("not a diagnosis.")


def test_crisis_words():
    assert CRISIS_WORDS.search("I want to take the whole box so I don't wake up")
    assert not CRISIS_WORDS.search("Can I take two sleeping pills tonight?")


from floor.final_gate import specialist_gate
from floor.store_data import lookup


def test_medicine_question_dose_numbers_removed():
    for r in ["Ibuprofen can be taken every 4-6 hours.", "Take 200 mg up to 3 times a day.", "Take it twice a day."]:
        out, v = specialist_gate(r, agent="medicine_question", store_facts=None)
        assert "package label" in out and not any(ch.isdigit() for ch in out), r


def test_medicine_question_without_numbers_passes():
    r = "It may cause drowsiness. Please follow the package label and ask our pharmacist."
    assert specialist_gate(r, agent="medicine_question", store_facts=None)[0] == r


def test_store_reply_grounded_in_demo_data():
    facts = lookup("What time do you close tonight?")
    assert "Demo Store" in facts and "21:00" in facts
    ok = "Our Demo Store closes at 21:00."
    assert specialist_gate(ok, agent="store", store_facts=facts)[0] == ok
    bad, v = specialist_gate("We close at 22:00 and open at 8:00.", agent="store", store_facts=facts)
    assert v.startswith("replaced") and "22:00" not in bad


def test_guessed_diagnosis_is_removed():
    r = ("You might have a mild case of hemorrhoids or an allergic reaction. Over-the-counter creams can soothe itching. "
         "If there is bleeding or it lasts over a week, please see a doctor.")
    out, v = specialist_gate(r, agent="symptoms", store_facts=None)
    assert "hemorrhoids" not in out and "allergic" not in out and "see a doctor" in out and v.startswith("removed")
    for g in ["It sounds like an infection.", "This could be eczema.", "It is likely caused by stress."]:
        assert g not in specialist_gate(g + " Please ask the pharmacist.", agent="symptoms", store_facts=None)[0]


def test_care_advice_without_guess_passes():
    r = "Gentle cleaning and an over-the-counter cream may help. See a doctor if it bleeds."
    assert specialist_gate(r, agent="symptoms", store_facts=None) == (r, "pass")


def test_feelings_empathy_not_treated_as_diagnosis():
    r = "It sounds like you're going through a lot. Would you like to tell me more?"
    assert specialist_gate(r, agent="feelings", store_facts=None) == (r, "pass")
