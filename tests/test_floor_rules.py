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
