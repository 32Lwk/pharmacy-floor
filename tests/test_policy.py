"""Policy tests: pure Python, no model download needed. Run: python -m pytest tests"""
from recipes.care_router.policy import apply_policy

Z = {"genital_area": 0.1, "anal_area": 0.1, "bleeding": 0.1, "strong_pain": 0.1, "long_duration": 0.1}

def f(**kw):
    return {**Z, **kw}

def test_genital_always_doctor_even_if_model_says_otc():
    assert apply_policy("otc_ok", 0.99, {}, f(genital_area=0.9)).route == "see_doctor"

def test_genital_never_upgraded_to_otc_by_anal_rule():
    assert apply_policy("see_doctor", 0.9, {}, f(genital_area=0.6, anal_area=0.8)).route == "see_doctor"

def test_anal_itch_without_danger_allows_otc():
    assert apply_policy("see_doctor", 0.8, {}, f(anal_area=0.8)).route == "otc_ok"

def test_anal_with_bleeding_pain_or_duration_goes_doctor():
    for k in ("bleeding", "strong_pain", "long_duration"):
        assert apply_policy("otc_ok", 0.9, {}, f(anal_area=0.8, **{k: 0.8})).route == "see_doctor"

def test_low_confidence_steps_safer():
    assert apply_policy("otc_ok", 0.3, {}, Z).route == "see_doctor"
    assert apply_policy("see_doctor", 0.3, {}, Z).route == "emergency"

def test_emergency_never_downgraded():
    assert apply_policy("emergency", 0.9, {}, f(anal_area=0.9)).route == "emergency"
