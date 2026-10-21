import json
from pathlib import Path

from floor import confidence as C

ROOT = Path(__file__).resolve().parents[1]


def test_thresholds_are_ordered():
    assert C.AGENT_HANDOFF < C.AGENT_PROCEED
    assert C.FLAG_NO < C.FLAG_YES
    assert C.ROUTE_HANDOFF < C.ROUTE_PROCEED
    assert C.REPLY_ON_TOPIC_HANDOFF < C.REPLY_ON_TOPIC_PROCEED


def test_band_edges_follow_one_convention():
    # p >= high -> high side; low <= p < high -> middle; p < low -> low side
    assert C.flag_band(C.FLAG_YES) == "yes" and C.flag_band(C.FLAG_NO) == "unsure" and C.flag_band(C.FLAG_NO - 0.01) == "no"
    assert C.band(C.AGENT_PROCEED, C.AGENT_PROCEED, C.AGENT_HANDOFF) == "proceed"
    assert C.band(C.AGENT_HANDOFF, C.AGENT_PROCEED, C.AGENT_HANDOFF) == "middle"


def test_recipes_read_thresholds_from_confidence():
    from recipes.care_router import policy
    assert policy.DEFAULT_CONFIDENCE_THRESHOLD == C.ROUTE_PROCEED
    assert policy.FLAG_THRESHOLDS["genital_area"] == C.CARE_FLAG_YES_GENITAL and policy.DEFAULT_FLAG_THRESHOLD == C.CARE_FLAG_YES


def test_shop_hint_is_asked_but_never_hesitant():
    from recipes.turn_router.questions import AGENT_HINTS, build_questions
    assert "shop_question" in build_questions(False, {}) and "shop_question" in AGENT_HINTS


def test_bands():
    assert C.band(0.9, C.AGENT_PROCEED, C.AGENT_HANDOFF) == "proceed"
    assert C.band(0.5, C.AGENT_PROCEED, C.AGENT_HANDOFF) == "middle"
    assert C.band(0.2, C.AGENT_PROCEED, C.AGENT_HANDOFF) == "handoff"
    assert [C.flag_band(p) for p in (0.7, 0.5, 0.1)] == ["yes", "unsure", "no"]


def test_added_questions_file_is_complete():
    qs = json.loads((ROOT / "rules" / "added_questions.json").read_text())
    for q in qs:
        assert {"key", "label", "question", "true", "false", "on_yes", "added_for"} <= set(q)
        assert "yes_at" not in q  # thresholds live only in floor/confidence.py
        assert q["on_yes"] in ("self_harm", "emergency_now", "danger_sign") and q["step"] in ("safety", "symptom")


def test_added_question_joins_the_turn_question_set():
    from recipes.turn_router.questions import build_questions, load_added
    q = build_questions(False, load_added("safety"))
    assert "self_harm" in q and "bleeding" not in q  # bleeding belongs to the Care Router step


def test_bleeding_added_to_care_router_and_counts_as_danger():
    from recipes.care_router.policy import apply_policy
    d = apply_policy("otc_ok", 0.9, {}, {"anal_area": 0.8, "bleeding": 0.79}, extra_danger=("bleeding",))
    assert d.route == "see_doctor"
    d = apply_policy("otc_ok", 0.9, {}, {"anal_area": 0.8, "bleeding": 0.19}, extra_danger=("bleeding",))
    assert d.route == "otc_ok"


def test_emergency_and_self_harm_never_go_to_an_llm():
    from floor.floor import NEVER_ASK_LLM
    assert {"emergency_now", "self_harm"} <= NEVER_ASK_LLM


def test_every_regression_case_names_an_expectation():
    for line in open(ROOT / "rules" / "regression_cases.jsonl"):
        c = json.loads(line)
        assert c["text"] and ("expect_agent" in c or "expect_not_agent" in c)
