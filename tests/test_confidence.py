import json
from pathlib import Path

from floor import confidence as C

ROOT = Path(__file__).resolve().parents[1]


def test_thresholds_are_ordered():
    assert C.AGENT_HANDOFF < C.AGENT_PROCEED
    assert C.FLAG_NO < C.FLAG_YES
    assert C.ROUTE_HANDOFF < C.ROUTE_PROCEED
    assert C.REPLY_ON_TOPIC_HANDOFF < C.REPLY_ON_TOPIC_PROCEED


def test_bands():
    assert C.band(0.9, C.AGENT_PROCEED, C.AGENT_HANDOFF) == "proceed"
    assert C.band(0.5, C.AGENT_PROCEED, C.AGENT_HANDOFF) == "middle"
    assert C.band(0.2, C.AGENT_PROCEED, C.AGENT_HANDOFF) == "handoff"
    assert [C.flag_band(p) for p in (0.7, 0.5, 0.1)] == ["yes", "unsure", "no"]


def test_added_questions_file_is_complete():
    qs = json.loads((ROOT / "rules" / "added_questions.json").read_text())
    for q in qs:
        assert {"key", "label", "question", "true", "false", "yes_at", "on_yes", "added_for"} <= set(q)
        assert q["on_yes"] in ("self_harm", "emergency_now", "danger_sign") and q["step"] in ("safety", "symptom")
        assert 0 < q["yes_at"] < 1


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
