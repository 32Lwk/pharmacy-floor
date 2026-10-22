"""No-model test: the panel's question count equals every typed question sent to Decider in the turn."""
from types import SimpleNamespace as NS

import pytest
from strands_decider.schema import ChoiceQuestion


class FakeEngine:
    """Answers every question; choice -> `pick` with confidence `conf`, yes/no -> P=`p` per key."""

    def __init__(self, pick, conf, p):
        self.pick, self.conf, self.p, self.asked = pick, conf, p, 0

    def ask(self, state, questions):
        self.asked += len(questions)
        out = {}
        for k, q in questions.items():
            if isinstance(q, ChoiceQuestion):
                opts = list(q.criteria)
                c = self.pick.get(k, self.pick["*"])
                c = c if c in opts else opts[0]
                out[k] = NS(type="choice", choice=c, confidence=self.conf,
                            probabilities={o: (self.conf if o == c else (1 - self.conf) / (len(opts) - 1)) for o in opts})
            else:
                out[k] = NS(type="noul", noul=self.p.get(k, 0.02))
        return NS(answers=out)


CASES = [  # store turn (dose check asked but used to be hidden), symptom turn, off-topic reply -> handoff
    ({"*": "store"}, 0.95, {"on_topic": 0.95}),
    ({"*": "symptoms", "route": "otc_ok"}, 0.95, {"mentions_symptom": 0.98, "on_topic": 0.95}),
    ({"*": "symptoms", "route": "otc_ok"}, 0.95, {"mentions_symptom": 0.98}),
    ({"*": "symptoms", "route": "otc_ok"}, 0.95, {"mentions_symptom": 0.98, "on_topic": 0.05}),
    ({"*": "medicine_question", "topic_check": "store"}, 0.5, {}),
]


@pytest.mark.parametrize("pick,conf,p", CASES)
def test_count_equals_questions_asked(monkeypatch, pick, conf, p):
    monkeypatch.setenv("FLOOR_LLM", "stub")
    from floor.floor import PharmacyFloor
    eng = FakeEngine(pick, conf, p)
    log = PharmacyFloor(engine=eng).handle("test message")
    assert log.decider_questions == eng.asked > 0
