"""Decider questions asked at later steps of a turn (not only routing).

Each is a typed question: the answer is a choice from a fixed list or a probability, never free text,
so it cannot invent a product or a fact. Each costs one small local Decider call (~0.5-1 s on CPU).
Wordings were compared on 11 hand-written good/bad replies (see rules/README.md), not on eval cases.
"""
from __future__ import annotations

import time

from strands_decider.schema import ChoiceQuestion, NoulQuestion

from recipes.turn_router import AGENTS

# Step 4 - before the helper (tool) runs: a second, independent look at what the user wants.
# Options in reverse order so a position bias cannot make both looks agree by accident.
TOOL_CHECK = {"topic_check": ChoiceQuestion(instructions="What is the user mainly asking for in this message?",
                                            criteria=dict(reversed(list(AGENTS.items()))))}

# Step 6 - before the reply goes out: is it grounded in what the user said, and does it overreach?
REPLY_CHECK = {
    "on_topic": NoulQuestion(
        instructions="Is the reply about the same body problem, medicine or shop question that the user asked about?",
        criteria={"true": "the reply answers the same thing the user asked about",
                  "false": "the reply talks about a different problem, medicine or product"}),
    "dose_or_diagnosis": NoulQuestion(
        instructions="Does the reply state a dose, an amount, a number of hours between doses, or a named likely illness?",
        criteria={"true": "a dose number, interval or likely illness name is stated",
                  "false": "no dose numbers and no illness is named as the likely cause"}),
}


def ask(engine, state: str, questions: dict):
    t0 = time.perf_counter()
    ans = engine.ask(state, questions).answers
    return ans, (time.perf_counter() - t0) * 1000
