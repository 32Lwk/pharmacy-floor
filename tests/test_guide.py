"""The reception LLM tries the wrong specialist; TurnRouterIntervention must Guide it to the Decider's pick."""
import json

from strands import Agent, tool

from floor.llm import StubModel
from recipes.turn_router import TurnRouterIntervention

calls = []


@tool
def store(message: str) -> str:
    """store"""
    calls.append("store"); return "store answer"


@tool
def feelings(message: str) -> str:
    """feelings"""
    calls.append("feelings"); return "feelings answer"


class WrongFirst(StubModel):
    n = 0

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kw):
        WrongFirst.n += 1
        last = messages[-1]
        results = [b["toolResult"] for b in last.get("content", []) if "toolResult" in b]
        yield {"messageStart": {"role": "assistant"}}
        if results and "answer" in json.dumps(results):
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": results[-1]["content"][0]["text"]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            return
        name = "feelings" if WrongFirst.n == 1 else "store"  # wrong first, then obeys
        yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{WrongFirst.n}", "name": name}}}}
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps({"message": "x"})}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}


def test_wrong_tool_is_guided_to_decider_choice():
    guard = TurnRouterIntervention(lambda t: "store")
    out = str(Agent(model=WrongFirst(), tools=[store, feelings], interventions=[guard], callback_handler=None)("where is aisle 3"))
    assert calls == ["store"] and "store answer" in out
