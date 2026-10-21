"""Wiring: an InterventionHandler that makes Decider 2B the tool selector of a reception agent.

- before_invocation: one Decider call; pins "[ROUTE: call tool X]" into the user turn (Transform).
- before_tool_call: if the LLM tries any other specialist tool, Guide it back to the chosen one.
The LLM never picks the specialist; it only executes the pinned choice.
"""
from __future__ import annotations

from typing import Callable

from strands.interventions import Guide, InterventionHandler, Proceed, Transform


def _last_user_text(messages) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return " ".join(b["text"] for b in m.get("content", []) if "text" in b)
    return ""


class TurnRouterIntervention(InterventionHandler):
    name = "turn-router"

    def __init__(self, decide: Callable[[str], str]):
        """decide(text) -> tool name. Put Decider + policy + hesitation handling inside it."""
        self.decide = decide
        self.chosen: str | None = None

    @property
    def on_error(self):
        return "throw"

    def before_invocation(self, event, **kwargs):
        text = _last_user_text(event.messages)
        if not text:
            return Proceed()
        self.chosen = self.decide(text)
        note = f"[ROUTE fixed by policy: call tool {self.chosen} exactly once with the user's message, then return its answer unchanged]"

        def apply(ev):
            for m in reversed(ev.messages):
                if m.get("role") == "user":
                    m["content"] = list(m["content"]) + [{"text": note}]
                    break
        return Transform(apply=apply, reason=f"route={self.chosen}")

    def before_tool_call(self, event, **kwargs):
        name = event.tool_use.get("name")
        if self.chosen and name != self.chosen:
            return Guide(feedback=f"Wrong specialist. The route is fixed: call {self.chosen}.")
        return Proceed()
