"""Offline stand-in for Amazon Bedrock. Makes NO network call.

It reads the route note the Care Router intervention appended and returns a fixed
template, so the whole Strands agent loop can be demoed without AWS access.
"""

from __future__ import annotations

import re
from typing import Any

from strands.models.model import Model

TEMPLATES = {
    "otc_ok": "[STUB — Bedrock not called] An over-the-counter medicine may help. Please ask the pharmacist. If it gets worse or lasts more than a few days, see a doctor. This is guidance, not a diagnosis.",
    "see_doctor": "[STUB — Bedrock not called] Please visit a clinic for this. I won't suggest a medicine here. This is guidance, not a diagnosis.",
    "emergency": "[STUB — Bedrock not called] Please get emergency care now. In Japan, call 119. This is guidance, not a diagnosis.",
}


class StubModel(Model):
    def __init__(self):
        self.config: dict[str, Any] = {"model_id": "stub"}

    def update_config(self, **cfg):
        self.config.update(cfg)

    def get_config(self):
        return self.config

    def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        text = " ".join(b.get("text", "") for m in messages if m.get("role") == "user" for b in m.get("content", []))
        found = re.findall(r"\[CARE ROUTE[^:]*: (\w+)\]", text)
        reply = TEMPLATES.get(found[-1] if found else "see_doctor")
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": reply}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
