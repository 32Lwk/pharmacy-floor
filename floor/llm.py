"""LLM provider. Bedrock (Amazon Nova Lite, us-east-1) when AWS keys are in the environment, else an offline stub.

FLOOR_LLM=stub forces the stub. FLOOR_LLM=bedrock forces Bedrock. Default "auto".
"""
from __future__ import annotations

import json
import os
import re
import uuid

from strands.models.model import Model

NOVA_LITE = os.environ.get("FLOOR_BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")
REGION = os.environ.get("FLOOR_BEDROCK_REGION", "us-east-1")


def mode() -> str:
    m = os.environ.get("FLOOR_LLM", "auto")
    if m == "auto":
        return "bedrock" if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY") else "stub"
    return m


def make_model(max_tokens: int = 400):
    if mode() == "bedrock":
        from .budget import GuardedBedrockModel
        if "nova-lite" not in NOVA_LITE:
            raise ValueError("Only Amazon Nova Lite is allowed (budget)")
        return GuardedBedrockModel(model_id=NOVA_LITE, region_name=REGION, max_tokens=max_tokens, temperature=0.3)
    return StubModel()


def model_label() -> str:
    return f"bedrock:{NOVA_LITE}" if mode() == "bedrock" else "stub (no LLM call)"


STUB_REPLIES = {
    "symptoms": {
        "otc_ok": "[stub] Sorry you're dealing with that. An over-the-counter option may help; our pharmacist can suggest one. If it gets worse or lasts more than a few days, please see a doctor. This is guidance, not a diagnosis.",
        "see_doctor": "[stub] Thank you for telling me. For this, please visit a clinic rather than choosing a medicine here. This is guidance, not a diagnosis.",
        "emergency": "[stub] Please get emergency care now. In Japan call 119.",
    },
    "medicine_question": "[stub] Good question. Please check the package insert and ask our pharmacist at the counter, who can look at your situation. This is general information only.",
    "feelings": "[stub] That sounds really hard. I'm here to listen. Would you like to tell me a bit more about what has been worrying you?",
    "store": "[stub] {store}",
    "small_talk": "[stub] Hello! I'm the pharmacy floor assistant. Ask me about a symptom, a medicine, or the shop.",
    "judge": "UNSURE",
}


def _text(messages, role="user"):
    return " ".join(b.get("text", "") for m in messages if m.get("role") == role for b in m.get("content", []))


class StubModel(Model):
    """Offline stand-in for Bedrock. Makes no network call. Handles tool calls so the agent loop runs."""

    def __init__(self):
        self.config = {"model_id": "stub"}

    def update_config(self, **c):
        self.config.update(c)

    def get_config(self):
        return self.config

    def structured_output(self, *a, **k):
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        sp = system_prompt or ""
        last = messages[-1]
        tool_results = [b["toolResult"] for b in last.get("content", []) if "toolResult" in b]
        yield {"messageStart": {"role": "assistant"}}
        if tool_results:  # reception: hand back the specialist's answer
            reply = " ".join(c.get("text", "") for c in tool_results[-1].get("content", []))
        elif tool_specs:  # reception: call the pinned tool
            m = re.findall(r"call tool (\w+)", _text(messages))
            name = m[-1] if m else "small_talk"
            user = re.sub(r"\[ROUTE[^\]]*\]", "", _text([last])).strip()
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": uuid.uuid4().hex[:8], "name": name}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps({"message": user})}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
            return
        else:
            role = (re.findall(r"ROLE=(\w+)", sp) or ["small_talk"])[0]
            r = STUB_REPLIES.get(role, STUB_REPLIES["small_talk"])
            if isinstance(r, dict):
                route = (re.findall(r"CARE ROUTE[^:]*: (\w+)", _text(messages)) or ["see_doctor"])[-1]
                r = r[route]
            if role == "store":
                facts = re.findall(r"STORE FACTS: (.*)", _text(messages))
                r = r.format(store=facts[-1] if facts else "Please ask a staff member at the counter.")
            reply = r
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": reply}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
