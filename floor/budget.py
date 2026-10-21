"""Hard Bedrock budget guard. Counts tokens from every Converse response and refuses new calls past the cap.

Price: Amazon Nova Lite on-demand, us-east-1: $0.06 / 1M input tokens, $0.24 / 1M output tokens
(AWS Price List AmazonBedrock offer, observed 2026-10-19). Caps: $1.00 or 300 calls, whichever first.
Running total is kept in logs/bedrock_usage.json (tokens and dollars only, never credentials).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from strands.models.bedrock import BedrockModel

PRICE_IN_PER_M = 0.06
PRICE_OUT_PER_M = 0.24
MAX_USD = 1.00
MAX_CALLS = 300
LOG = Path(__file__).resolve().parents[1] / "logs" / "bedrock_usage.json"
_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def load() -> dict:
    if LOG.exists():
        return json.loads(LOG.read_text())
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0, "history": []}


def _save(s: dict) -> None:
    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text(json.dumps(s, indent=1))


def check() -> None:
    s = load()
    if s["calls"] >= MAX_CALLS or s["usd"] >= MAX_USD:
        raise BudgetExceeded(f"Bedrock budget reached: {s['calls']} calls, ${s['usd']:.6f}")


def record(model_id: str, usage: dict, label: str = "") -> None:
    with _lock:
        s = load()
        i, o = int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))
        cost = i / 1e6 * PRICE_IN_PER_M + o / 1e6 * PRICE_OUT_PER_M
        s["calls"] += 1; s["input_tokens"] += i; s["output_tokens"] += o; s["usd"] = round(s["usd"] + cost, 8)
        s["history"].append({"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": model_id, "in": i, "out": o,
                             "usd": round(cost, 8), "label": label})
        _save(s)


class GuardedBedrockModel(BedrockModel):
    """BedrockModel that checks the cap before each call and records usage from the response metadata."""

    label = ""  # set per call site for the usage log

    async def stream(self, *args, **kwargs):
        check()
        seen = False
        async for ev in super().stream(*args, **kwargs):
            if "metadata" in ev and "usage" in ev["metadata"]:
                record(self.config.get("model_id", "?"), ev["metadata"]["usage"], self.label); seen = True
            yield ev
        if not seen:  # count the call even without usage, conservatively
            record(self.config.get("model_id", "?"), {"inputTokens": 0, "outputTokens": 0}, self.label + " (no usage)")
