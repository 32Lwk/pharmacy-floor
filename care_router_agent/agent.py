"""Care Router agent: Decider 2B picks the door, the LLM writes the words.

Modes (env CARE_ROUTER_LLM):
  stub     (default) offline template model, no AWS call at all
  bedrock  strands BedrockModel; needs AWS credentials + Bedrock model access.
           Model id from CARE_ROUTER_BEDROCK_MODEL_ID (unset -> Strands default).
"""

from __future__ import annotations

import argparse
import json
import os

from strands import Agent

from patterns.care_router import CareRouter, CareRouterIntervention

SYSTEM_PROMPT = (
    "You are a kind drugstore helper. A separate policy step has already decided where the user "
    "should go; it is given in a [CARE ROUTE] note. Follow it exactly. Never diagnose. "
    "Never name prescription drugs. Reply in 3-5 short sentences in the user's language."
)


def make_model():
    mode = os.environ.get("CARE_ROUTER_LLM", "stub")
    if mode == "bedrock":
        from strands.models.bedrock import BedrockModel

        mid = os.environ.get("CARE_ROUTER_BEDROCK_MODEL_ID")
        return BedrockModel(model_id=mid) if mid else BedrockModel()
    if mode == "stub":
        from .stub_model import StubModel

        return StubModel()
    raise ValueError(f"unknown CARE_ROUTER_LLM={mode!r}")


def build(router: CareRouter | None = None):
    router = router or CareRouter()
    guard = CareRouterIntervention(router)
    agent = Agent(model=make_model(), system_prompt=SYSTEM_PROMPT, interventions=[guard],
                  callback_handler=None)
    return agent, guard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("message", nargs="*")
    a = ap.parse_args()
    agent, guard = build()
    msgs = [" ".join(a.message)] if a.message else [
        "I have a mild headache since this morning, nothing else.",
        "My bottom has been itchy around the anus for two days.",
        "My bottom is itchy and there was blood on the toilet paper.",
        "I have itching in my genital area.",
        "Sudden chest pain and I can't breathe properly.",
    ]
    for m in msgs:
        agent.messages.clear()
        reply = agent(m)
        d = guard.last_decision
        print(json.dumps({"input": m, "decider_route": d.decider_route, "route": d.route,
                          "confidence": d.route_confidence, "probabilities": d.route_probabilities,
                          "flags": d.flags, "reasons": d.reasons, "decider_ms": round(d.latency_ms, 1)},
                         ensure_ascii=False))
        print("REPLY:", str(reply).strip(), "\n")


if __name__ == "__main__":
    main()
