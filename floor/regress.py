"""Run rules/regression_cases.jsonl through the full floor (stub LLM: no Bedrock calls).

  python -m floor.regress                          # current rules
  python -m floor.regress --before bleeding        # same, without that added question (shows the miss it fixed)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CASES = Path(__file__).resolve().parents[1] / "rules" / "regression_cases.jsonl"


def run(before: str | None = None, cases: Path = CASES) -> list[dict]:
    os.environ["FLOOR_LLM"] = "stub"
    from recipes.turn_router.questions import load_added

    from .floor import PharmacyFloor
    floor = PharmacyFloor()
    if before:
        added = {k: v for k, v in load_added().items() if k != before}
        floor.added = floor.turn_router.added = added
        from recipes.turn_router.questions import build_questions
        floor.turn_router.questions = build_questions(False, added)
        floor.care_router.questions.pop(before, None)
    rows = []
    for c in map(json.loads, open(cases)):
        log = floor.handle(c["text"])
        route = next((e.answer.split()[0] for e in log.entries if e.question == "route"), None)
        p = {k: next((e.confidence for e in log.entries if e.question == k), None) for k in c.get("show", [])}
        ok = (log.final_agent == c["expect_agent"]) if "expect_agent" in c else (log.final_agent != c["expect_not_agent"])
        if "expect_route" in c:
            ok = ok and route == c["expect_route"]
        rows.append({"id": c["id"], "ok": ok, "got": log.final_agent, "route": route, **{f"P({k})": v for k, v in p.items()}})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", help="drop this added question and the crisis-word gate to show the miss it fixes")
    rows = run(ap.parse_args().before)
    for r in rows:
        print(("PASS " if r["ok"] else "FAIL ") + json.dumps(r))
    print(f"{sum(r['ok'] for r in rows)}/{len(rows)} passed")
