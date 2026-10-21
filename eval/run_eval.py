"""Full Pharmacy Floor pipeline (stub LLM unless AWS keys are set) vs keyword rules on eval/cases.jsonl."""
import json, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.keyword_rules import keyword_decide
from floor.floor import PharmacyFloor

ROOT = Path(__file__).parent
ORDER = [None, "otc_ok", "see_doctor", "emergency"]
CASEFILE = sys.argv[1] if len(sys.argv) > 1 else "cases.jsonl"
cases = [json.loads(l) for l in open(ROOT / CASEFILE)]
floor = PharmacyFloor()
floor.handle("hello")  # warm-up
rows = []
for c in cases:
    log = floor.handle(c["text"])
    care = next((e.answer for e in reversed(log.entries) if e.step == "care_router" and e.question == "route"), None)
    if any(e.step == "policy" and e.question == "special_population" for e in log.entries):
        care = "see_doctor"
    ka, kc = keyword_decide(c["text"])
    rows.append({**c, "floor_agent": log.final_agent, "floor_care": care, "kw_agent": ka, "kw_care": kc,
                 "hesitations": [(e.question, e.answer, e.decided_by) for e in log.entries if e.step == "hesitation"],
                 "decider_ms": round(sum(e.ms or 0 for e in log.entries if e.decided_by == "decider-2b"), 1),
                 "total_ms": round(log.total_ms, 1)})
    print(c["id"], c["agent"], "->", log.final_agent, care, "| kw:", ka, kc, f'{log.total_ms:.0f}ms', flush=True)

def score(prefix):
    agent_ok = sum(r[f"{prefix}_agent"] == r["agent"] for r in rows)
    care_rows = [r for r in rows if r["care"]]
    care_ok = sum(r[f"{prefix}_care"] == r["care"] for r in care_rows)
    # under-triage: expected emergency but not emergency; or expected see_doctor but got otc_ok / nothing
    under = 0
    for r in rows:
        exp = "emergency" if r["agent"] == "emergency" else r["care"]
        got = "emergency" if r[f"{prefix}_agent"] == "emergency" else r[f"{prefix}_care"]
        if exp and ORDER.index(got) < ORDER.index(exp):
            under += 1
    return {"agent_correct": f"{agent_ok}/{len(rows)}", "care_correct": f"{care_ok}/{len(care_rows)}", "under_triage": under}

summary = {"n": len(rows), "llm": floor.llm_label(), "floor": score("floor"), "keyword_rules": score("kw"),
           "turn_ms": {"median": statistics.median(r["total_ms"] for r in rows), "max": max(r["total_ms"] for r in rows)},
           "decider_ms_per_turn_median": statistics.median(r["decider_ms"] for r in rows),
           "turns_with_hesitation": sum(bool(r["hesitations"]) for r in rows)}
json.dump({"summary": summary, "rows": rows}, open(ROOT / CASEFILE.replace("cases", "results").replace(".jsonl", ".json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
