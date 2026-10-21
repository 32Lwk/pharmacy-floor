"""Compare Decider 2B (+ fixed policy) vs rules-only on eval/cases.jsonl. Writes eval/results.json."""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.rules_only import rules_decide  # noqa: E402
from patterns.care_router import CareRouter  # noqa: E402
from patterns.care_router.policy import apply_policy  # noqa: E402
from patterns.care_router.questions import FLAG_KEYS  # noqa: E402

ROOT = Path(__file__).parent
cases = [json.loads(l) for l in open(ROOT / "cases.jsonl")]

t0 = time.perf_counter()
router = CareRouter()
load_s = time.perf_counter() - t0
router.raw("warm up")  # first call compiles/caches

rows, lat = [], []
for c in cases:
    resp, ms = router.raw(c["text"])
    lat.append(ms)
    r = resp.answers["route"]
    flags = {k: resp.answers[k].noul for k in FLAG_KEYS}
    pol = apply_policy(r.choice, r.confidence, dict(r.probabilities), flags)
    rules = rules_decide(c["text"])
    rows.append({**c, "decider_raw": r.choice, "confidence": r.confidence, "probabilities": dict(r.probabilities),
                 "flags": flags, "decider_policy": pol.route, "reasons": pol.reasons,
                 "rules_only": rules.route, "ms": round(ms, 1)})

def acc(key):
    return sum(r[key] == r["expected"] for r in rows)

def under(key):  # dangerous misses: final route less careful than expected
    order = ["otc_ok", "see_doctor", "emergency"]
    return sum(order.index(r[key]) < order.index(r["expected"]) for r in rows)

summary = {
    "n": len(rows), "device": "cpu", "model_load_s": round(load_s, 1),
    "latency_ms": {"median": round(statistics.median(lat), 1), "p90": round(sorted(lat)[int(0.9 * len(lat)) - 1], 1),
                   "min": round(min(lat), 1), "max": round(max(lat), 1)},
}
for k in ("decider_raw", "decider_policy", "rules_only"):
    summary[k] = {"correct": acc(k), "under_triage": under(k)}
json.dump({"summary": summary, "rows": rows}, open(ROOT / "results.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
for r in rows:
    mark = lambda k: "ok" if r[k] == r["expected"] else r[k]
    print(f'{r["id"]:>2} {r["expected"]:<10} raw={mark("decider_raw"):<10} pol={mark("decider_policy"):<10} rules={mark("rules_only"):<10} conf={r["confidence"]:.2f} {r["ms"]:.0f}ms')
