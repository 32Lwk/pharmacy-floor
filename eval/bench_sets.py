"""Latency: full 13-question turn set vs trimmed sets, same engine, CPU."""
import statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strands_decider.infer import load_engine
from recipes.turn_router.questions import build_questions, SAFETY_ALL, AGENTS
from recipes.care_router.questions import QUESTIONS as CARE

e = load_engine("StrandsAgents/strands-decider-2B-hobson-v19", device="cpu")
texts = ["I've had a sore throat since yesterday, what can I take?",
         "Where are the allergy tablets in this shop?",
         "Can I take ibuprofen together with my blood pressure pills?",
         "I feel so anxious lately and can't sleep, I just need someone to talk to."]
full = build_questions(True); turn = build_questions(False)
def label(qs):
    c = sum(q.type == "choice" for q in qs.values()); n = sum(q.type == "noul" for q in qs.values())
    return f"{c + n} questions ({c} choice + {n} noul)"
sets = {"full turn set " + label(full): full, "DEMO turn set " + label(turn): turn,
        "agent only " + label({"agent": full["agent"]}): {"agent": full["agent"]},
        "care router " + label(CARE): CARE}
for name, qs in sets.items():
    e.ask(texts[0], qs)
    lat = []
    for _ in range(2):
        for tx in texts:
            t0 = time.perf_counter(); e.ask(tx, qs); lat.append((time.perf_counter() - t0) * 1000)
    print(f"{name}: median {statistics.median(lat):.0f} ms, min {min(lat):.0f}, max {max(lat):.0f}, n={len(lat)}", flush=True)
