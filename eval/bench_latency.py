"""Latency of one Decider call: 1 choice question vs the full Care Router set (5 noul + 1 choice)."""
import statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strands_decider.infer import load_engine
from patterns.care_router.decider import DEFAULT_CHECKPOINT
from patterns.care_router.questions import QUESTIONS

e = load_engine(DEFAULT_CHECKPOINT, device="cpu")
text = "My bottom is itchy and there was blood on the toilet paper."
for name, qs in (("route only (1 choice)", {"route": QUESTIONS["route"]}), ("full set (1 choice + 5 noul)", QUESTIONS)):
    e.ask(text, qs)
    t = []
    for _ in range(10):
        t0 = time.perf_counter(); e.ask(text, qs); t.append((time.perf_counter() - t0) * 1000)
    print(f"{name}: median {statistics.median(t):.0f} ms, min {min(t):.0f}, max {max(t):.0f} (n=10, CPU)")
