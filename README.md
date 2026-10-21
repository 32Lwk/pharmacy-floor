# Care Router — the small model picks the door, the big model talks

A small, new hackathon agent for drugstore-style symptom questions.
**Strands Decider 2B** decides where the person should go first — `otc_ok`, `see_doctor`, or
`emergency` — plus five yes/no flags, each with a calibrated number. **Plain Python** enforces
the fixed rules. **Amazon Bedrock** writes the reply *after* the decision and cannot change it.

> Hackathon demo. General guidance, not a diagnosis. In an emergency call your local
> emergency number (119 in Japan). Decider 2B is an experimental model.

Pattern: **#policy** (classify the request before the agent acts). Reusable recipe:
[`patterns/care_router/`](patterns/care_router/README.md).

## Flow

```
user message
  └─ CareRouterIntervention.before_invocation   (Strands InterventionHandler)
       ├─ Decider 2B, one call: route (choice) + genital_area, anal_area,
       │   bleeding, strong_pain, long_duration (yes/no)
       ├─ policy.py (fixed rules)
       │    1. genital area           -> see_doctor, always, no OTC
       │    2. anal itch / hemorrhoids -> OTC allowed, unless bleeding,
       │                                 strong pain, or > 1 week -> see_doctor
       │    3. emergency is never downgraded
       │    4. route confidence < 0.5  -> one step safer
       └─ Transform: pin "[CARE ROUTE: ...]" into the user turn
  └─ LLM (Bedrock, or offline stub) writes 3-5 sentences inside that route
```

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
python -m pytest tests                       # policy tests, no model needed
CARE_ROUTER_LLM=stub python -m care_router_agent.agent      # offline, no AWS call
python eval/run_eval.py                      # Decider vs rules-only on 30 cases
```

First run downloads ~4.6 GB (`Qwen/Qwen3.5-2B-Base` + the Decider adapter).

With Bedrock (needs AWS credentials and model access):

```bash
CARE_ROUTER_LLM=bedrock CARE_ROUTER_BEDROCK_MODEL_ID=<your model id> python -m care_router_agent.agent "I have a mild headache"
```

## Measured results (CPU only, 8 vCPU, no GPU)

30 made-up English cases in `eval/cases.jsonl` (12 otc_ok, 12 see_doctor, 6 emergency), written by
the author. "Under-triage" = the final route is less careful than expected (the costly error).

| Method | Correct | Under-triage |
|---|---|---|
| Rules only (keyword regex + same policy) | 23 / 30 | 6 |
| Decider 2B route only (no policy) | 27 / 30 | 0 |
| **Decider 2B + policy (this agent)** | **28 / 30** | **0** |

The two remaining misses are both over-cautious: a small cut (confidence 0.44, stepped up to
see_doctor) and a 4-day fever that Decider called emergency.

Latency on this CPU box (one `engine.ask` with 1 choice + 5 yes/no questions): median
**2593 ms** (p90 2736, n=30). Route question alone: median 734 ms (n=10). This is much slower than
the Strands team's GPU numbers (~115 ms on RTX 3090) — we did not measure on a GPU.
Full per-case output: `eval/results.json`.

Lessons from these runs
- Atomic yes/no questions worked better than one combined question (see the pattern README).
- The genital flag separates thinly for vague words: "rash down there on my private parts" scored
  0.52 vs ≤ 0.33 on unrelated cases. That is why the genital threshold is 0.4, and why it is a rule.

## Status

- Works offline end-to-end with the stub model (no network call).
- Bedrock mode is wired but **not yet run**.
- API names were checked against installed source: see `docs/API_NOTES.md`.

License: Apache-2.0 (same as Decider 2B).
