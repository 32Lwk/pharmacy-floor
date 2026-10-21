# Pharmacy Floor — a multi-agent drugstore helper where Strands Decider 2B makes every decision

A reception agent and five specialist agents (symptoms, medicine questions, feelings, store,
small talk), plus emergency handling. **Strands Decider 2B** decides who answers and every
safety yes/no. **Plain Python** enforces the hard rules. **Amazon Bedrock** (Amazon Nova Lite)
writes the text. Unsure Decider answers (the "hesitation band") are handed to Bedrock.

> Hackathon demo, English conversations. General guidance, not a diagnosis. In an emergency call
> your local emergency number (119 in Japan). Decider 2B is an experimental model.

Declared pattern: **#tool-selection**. Reusable recipes:
- [`recipes/turn_router/`](recipes/turn_router/README.md) — Decider as the tool selector of a reception agent
- [`recipes/care_router/`](recipes/care_router/README.md) — three-door triage with hard-rule overrides (#policy-style safety layer)

## One turn

```
user message
 ├─ pre-gate (Python): crisis words -> fixed crisis reply, no model
 ├─ Turn Router — ONE Decider call, 6 questions:
 │     agent (choice: symptoms / medicine_question / feelings / store / small_talk)
 │     emergency_now, self_harm, special_population, manipulation, mentions_symptom (yes/no)
 ├─ hesitation band: agent conf < 0.5, or a yes/no with 0.4 <= P < 0.6
 │     -> Bedrock classifies that one question (offline: safer default)
 ├─ safety layer (every turn, whatever the specialist):
 │     emergency_now / self_harm -> fixed emergency text, no LLM
 │     symptom mentioned anywhere -> Care Router (second Decider call):
 │        genital area -> see_doctor, always · anal itch / hemorrhoids -> OTC unless bleeding,
 │        strong pain, > 1 week · emergency never downgraded · low confidence -> one step safer
 │        special population + otc_ok -> see_doctor
 ├─ reception Agent(tools=[5 specialist agents-as-tools], interventions=[TurnRouterIntervention])
 │     before_invocation: Transform pins "[ROUTE: call tool X]"
 │     before_tool_call: any other tool -> Guide back to X (tested in tests/test_guide.py)
 │     symptoms tool runs its own Agent with CareRouterIntervention; other tools get the care note
 └─ final gate (Python): no OTC wording on see_doctor turns, doctor mention required,
       disclaimer on symptom turns
```

Every step is logged (question, answer, confidence, who decided, ms) and shown in the demo UI.

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -r requirements.txt
python -m pytest tests                 # no model needed
python -m floor.cli                    # demo turns in the terminal
python web/server.py 8765              # http://127.0.0.1:8765  chat + decision panel (local only)
python eval/run_eval.py                # 40 dev cases;  python eval/run_eval.py cases_holdout.jsonl  for 12 held-out
```

LLM mode: `FLOOR_LLM=auto` (default) uses Bedrock `us.amazon.nova-lite-v1:0` in `us-east-1` when
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set, otherwise an offline stub (no network call).
`FLOOR_LLM=stub` forces the stub.

## Measured results (CPU only, 8 vCPU, no GPU; stub text model)

Decider latency per call, same CPU (`eval/bench_sets.py`, n=8 each):

| Question set | Median |
|---|---|
| 13 questions (2 choice + 11 yes/no) | 8557 ms |
| 6 questions (1 choice + 5 yes/no) | 2614 ms |
| 5 questions | 2205 ms |
| agent choice only | 541 ms |

The demo uses the 6-question turn set. Whole turn (Decider calls + stub): median 5.7 s on the 40 dev
cases, ~3.5 s for turns without a symptom.

Routing and safety. "Under-triage" = the final outcome is less careful than expected.

| Set | Method | Agent correct | Care route correct | Under-triage |
|---|---|---|---|---|
| 40 dev cases | Pharmacy Floor (Decider) | 33/40 | 17/17 | 0 |
| 40 dev cases | Keyword rules | 40/40 | 16/17 | 0 |
| 12 held-out cases | Pharmacy Floor (Decider) | 9/12 | 4/6 | 0 |
| 12 held-out cases | Keyword rules | 7/12 | 1/6 | 5 |

Honest notes
- The keyword rules were written while looking at the 40 dev cases, so 40/40 there is overfit. The 12
  held-out cases were written after all rules and thresholds were frozen; there the keywords
  under-triaged 5 cases and the Decider floor none.
- Most Decider "agent misses" on the dev set are symptoms hidden inside a medicine or feelings question
  being sent to the symptoms specialist; the care outcome was still right.
- Over-triage seen: "knee aches after hiking" was sent to emergency on the held-out set.
- Hand-tuning: the `self_harm` wording and the crisis-word pre-gate were added after Decider scored an
  indirect overdose message at P = 0.15 (case 40). The store option text and the hesitation band
  (0.4-0.6) were also adjusted once after the first dev run.
- Bedrock comparison: not run yet (no AWS keys on the build machine).

License: Apache-2.0.
