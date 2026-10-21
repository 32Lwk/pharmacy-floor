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
 │     default (FLOOR_RECEPTION=direct): the Decider-chosen tool is invoked directly
 │       (reception.tool.<name>) - no LLM call is spent on tool selection; before_tool_call still runs
 │     FLOOR_RECEPTION=llm: before_invocation pins "[ROUTE: call tool X]"; the reception LLM calls a
 │       tool; before_tool_call Guides any other tool back to X (both paths tested in tests/test_guide.py)
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

Decider latency per call, same CPU (`eval/bench_sets.py`, current code, n=8 each, 4 texts x 2):

| Question set | Median |
|---|---|
| full turn set: 14 questions (2 choice + 12 yes/no) | 8883 ms |
| **demo turn set: 6 questions (1 choice + 5 yes/no)** — used every turn | **2518 ms** |
| agent choice only (1 question) | 544 ms |
| Care Router: 6 questions (1 choice + 5 yes/no) — only on turns that mention a symptom | 1728 ms |

Whole turn with the stub text model (from `eval/results*.json`): turns with a symptom median 6291 ms
(n=21, dev) / 5801 ms (n=6, held-out); turns without a symptom 3835 ms (n=17) / 3537 ms (n=6).

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
- Bedrock-only routing (Nova Lite, one JSON classification call per message, no Decider) on the 12 held-out
  cases: agent 10/12, care 5/6, under-triage 2 ("my grandfather suddenly can't move his left arm" sent to
  symptoms / see_doctor instead of emergency; a syrup question for a 2-year-old marked otc_ok). Decider floor on the same cases: agent 9/12,
  care 4/6, under-triage 0. Bedrock median 478 ms per call over the network.

## Per-turn timing with Bedrock (direct mode, 9 turns, `eval/bedrock_samples.py`)

Before: the reception LLM made 2 calls (choose tool, then rewrite) plus 1 specialist call per turn.
Now: 1 Nova Lite call per turn (the specialist), max 220 output tokens. Text step: 355-734 ms.
Whole turns were 3.5-10.3 s, almost all of it Decider on CPU (3.5-8.8 s; the 8.8 s turn also ran
Care Router and the hesitation band). On a GPU the Decider part should drop sharply (not measured).

## Bedrock budget guard

`floor/budget.py` wraps `BedrockModel`: before every call it checks the running total in
`logs/bedrock_usage.json` and refuses once **$1.00 or 300 calls** is reached; after every call it adds the
`usage` token counts from the Converse response at Nova Lite prices ($0.06 / 1M input, $0.24 / 1M output,
us-east-1 on-demand). Only Nova Lite model IDs are allowed.

License: Apache-2.0.
