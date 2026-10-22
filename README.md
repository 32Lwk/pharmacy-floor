# Pharmacy Floor — a drugstore helper where Strands Decider 2B makes every decision

A reception agent and five specialist agents (symptoms, medicine questions, feelings, store,
small talk), plus emergency handling. **Strands Decider 2B** decides who answers, every safety
yes/no, and checks the written reply. **Plain Python** enforces the hard rules. **Amazon Bedrock**
(Amazon Nova Lite) only writes the text.

**The idea.** A decision-only model gives *typed* answers (a choice from a fixed list, or a yes/no
probability). It cannot invent a product, a dose or a fact, its confidence is a number you can act on,
and one more question costs a fraction of a second on CPU. So the floor asks many small questions at
every step of every turn: about 16 per symptom turn and about 10 on other turns. An LLM call could not sit at all of these steps.

> Hackathon demo, English conversations. General guidance, not a diagnosis. In an emergency call
> your local emergency number (119 in Japan). Decider 2B is an experimental model.

Declared pattern: **#tool-selection**. Reusable recipes:
- [`recipes/turn_router/`](recipes/turn_router/README.md) — Decider as the tool selector of a reception agent
- [`recipes/care_router/`](recipes/care_router/README.md) — three-door triage with hard-rule overrides (#policy-style safety layer)
- [`rules/`](rules/README.md) — **add one question, fix it forever** (worked example: the bleeding question)

![Decision panel](docs/screenshots/01_blood_see_doctor_steps.png)

## One turn: typed decisions at every step

```
user message
 ├─ pre-check (Python): crisis words -> fixed crisis-support reply, no model
 ├─ 1 Who should answer?     Decider choice over 5 helpers + a "shop question?" yes/no hint (same Decider call as step 2)
 │     pick < 0.60, shop_question >= 0.60 and no body problem -> store helper (on 40 dev messages: shop 0.57-0.95, others <= 0.23)
 ├─ 2 Safety checks          emergency_now, self_harm, special_population, manipulation, mentions_symptom
 │     emergency / self-harm: unsure (0.40 <= P < 0.60) = YES -> fixed message. Never sent to an LLM.
 │     other flags: unsure -> one Bedrock yes/no tie-break (offline: safer default)
 ├─ 3 Before the helper runs pick >= 0.60: go on, no extra call. Pick < 0.60: a second, independent Decider look
 │     ("what is the user asking for?", reversed options): agrees -> go on · differs -> Bedrock tie-break
 │     · differs AND pick < 0.35 -> pharmacist
 ├─ 4 Symptom check          Care Router (Decider), whenever a body problem is mentioned
 │     genital area -> see_doctor, always · anal itch -> OTC unless a danger sign (bleeding, strong pain, > 1 week)
 │     emergency never downgraded · route < 0.50 -> one step safer · route < 0.40 and no hard rule -> pharmacist
 ├─ 5 Writing the reply      the Decider-chosen Strands tool (agents-as-tools) writes with Nova Lite;
 │                           TurnRouterIntervention pins the tool (before_invocation / before_tool_call Guide)
 └─ 6 Before the reply goes out
       Python: no guessed diagnosis, no dose numbers, store numbers must be in store data, doctor mention, disclaimer
       Decider: "reply answers what was asked?" P < 0.40 -> pharmacist, 0.40 <= P < 0.60 -> add a pharmacist note
                "reply gives a dose or names an illness?" >= 0.60 -> pharmacist
```

The panel shows every step, grouped and in plain English, colored by what happened:
OK (sure, go on) · Extra care (one step safer) · Double-check (unsure, second look) ·
Pharmacist (too unsure: an **Ask the pharmacist** card replaces the reply) · Stop (fixed safety message).
"show technical details" adds the question key, who decided (Decider / Rule / Bedrock) and ms.

### Confidence thresholds (`floor/confidence.py`, the only place they are defined)

The recipes import them from there. One band convention everywhere: for a pair (low, high),
`P >= high` is the high side (proceed / yes), `low <= P < high` is the middle (unsure), `P < low` is the low side
(hand-off / no).

| Constant | Value | Meaning |
|---|---|---|
| `AGENT_PROCEED` | 0.60 | step 1 pick P >= 0.60: use it, skip step 3's second look |
| `AGENT_HANDOFF` | 0.35 | pick P < 0.35 and the second look disagrees: pharmacist (0.35 <= P < 0.60 and disagree: Bedrock tie-break) |
| `FLAG_YES` / `FLAG_NO` | 0.60 / 0.40 | safety yes/no: P >= 0.60 yes · 0.40 <= P < 0.60 unsure · P < 0.40 no |
| `SYMPTOM_LEAN` | 0.30 | offline tie-break: mentions_symptom >= 0.30 sends an unsure pick to the symptoms helper |
| `ROUTE_PROCEED` | 0.50 | Care Router route below this: one step safer (existing rule) |
| `ROUTE_HANDOFF` | 0.40 | route below this with no hard rule deciding: pharmacist |
| `CARE_FLAG_YES` / `CARE_FLAG_YES_GENITAL` | 0.50 / 0.40 | Care Router yes/no flags (incl. added questions) / the genital flag, which trips earlier on purpose |
| `REPLY_ON_TOPIC_PROCEED` / `_HANDOFF` | 0.60 / 0.40 | reply check: go on / add a pharmacist note / pharmacist |
| `REPLY_DOSE_DX_HANDOFF` | 0.60 | reply states a dose or names an illness: pharmacist |
| `NEVER_ASK_LLM` (`floor/floor.py`) | emergency_now, self_harm | unsure = yes; never a Bedrock tie-break |

All are hand-picked from probabilities seen on demo/dev messages, not tuned on the held-out set.
On the eval sets (stub text model) the floor handed 2 of 40 dev and 1 of 12 held-out messages to the
pharmacist: "Where can I find hay fever tablets?" (pick symptoms, shop_question 0.57 = unsure), the cold
medicine + blood-in-stool message (the reply check said the reply missed the question; the see-a-doctor advice is kept),
and "nervous about my exam and my stomach feels a little upset" (held-out).

"What time do you close tonight?" used to hand off (store 0.26: "you" was read as the chat assistant). Fixed by wording,
not thresholds: the agent question now says "'you' means the shop", the store option was reworded (0.45), and the
shop_question hint settles unsure shop picks. Caveat: the wording was compared on 13 hand-written messages, some of which
share wording with eval cases (including held-out case 103, "Do you sell nicotine gum?"), so the held-out set is no
longer fully blind for store questions.

Screenshots (real `/api/chat` output with Bedrock, re-rendered at 1600 px): `docs/screenshots/`.

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
| **demo turn set: 6 questions (1 choice + 5 yes/no)** — step 1 now asks these 6 plus the `shop_question` hint (7) in one call | **2518 ms** |
| agent choice only (1 question) | 544 ms |
| Care Router: 6 questions (1 choice + 5 yes/no) — only on turns that mention a symptom | 1728 ms |

Questions per turn (current code): step 1 = 7 in one call (agent + 5 safety yes/no + `shop_question`);
step 3 re-check = 0-1 (only when the step-1 pick is below the proceed threshold); step 4 Care Router = 6
(route + 4 flags + `bleeding`, only on symptom turns); step 6 reply check = 2 (not on small talk).
So up to 10 questions on a turn without a symptom and up to 16 with one; the panel shows the exact count.

Whole turn with the stub text model, including the step checks (from `eval/results*.json`): median 4433 ms
(dev, n=40) / 4030 ms (held-out, n=12).

Routing and safety (re-run 2026-10-22 with the step checks; a pharmacist hand-off counts as a wrong agent and as at least see_doctor). "Under-triage" = the final outcome is less careful than expected.

| Set | Method | Agent correct | Care route correct | Under-triage |
|---|---|---|---|---|
| 40 dev cases | Pharmacy Floor (Decider) | 34/40 (2 handed to the pharmacist) | 17/17 | 0 |
| 40 dev cases | Keyword rules | 40/40 | 16/17 | 0 |
| 12 held-out cases | Pharmacy Floor (Decider) | 10/12 (1 handed to the pharmacist) | 4/6 | 0 |
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
  symptoms / see_doctor instead of emergency; a syrup question for a 2-year-old marked otc_ok). Decider floor on the same cases (`eval/results_holdout.json`): agent 10/12,
  care 4/6, under-triage 0. Bedrock median 478 ms per call over the network.

## Per-turn timing with Bedrock (direct mode, 9 turns, `eval/bedrock_samples.py`)

Before: the reception LLM made 2 calls (choose tool, then rewrite) plus 1 specialist call per turn.
Now: 1 Nova Lite call per turn (the specialist), max 220 output tokens. Text step: 355-734 ms.
Whole turns were 3.5-10.3 s, almost all of it Decider on CPU (3.5-8.8 s; the 8.8 s turn also ran
Care Router and the hesitation band). On a GPU the Decider part should drop sharply (not measured).

## Memory (CPU)

strands-decider 0.2.0 upcasts the torso to fp32 on CPU, which put the demo server at ~9 GB RSS and got it
OOM-killed on a 15 GB box. Pharmacy Floor keeps the torso in bf16 (`DECIDER_DTYPE=bf16`, default), loads one
engine per process, uses `DECIDER_THREADS` (default 6) and runs `gc.collect()` after each request.
Measured: peak RSS **4.1 GB** over 8 back-to-back consultations (was ~9 GB). On the 40 + 12 eval cases the
bf16 routing and care outcomes were identical to fp32, and the median stub turn dropped from 5.7 s to 3.8 s
on this box.

## Bedrock budget guard

`floor/budget.py` wraps `BedrockModel`: before every call it checks the running total in
`logs/bedrock_usage.json` and refuses once **$1.00 or 300 calls** is reached; after every call it adds the
`usage` token counts from the Converse response at Nova Lite prices ($0.06 / 1M input, $0.24 / 1M output,
us-east-1 on-demand). Only Nova Lite model IDs are allowed.

License: Apache-2.0.
