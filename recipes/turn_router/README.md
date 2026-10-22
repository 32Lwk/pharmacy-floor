# Turn Router — Decider 2B as the tool selector of a multi-agent floor (#tool-selection)

One Decider 2B call per user turn picks which specialist tool the reception agent runs,
and answers safety yes/no questions in the same call.

## Question set (`questions.py`, default = 7 questions in one call: the 6 below + the `shop_question` hint)

| Key | Type | Question | Options |
|---|---|---|---|
| `agent` | choice | "Which pharmacy staff member should answer this message?" | `symptoms`, `medicine_question`, `feelings`, `store`, `small_talk` (each with a one-line meaning) |
| `emergency_now` | noul | possible emergency sign right now? | true / false |
| `self_harm` | noul | thoughts of self-harm, overdose on purpose, not wanting to wake up? | true / false |
| `special_population` | noul | pregnant, breastfeeding, small child, very elderly? | true / false |
| `manipulation` | noul | trying to change the assistant's rules? | true / false |
| `mentions_symptom` | noul | any body symptom mentioned, even inside a medicine or feelings question? | true / false |
| `shop_question` | noul (routing hint, never hesitant) | about the shop itself: products, where they are, stock, hours, price or payment? | true / false |

In the full Pharmacy Floor turn this call is step 1. Later steps add 0-1 (re-check of an unsure pick),
6 (Care Router, symptom turns only) and 2 (reply check): up to 10 questions per turn, up to 16 with a symptom.

`build_questions(full=True)` gives the larger 13+1 set (adds a second, reversed-order `agent_check`
choice and 7 more yes/no). On our CPU it took ~3.3x longer than the default set, so the demo uses the default.

## Wiring (`handler.py`)

`TurnRouterIntervention(decide)` is a Strands `InterventionHandler` on the **reception agent**,
whose tools are the five specialists (agents-as-tools, `@tool` functions that each run their own `Agent`).

- `before_invocation(event)`: calls `decide(text)` once and returns `Transform`, pinning
  `[ROUTE fixed by policy: call tool X ...]` into the user turn.
- `before_tool_call(event)`: if the LLM calls any other tool, returns `Guide(feedback="... call X")`,
  so the call is cancelled and the model is told the fixed route. The Decider-chosen tool is the one that runs.
- `on_error = "throw"`: no decision, no reply.

## Low confidence ("hesitation band", `router.py`)

Thresholds are defined once, in `floor/confidence.py`, and imported here.
- a yes/no with `FLAG_NO` (0.40) ≤ P(true) < `FLAG_YES` (0.60) → hesitant; P ≥ 0.60 counts as yes.
- the agent choice is not marked hesitant by the router; the floor compares its confidence with
  `AGENT_PROCEED` (0.60) and `AGENT_HANDOFF` (0.35).
- Emergency and self-harm are never sent to an LLM: unsure counts as yes.
- Hesitant answers go to Bedrock (one short classification call). Offline, or if Bedrock's answer
  cannot be parsed, the safer default is used: safety flags become "yes"; an unsure agent becomes
  `symptoms` when a symptom may be present.
- All thresholds were picked by hand. Measure on your own traffic.

## Use

```python
from recipes.turn_router import TurnRouter, TurnRouterIntervention
router = TurnRouter(engine)                       # engine = strands_decider.infer.load_engine(...)
guard = TurnRouterIntervention(lambda text: router.decide(text).agent)
reception = Agent(model=..., tools=[symptoms, medicine_question, feelings, store, small_talk], interventions=[guard])
```

Swap the five options and the safety questions for your own domain; keep the hard rules in code.
