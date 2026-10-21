# Care Router — a reusable Strands Decider 2B decision pattern (#policy)

**Three-door triage with a hard-rule override.** Decider 2B classifies the incoming
request *before the agent acts*; plain Python enforces the rules; the LLM only writes text
inside the chosen door.

## Pieces

| File | What it is |
|---|---|
| `questions.py` | Typed questions: one `ChoiceQuestion` (`route`: `otc_ok` / `see_doctor` / `emergency`) and five `NoulQuestion`s (`genital_area`, `anal_area`, `bleeding`, `strong_pain`, `long_duration`), each option with a one-line meaning. |
| `policy.py` | `apply_policy(...)`: hard rules + "low confidence → one step safer". No model inside. |
| `decider.py` | `CareRouter`: loads Decider 2B locally (`strands_decider.infer.load_engine`) and asks all questions about one `state` in a single `engine.ask(...)` call. |
| `handler.py` | `CareRouterIntervention`: a Strands `InterventionHandler` using `before_invocation`. It returns `Transform` to pin the route into the user turn before any model call. |

## Use it

```python
from strands import Agent
from recipes.care_router import CareRouter, CareRouterIntervention

guard = CareRouterIntervention(CareRouter(device="cpu"))   # or "cuda" / "mps"
agent = Agent(model=your_model, system_prompt="Follow the [CARE ROUTE] note.", interventions=[guard])
agent("My bottom is itchy and there was blood on the toilet paper.")
print(guard.last_decision.route, guard.last_decision.reasons)
```

## Adapt it to your domain

1. Swap the three doors for yours, ordered from least to most escalation
   (e.g. `self_serve` / `human_agent` / `urgent`). "One step safer" means "one step right".
2. Keep hard rules in `policy.py`, not in the prompt and not in a probability.
3. Ask **atomic** yes/no questions. In our tests, one combined question
   ("bleeding, strong pain, or long duration?") missed "blood on the toilet paper"
   (P(true)=0.19); a separate "any blood or bleeding?" question caught it (0.79).
4. Set thresholds on **your own** traffic. Ours live in `floor/confidence.py` (`ROUTE_PROCEED` 0.50,
   `CARE_FLAG_YES_GENITAL` 0.40, `CARE_FLAG_YES` 0.50) and were picked by hand on 30 made-up cases.
5. Fail closed: the handler uses `on_error = "throw"`, so no decision means no reply.
