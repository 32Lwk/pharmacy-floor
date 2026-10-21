# Real API notes (checked 2026-10-22 against installed code, not guessed)

Sources
- Blog: https://strandsagents.com/blog/introducing-strands-decider/ (CLI usage, `Decider.noul` example, `InterventionHandler` / `before_tool_call`, `Agent(interventions=[...])`, Proceed/Deny/Confirm/Guide, ~115 ms median on RTX 3090, ~153 ms on M3).
- Model card: https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19 (LoRA adapter on `Qwen/Qwen3.5-2B-Base` + readout head, Apache-2.0; code at https://github.com/strands-labs/strands-decider).
- PyPI `strands-decider` 0.2.0 and `strands-agents` 1.61.1, read from the installed source.

strands-decider 0.2.0
- CLI: `strands-decider ask <checkpoint> --state ... --choice "q?=a,b" --noul "q?" --score "q?=low,high" [--device cpu|cuda|mps|mlx] [--json]`; `strands-decider serve` (HTTP `POST /v1/systemone`).
- Python: `strands_decider.infer.load_engine(checkpoint, device="cuda"|"mps"|"cpu"|"mlx")` -> `SystemOneEngine`; `engine.ask(state, questions)` / `engine.evaluate(SystemOneRequest)` -> `SystemOneResponse`.
- Question types in `strands_decider.schema`: `NoulQuestion(instructions, criteria={"true","false"})`, `ChoiceQuestion(instructions, criteria={name: description})`, `ScoreQuestion(instructions, criteria=[ascending levels])`.
- Answers: `NoulAnswer.noul` (P(true)), `ChoiceAnswer.choice/.probabilities/.confidence`, `ScoreAnswer.score/.legend/.probabilities/.confidence`.
- `Decider.noul(...)` from the blog is NOT in the 0.2.0 package (the package `__init__` exports nothing); the blog says integration libraries are coming. This repo uses the schema classes directly.

strands-agents 1.61.1
- `from strands.interventions import InterventionHandler, Proceed, Deny, Guide, Confirm, Transform`
- Handler methods: `before_invocation`, `before_tool_call`, `after_tool_call`, `before_model_call`, `after_model_call`; property `on_error`.
- `Agent(model=..., interventions=[...])`. Bedrock: `strands.models.bedrock.BedrockModel`, default id `global.anthropic.claude-sonnet-4-6` in this version.

Where it runs: locally, in-process, CPU on this box (8 vCPU, no GPU). Warning on CPU: `flash-linear-attention` not installed, so transformers uses a slower reference kernel.
