"""Small, budget-guarded Bedrock run: 5 demo turns, one hesitation turn, Bedrock-only routing on the 12 held-out cases."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strands import Agent
from floor import budget
from floor.floor import PharmacyFloor
from floor.llm import make_model, model_label

OUT = Path(__file__).resolve().parents[2] / "bedrock_samples.md"
DEMO = ["I have a mild headache since this morning.", "My bottom has been itchy around the anus for two days.",
        "My bottom is itchy and there was blood on the toilet paper.", "I have itching in my genital area.",
        "Sudden chest pain and I can't breathe properly."]
HESITATE = "Where can I find hay fever tablets?"
OTHERS = ["What time do you close tonight?", "How many times a day can I take ibuprofen tablets?",
          "I feel so anxious lately and just need someone to talk to."]
md = [f"# Pharmacy Floor — real Bedrock replies\n\nModel: `{model_label()}`. Generated {time.strftime('%Y-%m-%d %H:%M %Z')}.\n"]
floor = PharmacyFloor()
for title, msgs in (("Demo turns", DEMO), ("Other specialists (store, medicine_question, feelings)", OTHERS), ("Hesitation-band turn", [HESITATE])):
    md.append(f"\n## {title}\n")
    for m in msgs:
        log = floor.handle(m)
        md.append(f"\n### User: {m}\n\n- Specialist: `{log.final_agent}` · turn {log.total_ms:.0f} ms\n")
        for e in log.entries:
            if e.step == "turn_router" and e.question != "agent" and (e.confidence or 0) < 0.4:
                continue
            c = "" if e.confidence is None else f" ({e.confidence:.2f})"
            md.append(f"- `{e.step}` {e.question}: **{e.answer}**{c} — {e.decided_by}" + (f", {e.ms:.0f} ms" if e.ms else "") + "\n")
        md.append(f"\n> {log.reply}\n")
        tx = next((e.ms for e in log.entries if e.step == "text"), 0)
        dm = sum(e.ms or 0 for e in log.entries if e.decided_by == "decider-2b")
        print(f"{m[:45]:<45} -> {log.final_agent:<17} total {log.total_ms:6.0f} ms | decider {dm:5.0f} | text {tx or 0:5.0f}", flush=True)

if "--turns-only" in sys.argv:
    old = OUT.read_text() if OUT.exists() else ""
    tail = old[old.find("\n## Bedrock-only"):] if "## Bedrock-only" in old else ""
    s = budget.load()
    OUT.write_text("".join(md) + tail.split("\n## Spend")[0] + f"\n## Spend (cumulative)\n\n{s['calls']} Bedrock calls, {s['input_tokens']} input + {s['output_tokens']} output tokens = **${s['usd']:.6f}** (Nova Lite $0.06/M in, $0.24/M out). Cap: $1.00 or 300 calls.\n")
    print("spend", s["calls"], s["usd"]); sys.exit(0)

# Bedrock-only routing (no Decider) on held-out cases
AGENTS = ["symptoms", "medicine_question", "feelings", "store", "small_talk", "emergency"]
PROMPT = ("Classify this drugstore chat message. Reply ONLY JSON like "
          '{"agent": one of ' + json.dumps(AGENTS) + ', "care": one of ["otc_ok","see_doctor","emergency",null]}. '
          "care is null if no body symptom is mentioned. Genital-area symptoms always see_doctor.\nMessage: ")
cases = [json.loads(l) for l in open(Path(__file__).parent / "cases_holdout.jsonl")]
floor_rows = {r["id"]: r for r in json.load(open(Path(__file__).parent / "results_holdout.json"))["rows"]}
md.append("\n## Bedrock-only routing vs Decider (12 held-out cases)\n\n| # | expected agent / care | Bedrock only | ms | Decider floor |\n|---|---|---|---|---|\n")
ok_a = ok_c = under = 0; lat = []
ORDER = [None, "otc_ok", "see_doctor", "emergency"]
for c in cases:
    m = make_model(max_tokens=40); m.label = f"route-only:{c['id']}"
    t0 = time.perf_counter()
    try:
        raw = str(Agent(model=m, callback_handler=None)(PROMPT + c["text"])).strip()
        j = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except Exception as e:
        j = {"agent": f"error:{type(e).__name__}", "care": None}
    ms = (time.perf_counter() - t0) * 1000; lat.append(ms)
    a, k = j.get("agent"), j.get("care")
    ok_a += a == c["agent"]; ok_c += bool(c["care"]) and k == c["care"]
    exp = "emergency" if c["agent"] == "emergency" else c["care"]
    got = "emergency" if a == "emergency" else (k if k in ORDER else None)
    under += bool(exp) and ORDER.index(got) < ORDER.index(exp)
    f = floor_rows[c["id"]]
    md.append(f"| {c['id']} | {c['agent']} / {c['care']} | {a} / {k} | {ms:.0f} | {f['floor_agent']} / {f['floor_care']} |\n")
lat.sort()
md.append(f"\nBedrock only: agent {ok_a}/12, care {ok_c}/6, under-triage {under}, median latency {lat[len(lat)//2]:.0f} ms (network round trip from Japan box to us-east-1).\n")
s = budget.load()
md.append(f"\n## Spend\n\n{s['calls']} Bedrock calls, {s['input_tokens']} input + {s['output_tokens']} output tokens = **${s['usd']:.6f}** "
          f"(Nova Lite $0.06/M in, $0.24/M out). Cap: $1.00 or 300 calls.\n")
OUT.write_text("".join(md))
print("agent", ok_a, "care", ok_c, "under", under, "spend", s)
