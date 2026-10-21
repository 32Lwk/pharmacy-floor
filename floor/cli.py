import json, sys
from .floor import PharmacyFloor

DEMO = ["Hi there!", "Where can I find hay fever tablets?", "I have a mild headache since this morning.",
        "My bottom has been itchy around the anus for two days.", "I have itching in my genital area.",
        "Can I take this cold medicine? Also I've had blood in my stool this week.",
        "I'm so worried, I've had chest pain and can't breathe properly."]

if __name__ == "__main__":
    f = PharmacyFloor()
    print("LLM:", f.llm_label())
    for m in (sys.argv[1:] or DEMO):
        log = f.handle(m)
        print(json.dumps({"user": m, "agent": log.final_agent, "ms": round(log.total_ms),
                          "decisions": [(e.step, e.question, e.answer, e.decided_by) for e in log.entries
                                        if e.step != "turn_router" or e.question == "agent" or (e.confidence or 0) >= 0.3]}, ensure_ascii=False))
        print("REPLY:", log.reply, "\n", flush=True)
