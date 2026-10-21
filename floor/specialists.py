"""Five specialist Strands agents, exposed to the reception agent as tools (agents-as-tools)."""
from __future__ import annotations

import re

from strands import Agent, tool

from recipes.care_router import CareRouterIntervention

from .llm import make_model
from .store_data import lookup

BASE = ("You work on a drugstore floor in Japan. Reply in 2-4 short sentences in English. Never diagnose: never name a likely condition or cause "
        "(no 'you might have ...', 'it sounds like ...', 'this could be ...'); describe care options and signs to see a doctor instead. "
        "Never name prescription-only drugs. If a [CARE ROUTE] note is present, follow it exactly.")

PROMPTS = {
    "symptoms": "ROLE=symptoms. You help with body symptoms. " + BASE,
    "medicine_question": "ROLE=medicine_question. You answer general questions about over-the-counter medicines. NEVER state a dose, an amount, an interval or a number of times per day; for any dosing question say to follow the package label and ask the pharmacist. " + BASE,
    "feelings": "ROLE=feelings. You listen kindly to worries. You do not give medical advice. " + BASE,
    "store": "ROLE=store. You help find things in the Demo Store. State ONLY what is in the STORE FACTS; if something is not there, say a staff member at the counter can help. Mention that this is a demo store. " + BASE,
    "small_talk": "ROLE=small_talk. You handle greetings and questions about this assistant. " + BASE,
}


class Specialists:
    def __init__(self, care_guard: CareRouterIntervention):
        self.care_guard = care_guard
        self.care_note: str | None = None
        self.last_output: str | None = None
        self.store_facts: str | None = None  # set by the floor when Care Router ran this turn

    def _run(self, role: str, message: str, interventions=None) -> str:
        m = make_model(max_tokens=220); m.label = f"specialist:{role}"
        agent = Agent(model=m, system_prompt=PROMPTS[role], interventions=interventions or [],
                      callback_handler=None)
        out = re.sub(r"<thinking>.*?</thinking>", "", str(agent(message)), flags=re.S).strip()
        self.last_output = out
        return out

    def tools(self):
        sp = self

        @tool
        def symptoms(message: str) -> str:
            """Specialist for body symptoms. Care Router runs before it (InterventionHandler)."""
            return sp._run("symptoms", message, [sp.care_guard])

        @tool
        def medicine_question(message: str) -> str:
            """Specialist for questions about a specific over-the-counter medicine."""
            return sp._run("medicine_question", message + (f"\n{sp.care_note}" if sp.care_note else ""))

        @tool
        def feelings(message: str) -> str:
            """Specialist that listens to worries and feelings."""
            return sp._run("feelings", message + (f"\n{sp.care_note}" if sp.care_note else ""))

        @tool
        def store(message: str) -> str:
            """Specialist for shop questions: aisles, stock, hours."""
            sp.store_facts = lookup(message)
            return sp._run("store", f"{message}\nSTORE FACTS: {sp.store_facts}")

        @tool
        def small_talk(message: str) -> str:
            """Specialist for greetings and questions about the assistant."""
            return sp._run("small_talk", message)

        return [symptoms, medicine_question, feelings, store, small_talk]
