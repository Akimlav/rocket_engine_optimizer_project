"""Retrieve-augment-generate pipeline for engine simulation runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Config, load
from .llm import LocalLLM
from .store import Store

SYSTEM_PROMPT = """You are an assistant for a rocket-engine nozzle-optimizer project.
You answer questions by reasoning strictly over the provided simulation run records.
Each record has a run_name, inputs (fuel, oxidizer, chamber/tank pressure, throat
radius, injector area, etc.), and performance metrics (thrust, Isp, exit Mach,
area ratio, ...). Some records also contain optimization results.

Rules:
- Cite run_name whenever you reference a specific run.
- Compare numerically when the question asks for a "best" or "which" comparison.
- Report physical units. Thrust in N, Isp in s, pressures in Pa, lengths in m.
- If the records do not contain the answer, say so — do NOT invent numbers.
""".strip()


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[str]
    hits: list[dict[str, Any]]


class RAG:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load()
        self.llm = LocalLLM(self.cfg)
        self.store = Store(self.cfg, llm=self.llm)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        hits = self.store.query(question, top_k=top_k)
        if not hits:
            return Answer(
                question=question,
                answer="No indexed runs yet. Run `python -m rag_memory.cli index` first.",
                sources=[],
                hits=[],
            )
        context = "\n\n---\n\n".join(
            f"[run_name={h['run_id']}]\n{h['document']}" for h in hits
        )
        user_msg = f"Question: {question}\n\nRelevant run records:\n{context}\n\nAnswer:"
        reply = self.llm.chat(SYSTEM_PROMPT, user_msg)
        return Answer(
            question=question,
            answer=reply.strip(),
            sources=[h["run_id"] for h in hits],
            hits=hits,
        )
