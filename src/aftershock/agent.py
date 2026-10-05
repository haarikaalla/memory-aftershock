"""A transparent retrieval-plus-memory agent for demos and API use."""

from __future__ import annotations

from dataclasses import dataclass

from .retrieval import BM25Index, Document
from .store import MemoryStore


@dataclass(frozen=True)
class AgentAnswer:
    answer: str
    citations: tuple[str, ...]
    tool_trace: tuple[dict, ...]


class EvidenceAgent:
    def __init__(self, store: MemoryStore, *, context_budget_chars: int = 1200) -> None:
        self.store = store
        self.context_budget_chars = context_budget_chars

    def answer(self, query: str, *, top_k: int = 5) -> AgentAnswer:
        active = [memory for memory in self.store.all_latest() if memory.status == "active"]
        docs = [
            Document(
                doc_id=memory.memory_id,
                text=memory.text,
                timestamp=str(memory.timestamp),
                metadata={"memory_id": memory.memory_id, "entity": memory.entity, "status": memory.status},
            )
            for memory in active
        ]
        hits = BM25Index(docs).search(query, top_k=top_k)
        trace = [{"tool": "retrieve_memory", "query": query, "hits": [hit.doc_id for hit in hits]}]
        if not hits:
            return AgentAnswer(
                answer="I do not have enough memory evidence to answer.",
                citations=(),
                tool_trace=tuple(trace),
            )
        snippets: list[str] = []
        used = 0
        for hit in hits:
            memory = self.store.latest(hit.doc_id)
            piece = f"[{memory.memory_id}] {memory.text}"
            if used + len(piece) > self.context_budget_chars:
                break
            snippets.append(piece)
            used += len(piece)
        return AgentAnswer(
            answer="Relevant memory evidence:\n" + "\n".join(snippets),
            citations=tuple(hit.doc_id for hit in hits[: len(snippets)]),
            tool_trace=tuple(trace),
        )
