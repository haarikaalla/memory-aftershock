"""FastAPI app for the Memory Aftershock demo."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agent import EvidenceAgent
from .repair import RepairEngine, Verdict
from .store import Memory, MemoryStore

app = FastAPI(title="Memory Aftershock", version="0.1.0")


class AskRequest(BaseModel):
    query: str


class RepairRequest(BaseModel):
    root_id: str
    corrected_text: str
    budget: int = 2


def demo_store() -> MemoryStore:
    store = MemoryStore()
    with store.transaction():
        store.put(Memory("m0", "Maya prefers vegetarian restaurants.", "maya", 0, sources=("demo",)))
        store.put(
            Memory(
                "m1",
                "Recommend vegetarian restaurants for Maya in travel plans.",
                "maya",
                1,
                sources=("demo",),
                kind="recommendation",
            )
        )
        store.put(Memory("m2", "Maya asked for concise summaries.", "maya", 2, sources=("demo-note",)))
        store.add_edge("m0", "m1")
    return store


STORE = demo_store()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")


@app.get("/memories")
def memories() -> list[dict]:
    return [memory.__dict__ for memory in STORE.all_latest()]


@app.get("/graph")
def graph() -> dict:
    memories = STORE.all_latest()
    return {
        "nodes": [
            {
                "id": memory.memory_id,
                "label": memory.text,
                "status": memory.status,
                "kind": memory.kind,
                "timestamp": memory.timestamp,
            }
            for memory in memories
        ],
        "edges": [{"source": parent, "target": child} for parent, child in STORE.edges()],
        "events": STORE.events()[-8:],
    }


@app.get("/plan/{root_id}")
def plan(root_id: str, strategy: str = "recorded", budget: int = 3) -> dict:
    candidates = RepairEngine(STORE).plan(root_id, strategy=strategy, budget=budget)
    return {
        "root_id": root_id,
        "strategy": strategy,
        "budget": budget,
        "candidates": [candidate.__dict__ for candidate in candidates],
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    answer = EvidenceAgent(STORE).answer(request.query)
    return {
        "answer": answer.answer,
        "citations": answer.citations,
        "tool_trace": answer.tool_trace,
    }


@app.post("/repair")
def repair(request: RepairRequest) -> dict:
    def verifier(memory: Memory) -> Verdict:
        if request.root_id == "m0" and memory.memory_id == "m1":
            return Verdict(
                status="invalid",
                evidence="Demo verifier: recommendation inherited corrected preference.",
                replacement="Re-check restaurant recommendations against Maya's current profile.",
            )
        return Verdict(status="valid", evidence="Demo verifier: no dependency found.")

    result = RepairEngine(STORE).repair(
        request.root_id,
        corrected_root_text=request.corrected_text,
        strategy="recorded",
        budget=request.budget,
        verifier=verifier,
    )
    return result.__dict__


from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
