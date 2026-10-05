"""Budgeted repair engine for memory aftershocks."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass

from .model import DependencyModel
from .store import Memory, MemoryStore


@dataclass(frozen=True)
class Verdict:
    status: str
    evidence: str
    replacement: str | None = None


@dataclass(frozen=True)
class RepairCandidate:
    memory_id: str
    risk: float
    reason: str


@dataclass(frozen=True)
class RepairResult:
    checked: tuple[str, ...]
    invalidated: tuple[str, ...]
    replaced: tuple[str, ...]
    quarantined: tuple[str, ...]
    preserved: tuple[str, ...]
    trace: tuple[dict, ...]


Verifier = Callable[[Memory], Verdict]


class RepairEngine:
    def __init__(
        self,
        store: MemoryStore,
        *,
        model: DependencyModel | None = None,
        inferred_threshold: float = 0.45,
    ) -> None:
        self.store = store
        self.model = model
        self.inferred_threshold = inferred_threshold

    def plan(
        self,
        root_id: str,
        *,
        strategy: str,
        budget: int,
    ) -> list[RepairCandidate]:
        memories = {memory.memory_id: memory for memory in self.store.all_latest()}
        root = memories[root_id]
        edge_children: defaultdict[str, list[str]] = defaultdict(list)
        for parent_id, child_id in self.store.edges():
            edge_children[parent_id].append(child_id)
        risks: dict[str, tuple[float, str]] = {}
        if strategy in {"recorded", "learned"}:
            queue: deque[tuple[str, float]] = deque([(root_id, 1.0)])
            while queue:
                parent_id, parent_risk = queue.popleft()
                for child_id in edge_children[parent_id]:
                    child = memories[child_id]
                    if child.status in {"invalid", "superseded"}:
                        continue
                    risk = max(risks.get(child_id, (0.0, ""))[0], parent_risk)
                    risks[child_id] = (risk, "recorded-edge")
                    queue.append((child_id, risk))
        if strategy == "learned" and self.model is not None:
            for child in memories.values():
                if child.memory_id == root_id or child.timestamp <= root.timestamp:
                    continue
                if child.status in {"invalid", "superseded"}:
                    continue
                proba = self.model.predict_proba(root, child)
                if proba >= self.inferred_threshold:
                    old = risks.get(child.memory_id, (0.0, ""))[0]
                    if proba > old:
                        risks[child.memory_id] = (proba, "learned-edge")
        if strategy == "exhaustive":
            for child in memories.values():
                if child.memory_id != root_id and child.timestamp > root.timestamp:
                    risks[child.memory_id] = (1.0, "chronological-sweep")
        ranked = sorted(
            (
                RepairCandidate(memory_id=memory_id, risk=risk, reason=reason)
                for memory_id, (risk, reason) in risks.items()
            ),
            key=lambda item: (-item.risk, memories[item.memory_id].timestamp, item.memory_id),
        )
        return ranked[: max(budget, 0)]

    def repair(
        self,
        root_id: str,
        *,
        corrected_root_text: str,
        strategy: str,
        budget: int,
        verifier: Verifier,
    ) -> RepairResult:
        trace: list[dict] = []
        checked: list[str] = []
        invalidated: list[str] = []
        replaced: list[str] = []
        quarantined: list[str] = []
        preserved: list[str] = []
        candidates = self.plan(root_id, strategy=strategy, budget=budget)
        with self.store.transaction():
            root = self.store.latest(root_id)
            self.store.put(root.replacement(corrected_root_text), event_type="root-corrected")
            for candidate in candidates:
                memory = self.store.latest(candidate.memory_id)
                verdict = verifier(memory)
                checked.append(memory.memory_id)
                trace.append(
                    {
                        "memory_id": memory.memory_id,
                        "risk": candidate.risk,
                        "reason": candidate.reason,
                        "verdict": verdict.status,
                    }
                )
                if verdict.status == "invalid":
                    if verdict.replacement:
                        self.store.put(memory.replacement(verdict.replacement), event_type="aftershock-replaced")
                        replaced.append(memory.memory_id)
                    else:
                        self.store.put(memory.with_status("invalid"), event_type="aftershock-invalidated")
                        invalidated.append(memory.memory_id)
                elif verdict.status == "unknown":
                    self.store.put(memory.with_status("needs_review"), event_type="aftershock-quarantined")
                    quarantined.append(memory.memory_id)
                else:
                    preserved.append(memory.memory_id)
            self.store.event(root_id, "repair-trace", {"strategy": strategy, "budget": budget, "trace": trace})
        return RepairResult(
            checked=tuple(checked),
            invalidated=tuple(invalidated),
            replaced=tuple(replaced),
            quarantined=tuple(quarantined),
            preserved=tuple(preserved),
            trace=tuple(trace),
        )
