"""Synthetic aftershock benchmark with known hidden dependency truth."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .store import Memory, MemoryStore

PRODUCTS = ["visa-card", "linux-laptop", "asthma-plan", "data-course", "travel-profile"]
ENTITIES = ["maya", "ravi", "sofia", "liam", "nora"]
PREFERENCES = ["vegetarian", "low latency", "budget", "privacy", "accessibility"]


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    memories: tuple[Memory, ...]
    true_edges: tuple[tuple[str, str], ...]
    observed_edges: tuple[tuple[str, str], ...]
    corrupted_root: str
    correct_root_text: str
    impacted_ids: tuple[str, ...]


def generate_cases(n_cases: int = 160, *, seed: int = 13) -> list[SyntheticCase]:
    rng = random.Random(seed)
    cases: list[SyntheticCase] = []
    for idx in range(n_cases):
        entity = rng.choice(ENTITIES)
        product = rng.choice(PRODUCTS)
        preference = rng.choice(PREFERENCES)
        root_id = f"c{idx:03d}-m0"
        old_value = f"{entity} prefers {preference} for {product}."
        corrected = f"{entity} no longer prefers {preference} for {product}; use verified current profile."
        memories = [
            Memory(root_id, old_value, entity, 0, sources=(f"case-{idx}",), kind="preference"),
            Memory(
                f"c{idx:03d}-m1",
                f"Unrelated note: {entity} asked for concise summaries.",
                entity,
                1,
                sources=(f"case-{idx}-note",),
                kind="note",
            ),
            Memory(
                f"c{idx:03d}-m2",
                f"{entity} prefers weekly status reports for collaborative work.",
                entity,
                2,
                sources=(f"case-{idx}-note",),
                kind="note",
            ),
            Memory(
                f"c{idx:03d}-m3",
                f"Recommend {product} options because {entity} prefers {preference}.",
                entity,
                3,
                sources=(f"case-{idx}",),
                kind="recommendation",
            ),
            Memory(
                f"c{idx:03d}-m4",
                f"Schedule follow-up around {preference} constraints for {entity}.",
                entity,
                4,
                sources=(f"case-{idx}",),
                kind="plan",
            ),
        ]
        true_edges = ((root_id, memories[3].memory_id), (root_id, memories[4].memory_id))
        observed_edges = tuple(edge for edge in true_edges if rng.random() > 0.35)
        cases.append(
            SyntheticCase(
                case_id=f"case-{idx:03d}",
                memories=tuple(memories),
                true_edges=true_edges,
                observed_edges=observed_edges,
                corrupted_root=root_id,
                correct_root_text=corrected,
                impacted_ids=(memories[3].memory_id, memories[4].memory_id),
            )
        )
    return cases


def load_case_into_store(case: SyntheticCase, *, observed: bool = True) -> MemoryStore:
    store = MemoryStore()
    with store.transaction():
        for memory in case.memories:
            store.put(memory)
        for parent_id, child_id in case.observed_edges if observed else case.true_edges:
            store.add_edge(parent_id, child_id)
    return store


def dependency_examples(cases: list[SyntheticCase]) -> list[tuple[Memory, Memory, int]]:
    examples: list[tuple[Memory, Memory, int]] = []
    for case in cases:
        truth = set(case.true_edges)
        memories = list(case.memories)
        for parent in memories:
            for child in memories:
                if parent.timestamp < child.timestamp:
                    examples.append((parent, child, 1 if (parent.memory_id, child.memory_id) in truth else 0))
    return examples
