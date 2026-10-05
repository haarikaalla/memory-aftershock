"""Evaluation entry points for retrieval and budgeted repair."""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path

from .datasets import load_longmemeval, sha256_file
from .model import train_dependency_model
from .repair import RepairEngine, Verdict
from .retrieval import MemoryRetriever
from .synthetic import dependency_examples, generate_cases, load_case_into_store


def evaluate_longmemeval_retrieval(dataset_path: str | Path, *, top_ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    data = load_longmemeval(dataset_path)
    per_type: dict[str, Counter] = {}
    totals = Counter()
    for item in data:
        truth = set(item.get("answer_session_ids") or [])
        if not truth or str(item["question_id"]).endswith("_abs"):
            continue
        retriever = MemoryRetriever.from_longmemeval_item(item)
        hits = retriever.search_sessions(
            str(item["question"]), question_date=str(item.get("question_date", "")), top_k=max(top_ks)
        )
        retrieved = [hit.doc_id for hit in hits]
        bucket = per_type.setdefault(item["question_type"], Counter())
        totals["questions"] += 1
        bucket["questions"] += 1
        for k in top_ks:
            hit = bool(truth & set(retrieved[:k]))
            full = truth.issubset(set(retrieved[:k]))
            totals[f"hit@{k}"] += int(hit)
            totals[f"full@{k}"] += int(full)
            bucket[f"hit@{k}"] += int(hit)
            bucket[f"full@{k}"] += int(full)
    metrics = _counter_to_rates(totals, top_ks)
    metrics["dataset_sha256"] = sha256_file(Path(dataset_path))
    metrics["by_question_type"] = {
        question_type: _counter_to_rates(counter, top_ks) for question_type, counter in sorted(per_type.items())
    }
    return metrics


def evaluate_repair(*, n_cases: int = 160, budget: int = 2, seed: int = 13) -> dict:
    cases = generate_cases(n_cases=n_cases, seed=seed)
    split = int(len(cases) * 0.65)
    model, model_metrics = train_dependency_model(dependency_examples(cases[:split]))
    strategies = ["recorded", "learned", "exhaustive"]
    results = {
        strategy: _evaluate_strategy(cases[split:], strategy=strategy, budget=budget, model=model)
        for strategy in strategies
    }
    return {"model": model_metrics, "repair": results, "n_cases": n_cases, "seed": seed}


def evaluate_repair_budget_curve(
    *, n_cases: int = 160, budgets: tuple[int, ...] = (1, 2, 3, 4), seed: int = 13
) -> dict:
    cases = generate_cases(n_cases=n_cases, seed=seed)
    split = int(len(cases) * 0.65)
    model, model_metrics = train_dependency_model(dependency_examples(cases[:split]))
    test_cases = cases[split:]
    curve = []
    for budget in budgets:
        row = {"budget": budget}
        for strategy in ("recorded", "learned", "exhaustive"):
            row[strategy] = _evaluate_strategy(
                test_cases,
                strategy=strategy,
                budget=budget,
                model=model,
            )
        curve.append(row)
    return {
        "model": model_metrics,
        "budget_curve": curve,
        "n_cases": n_cases,
        "seed": seed,
        "budgets": list(budgets),
    }


def _evaluate_strategy(cases, *, strategy: str, budget: int, model) -> dict:
    precision_values: list[float] = []
    recall_values: list[float] = []
    preservation_values: list[float] = []
    checked_values: list[int] = []
    for case in cases:
        store = load_case_into_store(case, observed=True)
        verifier = _synthetic_verifier(frozenset(case.impacted_ids), case.corrupted_root)

        engine = RepairEngine(store, model=model if strategy == "learned" else None)
        result = engine.repair(
            case.corrupted_root,
            corrected_root_text=case.correct_root_text,
            strategy=strategy,
            budget=budget,
            verifier=verifier,
        )
        found = set(result.invalidated) | set(result.replaced) | set(result.quarantined)
        truth = set(case.impacted_ids)
        checked = set(result.checked)
        precision_values.append(len(found & truth) / len(found) if found else 0.0)
        recall_values.append(len(found & truth) / len(truth))
        non_impacted = {memory.memory_id for memory in case.memories} - truth - {case.corrupted_root}
        touched_non_impacted = len(checked & non_impacted)
        preservation_values.append(1.0 - touched_non_impacted / len(non_impacted))
        checked_values.append(len(checked))
    return {
        "repair_precision": statistics.fmean(precision_values),
        "repair_recall": statistics.fmean(recall_values),
        "correct_memory_preservation": statistics.fmean(preservation_values),
        "mean_checked": statistics.fmean(checked_values),
        "budget": budget,
        "cases": len(cases),
    }


def _counter_to_rates(counter: Counter, top_ks: tuple[int, ...]) -> dict:
    questions = int(counter["questions"])
    metrics: dict[str, float | int] = {"questions": questions}
    for k in top_ks:
        metrics[f"hit@{k}"] = counter[f"hit@{k}"] / questions if questions else 0.0
        metrics[f"full_evidence@{k}"] = counter[f"full@{k}"] / questions if questions else 0.0
    return metrics


def _synthetic_verifier(impacted: frozenset[str], corrupted_root: str):
    def verifier(memory):
        if memory.memory_id in impacted:
            return Verdict(
                status="invalid",
                evidence="Synthetic oracle: downstream memory depended on corrected root.",
                replacement=f"Needs revision after correction to {corrupted_root}.",
            )
        return Verdict(status="valid", evidence="Synthetic oracle: unrelated memory.")

    return verifier
