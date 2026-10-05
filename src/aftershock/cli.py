"""Command line interface for Memory Aftershock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import download_longmemeval_s, sha256_file
from .evaluation import (
    evaluate_longmemeval_retrieval,
    evaluate_repair,
    evaluate_repair_budget_curve,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory Aftershock")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download-data", help="Download LongMemEval-cleaned small split.")
    download.add_argument("--output", default="data/raw/longmemeval_s_cleaned.json")

    retrieval = sub.add_parser("eval-retrieval", help="Evaluate leakage-safe session retrieval.")
    retrieval.add_argument("--dataset", required=True)
    retrieval.add_argument("--output", default="results/longmemeval_retrieval.json")

    repair = sub.add_parser("eval-repair", help="Evaluate synthetic aftershock repair.")
    repair.add_argument("--cases", type=int, default=160)
    repair.add_argument("--budget", type=int, default=2)
    repair.add_argument("--seed", type=int, default=13)
    repair.add_argument("--output", default="results/repair_benchmark.json")

    curve = sub.add_parser("eval-budget-curve", help="Evaluate repair quality across budgets.")
    curve.add_argument("--cases", type=int, default=160)
    curve.add_argument("--budgets", default="1,2,3,4")
    curve.add_argument("--seed", type=int, default=13)
    curve.add_argument("--output", default="results/budget_curve.json")

    serve = sub.add_parser("serve", help="Run the demo API and dashboard.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    if args.command == "download-data":
        path = download_longmemeval_s(args.output)
        print(json.dumps({"path": str(path), "sha256": sha256_file(path)}, indent=2))
    elif args.command == "eval-retrieval":
        result = evaluate_longmemeval_retrieval(args.dataset)
        _write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "eval-repair":
        result = evaluate_repair(n_cases=args.cases, budget=args.budget, seed=args.seed)
        _write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "eval-budget-curve":
        budgets = tuple(int(value.strip()) for value in args.budgets.split(",") if value.strip())
        result = evaluate_repair_budget_curve(n_cases=args.cases, budgets=budgets, seed=args.seed)
        _write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("aftershock.api:app", host=args.host, port=args.port, reload=False)


def _write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
