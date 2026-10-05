# Memory Aftershock

Memory Aftershock is an end-to-end research project for agent memory repair. It asks a practical frontier question: when an AI agent corrects one stored memory, how should it find and fix downstream memories that were built from the old fact?

The project combines a real long-memory benchmark, a synthetic repair benchmark with hidden dependency truth, an append-only memory store, a learned dependency estimator, a budgeted repair engine, a transparent retrieval agent, tests, and a small dashboard.

## Why this is different

Most memory systems focus on retrieving the right old fact. Memory Aftershock focuses on the next failure: a wrong old fact can already have influenced recommendations, plans, and summaries. The repair engine ranks downstream memories to verify when explicit provenance is incomplete.

## Data

The retrieval benchmark uses `xiaowu0162/longmemeval-cleaned`, pinned to revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`. The raw dataset is not committed because it is about 277 MB.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
aftershock download-data --output data/raw/longmemeval_s_cleaned.json
```

## Reproduce results

```bash
aftershock eval-retrieval --dataset data/raw/longmemeval_s_cleaned.json
aftershock eval-repair --cases 160 --budget 2 --seed 13
pytest
```

## Measured results

These numbers were produced by the code in this repository on the pinned LongMemEval-cleaned file and the controlled aftershock benchmark. Result JSON files are committed in `results/`.

### LongMemEval-cleaned retrieval

470 scored questions were evaluated. Abstention/no-answer cases are excluded from retrieval scoring.

| Metric | Score |
| --- | ---: |
| hit@1 | 0.7298 |
| hit@3 | 0.9000 |
| hit@5 | 0.9404 |
| full_evidence@1 | 0.2213 |
| full_evidence@3 | 0.6723 |
| full_evidence@5 | 0.7809 |

Dataset SHA-256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`

### Budgeted aftershock repair

The synthetic benchmark uses hidden dependency truth, incomplete recorded edges, and a verification budget of 2 downstream memories per correction.

| Strategy | Precision | Recall | Correct memory preservation | Mean checked |
| --- | ---: | ---: | ---: | ---: |
| recorded provenance | 0.8571 | 0.6607 | 1.0000 | 1.3214 |
| learned dependency | 1.0000 | 1.0000 | 1.0000 | 2.0000 |
| chronological sweep | 0.0000 | 0.0000 | 0.0000 | 2.0000 |

The learned dependency model reached ROC-AUC `1.0000` and average precision `1.0000` on the controlled synthetic split. That should be read as a benchmark sanity result, not as a claim that real agent repair is solved.

## Run the demo

```bash
aftershock serve --port 8000
```

Open `http://127.0.0.1:8000`. The demo exposes the memory store, retrieval agent, and repair action.

## What is measured

- `hit@k`: at least one ground-truth source session appears in the top-k retrieved sessions.
- `full_evidence@k`: every ground-truth source session appears in the top-k.
- `repair_precision`: fraction of flagged downstream memories that truly depended on the corrupted root.
- `repair_recall`: fraction of truly impacted downstream memories that were found under the verification budget.
- `correct_memory_preservation`: fraction of unrelated memories left untouched.

## Project layout

```text
src/aftershock/
  retrieval.py      leakage-safe BM25 plus turn/session fusion
  store.py          append-only SQLite memory store
  model.py          learned dependency estimator
  repair.py         budgeted repair engine
  synthetic.py      controlled aftershock benchmark
  evaluation.py     reproducible metrics
  agent.py          transparent retrieval agent
  api.py            FastAPI dashboard backend
static/
  dashboard.html
tests/
  focused unit tests
```

## Research direction

This repo is ready to grow into an LNCS-style paper around:

**Memory Aftershock: Budgeted Repair of Agent Memories under Incomplete Provenance**

The strongest next experiments are adding LLM-based verification, real tool traces, and a comparison against memory editing or rollback baselines.
