"""Dataset loading and manifest helpers."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

LONGMEMEVAL_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
LONGMEMEVAL_S_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
    f"{LONGMEMEVAL_REVISION}/longmemeval_s_cleaned.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_longmemeval(path: str | Path) -> list[dict]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError("LongMemEval file must contain a list of question objects.")
    required = {"question_id", "question", "question_date", "answer_session_ids", "haystack_sessions"}
    missing = required - set(data[0])
    if missing:
        raise ValueError(f"LongMemEval item is missing fields: {sorted(missing)}")
    return data


def download_longmemeval_s(target: str | Path) -> Path:
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(LONGMEMEVAL_S_URL, target_path)
    return target_path
