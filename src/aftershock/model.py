"""Learned missing-dependency estimator."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .store import Memory
from .text import jaccard, tokenize

FEATURE_NAMES = [
    "bias",
    "token_jaccard",
    "parent_coverage",
    "same_entity",
    "time_gap_inverse",
    "shared_source",
]


@dataclass(frozen=True)
class DependencyFeatures:
    values: tuple[float, ...]

    @classmethod
    def from_memories(cls, parent: Memory, child: Memory) -> DependencyFeatures:
        if parent.timestamp >= child.timestamp:
            return cls((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        parent_tokens = set(tokenize(parent.text))
        child_tokens = set(tokenize(child.text))
        coverage = len(parent_tokens & child_tokens) / len(parent_tokens) if parent_tokens else 0.0
        gap = child.timestamp - parent.timestamp
        shared_source = 1.0 if set(parent.sources) & set(child.sources) else 0.0
        return cls(
            (
                1.0,
                jaccard(parent_tokens, child_tokens),
                coverage,
                1.0 if parent.entity == child.entity else 0.0,
                1.0 / (1.0 + gap),
                shared_source,
            )
        )


@dataclass
class DependencyModel:
    weights: tuple[float, ...]
    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES)

    def predict_proba(self, parent: Memory, child: Memory) -> float:
        features = DependencyFeatures.from_memories(parent, child).values
        z = sum(weight * value for weight, value in zip(self.weights, features, strict=True))
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def save(self, path: str | Path, *, metadata: dict | None = None) -> None:
        payload = {
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
            "metadata": metadata or {},
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DependencyModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(weights=tuple(float(value) for value in payload["weights"]))


def train_dependency_model(examples: list[tuple[Memory, Memory, int]]) -> tuple[DependencyModel, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    rows = [DependencyFeatures.from_memories(parent, child).values for parent, child, _ in examples]
    labels = [label for _, _, label in examples]
    x_train, x_test, y_train, y_test = train_test_split(
        rows, labels, test_size=0.25, random_state=7, stratify=labels
    )
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=7)
    clf.fit(x_train, y_train)
    probabilities = clf.predict_proba(x_test)[:, 1]
    weights = tuple(float(v) for v in [clf.intercept_[0], *clf.coef_[0][1:]])
    weights = (float(clf.intercept_[0] + clf.coef_[0][0]), *weights[1:])
    metrics = {
        "examples": len(examples),
        "train_examples": len(x_train),
        "test_examples": len(x_test),
        "positive_rate": sum(labels) / len(labels),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
    }
    return DependencyModel(weights=weights), metrics
