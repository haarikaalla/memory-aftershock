"""Leakage-safe LongMemEval retrieval utilities."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from .text import tokenize


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    timestamp: str = ""
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float
    metadata: dict[str, str]


class BM25Index:
    """A compact BM25 implementation with deterministic tie-breaking."""

    def __init__(self, documents: Iterable[Document], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.doc_tokens: dict[str, Counter[str]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.doc_meta: dict[str, dict[str, str]] = {}
        self.inverted: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.avgdl = 0.0
        self._build()

    def _build(self) -> None:
        total_len = 0
        df: Counter[str] = Counter()
        for doc in self.documents:
            counts = Counter(tokenize(doc.text))
            self.doc_tokens[doc.doc_id] = counts
            self.doc_lengths[doc.doc_id] = sum(counts.values())
            self.doc_meta[doc.doc_id] = dict(doc.metadata or {})
            total_len += self.doc_lengths[doc.doc_id]
            for token, freq in counts.items():
                df[token] += 1
                self.inverted[token].append((doc.doc_id, freq))
        self.avgdl = total_len / len(self.documents) if self.documents else 0.0
        self.idf = {
            token: math.log(1.0 + (len(self.documents) - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def search(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        if not self.documents:
            return []
        scores: defaultdict[str, float] = defaultdict(float)
        for token in tokenize(query):
            for doc_id, tf in self.inverted.get(token, []):
                dl = self.doc_lengths[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[doc_id] += self.idf.get(token, 0.0) * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [SearchHit(doc_id=doc_id, score=score, metadata=self.doc_meta[doc_id]) for doc_id, score in ranked]


class MemoryRetriever:
    """Hybrid retriever over LongMemEval sessions and turns without using answer labels."""

    def __init__(self, session_docs: list[Document], turn_docs: list[Document]) -> None:
        self.session_index = BM25Index(session_docs)
        self.turn_index = BM25Index(turn_docs)

    @classmethod
    def from_longmemeval_item(cls, item: dict) -> MemoryRetriever:
        session_docs: list[Document] = []
        turn_docs: list[Document] = []
        for session_id, session_date, session in zip(
            item["haystack_session_ids"],
            item["haystack_dates"],
            item["haystack_sessions"],
            strict=True,
        ):
            turn_texts: list[str] = []
            for turn_idx, turn in enumerate(session):
                role = str(turn.get("role", "unknown"))
                content = str(turn.get("content", ""))
                turn_texts.append(f"{role}: {content}")
                turn_docs.append(
                    Document(
                        doc_id=f"{session_id}::turn-{turn_idx}",
                        text=f"{role}: {content}",
                        timestamp=session_date,
                        metadata={"session_id": session_id, "date": session_date, "role": role},
                    )
                )
            session_docs.append(
                Document(
                    doc_id=session_id,
                    text="\n".join(turn_texts),
                    timestamp=session_date,
                    metadata={"session_id": session_id, "date": session_date},
                )
            )
        return cls(session_docs=session_docs, turn_docs=turn_docs)

    def search_sessions(self, query: str, *, question_date: str = "", top_k: int = 10) -> list[SearchHit]:
        session_hits = self.session_index.search(query, top_k=top_k * 4)
        turn_hits = self.turn_index.search(query, top_k=top_k * 8)
        fused: defaultdict[str, float] = defaultdict(float)
        meta: dict[str, dict[str, str]] = {}
        for rank, hit in enumerate(session_hits, start=1):
            session_id = hit.metadata["session_id"]
            fused[session_id] += 1.0 / (60 + rank)
            meta[session_id] = hit.metadata
        for rank, hit in enumerate(turn_hits, start=1):
            session_id = hit.metadata["session_id"]
            fused[session_id] += 1.0 / (60 + rank)
            meta[session_id] = {"session_id": session_id, "date": hit.metadata["date"]}
        if question_date:
            query_date = _parse_date(question_date)
            for session_id, values in meta.items():
                session_date = _parse_date(values.get("date", ""))
                if query_date and session_date and session_date <= query_date:
                    fused[session_id] += 0.0001
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [SearchHit(doc_id=session_id, score=score, metadata=meta[session_id]) for session_id, score in ranked]


def _parse_date(value: str) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=UTC)
        except ValueError:
            pass
    return None
