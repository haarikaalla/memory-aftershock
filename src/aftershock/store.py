"""Append-only memory store with dependency edges and audit events."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Memory:
    memory_id: str
    text: str
    entity: str
    timestamp: int
    status: str = "active"
    revision: int = 1
    sources: tuple[str, ...] = ()
    kind: str = "fact"

    def with_status(self, status: str) -> Memory:
        return Memory(
            memory_id=self.memory_id,
            text=self.text,
            entity=self.entity,
            timestamp=self.timestamp,
            status=status,
            revision=self.revision,
            sources=self.sources,
            kind=self.kind,
        )

    def replacement(self, text: str, *, status: str = "active") -> Memory:
        return Memory(
            memory_id=self.memory_id,
            text=text,
            entity=self.entity,
            timestamp=self.timestamp,
            status=status,
            revision=self.revision + 1,
            sources=self.sources,
            kind=self.kind,
        )


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.connection.executescript(
            """
            create table if not exists revisions (
                memory_id text not null,
                revision integer not null,
                payload text not null,
                primary key(memory_id, revision)
            );
            create table if not exists edges (
                parent_id text not null,
                child_id text not null,
                primary key(parent_id, child_id)
            );
            create table if not exists events (
                event_id integer primary key autoincrement,
                memory_id text not null,
                event_type text not null,
                payload text not null,
                created_at datetime default current_timestamp
            );
            """
        )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        try:
            self.connection.execute("begin")
            yield
        except Exception:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def put(self, memory: Memory, *, event_type: str = "put") -> None:
        payload = asdict(memory)
        payload["sources"] = list(memory.sources)
        self.connection.execute(
            "insert into revisions(memory_id, revision, payload) values (?, ?, ?)",
            (memory.memory_id, memory.revision, json.dumps(payload, sort_keys=True)),
        )
        self.event(memory.memory_id, event_type, payload)

    def event(self, memory_id: str, event_type: str, payload: dict) -> None:
        self.connection.execute(
            "insert into events(memory_id, event_type, payload) values (?, ?, ?)",
            (memory_id, event_type, json.dumps(payload, sort_keys=True)),
        )

    def latest(self, memory_id: str) -> Memory:
        row = self.connection.execute(
            """
            select payload from revisions
            where memory_id = ?
            order by revision desc
            limit 1
            """,
            (memory_id,),
        ).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return _memory_from_payload(json.loads(row["payload"]))

    def all_latest(self) -> list[Memory]:
        rows = self.connection.execute(
            """
            select r.payload from revisions r
            join (
              select memory_id, max(revision) revision
              from revisions
              group by memory_id
            ) latest
            on r.memory_id = latest.memory_id and r.revision = latest.revision
            order by json_extract(r.payload, '$.timestamp'), r.memory_id
            """
        ).fetchall()
        return [_memory_from_payload(json.loads(row["payload"])) for row in rows]

    def add_edge(self, parent_id: str, child_id: str) -> None:
        parent = self.latest(parent_id)
        child = self.latest(child_id)
        if parent.timestamp >= child.timestamp:
            raise ValueError("Dependency edges must point forward in time.")
        if self._has_path(child_id, parent_id):
            raise ValueError("Dependency edge would create a cycle.")
        self.connection.execute(
            "insert or ignore into edges(parent_id, child_id) values (?, ?)",
            (parent_id, child_id),
        )

    def edges(self) -> list[tuple[str, str]]:
        rows = self.connection.execute(
            "select parent_id, child_id from edges order by parent_id, child_id"
        ).fetchall()
        return [(row["parent_id"], row["child_id"]) for row in rows]

    def children(self, memory_id: str) -> list[str]:
        rows = self.connection.execute(
            "select child_id from edges where parent_id = ? order by child_id",
            (memory_id,),
        ).fetchall()
        return [row["child_id"] for row in rows]

    def events(self) -> list[dict]:
        rows = self.connection.execute(
            "select event_id, memory_id, event_type, payload, created_at from events order by event_id"
        ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "memory_id": row["memory_id"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _has_path(self, start_id: str, target_id: str) -> bool:
        frontier = [start_id]
        seen: set[str] = set()
        while frontier:
            current = frontier.pop()
            if current == target_id:
                return True
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(self.children(current))
        return False


def _memory_from_payload(payload: dict) -> Memory:
    return Memory(
        memory_id=payload["memory_id"],
        text=payload["text"],
        entity=payload["entity"],
        timestamp=int(payload["timestamp"]),
        status=payload.get("status", "active"),
        revision=int(payload.get("revision", 1)),
        sources=tuple(payload.get("sources", ())),
        kind=payload.get("kind", "fact"),
    )
