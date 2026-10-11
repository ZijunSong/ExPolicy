"""Event-sourcing logger. Every critical event is appended as JSONL."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import orjson

from expolicy.core.types import EventType, JSONDict


class EventLogger:
    def __init__(self, path: str | Path, *, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        if not self.path.exists():
            self.path.touch()

    def emit(self, event_type: str | EventType, payload: JSONDict | None = None) -> JSONDict:
        name = event_type.value if isinstance(event_type, EventType) else event_type
        record = {
            "event_id": uuid.uuid4().hex,
            "run_id": self.run_id,
            "timestamp": time.time(),
            "event_type": name,
            "payload": payload or {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(orjson.dumps(record))
            handle.write(b"\n")
        return record

    def read(self) -> list[JSONDict]:
        if not self.path.exists():
            return []
        records: list[JSONDict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(orjson.loads(line))
        return records

    def by_type(self, event_type: str) -> list[JSONDict]:
        return [row for row in self.read() if row.get("event_type") == event_type]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(orjson.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(orjson.dumps(row))
        handle.write(b"\n")
