"""Run manifests, ledgers, and immutable summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from expolicy.paper_a.schema import write_rows_csv
from expolicy.r9_lab.io_util import append_jsonl, atomic_write_json, sha256_json


def _jsonable(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _jsonable(value) for key, value in payload.items() if not str(key).startswith("_")}
    if isinstance(payload, (list, tuple)):
        return [_jsonable(value) for value in payload]
    if hasattr(payload, "item"):
        return payload.item()
    return payload


def write_manifest(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = _jsonable(dict(payload))
    body["frozen"] = True
    body.pop("content_hash", None)
    body["content_hash"] = sha256_json(body)
    atomic_write_json(path, body)
    return body


def write_summary(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = _jsonable(dict(payload))
    body.pop("content_hash", None)
    body["content_hash"] = sha256_json(body)
    atomic_write_json(path, body)
    return body


def assert_summary_intact(path: Path) -> None:
    stored = json.loads(path.read_text(encoding="utf-8"))
    expected = stored.get("content_hash")
    body = {key: value for key, value in stored.items() if key != "content_hash"}
    actual = sha256_json(body)
    if expected != actual:
        raise RuntimeError("immutable summary hash mismatch")


def append_ledger(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        append_jsonl(path, _jsonable(dict(row)))


def write_failures(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    append_ledger(path, rows)


def write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_rows_csv(path, rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
