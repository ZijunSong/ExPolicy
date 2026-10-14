"""Task-level rows. Missing measurements stay missing and are not filled with zero."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from expolicy.paper_d.constants import LOG_FIELDS, STATUS_MISSING


def empty_log(**fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {key: None for key in LOG_FIELDS}
    row["infra_status"] = STATUS_MISSING
    row["legacy_u"] = None
    row.update(fields)
    return row


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in columns})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
