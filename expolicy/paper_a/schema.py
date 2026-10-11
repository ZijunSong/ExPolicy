"""Long-form result rows. Missing measurements stay missing; they are not stored as zero."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from expolicy.paper_a.constants import EXTRA_ROW_FIELDS, ROW_FIELDS, STATUS_MISSING


def empty_row(**fields: Any) -> dict[str, Any]:
    row = {key: None for key in ROW_FIELDS}
    row.update({key: None for key in EXTRA_ROW_FIELDS})
    row["infra_status"] = STATUS_MISSING
    row["legacy"] = False
    row["utility_scope"] = "cell"
    row.update(fields)
    return row


def numeric_or_blank(value: Any) -> Any:
    if value is None:
        return ""
    return value


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(ROW_FIELDS) + list(EXTRA_ROW_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: numeric_or_blank(row.get(key)) for key in header})


def read_rows_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    numeric = {
        "y_before",
        "y_after",
        "delta_shadow",
        "delta_retention",
        "utility",
        "tokens",
        "tool_calls",
        "elapsed_s",
        "retention_penalty",
    }
    for raw in raw_rows:
        row: dict[str, Any] = dict(raw)
        for key in numeric:
            text = row.get(key)
            row[key] = None if text in (None, "") else float(text)
        if row.get("tokens") is not None:
            row["tokens"] = int(row["tokens"])
        if row.get("tool_calls") is not None:
            row["tool_calls"] = int(row["tool_calls"])
        flag = str(row.get("legacy") or "").lower()
        row["legacy"] = flag in {"1", "true", "yes"}
        parsed.append(row)
    return parsed
