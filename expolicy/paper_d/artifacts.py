"""Write the D-v1 products. Task rows stay at task level in the parquet matrix."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from expolicy.paper_d.constants import MATRIX_FIELDS
from expolicy.paper_d.decision import render_audit_md, render_decision_md, render_protocol_md
from expolicy.paper_d.schema import write_csv, write_jsonl

HEADROOM_COLUMNS = (
    "configuration_id",
    "estimate_name",
    "global_gap",
    "within_gap",
    "selection_gain",
    "feature_gain",
    "global_lower",
    "global_upper",
    "within_lower",
    "within_upper",
    "is_credible_upper_bound",
    "is_oracle",
    "clipped",
    "finite_sample_guarantee",
    "weight_mode",
    "below_fixed_requires_audit",
    "structural_impossibility",
    "generalization",
    "notes",
)
CONTROL_COLUMNS = (
    "control",
    "known_structure",
    "expected_global_gap",
    "expected_within_gap",
    "observed_naive_global",
    "observed_naive_within",
    "observed_crossfit",
    "observed_naive_k2",
    "observed_naive_k6",
    "primary_status",
    "passed",
    "reasons",
)
PROSPECTIVE_COLUMNS = (
    "configuration_id",
    "card_hash",
    "predicted_status",
    "actual_gain",
    "false_go",
    "false_nogo",
    "validation_status",
    "diagnostic_cost",
    "training_cost",
    "correlation_reported",
    "significance_claimed",
    "net_savings",
)


def write_products(
    directory: Path,
    *,
    audit: Mapping[str, Any],
    inventory: Mapping[str, Any],
    decision: Mapping[str, Any],
    control_reports: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    headroom_rows: Sequence[Mapping[str, Any]],
    cards: Sequence[Mapping[str, Any]],
    prospective_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "D_PROTOCOL.md").write_text(render_protocol_md(), encoding="utf-8")
    (directory / "D_MEASUREMENT_AUDIT.md").write_text(render_audit_md(audit, inventory), encoding="utf-8")
    (directory / "D_DECISION.md").write_text(render_decision_md(decision, control_reports), encoding="utf-8")
    write_csv(directory / "D_KNOWN_CONTROLS.csv", [_control_row(report) for report in control_reports], CONTROL_COLUMNS)
    write_csv(directory / "D_HEADROOM_ESTIMATES.csv", headroom_rows, HEADROOM_COLUMNS)
    write_csv(directory / "D_PROSPECTIVE_VALIDATION.csv", prospective_rows, PROSPECTIVE_COLUMNS)
    write_jsonl(directory / "D_DIAGNOSTIC_CARDS.jsonl", cards)
    _write_parquet(directory / "D_TASK_MATRIX.parquet", task_rows)
    return {path.name: str(path) for path in sorted(directory.iterdir()) if path.is_file()}


def headroom_row(configuration_id: str, estimate: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {key: None for key in HEADROOM_COLUMNS}
    row["configuration_id"] = configuration_id
    for key in HEADROOM_COLUMNS:
        if key in estimate:
            row[key] = estimate[key]
    row["estimate_name"] = estimate["estimate_name"]
    return row


def _control_row(report: Mapping[str, Any]) -> dict[str, Any]:
    observed = report["observed"]
    expected = report["expected"]
    return {
        "control": report["name"],
        "known_structure": report["known_structure"],
        "expected_global_gap": expected.get("global_gap"),
        "expected_within_gap": expected.get("within_gap"),
        "observed_naive_global": observed.get("naive_global"),
        "observed_naive_within": observed.get("naive_within"),
        "observed_crossfit": observed.get("crossfit"),
        "observed_naive_k2": observed.get("naive_k2"),
        "observed_naive_k6": observed.get("naive_k6"),
        "primary_status": observed.get("primary"),
        "passed": report["passed"],
        "reasons": "; ".join(report.get("reasons") or []),
    }


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = list(MATRIX_FIELDS)
    arrays = []
    for key in columns:
        values = [row.get(key) for row in rows]
        arrays.append(pa.array(values))
    table = pa.Table.from_arrays(arrays, names=columns)
    pq.write_table(table, path)
