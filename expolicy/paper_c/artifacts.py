"""Write the C-v1 products. Missing rows stay in the failure list."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from expolicy.paper_c.decision import (
    render_assimilator_audit_md,
    render_decision_md,
    render_visibility_md,
)
from expolicy.paper_c.schema import write_csv, write_jsonl

GAIN_COLUMNS = (
    "sequence_id",
    "state_hash",
    "family",
    "batch_index",
    "source_template",
    "gain_select",
    "gain_eval",
    "delta_retention",
    "infra_status",
)
CURVE_COLUMNS = (
    "policy",
    "sequence_seed",
    "round",
    "state_hash",
    "interaction_cost",
    "episode_cost",
    "token_cost",
    "future_score",
    "shadow",
    "retention",
)


def write_products(
    directory: Path,
    *,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    decision: Mapping[str, Any],
    assimilator_audit: Mapping[str, Any],
    c0_judgment: Mapping[str, Any],
    gain_rows: Sequence[Mapping[str, Any]],
    visibility: Mapping[str, Any],
    curves: Sequence[Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
    logs: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    protocol_path = directory / "C_PROTOCOL.yaml"
    protocol_path.write_text(yaml.safe_dump(dict(protocol), sort_keys=False, allow_unicode=True), encoding="utf-8")
    (directory / "C_ASSIMILATOR_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "C_ASSIMILATOR_AUDIT.md").write_text(
        render_assimilator_audit_md(assimilator_audit, c0_judgment),
        encoding="utf-8",
    )
    (directory / "C_FEATURE_VISIBILITY_AUDIT.md").write_text(render_visibility_md(visibility), encoding="utf-8")
    (directory / "C_DECISION.md").write_text(render_decision_md(decision), encoding="utf-8")
    write_csv(directory / "C_GAIN_MATRIX.csv", gain_rows, GAIN_COLUMNS)
    write_csv(directory / "C_LEARNING_CURVES.csv", curves, CURVE_COLUMNS)
    write_jsonl(directory / "C_COST_LEDGER.jsonl", ledger_rows)
    write_jsonl(directory / "C_RUN_LOG.jsonl", logs)
    write_jsonl(directory / "C_FAILURES.jsonl", failures)
    return {path.name: str(path) for path in sorted(directory.iterdir()) if path.is_file()}
