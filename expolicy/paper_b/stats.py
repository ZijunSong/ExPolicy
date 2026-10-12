"""Paired, source-group bootstrap. Tasks that share a template are one cluster."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from expolicy.paper_a.stats import grouped_bootstrap_means


def mean_success(rows: Sequence[Mapping[str, Any]], condition: str, splits: set[str]) -> float | None:
    values = [float(row["success"]) for row in rows if row["condition"] == condition and row["split"] in splits]
    if not values:
        return None
    return float(np.mean(values))


def paired_comparison(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    splits: set[str],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    paired = _paired(rows, left, right, splits)
    if not paired:
        return {"diff": None, "ci": None, "n": 0, "n_groups": 0, "alpha": alpha}
    diffs = [item["diff"] for item in paired]
    groups = [item["group"] for item in paired]
    point = float(np.mean(diffs))
    samples = grouped_bootstrap_means(diffs, groups, n_boot=n_boot, seed=seed)
    n_groups = len(set(groups))
    if samples is None:
        return {"diff": point, "ci": None, "n": len(diffs), "n_groups": n_groups, "alpha": alpha}
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return {"diff": point, "ci": [lo, hi], "n": len(diffs), "n_groups": n_groups, "alpha": alpha}


def family_diffs(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    splits: set[str],
    families: Sequence[str],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for family in families:
        paired = [item for item in _paired(rows, left, right, splits) if item["family"] == family]
        out[family] = None if not paired else float(np.mean([item["diff"] for item in paired]))
    return out


def retention_delta(rows: Sequence[Mapping[str, Any]], left: str, right: str, split: str) -> float | None:
    paired = _paired(rows, left, right, {split})
    if not paired:
        return None
    return float(np.mean([item["diff"] for item in paired]))


def _paired(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    splits: set[str],
) -> list[dict[str, Any]]:
    by_condition: dict[str, dict[str, Mapping[str, Any]]] = {left: {}, right: {}}
    for row in rows:
        if row["condition"] in by_condition and row["split"] in splits:
            by_condition[row["condition"]][str(row["task_id"])] = row
    paired = []
    for task_id, lhs in sorted(by_condition[left].items()):
        rhs = by_condition[right].get(task_id)
        if rhs is None:
            continue
        paired.append(
            {
                "task_id": task_id,
                "diff": float(lhs["success"]) - float(rhs["success"]),
                "group": str(lhs["source_group"]),
                "family": str(lhs["family"]),
            }
        )
    return paired
