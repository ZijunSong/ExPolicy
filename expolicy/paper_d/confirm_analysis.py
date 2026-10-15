"""Batch-by-task intervals for the three utility contrasts. No bootstrap p-values."""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from expolicy.paper_d.confirm_protocol import PROTOCOL
from expolicy.paper_d.confirm_utility import update_utility


def _role_mean(table: Mapping[str, Mapping[str, float]], role_tasks: Sequence[str], condition: str) -> float | None:
    values = []
    for task_id in role_tasks:
        cell = table.get(task_id, {}).get(condition)
        if cell is None:
            return None
        values.append(float(cell))
    if not values:
        return None
    return sum(values) / len(values)


def batch_effects(batch: Mapping[str, Any], future_ids: Sequence[str], retention_ids: Sequence[str]) -> dict[str, Any]:
    """One batch. Missing retention makes every utility unavailable."""
    future = batch.get("future") or {}
    retention = batch.get("retention") or {}
    out: dict[str, Any] = {"batch_id": batch.get("batch_id"), "n_written": batch.get("n_written"), "content_batch": bool(batch.get("n_written"))}
    for condition in ("REAL", "EMPTY", "MATCHED_WRONG"):
        future_gain = None
        retention_gain = None
        base_f = _role_mean(future, future_ids, "NO_UPDATE")
        cond_f = _role_mean(future, future_ids, condition)
        base_r = _role_mean(retention, retention_ids, "NO_UPDATE")
        cond_r = _role_mean(retention, retention_ids, condition)
        if base_f is not None and cond_f is not None:
            future_gain = cond_f - base_f
        if base_r is not None and cond_r is not None:
            retention_gain = cond_r - base_r
        out[condition] = {
            "future_gain": future_gain,
            "retention_gain": retention_gain,
            "utility": update_utility(future_gain, retention_gain),
        }
    real_u = out["REAL"]["utility"]
    out["gaps"] = {
        "U-REAL-NO_UPDATE": real_u,
        "U-REAL-EMPTY": None if real_u is None or out["EMPTY"]["utility"] is None else real_u - out["EMPTY"]["utility"],
        "U-REAL-WRONG": None if real_u is None or out["MATCHED_WRONG"]["utility"] is None else real_u - out["MATCHED_WRONG"]["utility"],
    }
    return out


def overall_from_batches(batch_rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    result = {}
    for name in PROTOCOL["primary_estimates"]:
        values = [row["gaps"][name] for row in batch_rows]
        result[name] = None if any(value is None for value in values) or not values else sum(values) / len(values)
    return result


def _resample_ids(ids: Sequence[str], rng: random.Random) -> list[str]:
    return [ids[rng.randrange(len(ids))] for _ in ids]


def _resampled_table(table: Mapping[str, Mapping[str, float]], ids: Sequence[str]) -> dict[str, Mapping[str, float]]:
    """Duplicate ids are kept as repeated tasks in the resampled mean via a list, not a dict overwrite."""
    return {f"{task_id}#{index}": table[task_id] for index, task_id in enumerate(ids)}


def cross_bootstrap(
    batches: Sequence[Mapping[str, Any]],
    future_ids: Sequence[str],
    retention_ids: Sequence[str],
    *,
    draws: int | None = None,
    seed: int = 20261015,
) -> dict[str, Any]:
    draws = int(PROTOCOL["bootstrap_draws"] if draws is None else draws)
    rng = random.Random(seed)
    samples = {name: [] for name in PROTOCOL["primary_estimates"]}
    if not batches or not future_ids or not retention_ids:
        return {"estimable": False, "samples": samples}
    for _ in range(draws):
        chosen = [batches[rng.randrange(len(batches))] for _ in batches]
        future_draw = _resample_ids(list(future_ids), rng)
        retention_draw = _resample_ids(list(retention_ids), rng)
        rebuilt = []
        for batch in chosen:
            rebuilt.append(
                {
                    "batch_id": batch.get("batch_id"),
                    "n_written": batch.get("n_written"),
                    "future": _resampled_table(batch["future"], future_draw),
                    "retention": _resampled_table(batch["retention"], retention_draw),
                }
            )
        # resampled ids are synthetic keys; pass those keys as the task lists
        effects = [
            batch_effects(batch, [f"{task_id}#{index}" for index, task_id in enumerate(future_draw)], [f"{task_id}#{index}" for index, task_id in enumerate(retention_draw)])
            for batch in rebuilt
        ]
        overall = overall_from_batches(effects)
        if any(overall[name] is None for name in samples):
            return {"estimable": False, "samples": samples}
        for name in samples:
            samples[name].append(float(overall[name]))
    point_batches = [batch_effects(batch, future_ids, retention_ids) for batch in batches]
    points = overall_from_batches(point_batches)
    tail = float(PROTOCOL["simultaneous_tail"])
    intervals = {}
    for name, values in samples.items():
        ordered = sorted(values)
        low_i = int(tail * draws)
        high_i = min(draws - 1, int((1.0 - tail) * draws))
        marginal_low = ordered[int(0.025 * draws)]
        marginal_high = ordered[min(draws - 1, int(0.975 * draws) - 1)]
        intervals[name] = {
            "point": points[name],
            "simultaneous_low": ordered[low_i],
            "simultaneous_high": ordered[high_i],
            "marginal_low": marginal_low,
            "marginal_high": marginal_high,
        }
    return {"estimable": True, "intervals": intervals, "batch_rows": point_batches}


def leave_one_batch_out(batches: Sequence[Mapping[str, Any]], future_ids: Sequence[str], retention_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for index, batch in enumerate(batches):
        kept = [item for item_index, item in enumerate(batches) if item_index != index]
        effects = [batch_effects(item, future_ids, retention_ids) for item in kept]
        overall = overall_from_batches(effects)
        rows.append({"left_out": batch.get("batch_id"), **overall})
    return rows


def leave_one_task_out(batches: Sequence[Mapping[str, Any]], future_ids: Sequence[str], retention_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for task_id in future_ids:
        kept = [item for item in future_ids if item != task_id]
        overall = overall_from_batches([batch_effects(batch, kept, retention_ids) for batch in batches])
        rows.append({"left_out": task_id, "role": "future", **overall})
    for task_id in retention_ids:
        kept = [item for item in retention_ids if item != task_id]
        overall = overall_from_batches([batch_effects(batch, future_ids, kept) for batch in batches])
        rows.append({"left_out": task_id, "role": "retention", **overall})
    return rows


def positive_batch_count(batch_rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for row in batch_rows if (row["gaps"]["U-REAL-NO_UPDATE"] or 0) > 0 and row["gaps"]["U-REAL-NO_UPDATE"] is not None)


def robustness_ok(batch_rows: Sequence[Mapping[str, Any]], loo: Sequence[Mapping[str, Any]]) -> bool:
    if positive_batch_count(batch_rows) < int(PROTOCOL["min_positive_batches"]):
        return False
    for row in loo:
        for name in PROTOCOL["primary_estimates"]:
            value = row.get(name)
            if value is None or value < 0:
                return False
    return True


def decide(report: Mapping[str, Any]) -> str:
    if report.get("execution_blocked") or report.get("control_blocked"):
        return "D_CONFIRM_BLOCKED"
    intervals = (report.get("bootstrap") or {}).get("intervals") or {}
    if not intervals or any(intervals[name]["point"] is None for name in PROTOCOL["primary_estimates"]):
        return "D_CONFIRM_INCONCLUSIVE"
    points = {name: intervals[name]["point"] for name in PROTOCOL["primary_estimates"]}
    lows = {name: intervals[name]["simultaneous_low"] for name in PROTOCOL["primary_estimates"]}
    beats_baseline = points["U-REAL-NO_UPDATE"] > 0
    beats_controls = points["U-REAL-EMPTY"] > 0 and points["U-REAL-WRONG"] > 0
    if beats_baseline and not beats_controls and not report.get("single_batch_or_task"):
        return "D_CONTENT_NOT_IDENTIFIED"
    if report.get("single_batch_or_task") or not all(points[name] > 0 for name in points):
        return "D_CONFIRM_NOT_REPLICATED"
    if not report.get("robustness_ok"):
        return "D_CONFIRM_NOT_REPLICATED"
    if all(lows[name] > float(PROTOCOL["delta_reference"]) for name in lows) and report.get("controls_valid"):
        return "D_CONTENT_CONFIRMED_IN_SCOPE"
    if all(points[name] > 0 for name in points):
        return "D_CONFIRM_CANDIDATE_REPLICATED"
    return "D_CONFIRM_INCONCLUSIVE"
