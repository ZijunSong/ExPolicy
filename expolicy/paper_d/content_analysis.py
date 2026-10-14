"""Plan-aligned effects. A missing retention cell does not become a zero penalty."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from expolicy.paper_d.content_protocol import PROTOCOL


CELL_KEY = ("block_id", "state_id", "batch_id", "condition", "task_id", "probe_role", "repeat_id")


def _key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[name]) for name in CELL_KEY)


def align_plan(
    plan: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    seen: dict[tuple[str, ...], Mapping[str, Any]] = {}
    duplicates = []
    for row in rows:
        key = _key(row)
        if key in seen:
            duplicates.append(key)
            continue
        seen[key] = row
    if duplicates:
        raise RuntimeError(f"duplicate result keys: {duplicates[:3]}")
    missing = []
    paired: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for cell in plan:
        key = _key(cell)
        found = seen.get(key)
        if found is None or found.get("outcome") is None or found.get("infra_status") != "ok":
            missing.append({name: cell[name] for name in CELL_KEY})
            continue
        if found.get("structural_status") not in {None, "ok"}:
            missing.append({name: cell[name] for name in CELL_KEY})
            continue
        paired[key] = found
    return {
        "planned": len(plan),
        "paired": len(paired),
        "missing": missing,
        "complete": not missing,
        "outcomes": paired,
    }


def _task_mean(
    outcomes: Mapping[tuple[str, ...], Mapping[str, Any]],
    *,
    block_id: str,
    state_id: str,
    batch_id: str,
    condition: str,
    task_id: str,
    role: str,
) -> float | None:
    values = []
    for repeat_id in range(int(PROTOCOL["repeats"])):
        key = (block_id, state_id, batch_id, condition, task_id, role, str(repeat_id))
        row = outcomes.get(key)
        if row is None:
            return None
        values.append(float(row["outcome"]))
    return sum(values) / len(values)


def role_delta(
    outcomes: Mapping[tuple[str, ...], Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    *,
    block_id: str,
    state_id: str,
    batch_id: str,
    role: str,
    left: str,
    right: str,
) -> dict[str, Any]:
    wanted = [task for task in tasks if task["probe_role"] == role]
    deltas = []
    missing = []
    for task in wanted:
        a = _task_mean(
            outcomes, block_id=block_id, state_id=state_id, batch_id=batch_id,
            condition=left, task_id=task["task_id"], role=role,
        )
        b = _task_mean(
            outcomes, block_id=block_id, state_id=state_id, batch_id=batch_id,
            condition=right, task_id=task["task_id"], role=role,
        )
        if a is None or b is None:
            missing.append(task["task_id"])
            continue
        deltas.append({"task_id": task["task_id"], "family": task.get("task_family"), "delta": a - b})
    estimable = not missing and len(deltas) == len(wanted) and len(wanted) > 0
    mean = sum(item["delta"] for item in deltas) / len(deltas) if estimable else None
    return {"estimable": estimable, "mean": mean, "tasks": deltas, "missing": missing, "n": len(wanted)}


def utility_from_deltas(future_gain: float | None, retention_gain: float | None) -> float | None:
    if future_gain is None or retention_gain is None:
        return None
    return float(future_gain) + float(PROTOCOL["retention_weight"]) * min(0.0, float(retention_gain))


def batch_equal_task_deltas(per_batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average batch-level paired deltas only when every batch has the same complete task set."""
    if not per_batch or any(not item["estimable"] for item in per_batch):
        return {"estimable": False, "mean": None, "tasks": []}
    by_task: dict[str, list[float]] = defaultdict(list)
    families = {}
    for item in per_batch:
        for task in item["tasks"]:
            by_task[task["task_id"]].append(float(task["delta"]))
            families[task["task_id"]] = task.get("family")
    if any(len(values) != len(per_batch) for values in by_task.values()):
        return {"estimable": False, "mean": None, "tasks": []}
    tasks = [
        {"task_id": task_id, "family": families[task_id], "delta": sum(values) / len(values)}
        for task_id, values in sorted(by_task.items())
    ]
    mean = sum(item["delta"] for item in tasks) / len(tasks)
    return {"estimable": True, "mean": mean, "tasks": tasks}


def bootstrap_interval(task_deltas: Sequence[float], *, draws: int = 1000, seed: int = 20261014) -> dict[str, Any]:
    values = [float(item) for item in task_deltas]
    if not values:
        return {"estimable": False}
    point = sum(values) / len(values)
    rng = random.Random(seed)
    stats = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        stats.append(sum(sample) / len(sample))
    stats.sort()
    low = stats[int(0.025 * draws)]
    high = stats[max(0, int(0.975 * draws) - 1)]
    left = sum(1 for item in stats if item <= 0.0) / draws
    right = sum(1 for item in stats if item >= 0.0) / draws
    return {
        "estimable": True,
        "point": point,
        "low": low,
        "high": high,
        "width": high - low,
        "p": min(1.0, 2 * min(left, right)),
        "n_tasks": len(values),
    }


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * float(pvalue)))
        adjusted[name] = running
    return adjusted


def leave_one_task_out(task_deltas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    full = [float(item["delta"]) for item in task_deltas]
    if not full:
        return rows
    for index, item in enumerate(task_deltas):
        kept = full[:index] + full[index + 1 :]
        mean = sum(kept) / len(kept) if kept else None
        rows.append({"left_out": item["task_id"], "family": item.get("family"), "mean_without": mean})
    return rows


def single_task_dependent(task_deltas: Sequence[Mapping[str, Any]]) -> bool:
    full = [float(item["delta"]) for item in task_deltas]
    if not full:
        return False
    point = sum(full) / len(full)
    if point <= 0:
        return False
    for index in range(len(full)):
        kept = full[:index] + full[index + 1 :]
        without = sum(kept) / len(kept) if kept else 0.0
        if point - without >= float(PROTOCOL["single_task_share"]) * point:
            return True
    return False


def benefiting_tasks(contrasts: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """A future task benefits only when REAL beats baseline and both content controls."""
    tables = {name: {item["task_id"]: float(item["delta"]) for item in rows} for name, rows in contrasts.items()}
    ids = set.intersection(*(set(table) for table in tables.values())) if tables else set()
    benefited = []
    harmed = []
    families = set()
    for task_id in sorted(ids):
        if all(tables[name][task_id] > 0 for name in ("REAL-NO_UPDATE", "REAL-EMPTY", "REAL-MATCHED_WRONG")):
            benefited.append(task_id)
        if tables["REAL-NO_UPDATE"][task_id] < 0:
            harmed.append(task_id)
    family_by_id = {
        item["task_id"]: item.get("family")
        for item in contrasts.get("REAL-NO_UPDATE", [])
    }
    for task_id in benefited:
        if family_by_id.get(task_id) and family_by_id[task_id] != "unclustered":
            families.add(family_by_id[task_id])
    return {
        "benefited": benefited,
        "harmed": harmed,
        "n_benefited": len(benefited),
        "n_harmed": len(harmed),
        "n_families": len(families),
        "families": sorted(families),
    }


def decide(report: Mapping[str, Any]) -> str:
    if report.get("execution_blocked"):
        return "D_CONTENT_EXECUTION_BLOCKED"
    if report.get("control_invalid"):
        return "D_CONTENT_CONTROL_INVALID"
    future = report.get("future") or {}
    if not future.get("estimable"):
        return "D_CONTENT_INCONCLUSIVE"
    points = {name: future[name]["point"] for name in PROTOCOL["primary_contrasts"]}
    benefits = report.get("benefits") or {}
    beats_baseline = points["REAL-NO_UPDATE"] > 0
    beats_controls = points["REAL-EMPTY"] > 0 and points["REAL-MATCHED_WRONG"] > 0
    if beats_baseline and not beats_controls:
        return "D_CONTENT_UNSUPPORTED"
    if beats_baseline and beats_controls and (
        int(benefits.get("n_benefited") or 0) < int(PROTOCOL["min_benefiting_future_tasks"])
        or report.get("single_task_dependent")
    ):
        return "D_LOCAL_CASE_ONLY"
    utility = report.get("utility") or {}
    utility_point = (utility.get("REAL-NO_UPDATE") or {}).get("point")
    no_clear_utility_loss = utility_point is not None and utility_point >= -float(PROTOCOL["delta_reference"])
    content_bounds = [
        (future[name].get("low"))
        for name in ("REAL-EMPTY", "REAL-MATCHED_WRONG")
    ]
    utility_low = (utility.get("REAL-NO_UPDATE") or {}).get("low")
    if (
        beats_baseline
        and beats_controls
        and no_clear_utility_loss
        and all(low is not None and low > float(PROTOCOL["delta_reference"]) for low in content_bounds)
        and utility_low is not None
        and utility_low > float(PROTOCOL["delta_reference"])
    ):
        return "D_CONTENT_SUPPORTED_IN_SCOPE"
    if beats_baseline and beats_controls and no_clear_utility_loss and int(benefits.get("n_benefited") or 0) >= int(PROTOCOL["min_benefiting_future_tasks"]):
        return "D_CONTENT_TRANSFER_CANDIDATE"
    return "D_CONTENT_INCONCLUSIVE"
