"""Task-level matrix. Averages are derived later and do not replace the rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from expolicy.paper_d.constants import (
    CONTENT_REAL,
    LOG_FIELDS,
    MEASUREMENT_POPULATION,
    MEASUREMENT_SAMPLE,
    PROBE_FUTURE,
    PROBE_RETENTION,
    STATUS_OK,
    ProtocolError,
)
from expolicy.paper_d.utility import (
    legacy_utility_from_repeat,
    mean_or_none,
    utility_from_expectations,
)
from expolicy.r9_lab.io_util import sha256_json


def unit_id(row: Mapping[str, Any]) -> str:
    """Identity of x = (configuration, state, batch, environment). Repeat and seed are not part of x."""
    state = row.get("state_id") or row.get("state_distribution_id")
    env = row.get("env") or row.get("env_hash")
    content = row.get("content_mode") or CONTENT_REAL
    return "|".join(
        [
            str(row.get("configuration_id")),
            str(state),
            str(row.get("batch_id")),
            str(env),
            str(content),
        ]
    )


def build_task_level_matrix(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep every task row. Do not collapse to a mean utility and do not impute missing outcomes."""
    matrix: list[dict[str, Any]] = []
    for row in rows:
        missing = [key for key in LOG_FIELDS if key not in row]
        if missing:
            raise ProtocolError(f"task row is missing fields: {missing}")
        copied = dict(row)
        if copied.get("legacy_u") is None:
            copied["legacy_u"] = None
        for key in ("outcome_before", "outcome_after", "retention_before", "retention_after"):
            if key in copied and copied[key] is None:
                copied[key] = None
        matrix.append(copied)
    return matrix


def cache_fingerprint(row: Mapping[str, Any]) -> str:
    """Legacy fingerprint. Real reuse must use execution_cache_key."""
    payload = {
        "configuration_id": row.get("configuration_id"),
        "model_hash": row.get("model_hash"),
        "prompt_hash": row.get("prompt_hash"),
        "task_id": row.get("task_id"),
        "action": row.get("action"),
        "probe_role": row.get("probe_role"),
        "state_id": row.get("state_id") or row.get("state_distribution_id"),
    }
    return sha256_json(payload)


def execution_cache_key(row: Mapping[str, Any]) -> str:
    """Identity of one real execution. A cached_baseline flag is not this key."""
    payload = {
        "configuration_id": row.get("configuration_id"),
        "model_hash": row.get("model_hash"),
        "tokenizer_hash": row.get("tokenizer_hash"),
        "tool_schema_hash": row.get("tool_schema_hash"),
        "env_hash": row.get("env_hash"),
        "inference_config_hash": row.get("inference_config_hash"),
        "state_hash_before": row.get("state_hash_before") or row.get("state_id"),
        "batch_hash": row.get("batch_hash") or row.get("batch_id"),
        "task_id": row.get("task_id"),
        "raw_task_hash": row.get("raw_task_hash"),
        "action": row.get("action"),
        "content_mode": row.get("content_mode"),
        "probe_role": row.get("probe_role"),
        "repeat_id": row.get("repeat_id"),
    }
    return sha256_json(payload)


def charge_executions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bill each execution_id once. Repeating the id does not add cost; a new identity does."""
    seen: set[str] = set()
    total = 0.0
    for row in rows:
        execution_id = str(row.get("execution_id") or execution_cache_key(row))
        if execution_id in seen:
            continue
        seen.add(execution_id)
        total += float(row.get("cost") or 0.0)
    return {"cost": total, "n_executions": len(seen)}


def align_to_plan(
    rows: Sequence[Mapping[str, Any]],
    plan: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reject a panel whose observed cells are not exactly the frozen plan."""
    def key(item: Mapping[str, Any]) -> tuple[str, ...]:
        return (
            str(item.get("configuration_id")),
            str(item.get("state_id") or item.get("state_distribution_id")),
            str(item.get("batch_id")),
            str(item.get("action")),
            str(item.get("content_mode") or CONTENT_REAL),
            str(item.get("probe_role")),
            str(item.get("task_id")),
            str(item.get("repeat_id")),
        )

    expected = [key(item) for item in plan]
    observed = [key(item) for item in rows]
    reasons: list[str] = []
    if len(observed) != len(set(observed)):
        reasons.append("duplicate primary key")
    missing = set(expected) - set(observed)
    extra = set(observed) - set(expected)
    planned_actions = {item[3] for item in expected}
    observed_actions = {item[3] for item in observed}
    if planned_actions - observed_actions:
        reasons.append("planned action column is missing")
    elif missing:
        reasons.append("planned cell is missing")
    if extra:
        reasons.append("observed cell is not in the plan")
    planned_repeats = {(item[6], item[7]) for item in expected}
    for task_id, repeat_id in {(item[6], item[7]) for item in observed}:
        if any(item[6] == task_id for item in expected) and (task_id, repeat_id) not in planned_repeats:
            reasons.append("repeat does not match the plan")
            break
    return {"valid": not reasons, "reasons": reasons, "n_missing": len(missing), "n_extra": len(extra)}


def charge_cost(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Cache a paired baseline only when the fingerprint matches. A repeat that is not cached is paid again."""
    seen: dict[str, str] = {}
    total = 0.0
    cached_hits = 0
    for row in rows:
        key = "|".join([str(row.get("configuration_id")), str(row.get("action")), str(row.get("task_id")), str(row.get("probe_role"))])
        fingerprint = cache_fingerprint(row)
        if row.get("cached_baseline") is True and seen.get(key) == fingerprint:
            cached_hits += 1
            continue
        total += float(row.get("cost") or 0.0)
        seen[key] = fingerprint
    return {"cost": total, "cached_hits": cached_hits}


def cell_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    retention_weight: float,
) -> list[dict[str, Any]]:
    """One cell per (x, action). Utility is missing when any task in the cell has no successful measurement."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(unit_id(row), str(row.get("action")))].append(row)
    cells: list[dict[str, Any]] = []
    for (uid, action), items in sorted(grouped.items()):
        cells.append(_one_cell(uid, action, items, retention_weight=retention_weight))
    return cells


def utilities_from_cells(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Drop a unit whose actions do not share the same successful task ids. Do not recompute on the remainder."""
    by_unit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_unit[str(cell["unit_id"])].append(cell)
    utilities: dict[str, dict[str, float]] = {}
    envs: dict[str, str] = {}
    meta: dict[str, dict[str, Any]] = {}
    rejected_units: list[dict[str, Any]] = []
    for uid, unit_cells in by_unit.items():
        task_sets = [tuple(cell["future_task_ids"]) for cell in unit_cells]
        retention_sets = [tuple(cell["retention_task_ids"]) for cell in unit_cells]
        incomplete = any(cell["utility"] is None for cell in unit_cells)
        misaligned = any(item != task_sets[0] for item in task_sets) or any(
            item != retention_sets[0] for item in retention_sets
        )
        if incomplete or misaligned:
            rejected_units.append(
                {
                    "unit_id": uid,
                    "reason": "misaligned_or_incomplete_tasks" if misaligned or incomplete else "incomplete",
                    "actions": [cell["action"] for cell in unit_cells],
                }
            )
            continue
        utilities[uid] = {str(cell["action"]): float(cell["utility"]) for cell in unit_cells}
        envs[uid] = str(unit_cells[0]["env"])
        meta[uid] = {
            "configuration_id": unit_cells[0]["configuration_id"],
            "budget": unit_cells[0]["budget"],
            "measurement_kind": unit_cells[0]["measurement_kind"],
            "source_group": unit_cells[0]["source_group"],
            "batch_id": unit_cells[0]["batch_id"],
            "state_id": unit_cells[0]["state_id"],
        }
    return {"utilities": utilities, "envs": envs, "meta": meta, "rejected_units": rejected_units}


def panel_problem(cells: Sequence[Mapping[str, Any]], packed: Mapping[str, Any]) -> str:
    """Refuse mixed configurations, budgets, or measurement kinds. Do not average them together."""
    if packed["rejected_units"]:
        return "INCOMPLETE_ACTION_SET"
    metas = list(packed["meta"].values())
    if not metas:
        return "empty panel"
    configs = {item["configuration_id"] for item in metas}
    if len(configs) != 1:
        return "MIXED_CONFIGURATION"
    budgets = {item["budget"] for item in metas}
    if len(budgets) != 1:
        return "MIXED_BUDGET"
    kinds = {item["measurement_kind"] for item in metas}
    if len(kinds) != 1:
        return "MIXED_MEASUREMENT_KIND"
    action_sets = [set(packed["utilities"][uid]) for uid in packed["utilities"]]
    if any(item != action_sets[0] for item in action_sets):
        return "INCOMPLETE_ACTION_SET"
    if any(cell["infra_status"] != STATUS_OK for cell in cells if cell["unit_id"] in packed["utilities"]):
        return "INFRA_INCOMPLETE"
    return ""


def _one_cell(
    uid: str,
    action: str,
    items: Sequence[Mapping[str, Any]],
    *,
    retention_weight: float,
) -> dict[str, Any]:
    sample = items[0]
    kind = str(sample.get("measurement_kind") or MEASUREMENT_SAMPLE)
    if kind not in {MEASUREMENT_POPULATION, MEASUREMENT_SAMPLE}:
        raise ProtocolError(f"unknown measurement kind: {kind}")
    future = [row for row in items if str(row.get("probe_role")) == PROBE_FUTURE]
    retention = [row for row in items if str(row.get("probe_role")) == PROBE_RETENTION]
    future_ok, future_failed = _split_ok(future)
    retention_ok, retention_failed = _split_ok(retention)
    future_tasks = _task_means(future_ok, "outcome_before", "outcome_after")
    retention_tasks = _task_means(retention_ok, "retention_before", "retention_after")
    failed_future_tasks = {str(row.get("task_id")) for row in future_failed}
    failed_retention_tasks = {str(row.get("task_id")) for row in retention_failed}
    # A task that only failed is part of the planned set and blocks a subset mean.
    future_task_ids = tuple(sorted(set(future_tasks) | failed_future_tasks))
    retention_task_ids = tuple(sorted(set(retention_tasks) | failed_retention_tasks))
    utility = None
    legacy = None
    gain = None
    retention_delta = None
    penalty = None
    future_group_means: list[float] = []
    retention_group_means: list[float] = []
    group_sizes_equal = False
    complete = not failed_future_tasks and not failed_retention_tasks and future_tasks and retention_tasks
    if complete:
        future_before = mean_or_none([item["before"] for item in future_tasks.values()])
        future_after = mean_or_none([item["after"] for item in future_tasks.values()])
        ret_before = mean_or_none([item["before"] for item in retention_tasks.values()])
        ret_after = mean_or_none([item["after"] for item in retention_tasks.values()])
        assert future_before is not None and future_after is not None
        assert ret_before is not None and ret_after is not None
        packed = utility_from_expectations(
            future_before=future_before,
            future_after=future_after,
            retention_before=ret_before,
            retention_after=ret_after,
            retention_weight=retention_weight,
        )
        utility = packed["utility"]
        gain = packed["gain"]
        retention_delta = packed["retention_delta"]
        penalty = packed["penalty"]
        legacy = _legacy_over_repeats(future_ok, retention_ok, retention_weight)
        future_group_means, group_sizes_equal = _group_means(future_tasks)
        retention_group_means, retention_equal = _group_means(retention_tasks)
        group_sizes_equal = group_sizes_equal and retention_equal
    infra_values = {str(row.get("infra_status")) for row in items}
    if complete:
        infra = STATUS_OK
    elif STATUS_OK in infra_values and len(infra_values) > 1:
        infra = "infra_incomplete"
    else:
        infra = next(iter(infra_values))
    accepted_values = {row.get("accepted") for row in items}
    changed_values = {row.get("changed") for row in items}
    return {
        "unit_id": uid,
        "action": action,
        "configuration_id": sample.get("configuration_id"),
        "env": str(sample.get("env") or sample.get("env_hash")),
        "state_id": str(sample.get("state_id") or sample.get("state_distribution_id")),
        "batch_id": str(sample.get("batch_id")),
        "source_group": str(sample.get("source_group") or sample.get("batch_id")),
        "budget": sample.get("budget"),
        "measurement_kind": kind,
        "content_mode": sample.get("content_mode") or CONTENT_REAL,
        "sequence_id": sample.get("sequence_id"),
        "utility": utility,
        "legacy_utility": legacy,
        "gain": gain,
        "retention_delta": retention_delta,
        "penalty": penalty,
        "future_group_means": future_group_means,
        "retention_group_means": retention_group_means,
        "group_sizes_equal": group_sizes_equal,
        "n_tasks": len(future_tasks),
        "n_source_groups": len(future_group_means),
        "future_task_ids": future_task_ids,
        "retention_task_ids": retention_task_ids,
        "accepted": False if False in accepted_values else (True if accepted_values == {True} else None),
        "changed": False if False in changed_values else (True if changed_values == {True} else None),
        "infra_status": infra,
    }


def _split_ok(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    ok: list[Mapping[str, Any]] = []
    failed: list[Mapping[str, Any]] = []
    for row in rows:
        if str(row.get("infra_status")) == STATUS_OK:
            ok.append(row)
        else:
            failed.append(row)
    return ok, failed


def _probe_group(row: Mapping[str, Any]) -> str:
    """Probe intervals use probe groups. Experience groups stay on the policy split."""
    return str(row.get("probe_source_group") or row.get("source_group") or row.get("batch_id"))


def _task_means(
    rows: Sequence[Mapping[str, Any]],
    before_key: str,
    after_key: str,
) -> dict[str, dict[str, Any]]:
    pooled: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for row in rows:
        before = row.get(before_key)
        after = row.get(after_key)
        if before is None or after is None:
            raise ProtocolError("an ok probe row is missing its paired outcome")
        pooled[str(row.get("task_id"))].append((float(before), float(after), _probe_group(row)))
    means: dict[str, dict[str, Any]] = {}
    for task_id, pairs in pooled.items():
        group = pairs[0][2]
        if any(item[2] != group for item in pairs):
            raise ProtocolError("one task cannot belong to two source groups")
        means[task_id] = {
            "before": sum(item[0] for item in pairs) / len(pairs),
            "after": sum(item[1] for item in pairs) / len(pairs),
            "source_group": group,
            "delta": sum(item[1] - item[0] for item in pairs) / len(pairs),
        }
    return means


def _group_means(task_means: Mapping[str, Mapping[str, Any]]) -> tuple[list[float], bool]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for item in task_means.values():
        buckets[str(item["source_group"])].append(float(item["delta"]))
    sizes = [len(bucket) for bucket in buckets.values()]
    means = [sum(bucket) / len(bucket) for bucket in buckets.values()]
    return means, len(set(sizes)) <= 1


def _legacy_over_repeats(
    future_rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    retention_weight: float,
) -> float | None:
    """Average of per-repeat R9 utilities. This is not the protocol utility."""
    future_by_repeat: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    retention_by_repeat: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in future_rows:
        future_by_repeat[str(row.get("repeat_id"))].append(row)
    for row in retention_rows:
        retention_by_repeat[str(row.get("repeat_id"))].append(row)
    shared = sorted(set(future_by_repeat) & set(retention_by_repeat))
    if not shared:
        return None
    utilities: list[float] = []
    for repeat in shared:
        future_before = mean_or_none([float(row["outcome_before"]) for row in future_by_repeat[repeat]])
        future_after = mean_or_none([float(row["outcome_after"]) for row in future_by_repeat[repeat]])
        ret_before = mean_or_none([float(row["retention_before"]) for row in retention_by_repeat[repeat]])
        ret_after = mean_or_none([float(row["retention_after"]) for row in retention_by_repeat[repeat]])
        if None in {future_before, future_after, ret_before, ret_after}:
            continue
        assert future_before is not None and future_after is not None
        assert ret_before is not None and ret_after is not None
        utilities.append(
            float(
                legacy_utility_from_repeat(
                    future_before=future_before,
                    future_after=future_after,
                    retention_before=ret_before,
                    retention_after=ret_after,
                    retention_weight=retention_weight,
                )["utility"]
            )
        )
    return mean_or_none(utilities)
