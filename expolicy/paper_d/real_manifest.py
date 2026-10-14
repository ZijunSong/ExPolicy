"""Frozen execution plan. Observed actions do not define the planned panel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from expolicy.r9_lab.io_util import sha256_json


def build_run_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    environments = list(config["environments"])
    states = list(config["states"])
    batches = int(config["development_batches_per_env"])
    actions = list(config["primary_actions"])
    probes = int(config["future_tasks_per_x"]) + int(config["retention_tasks_per_x"])
    n_x = len(environments) * len(states) * batches
    matrix_episodes = n_x * (1 + len(actions)) * probes
    plan = {
        "protocol_version": config["protocol_version"],
        "n_x": n_x,
        "matrix_episodes_without_reuse": matrix_episodes,
        "history_episodes": len(environments) * 16,
        "acquisition_episodes": len(environments) * batches * int(config["experience_batch_size"]),
        "smoke_episodes": len(environments) * 2 * 2 * 3 * 8 * 2,
        "content_episodes": len(environments) * 2 * 2 * 2 * 2 * 16,
        "skillopt_episode_cap": 192,
        "cap": int(config["pilot_episode_cap_all_inclusive"]),
    }
    plan["planned_total"] = (
        plan["history_episodes"]
        + plan["acquisition_episodes"]
        + plan["smoke_episodes"]
        + plan["skillopt_episode_cap"]
        + plan["matrix_episodes_without_reuse"]
        + plan["content_episodes"]
    )
    plan["within_cap"] = plan["planned_total"] <= plan["cap"]
    plan["plan_hash"] = sha256_json(plan)
    return plan


def planned_cells(
    *,
    configuration_id: str,
    environments: Sequence[str],
    states: Sequence[str],
    batches: Sequence[str],
    actions: Sequence[str],
    tasks: Sequence[Mapping[str, str]],
    repeat_id: str = "0",
) -> list[dict[str, str]]:
    cells = []
    for env in environments:
        for state in states:
            for batch in batches:
                for action in actions:
                    for task in tasks:
                        cells.append(
                            {
                                "configuration_id": configuration_id,
                                "state_id": state,
                                "batch_id": batch,
                                "env": env,
                                "action": action,
                                "content_mode": "real",
                                "probe_role": task["probe_role"],
                                "task_id": task["task_id"],
                                "repeat_id": repeat_id,
                                "experience_source_group": batch,
                                "probe_source_group": task["probe_source_group"],
                            }
                        )
    return cells


def audit_split(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Renamed ids of the same content hash are not new tasks or new groups."""
    reasons = []
    by_hash: dict[str, set[str]] = {}
    for task in tasks:
        content = str(task.get("content_hash") or "")
        by_hash.setdefault(content, set()).add(str(task.get("task_id")))
    collisions = {key: sorted(ids) for key, ids in by_hash.items() if key and len(ids) > 1}
    if collisions:
        reasons.append("content hash is shared by renamed task ids")
    roles = {str(task.get("role")) for task in tasks}
    required = {"history", "acquisition", "diagnostic_select", "diagnostic_eval", "prospective_test"}
    missing_roles = sorted(required - roles)
    if missing_roles:
        reasons.append(f"missing data roles {missing_roles}")
    group_counts: dict[str, set[str]] = {}
    for task in tasks:
        if task.get("role") in {"diagnostic_select", "diagnostic_eval"}:
            group_counts.setdefault(str(task.get("probe_role")), set()).add(str(task.get("probe_source_group")))
    return {
        "valid": not reasons,
        "reasons": reasons,
        "renamed_collisions": collisions,
        "probe_groups": {role: sorted(groups) for role, groups in group_counts.items()},
    }
