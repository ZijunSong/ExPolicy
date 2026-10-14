"""No-op measurement audit. An unexplained offset blocks scientific conclusions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from expolicy.paper_d.constants import (
    CLONE_RESOURCES,
    MEASUREMENT_INVALID,
    NOOP_BRANCHES,
    NOOP_OFFSET,
    PROVENANCE_FIELDS,
    STATUS_OK,
)


def audit_measurement(bundle: Mapping[str, Any], *, offset: float = NOOP_OFFSET) -> dict[str, Any]:
    """Check clone isolation, provenance, pairing, replay, and the no-op raw gap.

    A zero raw no-op gap can still have a negative legacy penalty. That bias is recorded
    and does not by itself invalidate the measurement.
    """
    reasons: list[str] = []
    states = list(bundle.get("states") or [])
    batches = list(bundle.get("batches") or [])
    if len(states) < 2:
        reasons.append("fewer than two states")
    if len(batches) < 2:
        reasons.append("fewer than two experience batches")
    if bundle.get("seed_relabelled_batches"):
        reasons.append("同一个 batch 换 seed 不能当成新的 x")
    reasons.extend(_audit_states(states))
    reasons.extend(_audit_replay(bundle))
    reasons.extend(_audit_offset(bundle, offset))
    reasons.extend(_audit_infra(bundle))
    if "paired_rows" in bundle:
        reasons.extend(audit_paired_rows(bundle["paired_rows"], offset=offset))
    status = STATUS_OK if not reasons else MEASUREMENT_INVALID
    raw_shadow = _as_float(bundle.get("raw_shadow_delta"))
    legacy = _as_float(bundle.get("legacy_utility"))
    penalty_bias = (
        raw_shadow is not None
        and abs(raw_shadow) <= offset
        and legacy is not None
        and legacy < -offset
    )
    return {
        "status": status,
        "reasons": reasons,
        "scientific_conclusion_allowed": status == STATUS_OK,
        "penalty_negative_bias": penalty_bias,
        "independent_task_count": (bundle.get("replay") or {}).get("independent_task_count"),
        "long_term_covered": False,
    }


def audit_paired_rows(rows: Sequence[Mapping[str, Any]], *, offset: float = NOOP_OFFSET) -> list[str]:
    """BASE, DISCARD, and rejected-edit outcomes must share task ids, prompts, and model hashes."""
    reasons: list[str] = []
    by_branch: dict[str, list[Mapping[str, Any]]] = {branch: [] for branch in NOOP_BRANCHES}
    for row in rows:
        branch = str(row.get("branch") or row.get("action"))
        if branch not in by_branch:
            reasons.append(f"unexpected no-op branch {branch}")
            continue
        by_branch[branch].append(row)
    if any(not by_branch[branch] for branch in NOOP_BRANCHES):
        reasons.append("BASE, DISCARD, and REJECTED_EDIT must all be measured")
        return reasons
    task_sets = [{str(row.get("task_id")) for row in by_branch[branch]} for branch in NOOP_BRANCHES]
    if any(item != task_sets[0] for item in task_sets):
        reasons.append("baseline task set differs across no-op branches")
    hashes = {str(row.get("model_hash")) for branch in NOOP_BRANCHES for row in by_branch[branch]}
    prompts = {str(row.get("prompt_hash")) for branch in NOOP_BRANCHES for row in by_branch[branch]}
    if len(hashes) != 1:
        reasons.append("model hash differs across paired no-op branches")
    if len(prompts) != 1:
        reasons.append("prompt hash differs across paired no-op branches")
    base = {str(row["task_id"]): float(row["outcome"]) for row in by_branch["BASE"]}
    for branch in ("DISCARD", "REJECTED_EDIT"):
        deltas = [float(row["outcome"]) - base[str(row["task_id"])] for row in by_branch[branch] if str(row["task_id"]) in base]
        if deltas and abs(sum(deltas) / len(deltas)) > offset:
            reasons.append(f"unexplained raw no-op offset on {branch}")
    return reasons


def valid_noop_bundle() -> dict[str, Any]:
    """Synthetic audit fixture. It is not a LifelongAgentBench result."""
    task_ids = ["task-1", "task-2"]

    def resources(suffix: str) -> dict[str, dict[str, str]]:
        return {
            key: {"clone_id": f"{key}-{suffix}", "reset_id": f"reset-{key}-{suffix}"}
            for key in CLONE_RESOURCES
        }

    def branch(name: str, suffix: str) -> dict[str, Any]:
        body: dict[str, Any] = {field: f"{field}-{suffix}" for field in PROVENANCE_FIELDS if field != "task_ids"}
        body["task_ids"] = list(task_ids)
        body["model_hash"] = "model-a"
        body["tokenizer_hash"] = "tokenizer-a"
        body["prompt_hash"] = "prompt-a"
        body["tool_schema_hash"] = "tools-a"
        body["inference_config_hash"] = "infer-a"
        body["concurrency"] = 1
        body["env_reset_id"] = f"env-{suffix}"
        body["order_randomized"] = True
        body["order_seed"] = suffix
        body["resources"] = resources(suffix)
        body["branch"] = name
        return body

    def state(name: str) -> dict[str, Any]:
        return {
            "state_id": name,
            "branches": {
                "BASE": branch("BASE", f"{name}-base"),
                "DISCARD": branch("DISCARD", f"{name}-discard"),
                "REJECTED_EDIT": branch("REJECTED_EDIT", f"{name}-rejected"),
            },
        }

    return {
        "configuration_id": "audit-fixture",
        "states": [state("s0"), state("s1")],
        "batches": ["batch-0", "batch-1"],
        "replay": {
            "deterministic_match": True,
            "concurrency_repeats": 2,
            "independent_task_count": 2,
        },
        "task_ids": task_ids,
        "raw_shadow_delta": 0.0,
        "raw_retention_delta": 0.0,
        "legacy_utility": -0.03,
        "infra_failures": [{"task_id": "task-x", "reason": "timeout", "counted_in_cost": True}],
        "retry_policy": {"max_retries": 2, "count_toward_budget": True, "used_retries": 1},
        "dropped_task_ids": [],
        "seed_relabelled_batches": [],
        "environment_class": "synthetic_audit_fixture",
    }


def _audit_states(states: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    seen_clones: list[tuple[str, str]] = []
    model_hashes: set[str] = set()
    for state in states:
        branches = state.get("branches") or {}
        if any(branch not in branches for branch in NOOP_BRANCHES):
            reasons.append(f"state {state.get('state_id')} is missing a no-op branch")
            continue
        task_sets = []
        for branch_name in NOOP_BRANCHES:
            branch = branches[branch_name]
            reasons.extend(_missing_provenance(branch, f"{state.get('state_id')}:{branch_name}"))
            if branch.get("order_randomized") is not True:
                reasons.append(f"{branch_name} order was not randomized")
            model_hashes.add(str(branch.get("model_hash")))
            task_sets.append(tuple(branch.get("task_ids") or []))
            for resource in CLONE_RESOURCES:
                info = (branch.get("resources") or {}).get(resource) or {}
                clone_id = info.get("clone_id")
                reset_id = info.get("reset_id")
                if not clone_id or not reset_id:
                    reasons.append(f"missing clone for {resource}")
                    continue
                seen_clones.append((str(resource), str(clone_id)))
        if any(item != task_sets[0] for item in task_sets):
            reasons.append("paired branches use different task ids")
    if len(model_hashes) > 1:
        reasons.append("version mix: model hash is not constant")
    counts: dict[tuple[str, str], int] = {}
    for item in seen_clones:
        counts[item] = counts.get(item, 0) + 1
    if any(count > 1 for count in counts.values()):
        reasons.append("clone crosstalk: two branches share a resource clone")
    return reasons


def _missing_provenance(branch: Mapping[str, Any], label: str) -> list[str]:
    missing = [field for field in PROVENANCE_FIELDS if branch.get(field) in (None, "", [])]
    if missing:
        return [f"{label} missing provenance: {missing}"]
    return []


def _audit_replay(bundle: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    replay = bundle.get("replay") or {}
    if replay.get("deterministic_match") is not True:
        reasons.append("deterministic replay did not match")
    task_ids = list(bundle.get("task_ids") or [])
    claimed = replay.get("independent_task_count")
    if claimed is None:
        reasons.append("independent task count was not recorded")
    elif task_ids and int(claimed) != len(set(task_ids)):
        reasons.append("repeats were counted as independent tasks")
    return reasons


def _audit_offset(bundle: Mapping[str, Any], offset: float) -> list[str]:
    reasons: list[str] = []
    for key in ("raw_shadow_delta", "raw_retention_delta"):
        value = _as_float(bundle.get(key))
        if value is None:
            reasons.append(f"{key} was not separated from the penalty")
        elif abs(value) > offset:
            reasons.append(f"unexplained fixed offset in {key}")
    return reasons


def _audit_infra(bundle: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if "infra_failures" not in bundle:
        reasons.append("infrastructure failures were not listed")
    if bundle.get("dropped_task_ids"):
        reasons.append("tasks were dropped because a group failed")
    policy = bundle.get("retry_policy") or {}
    if "max_retries" not in policy:
        reasons.append("retry policy was not pre-specified")
    elif int(policy.get("used_retries") or 0) > int(policy["max_retries"]):
        reasons.append("retries exceeded the pre-specified limit")
    return reasons


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
