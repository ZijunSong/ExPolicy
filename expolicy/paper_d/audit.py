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
    if bundle.get("evidence_class") == "real":
        reasons.extend(_real_resource_reasons(states))
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
    """Pair no-op branches per task. Different tasks may have different prompt hashes."""
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
    grouped: dict[tuple[str, ...], dict[str, Mapping[str, Any]]] = {}
    seen_keys: set[tuple[str, tuple[str, ...]]] = set()
    for branch in NOOP_BRANCHES:
        for row in by_branch[branch]:
            key = _pair_key(row)
            if (branch, key) in seen_keys:
                reasons.append("duplicate primary key")
                continue
            seen_keys.add((branch, key))
            if row.get("outcome") is None:
                reasons.append(f"null outcome for {key}")
                continue
            grouped.setdefault(key, {})[branch] = row
    for key, branches in grouped.items():
        if any(branch not in branches for branch in NOOP_BRANCHES):
            reasons.append(f"incomplete no-op pair {key}")
            continue
        prompts = {str(branches[branch].get("prompt_hash")) for branch in NOOP_BRANCHES}
        models = {str(branches[branch].get("model_hash")) for branch in NOOP_BRANCHES}
        if len(prompts) != 1:
            reasons.append(f"initial prompt differs across no-op branches for task {key[3]}")
        if len(models) != 1:
            reasons.append(f"model hash differs across no-op branches for task {key[3]}")
    base = {
        _pair_key(row): float(row["outcome"])
        for row in by_branch["BASE"]
        if row.get("outcome") is not None
    }
    for branch in ("DISCARD", "REJECTED_EDIT"):
        deltas = []
        for row in by_branch[branch]:
            key = _pair_key(row)
            if key in base and row.get("outcome") is not None:
                deltas.append(float(row["outcome"]) - base[key])
        if deltas and abs(sum(deltas) / len(deltas)) > offset:
            reasons.append(f"unexplained raw no-op offset on {branch}")
    return reasons


def _pair_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("configuration_id") or ""),
        str(row.get("state_id") or ""),
        str(row.get("batch_id") or ""),
        str(row.get("task_id")),
        str(row.get("repeat_id") or "0"),
    )


def resource_isolation_failures(resource: Mapping[str, Any]) -> list[str]:
    """A distinct clone name is not evidence that a writable resource was isolated."""
    if resource.get("applicability") == "not_applicable":
        if not resource.get("reason"):
            return ["not_applicable resource is missing a reason"]
        return []
    if resource.get("mode") == "readonly":
        missing = [
            name
            for name in ("version", "session_isolated")
            if not resource.get(name)
        ]
        if missing:
            return [f"readonly resource missing {missing}"]
        return []
    if resource.get("write_neighbor_restore_passed") is not True:
        return ["writable resource has no write-neighbor-restore evidence"]
    return []


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
                if info.get("applicability") or info.get("mode") == "readonly":
                    for failure in resource_isolation_failures(info):
                        reasons.append(f"{branch_name}:{resource}: {failure}")
                    continue
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


def _real_resource_reasons(states: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for state in states:
        for branch_name, branch in (state.get("branches") or {}).items():
            resources = branch.get("resources") or {}
            for name in CLONE_RESOURCES:
                info = resources.get(name) or {}
                for failure in resource_isolation_failures(info):
                    reasons.append(f"{state.get('state_id')}:{branch_name}:{name}: {failure}")
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
