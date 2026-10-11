"""Shared official-LAB cell runner for operator sanity and the 36x4 matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from expolicy.r9_lab.assimilators.discard import apply_discard
from expolicy.r9_lab.assimilators.episodic import apply_episodic
from expolicy.r9_lab.assimilators.semantic import apply_semantic
from expolicy.r9_lab.assimilators.skillopt import apply_skillopt
from expolicy.r9_lab.benchmark.official_env import OfficialLabEnv, summarize_episodes
from expolicy.r9_lab.constants import A0_DISCARD, A1_EPISODIC, A2_SEMANTIC, A3_SKILLOPT
from expolicy.r9_lab.evaluation.utility import compute_utility
from expolicy.r9_lab.io_util import sha256_file
from expolicy.r9_lab.state.clone import clone_state
from expolicy.r9_lab.state.persistent_state import (
    EPISODIC_NAME,
    HASH_NAME,
    META_NAME,
    SEMANTIC_NAME,
    SKILL_NAME,
    PersistentState,
)
from expolicy.r9_lab.watchdog_util import write_status_live


INJECTION_MARKERS = {
    A1_EPISODIC: "Previous successful trajectories",
    A2_SEMANTIC: "Structured memory",
    A3_SKILLOPT: "Skill document:",
}


def record_to_item(rec: Any) -> dict[str, Any]:
    if hasattr(rec, "to_dict"):
        return rec.to_dict()
    return dict(rec)


def failure_type_stats(results: Sequence[Any]) -> dict[str, Any]:
    n = len(results)
    n_fail = sum(1 for r in results if not getattr(r, "success", False))
    n_infra = sum(1 for r in results if getattr(r, "infra_failure", False))
    n_parser_bad = sum(1 for r in results if not getattr(r, "parser_valid", True))
    return {
        "n": n,
        "n_fail": n_fail,
        "n_infra": n_infra,
        "n_parser_invalid": n_parser_bad,
    }


def state_size(state: PersistentState) -> dict[str, int]:
    return {
        "memory_size": len(state.episodic) + len(state.semantic),
        "skill_size": len(state.skill_markdown or ""),
        "n_episodic": len(state.episodic),
        "n_semantic": len(state.semantic),
    }


def fingerprint(state: PersistentState) -> dict[str, str]:
    root = state.root_path()
    out = {"content_hash": state.content_hash()}
    for name in (EPISODIC_NAME, SEMANTIC_NAME, SKILL_NAME, META_NAME, HASH_NAME):
        path = root / name
        if path.exists():
            out[name] = sha256_file(path)
    return out


def run_items(
    lab: OfficialLabEnv,
    client: Any,
    items: Sequence[dict[str, Any]],
    state: PersistentState | None,
    *,
    status_path: Path | None = None,
    gpu: str | None = None,
    stage: str = "",
    batch_id: str = "",
) -> list:
    results = []
    for i, item in enumerate(items):
        if status_path is not None:
            write_status_live(
                status_path,
                gpu=gpu,
                stage=stage,
                batch_id=batch_id,
                completed_tasks=i,
            )
        results.append(
            lab.run_episode(
                item["sample_index"],
                client,
                state=state,
                task_id=str(item.get("task_id") or ""),
                instruction_hint=str(item.get("instruction") or ""),
            )
        )
    return results


def as_experiences(results: Sequence[Any], *, env: str, bucket: str) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        row = r.as_experience(bucket=bucket)
        row["env"] = env
        rows.append(row)
    return rows


def apply_action(
    action: str,
    state: PersistentState,
    experiences: Sequence[dict[str, Any]],
    *,
    client: Any = None,
    validate_fn=None,
) -> dict[str, Any]:
    if action == A0_DISCARD:
        return apply_discard(state, experiences)
    if action == A1_EPISODIC:
        return apply_episodic(state, experiences)
    if action == A2_SEMANTIC:
        return apply_semantic(state, experiences, client=client)
    if action == A3_SKILLOPT:
        if validate_fn is None:
            validate_fn = lambda _skill: 0.0
        return apply_skillopt(state, experiences, validate_fn=validate_fn, client=client)
    raise ValueError(f"unknown action {action}")


def eval_success(
    lab: OfficialLabEnv,
    client: Any,
    items: Sequence[dict[str, Any]],
    state: PersistentState,
    *,
    status_path: Path | None = None,
    gpu: str | None = None,
    stage: str = "eval",
    batch_id: str = "",
) -> dict[str, Any]:
    before = fingerprint(state)
    results = run_items(
        lab,
        client,
        items,
        state,
        status_path=status_path,
        gpu=gpu,
        stage=stage,
        batch_id=batch_id,
    )
    after = fingerprint(state)
    if before != after:
        raise RuntimeError("read-only probe mutated persistent state")
    stats = summarize_episodes(results)
    stats["state_hash"] = before["content_hash"]
    return stats


def make_validate_fn(
    lab: OfficialLabEnv,
    client: Any,
    val_items: Sequence[dict[str, Any]],
    tmp: Path,
    *,
    status_path: Path | None = None,
    gpu: str | None = None,
    batch_id: str = "",
):
    def _fn(skill: str) -> float:
        root = tmp / "skillopt_val"
        st = PersistentState.create(root)
        st.skill_markdown = skill
        st.save()
        stats = eval_success(
            lab,
            client,
            val_items,
            st,
            status_path=status_path,
            gpu=gpu,
            stage="skillopt_val",
            batch_id=batch_id,
        )
        return float(stats.get("success") or 0.0)

    return _fn


def run_matrix_cell(
    *,
    env: str,
    action: str,
    acquire_items: Sequence[dict[str, Any]],
    shadow_items: Sequence[dict[str, Any]],
    retention_items: Sequence[dict[str, Any]],
    val_items: Sequence[dict[str, Any]],
    s0_dir: Path,
    branch_dir: Path,
    lab: OfficialLabEnv,
    client: Any,
    baseline: dict[str, float],
    status_path: Path | None = None,
    gpu: str | None = None,
    batch_id: str = "",
) -> dict[str, Any]:
    s0 = PersistentState.load(s0_dir) if (s0_dir / "STATE.json").exists() else PersistentState.create(s0_dir)
    branch = clone_state(s0, branch_dir)
    acquire = run_items(
        lab,
        client,
        acquire_items,
        branch,
        status_path=status_path,
        gpu=gpu,
        stage="matrix_acquire",
        batch_id=f"{batch_id}:{action}",
    )
    experiences = as_experiences(acquire, env=env, bucket="acquisition_pool")
    validate_fn = make_validate_fn(lab, client, val_items[:8], branch_dir / "_val")
    assim = apply_action(action, branch, experiences, client=client, validate_fn=validate_fn)
    shadow = eval_success(lab, client, shadow_items, branch)
    retention = eval_success(lab, client, retention_items, branch)
    util = compute_utility(
        shadow_before=float(baseline["shadow"]),
        shadow_after=float(shadow["success"]),
        retention_before=float(baseline["retention"]),
        retention_after=float(retention["success"]),
    )
    acq_stats = summarize_episodes(acquire)
    sizes = state_size(branch)
    return {
        "env": env,
        "action": action,
        "batch_id": batch_id,
        "assim": assim,
        "acquire_success": acq_stats["success"],
        "batch_success_rate": acq_stats["success"],
        "mean_steps": acq_stats["mean_steps"],
        "invalid_action_rate": acq_stats["invalid_action_rate"],
        "tokens": int(acq_stats["tokens"]) + int(shadow.get("tokens") or 0) + int(retention.get("tokens") or 0),
        "wall_time": float(acq_stats["wall_time"]) + float(shadow.get("wall_time") or 0) + float(retention.get("wall_time") or 0),
        "shadow_after": shadow["success"],
        "retention_after": retention["success"],
        "shadow_n": shadow["n"],
        "retention_n": retention["n"],
        "base_success_rate": baseline["shadow"],
        "failure_type_stats": failure_type_stats(acquire),
        "utility": util,
        "U": util["U"],
        "used_future_test": False,
        **sizes,
    }
