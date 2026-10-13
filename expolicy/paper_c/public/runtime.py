"""P0 runtime audit. Official construction failure does not fall back to a fake environment."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from expolicy.paper_c.public.budget import BudgetExhausted, EpisodeBudget
from expolicy.paper_c.public.e0 import EpisodicE0
from expolicy.r9_lab.benchmark.official_env import OfficialLabEnv, OfficialLabInfraError
from expolicy.r9_lab.state.clone import clone_state
from expolicy.r9_lab.state.persistent_state import PersistentState

EnvFactory = Callable[[str], Any]


def refuse_fake(env: Any) -> None:
    if type(env).__name__ == "FakeLabEnv":
        raise RuntimeError("public runtime refused FakeLabEnv")


def construct_official_or_stop(env_name: str) -> OfficialLabEnv:
    try:
        lab = OfficialLabEnv(env_name)
    except OfficialLabInfraError:
        raise
    refuse_fake(lab)
    return lab


def _experience_from_result(result: Any, *, family: str, bucket: str) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "env": result.env,
        "family": family,
        "instruction": result.instruction,
        "trajectory": result.trajectory,
        "success": bool(result.success) and not result.infra_failure,
        "verifier_outcome": "CORRECT" if result.success and not result.infra_failure else "INCORRECT",
        "infra_failure": bool(result.infra_failure),
        "bucket": bucket,
        "tokens": int(result.tokens),
        "source_verified": bool(result.success) and not result.infra_failure,
    }


def run_runtime_audit(
    *,
    lab: Any,
    client: Any,
    tasks: list[dict[str, Any]],
    work_dir: Path,
    cap: int = 64,
) -> dict[str, Any]:
    refuse_fake(lab)
    work_dir.mkdir(parents=True, exist_ok=True)
    budget = EpisodeBudget(cap, stage="runtime")
    e0 = EpisodicE0()
    state = PersistentState.create(work_dir / "state")
    e0.prepare_state(state)
    state.save()
    rows: list[dict[str, Any]] = []
    infra = 0
    for task in tasks:
        if budget.remaining < 1:
            break
        try:
            budget.reserve(1, kind="runtime_episode", detail=str(task["task_id"]))
        except BudgetExhausted:
            break
        before = state.content_hash()
        try:
            result = lab.run_episode(
                task["sample_index"],
                client,
                state=state,
                task_id=str(task["task_id"]),
                instruction_hint=str(task.get("instruction") or ""),
            )
        except OfficialLabInfraError as exc:
            infra += 1
            rows.append(
                {
                    "task_id": task["task_id"],
                    "success": False,
                    "infra_failure": True,
                    "state_hash_before": before,
                    "state_hash_after": state.content_hash(),
                    "n_written": 0,
                    "tokens": 0,
                    "instruction": "",
                    "trajectory": [],
                    "messages": [],
                    "error": str(exc),
                }
            )
            continue
        if result.infra_failure:
            infra += 1
        experience = _experience_from_result(result, family=str(task.get("family") or ""), bucket="runtime_smoke")
        applied = e0.apply(state, [experience] if experience["source_verified"] else [])
        rows.append(
            {
                "task_id": task["task_id"],
                "success": experience["success"],
                "infra_failure": result.infra_failure,
                "state_hash_before": before,
                "state_hash_after": state.content_hash(),
                "n_written": applied["n_written"],
                "tokens": result.tokens,
                "instruction": result.instruction,
                "trajectory": result.trajectory,
                "messages": result.messages,
            }
        )
    successes = [row for row in rows if row["success"]]
    injection = {"visible": False, "reason": "no verified success to inject"}
    if successes:
        preview = e0.injection_preview(state, str(successes[0]["instruction"] or "check"))
        probe = tasks[0]
        before_hash = state.content_hash()
        if budget.remaining >= 1:
            budget.reserve(1, kind="injection_check", detail=str(probe["task_id"]))
            checked = lab.run_episode(
                probe["sample_index"],
                client,
                state=state,
                task_id=str(probe["task_id"]) + ":inject-check",
                instruction_hint=str(probe.get("instruction") or ""),
            )
            blob = "\n".join(str(message.get("content") or "") for message in checked.messages)
            injection = {
                "visible": "Previous successful trajectories:" in blob and bool(preview["hit_ids"]),
                "hit_ids": preview["hit_ids"],
                "chars_after": preview["chars_after"],
                "state_unchanged": state.content_hash() == before_hash,
            }
            if state.content_hash() != before_hash:
                injection["visible"] = False
                injection["reason"] = "injection check mutated state"
    noop = _noop_branches(state, lab, client, tasks[:1], budget, work_dir)
    stopper = _check_stopper()
    status = "C_RUNTIME_VALID"
    reasons = []
    if infra and infra == len(rows):
        status = "BLOCKED_INFRA"
        reasons.append("全部 episode 都是基础设施失败")
    elif not successes:
        status = "INVALID_RUNTIME"
        reasons.append("没有官方验证成功的轨迹，无法检查注入")
    elif not injection.get("visible"):
        status = "INVALID_RUNTIME"
        reasons.append("成功记忆没有出现在后续模型输入中")
    elif not noop.get("paired"):
        status = "INVALID_RUNTIME"
        reasons.append("no-update、空 apply 和 round-trip 没有形成配对")
    elif not stopper["stopped"]:
        status = "INVALID_RUNTIME"
        reasons.append("预算停止器没有拦住超额 episode")
    return {
        "status": status,
        "reasons": reasons,
        "n_episodes": budget.used,
        "n_success": len(successes),
        "n_failure": len(rows) - len(successes),
        "infra": infra,
        "injection": injection,
        "noop": noop,
        "stopper": stopper,
        "rows": rows,
        "algorithm_hash": e0.algorithm_hash(),
    }


def _noop_branches(state, lab, client, tasks, budget: EpisodeBudget, work_dir: Path) -> dict[str, Any]:
    if not tasks or budget.remaining < 3:
        return {"paired": False, "reason": "not enough budget or tasks"}
    task = tasks[0]
    scores = {}
    for name, updater in (
        ("no-update", None),
        ("apply-empty", lambda clone: EpisodicE0().apply(clone, [])),
        ("round-trip", lambda clone: clone.save()),
    ):
        clone = clone_state(state, work_dir / name)
        if updater is not None:
            updater(clone)
        budget.reserve(1, kind="noop", detail=name)
        result = lab.run_episode(
            task["sample_index"],
            client,
            state=clone,
            task_id=f"{task['task_id']}:{name}",
            instruction_hint=str(task.get("instruction") or ""),
        )
        scores[name] = None if result.infra_failure else float(result.success)
    paired = all(value is not None for value in scores.values())
    deltas = {}
    if paired:
        deltas = {
            "apply_empty_minus_no_update": scores["apply-empty"] - scores["no-update"],
            "round_trip_minus_no_update": scores["round-trip"] - scores["no-update"],
        }
    return {"paired": paired, "scores": scores, "raw_deltas": deltas}


def _check_stopper() -> dict[str, Any]:
    budget = EpisodeBudget(1, stage="stopper-check")
    budget.reserve(1, kind="probe")
    stopped = False
    try:
        budget.reserve(1, kind="probe")
    except BudgetExhausted:
        stopped = True
    return {"stopped": stopped, "used_after_refusal": budget.used}
