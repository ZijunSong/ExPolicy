"""No-op measurement audit. A nonzero mean U is not, by itself, evidence of a bug."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from expolicy.paper_a.constants import (
    AUDIT_FAIL,
    AUDIT_INCONCLUSIVE,
    AUDIT_PASS,
    BRANCH_BASE,
    BRANCH_DISCARD,
    BRANCH_REJECTED,
    BRANCH_SENTINEL_PREFIX,
    NOOP_BRANCHES,
    RETENTION,
    SHADOW,
)
from expolicy.paper_a.stats import interval_claim, normal_mean_ci
from expolicy.r9_lab.assimilators.discard import apply_discard
from expolicy.r9_lab.assimilators.episodic import format_episodic_injection, retrieve_episodic
from expolicy.r9_lab.assimilators.semantic import format_semantic_injection, retrieve_semantic
from expolicy.r9_lab.assimilators.skillopt import apply_skillopt
from expolicy.r9_lab.evaluation.utility import compute_utility
from expolicy.r9_lab.io_util import sha256_json, sha256_text
from expolicy.r9_lab.state.clone import clone_state
from expolicy.r9_lab.state.persistent_state import PersistentState


Evaluator = Callable[..., list[dict[str, Any]]]


@dataclass
class AuditTask:
    probe_task_id: str
    eval_split: str
    instruction: str
    env: str = ""


@dataclass
class AuditReport:
    status: str
    reasons: list[str] = field(default_factory=list)
    components: dict[str, Any] = field(default_factory=dict)
    repeats: list[dict[str, Any]] = field(default_factory=list)
    penalty: dict[str, Any] = field(default_factory=dict)
    task_rows: list[dict[str, Any]] = field(default_factory=list)
    randomness_located: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": self.reasons,
            "components": self.components,
            "repeats": self.repeats,
            "penalty": self.penalty,
            "task_rows": self.task_rows,
            "randomness_located": self.randomness_located,
            "uses_discard_recentering": False,
        }


def component_fingerprint(
    state: PersistentState,
    queries: Sequence[str],
    *,
    system_prompt: str,
    tool_config: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    env_snapshot_id: str,
) -> dict[str, Any]:
    episodic_order = []
    semantic_order = []
    for query in queries:
        episodic_order.append([rec.task_id for rec in retrieve_episodic(state, query, trace=False)])
        semantic_order.append(
            [entry.dedup_key() for entry in retrieve_semantic(state, query, trace=False)]
        )
    return {
        "content_hash": state.content_hash(),
        "episodic_hash": sha256_json([rec.to_dict() for rec in state.episodic]),
        "semantic_hash": sha256_json([entry.to_dict() for entry in state.semantic]),
        "skill_hash": sha256_text(state.skill_markdown or ""),
        "committed_hash": sha256_json(list(state.committed_task_ids)),
        "extra_hash": sha256_json(dict(state.extra)),
        "retriever_order": {"episodic": episodic_order, "semantic": semantic_order},
        "system_prompt_hash": sha256_text(system_prompt),
        "tool_config_hash": sha256_json(dict(tool_config)),
        "model_identity_hash": sha256_json(dict(model_identity)),
        "env_snapshot_id": env_snapshot_id,
    }


def _component_diffs(fingerprints: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    base = fingerprints[BRANCH_BASE]
    diffs: dict[str, list[str]] = {}
    comparable = [
        "content_hash",
        "episodic_hash",
        "semantic_hash",
        "skill_hash",
        "committed_hash",
        "extra_hash",
        "retriever_order",
        "system_prompt_hash",
        "tool_config_hash",
        "model_identity_hash",
        "env_snapshot_id",
    ]
    for name, fp in fingerprints.items():
        bad = [key for key in comparable if fp.get(key) != base.get(key)]
        if bad:
            diffs[name] = bad
    return diffs


def _default_prepare(name: str, state: PersistentState, experiences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if name == BRANCH_BASE:
        return {"action": BRANCH_BASE, "changed": False, "accepted": False}
    if name == BRANCH_DISCARD:
        payload = apply_discard(state, experiences)
        payload["accepted"] = False
        return payload
    if name == BRANCH_REJECTED:
        payload = apply_skillopt(state, experiences, validate_fn=lambda _skill: 0.0, client=None)
        if payload.get("accepted"):
            raise RuntimeError("rejected-edit control accepted a skill update")
        return payload
    raise KeyError(name)


def _check_task_sets(tasks: Sequence[AuditTask]) -> list[str]:
    reasons = []
    splits = {task.eval_split for task in tasks}
    allowed = {SHADOW, RETENTION}
    extra = splits - allowed
    if extra:
        reasons.append(f"audit tasks include non-probe splits: {sorted(extra)}")
    if SHADOW not in splits or RETENTION not in splits:
        reasons.append("audit needs both shadow_reward_probe and retention_probe tasks")
    ids = [task.probe_task_id for task in tasks]
    if len(ids) != len(set(ids)):
        reasons.append("duplicate probe task ids in the audit set")
    return reasons


def _rendered_prefix(state: PersistentState, instruction: str) -> str:
    episodic = format_episodic_injection(retrieve_episodic(state, instruction, trace=False))
    semantic = format_semantic_injection(retrieve_semantic(state, instruction, trace=False))
    parts = [part for part in (episodic, semantic) if part]
    if state.skill_markdown.strip():
        parts.append(state.skill_markdown.strip())
    return "\n\n".join(parts)


def _prompt_leak(state: PersistentState, tasks: Sequence[AuditTask], branch: str) -> list[str]:
    sentinel = f"{BRANCH_SENTINEL_PREFIX}{branch}"
    leaks = []
    for task in tasks:
        prefix = _rendered_prefix(state, task.instruction)
        if sentinel in prefix or BRANCH_SENTINEL_PREFIX in prefix:
            leaks.append(task.probe_task_id)
    return leaks


def _pair_matrix(
    outcomes: Sequence[Mapping[str, Any]],
    tasks: Sequence[AuditTask],
) -> dict[str, dict[str, float | None]]:
    by_branch: dict[str, dict[str, float | None]] = {name: {} for name in NOOP_BRANCHES}
    for row in outcomes:
        branch = str(row["branch"])
        task_id = str(row["probe_task_id"])
        status = str(row.get("infra_status") or "ok")
        if status != "ok" or row.get("y") is None:
            by_branch.setdefault(branch, {})[task_id] = None
        else:
            by_branch.setdefault(branch, {})[task_id] = float(row["y"])
    for task in tasks:
        for branch in NOOP_BRANCHES:
            by_branch[branch].setdefault(task.probe_task_id, None)
    return by_branch


def _utility_from_maps(
    base: Mapping[str, float | None],
    other: Mapping[str, float | None],
    tasks: Sequence[AuditTask],
) -> dict[str, Any] | None:
    def _mean(split: str, source: Mapping[str, float | None]) -> float | None:
        vals = [source[task.probe_task_id] for task in tasks if task.eval_split == split]
        if not vals or any(v is None for v in vals):
            return None
        return float(sum(vals) / len(vals))

    numbers = {}
    for split, name in ((SHADOW, "shadow"), (RETENTION, "retention")):
        before = _mean(split, base)
        after = _mean(split, other)
        if before is None or after is None:
            return None
        numbers[name] = (before, after)
    util = compute_utility(
        shadow_before=numbers["shadow"][0],
        shadow_after=numbers["shadow"][1],
        retention_before=numbers["retention"][0],
        retention_after=numbers["retention"][1],
    )
    return util


def audit_noop(
    snapshot: PersistentState | Path | str,
    tasks: Sequence[AuditTask | Mapping[str, Any]],
    evaluate: Evaluator,
    *,
    experiences: Sequence[Mapping[str, Any]],
    work_dir: Path,
    model_identity: Mapping[str, Any],
    system_prompt: str,
    tool_config: Mapping[str, Any],
    env_snapshot_id: str,
    repeats: int = 3,
    equivalence: float = 0.01,
    ci_z: float = 1.96,
    formal_concurrency: int = 1,
    require_formal_concurrency_replay: bool = True,
    prepare_branch: Callable[[str, PersistentState, Sequence[Mapping[str, Any]]], dict[str, Any]] | None = None,
) -> AuditReport:
    """Compare BASE, DISCARD, and a rejected edit on the same tasks.

    Three repeats estimate run noise. They are not three experiment seeds, and the
    equivalence sample size is the number of tasks. Raw paired success differences
    decide the gate. The positive-part retention penalty can make mean U negative
    for a true no-op, so that mean is stored and then not used as a bug test.
    Nothing is recentered by subtracting the DISCARD row.
    """
    audit_tasks = [_as_task(task) for task in tasks]
    report = AuditReport(status=AUDIT_INCONCLUSIVE)
    report.reasons.extend(_check_task_sets(audit_tasks))
    if isinstance(snapshot, PersistentState):
        snapshot.save()
        source: PersistentState | Path = snapshot
    else:
        source = Path(snapshot)

    work_dir.mkdir(parents=True, exist_ok=True)
    prepare = prepare_branch or _default_prepare
    branches: dict[str, PersistentState] = {}
    for name in NOOP_BRANCHES:
        state = clone_state(source, work_dir / name / "state")
        prepare(name, state, experiences)
        branches[name] = state

    queries = [task.instruction for task in audit_tasks]
    fingerprints = {
        name: component_fingerprint(
            state,
            queries,
            system_prompt=system_prompt,
            tool_config=tool_config,
            model_identity=model_identity,
            env_snapshot_id=env_snapshot_id,
        )
        for name, state in branches.items()
    }
    report.components = {
        "fingerprints": {name: _public_fp(fp) for name, fp in fingerprints.items()},
        "diffs_vs_base": _component_diffs(fingerprints),
        "compared_fields": [
            "episodic_hash",
            "semantic_hash",
            "skill_hash",
            "retriever_order",
            "system_prompt_hash",
            "tool_config_hash",
            "env_snapshot_id",
            "model_identity_hash",
        ],
    }
    if report.components["diffs_vs_base"]:
        report.reasons.append(
            "no-op branches differ in deployment components, not only in a single state file hash: "
            + str(report.components["diffs_vs_base"])
        )

    for name, state in branches.items():
        leaks = _prompt_leak(state, audit_tasks, name)
        if leaks:
            report.reasons.append(f"branch sentinel entered the prompt for {name}: {leaks[:5]}")

    modes = ["serial"]
    if require_formal_concurrency_replay:
        modes.append("formal_concurrency")
    per_mode_deltas: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    penalty_rows = []
    for mode in modes:
        concurrency = 1 if mode == "serial" else int(formal_concurrency)
        if concurrency < 1:
            report.reasons.append("formal concurrency must be positive")
            continue
        if mode == "formal_concurrency" and concurrency > 1 and not _evaluator_supports_concurrency(evaluate):
            report.reasons.append(
                "formal concurrency is greater than 1 but this runner is not thread-safe; "
                "serial scores were not relabeled as a concurrent replay"
            )
            continue
        for repeat_id in range(int(repeats)):
            repeat_rows = []
            for name, state in branches.items():
                before_hash = state.content_hash()
                outcomes = evaluate(
                    state,
                    audit_tasks,
                    mode=mode,
                    repeat_id=repeat_id,
                    concurrency=concurrency,
                    branch=name,
                )
                after = PersistentState.load(state.root_path())
                if after.content_hash() != before_hash:
                    report.reasons.append(f"evaluation mutated {name} on {mode} repeat {repeat_id}")
                for row in outcomes:
                    row = dict(row)
                    row["branch"] = name
                    row["mode"] = mode
                    row["repeat_id"] = repeat_id
                    repeat_rows.append(row)
                    report.task_rows.append(row)
            paired = _pair_matrix(repeat_rows, audit_tasks)
            missing = [
                task.probe_task_id
                for task in audit_tasks
                if any(paired[name].get(task.probe_task_id) is None for name in NOOP_BRANCHES)
            ]
            if missing:
                report.reasons.append(f"missing paired outcomes on {mode} repeat {repeat_id}: {missing[:5]}")
                continue
            discard_util = _utility_from_maps(paired[BRANCH_BASE], paired[BRANCH_DISCARD], audit_tasks)
            rejected_util = _utility_from_maps(paired[BRANCH_BASE], paired[BRANCH_REJECTED], audit_tasks)
            if discard_util is None or rejected_util is None:
                report.reasons.append(f"could not form shadow and retention means on {mode} repeat {repeat_id}")
                continue
            penalty_rows.append({"mode": mode, "repeat_id": repeat_id, "discard": discard_util, "rejected": rejected_util})
            deltas = []
            for task in audit_tasks:
                base_y = float(paired[BRANCH_BASE][task.probe_task_id])  # type: ignore[arg-type]
                pair = []
                for name in (BRANCH_DISCARD, BRANCH_REJECTED):
                    other_y = float(paired[name][task.probe_task_id])  # type: ignore[arg-type]
                    pair.append(other_y - base_y)
                    report.task_rows.append(
                        {
                            "mode": mode,
                            "repeat_id": repeat_id,
                            "branch": name,
                            "probe_task_id": task.probe_task_id,
                            "eval_split": task.eval_split,
                            "paired_delta": other_y - base_y,
                            "note": "raw paired success difference; discard was not subtracted",
                        }
                    )
                deltas.append(sum(pair) / len(pair))
            per_mode_deltas[mode].append(np.asarray(deltas, dtype=float))
            report.repeats.append(
                {
                    "mode": mode,
                    "repeat_id": repeat_id,
                    "concurrency": concurrency,
                    "discard_utility": discard_util,
                    "rejected_utility": rejected_util,
                }
            )

    report.penalty = {
        "rows": penalty_rows,
        "note": (
            "U uses a positive-part retention penalty, so a true no-op can have mean U below zero "
            "under random scoring. The gate uses raw paired success differences."
        ),
        "recentered_on_discard": False,
    }
    report.status, extra = _decide_status(
        report.reasons,
        per_mode_deltas,
        equivalence=equivalence,
        ci_z=ci_z,
        require_formal=require_formal_concurrency_replay,
        formal_concurrency=formal_concurrency,
    )
    report.reasons.extend(extra["reasons"])
    report.randomness_located = extra["randomness_located"]
    report.penalty["equivalence"] = extra["equivalence"]
    report.penalty["stratified"] = stratified_paired_summary(report.task_rows)
    report.penalty["variation_observed"] = bool(extra.get("variation_observed"))
    report.penalty["stage_status"] = extra.get("stage_status")
    return report


def stratified_paired_summary(task_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Branch × probe-type × repeat means. Repeats are not multiplied into n."""
    groups: dict[tuple, list[float]] = {}
    for row in task_rows:
        if "paired_delta" not in row:
            continue
        key = (row.get("mode"), row.get("repeat_id"), row.get("branch"), row.get("eval_split"))
        groups.setdefault(key, []).append(float(row["paired_delta"]))
    out = []
    for key, vals in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        out.append(
            {
                "mode": key[0],
                "repeat_id": key[1],
                "branch": key[2],
                "eval_split": key[3],
                "n": len(vals),
                "n_pos": sum(1 for value in vals if value > 0),
                "n_neg": sum(1 for value in vals if value < 0),
                "mean": (sum(vals) / len(vals)) if vals else None,
            }
        )
    return out


def _evaluator_supports_concurrency(evaluate: Evaluator) -> bool:
    return bool(getattr(evaluate, "thread_safe", False))


def _decide_status(
    reasons: Sequence[str],
    per_mode_deltas: Mapping[str, Sequence[np.ndarray]],
    *,
    equivalence: float,
    ci_z: float,
    require_formal: bool,
    formal_concurrency: int,
) -> tuple[str, dict[str, Any]]:
    extra: dict[str, Any] = {"reasons": [], "randomness_located": None, "equivalence": {}}
    structural = [reason for reason in reasons if not reason.startswith("missing")]
    # Missing outcomes are also fatal for a pass, but a systematic component diff fails first.
    if any("differ in deployment" in reason or "mutated" in reason or "sentinel" in reason or "non-probe" in reason for reason in reasons):
        extra["reasons"].append("systematic measurement bias must be fixed before any pass is claimed")
        return AUDIT_FAIL, extra
    if any(reason.startswith("missing") or "could not form" in reason for reason in reasons):
        extra["reasons"].append("paired outcomes are incomplete")
        return AUDIT_INCONCLUSIVE, extra

    serial = list(per_mode_deltas.get("serial") or [])
    if not serial:
        extra["reasons"].append("no serial replay was recorded")
        return AUDIT_INCONCLUSIVE, extra
    serial_identical = all(np.array_equal(vec, serial[0]) for vec in serial)
    serial_zero = bool(np.all(serial[0] == 0.0))
    averaged = np.mean(np.stack(serial, axis=0), axis=0)
    # Sample size is the number of paired task differences inside one repeat, after averaging repeats.
    # Two branches x tasks: we test the task-level mean of branch-vs-base, so divide the stacked
    # pairs back into a per-task mean across the two no-op branches.
    center, lo, hi = normal_mean_ci(averaged.tolist(), z=ci_z)
    extra["equivalence"] = {
        "mean_paired_delta": center,
        "ci": [lo, hi],
        "bound": equivalence,
        "n_pairs": int(averaged.size),
        "serial_deterministic": serial_identical,
        "paired_deltas_identical": serial_identical,
        "interval_claim": interval_claim(averaged.tolist()),
        "note": "n_pairs is the number of tasks. The two no-op branches are averaged within a task, and repeats are not extra samples. A zero-width normal interval is not a population-noise claim.",
    }
    if serial_identical and serial_zero:
        serial_status = AUDIT_PASS
    elif serial_identical and not serial_zero:
        extra["reasons"].append("deterministic serial replay produced a nonzero no-op success difference")
        return AUDIT_FAIL, extra
    else:
        extra["variation_observed"] = True
        if hi < -equivalence or lo > equivalence:
            extra["reasons"].append("serial no-op difference sits outside the pre-registered equivalence interval")
            return AUDIT_FAIL, extra
        if lo >= -equivalence and hi <= equivalence:
            serial_status = AUDIT_PASS
        else:
            extra["reasons"].append("serial sample cannot fit inside the equivalence interval")
            serial_status = AUDIT_INCONCLUSIVE

    if not require_formal:
        return serial_status, extra
    formal = list(per_mode_deltas.get("formal_concurrency") or [])
    if not formal:
        extra["reasons"].append("formal-concurrency replay was not run")
        return AUDIT_INCONCLUSIVE, extra
    formal_identical = all(np.array_equal(vec, formal[0]) for vec in formal)
    formal_zero = bool(np.all(formal[0] == 0.0))
    if formal_concurrency == 1 and formal_identical and formal_zero and serial_status == AUDIT_PASS:
        extra["randomness_located"] = extra["randomness_located"] or "none_observed"
        return AUDIT_PASS, extra
    if formal_concurrency > 1:
        extra["variation_observed"] = True
    elif not formal_identical:
        extra["variation_observed"] = True
    formal_avg = np.mean(np.stack(formal, axis=0), axis=0)
    f_center, f_lo, f_hi = normal_mean_ci(formal_avg.tolist(), z=ci_z)
    extra["equivalence"]["formal"] = {
        "mean_paired_delta": f_center,
        "ci": [f_lo, f_hi],
        "deterministic": formal_identical,
    }
    if formal_identical and not formal_zero:
        extra["reasons"].append("formal replay is deterministic and the no-op scores still differ")
        return AUDIT_FAIL, extra
    if formal_identical and formal_zero and serial_status == AUDIT_PASS:
        extra["stage_status"] = {"serial": serial_status, "formal_concurrency": AUDIT_PASS}
        return AUDIT_PASS, extra
    # A later block must not promote an earlier inconclusive or failed block to PASS.
    if serial_status == AUDIT_PASS and f_lo >= -equivalence and f_hi <= equivalence:
        extra["stage_status"] = {"serial": serial_status, "formal_concurrency": AUDIT_PASS}
        return AUDIT_PASS, extra
    if serial_status != AUDIT_PASS:
        extra["stage_status"] = {"serial": serial_status, "formal_concurrency": AUDIT_INCONCLUSIVE}
        extra["reasons"].append("an earlier replay block did not pass, so a later block cannot supply the global PASS")
    if f_hi < -equivalence or f_lo > equivalence:
        extra["reasons"].append("formal replay difference is outside the equivalence interval")
        return AUDIT_FAIL, extra
    extra["reasons"].append("formal replay is not yet inside the equivalence interval")
    return AUDIT_INCONCLUSIVE, extra


def _as_task(task: AuditTask | Mapping[str, Any]) -> AuditTask:
    if isinstance(task, AuditTask):
        return task
    return AuditTask(
        probe_task_id=str(task["probe_task_id"]),
        eval_split=str(task["eval_split"]),
        instruction=str(task.get("instruction") or ""),
        env=str(task.get("env") or ""),
    )


def _public_fp(fp: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fp.items()}
