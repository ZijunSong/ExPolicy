"""Public C0 screen. Successes are screened once, then every arm is scored on the same tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from expolicy.paper_c.public.budget import BudgetExhausted, EpisodeBudget
from expolicy.paper_c.public.controls import make_content_controls
from expolicy.paper_c.public.e0 import EpisodicE0
from expolicy.paper_c.public.gates import judge_c0_screen
from expolicy.paper_c.public.runtime import _experience_from_result, refuse_fake
from expolicy.paper_c.utility import acquisition_gain
from expolicy.r9_lab.state.clone import clone_state
from expolicy.r9_lab.state.persistent_state import PersistentState


def _mean(flags: list[bool | None]) -> float | None:
    values = [1.0 if flag else 0.0 for flag in flags if flag is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _load_done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["key"])] = row
    return rows


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_c0_screen(
    *,
    lab: Any,
    client: Any,
    assignment: dict[str, Any],
    work_dir: Path,
    cap: int,
) -> dict[str, Any]:
    refuse_fake(lab)
    work_dir.mkdir(parents=True, exist_ok=True)
    budget = EpisodeBudget(cap, stage="c0-screen")
    log_path = work_dir / "episodes.jsonl"
    done = _load_done(log_path)
    e0 = EpisodicE0()
    sequences = []
    for index, tasks in enumerate(assignment["sequences"]):
        sequences.append(_one_sequence(
            lab, client, tasks, assignment, work_dir / f"seq{index}", budget, done, log_path, e0, index
        ))
    judged = _summarize(sequences)
    return {"sequences": sequences, "judgment": judged, "episodes_used": budget.used, "cap": cap}


def _run_saved(lab, client, task, state, budget: EpisodeBudget, done, log_path, key: str, bucket: str) -> dict[str, Any]:
    if key in done:
        return done[key]
    try:
        budget.reserve(1, kind=bucket, detail=key)
    except BudgetExhausted:
        return {"key": key, "missing": True, "infra_failure": False, "success": None}
    result = lab.run_episode(
        task["sample_index"],
        client,
        state=state,
        task_id=str(task["task_id"]),
        instruction_hint=str(task.get("instruction") or ""),
    )
    row = {
        "key": key,
        "task_id": task["task_id"],
        "family": task.get("family"),
        "success": None if result.infra_failure else bool(result.success),
        "infra_failure": bool(result.infra_failure),
        "tokens": int(result.tokens),
        "experience": _experience_from_result(result, family=str(task.get("family") or ""), bucket=bucket),
    }
    _append(log_path, row)
    done[key] = row
    return row


def _one_sequence(lab, client, tasks, assignment, work_dir, budget, done, log_path, e0, index) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    state = PersistentState.create(work_dir / "passive")
    e0.prepare_state(state)
    state.save()
    collected = []
    for task in tasks:
        row = _run_saved(lab, client, task, state, budget, done, log_path, f"s{index}:acq:{task['task_id']}", "acquisition_train")
        if row.get("experience"):
            collected.append(row["experience"])
    controls = make_content_controls(collected)
    start_hash = state.content_hash()
    arm_scores = {}
    probes = {
        "shadow": assignment["shadow_screen"],
        "retention": assignment["retention_screen"],
    }
    for arm, payload in controls["arms"].items():
        clone = clone_state(state, work_dir / arm)
        if arm != "no-update":
            e0.apply(clone, payload)
        scores = {}
        for role, probe_tasks in probes.items():
            flags = []
            for task in probe_tasks:
                row = _run_saved(
                    lab,
                    client,
                    task,
                    clone,
                    budget,
                    done,
                    log_path,
                    f"s{index}:{arm}:{role}:{task['task_id']}",
                    role,
                )
                flags.append(row.get("success"))
            scores[role] = _mean(flags)
        arm_scores[arm] = scores
    gains = {}
    base_s = arm_scores.get("no-update", {}).get("shadow")
    base_r = arm_scores.get("no-update", {}).get("retention")
    for arm, scores in arm_scores.items():
        if None in (base_s, base_r, scores.get("shadow"), scores.get("retention")):
            gains[arm] = {"gain": None, "delta_shadow": None, "delta_retention": None}
            continue
        packed = acquisition_gain(
            shadow_before=float(base_s),
            shadow_after=float(scores["shadow"]),
            retention_before=float(base_r),
            retention_after=float(scores["retention"]),
        )
        gains[arm] = {
            "gain": packed["g"],
            "delta_shadow": packed["delta_shadow"],
            "delta_retention": packed["delta_retention"],
            "raw_shadow": scores["shadow"],
            "raw_retention": scores["retention"],
        }
    return {
        "sequence": index,
        "state_hash": start_hash,
        "n_verified": controls["n_verified"],
        "counts_match": controls["counts_match"],
        "gains": gains,
        "control_invalid": bool(controls["invalid_donors"]),
    }


def _summarize(sequences: list[dict[str, Any]]) -> dict[str, Any]:
    if not sequences:
        return judge_c0_screen({})
    names = ("no-update", "real", "empty-slots", "matched-wrong", "query-only")
    arms: dict[str, Any] = {"counts_match": all(item["counts_match"] for item in sequences), "control_invalid": any(item["control_invalid"] for item in sequences)}
    for name in names:
        gains = [item["gains"].get(name, {}).get("gain") for item in sequences]
        deltas = [item["gains"].get(name, {}).get("delta_shadow") for item in sequences]
        retention = [item["gains"].get(name, {}).get("delta_retention") for item in sequences]
        usable = [float(value) for value in gains if value is not None]
        arms[name] = {
            "gain": None if not usable else sum(usable) / len(usable),
            "delta_shadow": None if not any(value is not None for value in deltas) else sum(float(v) for v in deltas if v is not None) / len([v for v in deltas if v is not None]),
            "delta_retention": None if not any(value is not None for value in retention) else sum(float(v) for v in retention if v is not None) / len([v for v in retention if v is not None]),
            "n": len(usable),
        }
    # Screen has only three sequences. Do not treat a zero-width interval as confirmation.
    judgment = judge_c0_screen(arms)
    judgment["development_only"] = True
    judgment["public_pass"] = False
    return judgment
