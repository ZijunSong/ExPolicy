"""OS repeated paired pilot. Assimilation is frozen once; probes only read it."""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path
from typing import Any

from expolicy.paper_d.official_worker import execute_official_job
from expolicy.paper_d.persistent_ledger import PersistentLedger
from expolicy.r9_lab.assimilators.discard import apply_discard
from expolicy.r9_lab.assimilators.episodic import apply_episodic, format_episodic_injection, retrieve_episodic
from expolicy.r9_lab.assimilators.semantic import commit_semantic_entries, generate_semantic_v11
from expolicy.r9_lab.io_util import sha256_json, write_json
from expolicy.r9_lab.llm_client import VLLMClient
from expolicy.r9_lab.state.persistent_state import PersistentState


ARMS = ("BASE", "DISCARD", "EPISODIC_REPLAY", "SEMANTIC_MEMORY")
REVISION = "d-noise-calibration-v1"


def _instruction(artifact: dict[str, Any]) -> str:
    trace = artifact.get("request_trace") or []
    if not trace:
        return ""
    body = trace[0]["request_wire"]
    if isinstance(body, str):
        body = json.loads(body)
    messages = body.get("messages") or []
    users = [str(item.get("content") or "") for item in messages if item.get("role") == "user" and item.get("content")]
    if users:
        return users[-1]
    return str(messages[0]["content"]) if messages else ""


def _load_kind(root: Path, kind: str, env: str) -> list[dict[str, Any]]:
    found = []
    for path in sorted(root.glob("executions/*/*/result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        job = data.get("job") or {}
        if job.get("job_kind") == kind and job.get("env") == env and data.get("infra_status") == "ok":
            data["instruction"] = _instruction(data)
            found.append(data)
    return found


def probe_tasks(root: Path) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for item in _load_kind(root, "smoke", "os"):
        job = item["job"]
        if job.get("state_id") != "S0" or job.get("branch") != "BASE" or int(job.get("repeat_id") or 0) != 0:
            continue
        seen[str(job["task_id"])] = {
            "task_id": job["task_id"],
            "sample_index": job["sample_index"],
            "probe_role": job["probe_role"],
            "instruction": item["instruction"],
            "development_task": True,
        }
    return sorted(seen.values(), key=lambda row: (row["probe_role"], int(row["sample_index"])))


def _clone(state: PersistentState, root: Path) -> PersistentState:
    return PersistentState(
        root=str(root),
        episodic=list(state.episodic),
        semantic=list(state.semantic),
        skill_markdown=state.skill_markdown,
        extra=dict(state.extra),
        committed_task_ids=list(state.committed_task_ids),
    )


def _experience(item: dict[str, Any]) -> dict[str, Any]:
    success = bool(item.get("success"))
    return {
        "task_id": str(item["job"]["task_id"]),
        "env": "os",
        "instruction": item["instruction"],
        "trajectory": list(item.get("trajectory") or []),
        "success": success,
        "verifier_outcome": "CORRECT" if success else "INCORRECT",
        "infra_status": "ok",
        "bucket": "acquisition_pool" if item["job"].get("job_kind") == "acquisition" else "calibration",
    }


def build_consumed_history(smoke_root: Path, destination: Path) -> dict[str, Any]:
    source = PersistentState.load(smoke_root / "states" / "os" / "S0")
    state = _clone(source, destination)
    experiences = [_experience(item) for item in _load_kind(smoke_root, "history", "os") if item.get("success")]
    report = apply_episodic(state, experiences)
    return {"written": report["n"], "n_experiences": len(experiences), "state_hash": state.content_hash()}


def acquisition_batches(smoke_root: Path) -> dict[str, list[dict[str, Any]]]:
    items = _load_kind(smoke_root, "acquisition", "os")
    items.sort(key=lambda item: int(item["job"]["sample_index"]))
    return {
        "os-B1": [_experience(item) for item in items[:8]],
        "os-B2": [_experience(item) for item in items[8:16]],
    }


def prepare_arm_state(
    base: PersistentState,
    arm: str,
    experiences: list[dict[str, Any]],
    destination: Path,
    client: VLLMClient | None,
) -> dict[str, Any]:
    state = _clone(base, destination)
    report: dict[str, Any] = {"arm": arm, "accepted": arm != "BASE", "changed": False, "operator_status": "ok"}
    if arm == "EPISODIC_REPLAY":
        applied = apply_episodic(state, experiences)
        report.update(accepted=True, changed=bool(applied["changed"]), detail=applied)
    elif arm == "SEMANTIC_MEMORY":
        entries, meta = generate_semantic_v11(experiences, client, record_dir=destination.parent / "semantic_raw" / destination.name)
        report["semantic_meta"] = {key: value for key, value in meta.items() if key != "raw_response_path"}
        report["raw_response_path"] = meta.get("raw_response_path")
        if meta.get("generation_status") != "ok":
            report["operator_status"] = "OPERATOR_INVALID"
            state.save()
        else:
            committed = commit_semantic_entries(state, experiences, entries)
            report.update(accepted=True, changed=bool(committed["changed"]), detail=committed)
    elif arm == "DISCARD":
        applied = apply_discard(state, experiences)
        state.save()
        report.update(accepted=True, changed=bool(applied["changed"]), detail=applied)
    else:
        state.save()
    report["state_hash"] = PersistentState.load(destination).content_hash()
    return report


def planned_cells(probes: list[dict[str, Any]], *, seed: int = 20261014) -> list[dict[str, Any]]:
    cells = []
    for state_id in ("S0", "S_history_consumed_v2"):
        for batch_id in ("os-B1", "os-B2"):
            for probe in probes:
                for repeat_id in (0, 1, 2):
                    order = list(ARMS)
                    rng = random.Random(f"{seed}:{state_id}:{batch_id}:{probe['task_id']}:{repeat_id}")
                    rng.shuffle(order)
                    for position, arm in enumerate(order):
                        cells.append(
                            {
                                "env": "os",
                                "state_id": state_id,
                                "batch_id": batch_id,
                                "branch": arm,
                                "task_id": probe["task_id"],
                                "sample_index": probe["sample_index"],
                                "probe_role": probe["probe_role"],
                                "repeat_id": repeat_id,
                                "execution_order": position,
                                "development_task": True,
                            }
                        )
    return cells


def _import_ledger(ledger: PersistentLedger, path: Path) -> None:
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT attempt_id, logical_execution_id, job_kind, planned_episodes, COALESCE(tokens, 0), state FROM attempts"
        ).fetchall()
    finally:
        db.close()
    ledger.import_historical([(str(a), str(b), str(c), int(d), int(e), str(state)) for a, b, c, d, e, state in rows])


def run_calibration(
    *,
    smoke_root: Path,
    output: Path,
    port: int,
    served_name: str,
    prior_ledger: Path,
    cap: int = 1954,
) -> dict[str, Any]:
    probes = probe_tasks(smoke_root)
    if len(probes) != 8:
        raise RuntimeError(f"expected 8 OS development probes, found {len(probes)}")
    history_dir = output / "states" / "os" / "S_history_consumed_v2"
    history = build_consumed_history(smoke_root, history_dir)
    visible = format_episodic_injection(retrieve_episodic(PersistentState.load(history_dir), probes[0]["instruction"]))
    if not visible:
        raise RuntimeError("rebuilt history is still invisible to the first probe")
    cells = planned_cells(probes)
    (output / "D_CALIBRATION_PLANNED_CELLS.jsonl").write_text(
        "".join(json.dumps(cell, ensure_ascii=False) + "\n" for cell in cells),
        encoding="utf-8",
    )
    batches = acquisition_batches(smoke_root)
    client = VLLMClient(port=port, model=served_name)
    arm_reports = []
    states: dict[tuple[str, str, str], Path] = {}
    bases = {
        "S0": PersistentState.load(smoke_root / "states" / "os" / "S0"),
        "S_history_consumed_v2": PersistentState.load(history_dir),
    }
    for state_id, base in bases.items():
        for batch_id, experiences in batches.items():
            for arm in ARMS:
                path = output / "states" / "os" / f"{state_id}__{batch_id}__{arm}"
                print(f"prepare {state_id} {batch_id} {arm}", flush=True)
                report = prepare_arm_state(base, arm, experiences, path, client)
                report.update(state_id=state_id, batch_id=batch_id)
                arm_reports.append(report)
                states[(state_id, batch_id, arm)] = path
    write_json(output / "D_OPERATOR_PREP.json", arm_reports)
    ledger = PersistentLedger(output / "D_COST_LEDGER.sqlite", cap=cap)
    _import_ledger(ledger, prior_ledger)
    invalid_arms = {
        (item["state_id"], item["batch_id"], item["arm"])
        for item in arm_reports
        if item.get("operator_status") == "OPERATOR_INVALID"
    }
    results = []
    ordered = sorted(cells, key=lambda item: (item["state_id"], item["batch_id"], item["task_id"], item["repeat_id"], item["execution_order"]))
    for index, cell in enumerate(ordered, start=1):
        if (cell["state_id"], cell["batch_id"], cell["branch"]) in invalid_arms:
            results.append({**cell, "operator_status": "OPERATOR_INVALID", "outcome": None})
            continue
        state_path = states[(cell["state_id"], cell["batch_id"], cell["branch"])]
        state_hash = PersistentState.load(state_path).content_hash()
        identity = sha256_json(
            {
                "env": "os",
                "state": state_hash,
                "batch": cell["batch_id"],
                "branch": cell["branch"],
                "task": cell["task_id"],
                "sample": cell["sample_index"],
                "role": cell["probe_role"],
                "repeat": cell["repeat_id"],
                "revision": REVISION,
                "model": served_name,
                "seed": 0,
            }
        )[:24]
        job = {
            **cell,
            "job_kind": "calibration",
            "state_path": str(state_path),
            "state_content_hash": state_hash,
            "logical_execution_id": identity,
            "attempt_id": f"{identity}-{output.name}-a0",
            "result_path": str(output / "executions" / identity / f"{identity}-a0" / "result.json"),
            "port": port,
            "served_name": served_name,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1024,
            "seed": 0,
            "execution_revision": REVISION,
            "instruction_hint": next(item["instruction"] for item in probes if item["task_id"] == cell["task_id"]),
        }
        artifact = execute_official_job(job, ledger=ledger, run_root=output)
        print(
            f"calibration {index}/{len(ordered)} {cell['task_id']} {cell['branch']} outcome={artifact.get('outcome')}",
            flush=True,
        )
        results.append(
            {
                **cell,
                "outcome": artifact.get("outcome"),
                "infra_status": artifact.get("infra_status"),
                "tokens": artifact.get("tokens"),
                "initial_request_hash": artifact.get("initial_request_hash"),
                "operator_status": "ok",
            }
        )
    (output / "D_CALIBRATION_TASK_RESULTS.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    write_json(
        output / "D_CALIBRATION_SUMMARY.json",
        {"history": history, "spent": ledger.spent(), "n_results": len(results), "injection_nonempty": bool(visible)},
    )
    summary = summarize_results(output)
    summary.update(spent=ledger.spent(), n_results=len(results))
    write_json(output / "D_CALIBRATION_SUMMARY.json", summary)
    return summary


def summarize_results(output: Path) -> dict[str, Any]:
    """Aggregate a finished or partial calibration directory. Missing arms stay missing."""
    rows = []
    for path in sorted(output.glob("executions/*/*/result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        job = data.get("job") or {}
        if job.get("job_kind") != "calibration" or data.get("outcome") is None:
            continue
        rows.append(
            {
                "state_id": job.get("state_id"),
                "batch_id": job.get("batch_id"),
                "branch": job.get("branch"),
                "task_id": job.get("task_id"),
                "probe_role": job.get("probe_role"),
                "repeat_id": int(job.get("repeat_id") or 0),
                "outcome": float(data["outcome"]),
            }
        )
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        key = (row["state_id"], row["batch_id"], row["branch"], row["task_id"], row["probe_role"])
        grouped.setdefault(key, []).append(row["outcome"])
    means = {key: sum(values) / len(values) for key, values in grouped.items()}
    effects = []
    states = sorted({key[0] for key in means})
    batches = sorted({key[1] for key in means})
    arms = sorted({key[2] for key in means if key[2] != "BASE"})
    for state_id in states:
        for batch_id in batches:
            for arm in arms:
                future_base, future_arm, ret_base, ret_arm = [], [], [], []
                for key, value in means.items():
                    if key[0] != state_id or key[1] != batch_id:
                        continue
                    role = key[4]
                    if key[2] == "BASE" and role == "future":
                        future_base.append(value)
                    if key[2] == arm and role == "future":
                        future_arm.append(value)
                    if key[2] == "BASE" and role == "retention":
                        ret_base.append(value)
                    if key[2] == arm and role == "retention":
                        ret_arm.append(value)
                if not future_arm or not future_base:
                    continue
                future_gain = sum(future_arm) / len(future_arm) - sum(future_base) / len(future_base)
                retention_gain = (
                    sum(ret_arm) / len(ret_arm) - sum(ret_base) / len(ret_base) if ret_arm and ret_base else None
                )
                penalty = 0.0 if retention_gain is None else 0.5 * min(0.0, retention_gain)
                effects.append(
                    {
                        "state_id": state_id,
                        "batch_id": batch_id,
                        "arm": arm,
                        "future_gain": future_gain,
                        "retention_gain": retention_gain,
                        "utility": future_gain + penalty,
                        "n_future_tasks": len(future_arm),
                        "n_retention_tasks": len(ret_arm),
                    }
                )
    return {"n_scored_rows": len(rows), "effects": effects}
