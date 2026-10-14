"""Build per-environment jobs and run them one at a time."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from expolicy.paper_d.constants import REPO_ROOT, ProtocolError
from expolicy.paper_d.official_worker import EXECUTION_REVISION, execute_official_job
from expolicy.paper_d.persistent_ledger import PersistentLedger
from expolicy.r9_lab.benchmark.split import assert_production_split, load_split
from expolicy.r9_lab.io_util import sha256_json
from expolicy.r9_lab.state.persistent_state import EpisodicRecord, PersistentState


def vertical_jobs(config: Mapping[str, Any], run_root: Path, *, state_hash: str) -> list[dict[str, Any]]:
    """One frozen shadow probe per environment, then the two no-op branches."""
    split_path = Path(str(config["split_path"]))
    if not split_path.is_absolute():
        split_path = REPO_ROOT / split_path
    payload = load_split(split_path)
    assert_production_split(payload, allow_synthetic=False)
    model = config["model"]
    jobs: list[dict[str, Any]] = []
    for env in ("db", "os", "kg"):
        task = _first_shadow(payload, env)
        for branch in ("BASE", "DISCARD", "REJECTED_EDIT"):
            logical = sha256_json(
                {
                    "env": env,
                    "task_id": task["task_id"],
                    "sample_index": task["sample_index"],
                    "branch": branch,
                    "state": "S0",
                    "revision": EXECUTION_REVISION,
                }
            )[:24]
            attempt = f"{logical}-a0"
            jobs.append(
                {
                    "run_id": run_root.name,
                    "logical_execution_id": logical,
                    "attempt_id": attempt,
                    "job_kind": "vertical",
                    "configuration_id": "d-v1.1-real",
                    "protocol_hash": config["config_hash"],
                    "execution_revision": EXECUTION_REVISION,
                    "env": env,
                    "state_id": "S0",
                    "state_path": str(run_root / "states" / "S0"),
                    "state_content_hash": state_hash,
                    "batch_id": "vertical-s0",
                    "branch": branch,
                    "task_id": task["task_id"],
                    "sample_index": task["sample_index"],
                    "instruction_hint": task["instruction"],
                    "probe_role": "future",
                    "probe_source_group": "shadow_reward_probe",
                    "repeat_id": "0",
                    "port": int(config["servers"][0]["port"]),
                    "served_name": model["served_name"],
                    "temperature": float(model["temperature"]),
                    "top_p": float(model["top_p"]),
                    "max_tokens": int(model["max_tokens"]),
                    "seed": int(model["inference_seed"]),
                    "result_path": str(run_root / "executions" / logical / attempt / "result.json"),
                }
            )
    return jobs


def prepare_s0(run_root: Path) -> str:
    path = run_root / "states" / "S0"
    if (path / "STATE.json").exists():
        return PersistentState.load(path).content_hash()
    return PersistentState.create(path).content_hash()


def run_jobs(jobs: Sequence[Mapping[str, Any]], *, ledger: PersistentLedger, run_root: Path) -> list[dict[str, Any]]:
    artifacts = []
    for job in jobs:
        artifacts.append(execute_official_job(job, ledger=ledger, run_root=run_root))
    return artifacts


def _first_shadow(payload: Mapping[str, Any], env: str) -> dict[str, Any]:
    rows = list(payload["envs"][env]["shadow_reward_probe"])
    rows.sort(key=lambda row: int(row["sample_index"]))
    chosen = rows[0]
    if not str(chosen["task_id"]).startswith(f"{env}:") or str(chosen["sample_index"]) not in str(chosen["task_id"]):
        raise ValueError(f"split task for {env} is not an official sample mapping")
    return chosen


def smoke_cells(split_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """576 planned smoke cells. Each environment uses only its own tasks."""
    cells = []
    for env in ("db", "os", "kg"):
        future = _sorted_bucket(split_payload, env, "shadow_reward_probe")[:4]
        retention = _sorted_bucket(split_payload, env, "retention_probe")[:4]
        probes = [("future", row) for row in future] + [("retention", row) for row in retention]
        for state_id in ("S0", "S_history"):
            for batch_index in (1, 2):
                for repeat_id in ("0", "1"):
                    for role, task in probes:
                        for branch in ("BASE", "DISCARD", "REJECTED_EDIT"):
                            if not str(task["task_id"]).startswith(f"{env}:") or str(task["sample_index"]) != str(task["task_id"]).split(":", 1)[1]:
                                raise ValueError(f"official sample mapping failed for {task['task_id']}")
                            cells.append(
                                {
                                    "env": env,
                                    "state_id": state_id,
                                    "batch_id": f"{env}-B{batch_index}",
                                    "branch": branch,
                                    "task_id": task["task_id"],
                                    "sample_index": str(task["sample_index"]),
                                    "probe_role": role,
                                    "repeat_id": repeat_id,
                                    "instruction": task["instruction"],
                                }
                            )
    if len(cells) != 576 or any(sum(1 for cell in cells if cell["env"] == env) != 192 for env in ("db", "os", "kg")):
        raise ValueError("smoke plan is not 192 cells per environment")
    return cells


def prepare_noop_branches(state: PersistentState, experiences: list[dict[str, Any]], dest_root: Path) -> dict[str, Any]:
    """Apply discard and a real reject path on independent clones. Renaming a branch is not enough."""
    from expolicy.r9_lab.assimilators.discard import apply_discard

    dest_root.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for branch, mutator in (
        ("BASE", None),
        ("DISCARD", lambda cloned: apply_discard(cloned, experiences)),
        ("REJECTED_EDIT", lambda cloned: _reject_edit(cloned, experiences)),
    ):
        dest = dest_root / branch
        if (dest / "STATE.json").exists():
            loaded = PersistentState.load(dest)
        else:
            _copy_frozen(state.root_path(), dest)
            loaded = PersistentState.load(dest)
            if mutator is not None:
                mutator(loaded)
                loaded.save()
        provenance[branch] = {
            "path": str(dest),
            "content_hash": loaded.content_hash(),
            "accepted": False if branch == "REJECTED_EDIT" else (True if branch == "DISCARD" else None),
            "changed": loaded.content_hash() != state.content_hash(),
        }
    if provenance["REJECTED_EDIT"]["accepted"] is not False:
        raise ProtocolError("rejected edit was committed")
    return provenance


def _reject_edit(state: PersistentState, experiences: list[dict[str, Any]]) -> dict[str, Any]:
    before = state.content_hash()
    proposal = {"backend": "local_reject_verifier", "task_ids": [item.get("task_id") for item in experiences]}
    reason = "verifier rejected the proposal before commit"
    after = state.content_hash()
    if before != after:
        raise ProtocolError("reject path changed state")
    return {"accepted": False, "changed": False, "reason": reason, "proposal": proposal}


def _copy_frozen(src: Path, dest: Path) -> None:
    import shutil

    if dest.exists():
        raise ProtocolError(f"refusing to overwrite an existing state: {dest}")
    shutil.copytree(src, dest)


def _sorted_bucket(payload: Mapping[str, Any], env: str, bucket: str) -> list[dict[str, Any]]:
    rows = list(payload["envs"][env][bucket])
    rows.sort(key=lambda row: int(row["sample_index"]))
    return rows


def run_full_smoke(config: Mapping[str, Any], run_root: Path, *, prior_ledgers: list[Path] | None = None) -> dict[str, Any]:
    """Collect the smoke batches, prepare real no-op branches, then run all 576 probe cells."""
    run_root.mkdir(parents=True, exist_ok=True)
    split_path = Path(str(config["split_path"]))
    if not split_path.is_absolute():
        split_path = REPO_ROOT / split_path
    payload = load_split(split_path)
    assert_production_split(payload, allow_synthetic=False)
    cells = smoke_cells(payload)
    write_jsonl(run_root / "D_SMOKE_PLANNED_CELLS.jsonl", [{k: v for k, v in cell.items() if k != "instruction"} for cell in cells])
    ledger = PersistentLedger(run_root / "D_COST_LEDGER.sqlite", cap=int(config["pilot_episode_cap_all_inclusive"]))
    for prior in prior_ledgers or []:
        if prior.is_file():
            _import_prior(ledger, prior)
    s0_hash = prepare_s0(run_root)
    _collect_histories_and_batches(config, payload, run_root, ledger, s0_hash)
    provenance_rows = _prepare_all_branches(payload, run_root, s0_hash)
    write_jsonl(run_root / "D_NOOP_OPERATOR_PROVENANCE.jsonl", provenance_rows)
    jobs = _probe_jobs(config, run_root, cells, provenance_rows)
    artifacts = run_jobs(jobs, ledger=ledger, run_root=run_root)
    spent = ledger.spent()
    ledger.close()
    return {
        "episodes": len(artifacts),
        "planned": len(cells),
        "spent": spent,
        "artifact_paths": [str((row.get("job") or {}).get("result_path") or "") for row in artifacts],
        "structural_invalid": sum(1 for row in artifacts if row.get("structural_status") == "STRUCTURAL_INVALID"),
    }


def _collect_histories_and_batches(
    config: Mapping[str, Any],
    payload: Mapping[str, Any],
    run_root: Path,
    ledger: PersistentLedger,
    s0_hash: str,
) -> None:
    jobs = []
    for env in ("db", "os", "kg"):
        history = _sorted_bucket(payload, env, "calibration")[:16]
        acquired = _sorted_bucket(payload, env, "acquisition_pool")[:16]
        for kind, rows in (("history", history), ("acquisition", acquired)):
            for row in rows:
                logical = sha256_json({"kind": kind, "task": row["task_id"], "revision": EXECUTION_REVISION})[:24]
                jobs.append(
                    _job(
                        config,
                        run_root,
                        logical=logical,
                        kind=kind,
                        env=env,
                        state_id="S0",
                        state_hash=s0_hash,
                        batch_id=f"{env}-{kind}",
                        branch="BASE",
                        task=row,
                        role=kind,
                        repeat_id="0",
                    )
                )
    run_jobs(jobs, ledger=ledger, run_root=run_root)
    for env in ("db", "os", "kg"):
        history_state = PersistentState.create(run_root / "states" / env / "S_history_source") if not (run_root / "states" / env / "S_history_source" / "STATE.json").exists() else PersistentState.load(run_root / "states" / env / "S_history_source")
        if not history_state.episodic:
            for row in _sorted_bucket(payload, env, "calibration")[:16]:
                logical = sha256_json({"kind": "history", "task": row["task_id"], "revision": EXECUTION_REVISION})[:24]
                artifact = json.loads((run_root / "executions" / logical / f"{logical}-a0" / "result.json").read_text(encoding="utf-8"))
                history_state.episodic.append(
                    EpisodicRecord(
                        task_id=row["task_id"],
                        env=env,
                        instruction=row["instruction"],
                        trajectory=list(artifact.get("trajectory") or []),
                        success=bool(artifact.get("success")),
                        verifier_outcome="CORRECT" if artifact.get("success") else "INCORRECT",
                    )
                )
            history_state.save()


def _prepare_all_branches(payload: Mapping[str, Any], run_root: Path, s0_hash: str) -> list[dict[str, Any]]:
    rows = []
    for env in ("db", "os", "kg"):
        acquired = _sorted_bucket(payload, env, "acquisition_pool")[:16]
        experiences = [{"task_id": row["task_id"], "env": env, "instruction": row["instruction"]} for row in acquired]
        s0 = PersistentState.load(run_root / "states" / "S0")
        if s0.content_hash() != s0_hash:
            raise ProtocolError("S0 changed after it was frozen")
        history = PersistentState.load(run_root / "states" / env / "S_history_source")
        for state_id, state in (("S0", s0), ("S_history", history)):
            for batch_index, chunk in ((1, experiences[:8]), (2, experiences[8:16])):
                provenance = prepare_noop_branches(state, chunk, run_root / "states" / env / state_id / f"B{batch_index}")
                for branch, info in provenance.items():
                    rows.append({"env": env, "state_id": state_id, "batch_id": f"{env}-B{batch_index}", "branch": branch, **info})
    return rows


def _probe_jobs(
    config: Mapping[str, Any],
    run_root: Path,
    cells: Sequence[Mapping[str, Any]],
    provenance_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    index = {(row["env"], row["state_id"], row["batch_id"], row["branch"]): row for row in provenance_rows}
    jobs = []
    for cell in cells:
        info = index[(cell["env"], cell["state_id"], cell["batch_id"], cell["branch"])]
        logical = sha256_json(
            {
                "env": cell["env"],
                "state": info["content_hash"],
                "batch": cell["batch_id"],
                "branch": cell["branch"],
                "task": cell["task_id"],
                "sample": cell["sample_index"],
                "role": cell["probe_role"],
                "repeat": cell["repeat_id"],
                "revision": EXECUTION_REVISION,
            }
        )[:24]
        jobs.append(
            _job(
                config,
                run_root,
                logical=logical,
                kind="smoke",
                env=cell["env"],
                state_id=cell["state_id"],
                state_hash=info["content_hash"],
                state_path=info["path"],
                batch_id=cell["batch_id"],
                branch=cell["branch"],
                task=cell,
                role=cell["probe_role"],
                repeat_id=cell["repeat_id"],
            )
        )
    return jobs


def _job(
    config: Mapping[str, Any],
    run_root: Path,
    *,
    logical: str,
    kind: str,
    env: str,
    state_id: str,
    state_hash: str,
    batch_id: str,
    branch: str,
    task: Mapping[str, Any],
    role: str,
    repeat_id: str,
    state_path: str | None = None,
) -> dict[str, Any]:
    model = config["model"]
    attempt = f"{logical}-a0"
    return {
        "run_id": run_root.name,
        "logical_execution_id": logical,
        "attempt_id": attempt,
        "job_kind": kind,
        "configuration_id": "d-v1.1-real",
        "protocol_hash": config["config_hash"],
        "execution_revision": EXECUTION_REVISION,
        "env": env,
        "state_id": state_id,
        "state_path": state_path or str(run_root / "states" / "S0"),
        "state_content_hash": state_hash,
        "batch_id": batch_id,
        "branch": branch,
        "task_id": task["task_id"],
        "sample_index": str(task["sample_index"]),
        "instruction_hint": task["instruction"],
        "probe_role": role,
        "probe_source_group": role,
        "repeat_id": repeat_id,
        "port": int(config["servers"][0]["port"]),
        "served_name": model["served_name"],
        "temperature": float(model["temperature"]),
        "top_p": float(model["top_p"]),
        "max_tokens": int(model["max_tokens"]),
        "seed": int(model["inference_seed"]),
        "result_path": str(run_root / "executions" / logical / attempt / "result.json"),
    }


def _import_prior(ledger: PersistentLedger, path: Path) -> None:
    import sqlite3

    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT attempt_id, logical_execution_id, job_kind, planned_episodes, COALESCE(tokens, 0) FROM attempts"
        ).fetchall()
    finally:
        db.close()
    ledger.import_historical([(str(a), str(b), str(c), int(d), int(e)) for a, b, c, d, e in rows])


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
