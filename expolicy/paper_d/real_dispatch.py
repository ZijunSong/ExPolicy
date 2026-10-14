"""Build per-environment jobs and run them one at a time."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from expolicy.paper_d.constants import REPO_ROOT
from expolicy.paper_d.official_worker import EXECUTION_REVISION, execute_official_job
from expolicy.paper_d.persistent_ledger import PersistentLedger
from expolicy.r9_lab.benchmark.split import assert_production_split, load_split
from expolicy.r9_lab.io_util import sha256_json
from expolicy.r9_lab.state.persistent_state import PersistentState


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
