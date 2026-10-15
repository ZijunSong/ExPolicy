"""Acquire confirmation sources and run only a length-gated evaluation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from expolicy.paper_d.confirm_controls import choose_wrong_bundle, empty_has_source_text, load_tokenizer_counter, memory_text
from expolicy.paper_d.confirm_plan import execution_identity
from expolicy.paper_d.confirm_protocol import PROTOCOL
from expolicy.paper_d.constants import ProtocolError
from expolicy.paper_d.content_controls import clone_state, write_empty, write_no_update, write_real
from expolicy.paper_d.official_worker import execute_official_job
from expolicy.paper_d.persistent_ledger import PersistentLedger
from expolicy.r9_lab.assimilators.episodic import apply_episodic
from expolicy.r9_lab.io_util import sha256_json, write_json
from expolicy.r9_lab.state.persistent_state import PersistentState


PRIOR_LEDGER = Path("outputs/paper_d_content_os/20261014-content-os/D_CONTENT_COST_LEDGER.sqlite")
S0_SOURCE = Path("outputs/paper_d_real_v11/20261014-smoke-r2/states/os/S0")


def import_ledger(ledger: PersistentLedger, path: Path) -> None:
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT attempt_id, logical_execution_id, job_kind, planned_episodes, COALESCE(tokens, 0), state FROM attempts"
        ).fetchall()
    finally:
        db.close()
    ledger.import_historical([(str(a), str(b), str(c), int(d), int(e), str(state)) for a, b, c, d, e, state in rows])


def _user_text(artifact: dict[str, Any]) -> str:
    trace = artifact.get("request_trace") or []
    if not trace:
        return ""
    body = trace[0]["request_wire"]
    if isinstance(body, str):
        body = json.loads(body)
    return "\n".join(str(item.get("content") or "") for item in body.get("messages") or [] if item.get("role") == "user")


def run_job(cell: dict[str, Any], *, output: Path, ledger: PersistentLedger, state_path: Path, state_hash: str, control_hash: str, attempt_index: int, job_kind: str) -> dict[str, Any]:
    identity = execution_identity(cell, state_hash=state_hash, control_hash=control_hash)
    attempt = f"{identity}-{output.name}-a{attempt_index}"
    job = {
        **cell,
        "env": "os",
        "job_kind": job_kind,
        "state_path": str(state_path),
        "state_content_hash": state_hash,
        "logical_execution_id": identity,
        "attempt_id": attempt,
        "result_path": str(output / "executions" / identity / f"{identity}-a{attempt_index}" / "result.json"),
        "port": PROTOCOL["port"],
        "served_name": PROTOCOL["served_name"],
        "temperature": PROTOCOL["temperature"],
        "top_p": PROTOCOL["top_p"],
        "max_tokens": PROTOCOL["max_tokens"],
        "seed": PROTOCOL["inference_seed"],
        "execution_revision": PROTOCOL["stage_id"],
        "instruction_hint": cell["instruction"],
    }
    try:
        artifact = execute_official_job(job, ledger=ledger, run_root=output)
    except ProtocolError as exc:
        if "cannot be restarted" not in str(exc):
            raise
        return {"restart_blocked": True, "attempt_id": attempt}
    return {
        "task_id": cell["task_id"],
        "batch_id": cell.get("batch_id"),
        "condition": cell.get("condition"),
        "probe_role": cell.get("probe_role"),
        "repeat_id": cell.get("repeat_id"),
        "logical_execution_id": identity,
        "attempt_id": attempt,
        "attempt_index": attempt_index,
        "cache_hit": bool(artifact.get("cache_hit")),
        "outcome": artifact.get("outcome"),
        "infra_status": artifact.get("infra_status"),
        "structural_status": artifact.get("structural_status"),
        "tokens": artifact.get("tokens"),
        "tool_calls": artifact.get("tool_calls"),
        "wall_seconds": artifact.get("wall_seconds"),
        "request_wire_hash": artifact.get("initial_request_hash"),
        "user_contains_official_instruction": str(cell["instruction"])[:80] in _user_text(artifact),
        "trajectory": artifact.get("trajectory") or [],
        "success": bool(artifact.get("success")),
        "state_hash": state_hash,
        "control_transform_hash": control_hash,
    }


def _existing(cell: dict[str, Any], output: Path, state_hash: str, control_hash: str) -> dict[str, Any] | None:
    identity = execution_identity(cell, state_hash=state_hash, control_hash=control_hash)
    for attempt in range(3):
        path = output / "executions" / identity / f"{identity}-a{attempt}" / "result.json"
        if not path.is_file():
            continue
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("infra_status") == "ok" and artifact.get("outcome") is not None:
            return {
                "task_id": cell["task_id"],
                "batch_id": cell.get("batch_id"),
                "condition": cell.get("condition"),
                "probe_role": cell.get("probe_role"),
                "repeat_id": cell.get("repeat_id"),
                "logical_execution_id": identity,
                "attempt_id": f"{identity}-{output.name}-a{attempt}",
                "attempt_index": attempt,
                "cache_hit": True,
                "outcome": artifact.get("outcome"),
                "infra_status": "ok",
                "structural_status": artifact.get("structural_status"),
                "tokens": artifact.get("tokens"),
                "tool_calls": artifact.get("tool_calls"),
                "wall_seconds": artifact.get("wall_seconds"),
                "request_wire_hash": artifact.get("initial_request_hash"),
                "user_contains_official_instruction": True,
                "trajectory": artifact.get("trajectory") or [],
                "success": bool(artifact.get("success")),
                "state_hash": state_hash,
                "control_transform_hash": control_hash,
            }
    return None


def run_with_retry(cell: dict[str, Any], **kwargs: Any) -> dict[str, Any] | None:
    found = _existing(cell, kwargs["output"], kwargs["state_hash"], kwargs["control_hash"])
    if found is not None:
        return found
    retries = kwargs.pop("retries")
    for attempt in range(3):
        if attempt and retries[0] >= int(PROTOCOL["retry_pool"]):
            break
        row = run_job(cell, attempt_index=attempt, **kwargs)
        if row.get("restart_blocked"):
            continue
        if attempt and not row.get("cache_hit"):
            retries[0] += 1
        if row.get("infra_status") == "ok":
            return row
    return row if "row" in locals() and not row.get("restart_blocked") else None


def experience_from(task: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    success = bool(artifact.get("success")) and artifact.get("infra_status") == "ok"
    return {
        "task_id": task["task_id"],
        "env": "os",
        "instruction": task["instruction"],
        "trajectory": list(artifact.get("trajectory") or []),
        "success": success,
        "verifier_outcome": "CORRECT" if success else "INCORRECT",
        "infra_status": artifact.get("infra_status"),
        "skill_list": task.get("skill_list") or [],
        "bucket": "acquisition_pool",
    }


def select_written(experiences: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [item for item in experiences if item["success"] and item["infra_status"] == "ok"]
    limit = int(PROTOCOL["max_successes_written"])
    return {
        "written": successes[:limit],
        "excluded_by_cap": [item["task_id"] for item in successes[limit:]],
        "n_success": len(successes),
        "n_written": min(len(successes), limit),
        "status": "EMPTY_UPDATE_BATCH" if not successes else "content_batch",
    }


def audit_controls(pools: dict[str, Any], source_results: dict[str, Any], donor_successes: list[dict[str, Any]], counter) -> dict[str, Any]:
    instructions = [task["instruction"] for task in pools["evaluation"]]
    used: set[str] = set()
    batches = []
    blocked = None
    for batch in source_results["batches"]:
        written = [item for item in batch["written"] if item["task_id"] not in used]
        available = [item for item in donor_successes if item["task_id"] not in used]
        choice = choose_wrong_bundle(written, available, instructions, counter)
        if choice["status"] == "matched":
            for donor in choice["donors"]:
                used.add(donor["task_id"])
        elif choice["status"] != "EMPTY_UPDATE_BATCH":
            blocked = f"{batch['batch_id']}: {choice['status']}"
        leak = []
        if written:
            for instruction in instructions:
                leak.extend(empty_has_source_text(written, instruction))
                if leak:
                    break
        if leak:
            blocked = f"{batch['batch_id']}: EMPTY still contains source text"
        batches.append({"batch_id": batch["batch_id"], "n_written": len(written), "choice": choice, "empty_leaks": leak})
    return {"blocked": blocked, "batches": batches, "controls_valid": blocked is None}
