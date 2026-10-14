"""Offline smoke reanalysis. This module does not start an environment."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from expolicy.r9_lab.assimilators.episodic import format_episodic_injection, retrieve_episodic
from expolicy.r9_lab.state.persistent_state import PersistentState


TRIAD_KEY = ("env", "state_id", "batch_id", "task_id", "probe_role", "repeat_id")


def load_smoke_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("executions/*/*/result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        job = data.get("job") or {}
        if job.get("job_kind") != "smoke":
            continue
        if data.get("infra_status") != "ok" or data.get("outcome") is None:
            continue
        wire = ""
        trace = data.get("request_trace") or []
        if trace:
            wire = str(trace[0].get("request_wire_hash") or "")
        rows.append(
            {
                "env": job.get("env"),
                "state_id": job.get("state_id"),
                "state_content_hash": job.get("state_content_hash"),
                "batch_id": job.get("batch_id"),
                "task_id": job.get("task_id"),
                "sample_index": job.get("sample_index"),
                "probe_role": job.get("probe_role"),
                "repeat_id": job.get("repeat_id"),
                "branch": job.get("branch"),
                "outcome": data.get("outcome"),
                "initial_request_wire_hash": wire or data.get("initial_request_hash"),
                "model_service_id": job.get("served_name"),
                "inference_config_hash": job.get("execution_revision"),
                "state_unchanged": data.get("state_unchanged"),
                "structural_status": data.get("structural_status"),
                "infra_status": data.get("infra_status"),
                "attempt_id": job.get("attempt_id"),
                "origin_run_id": root.name,
                "result_path": str(path),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def triad_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[tuple(str(row[key]) for key in TRIAD_KEY)][str(row["branch"])] = row
    out = []
    for key, branches in sorted(grouped.items()):
        if set(branches) != {"BASE", "DISCARD", "REJECTED_EDIT"}:
            continue
        outcomes = {name: branches[name]["outcome"] for name in branches}
        hashes = {branches[name]["initial_request_wire_hash"] for name in branches}
        out.append(
            {
                "env": key[0],
                "state_id": key[1],
                "batch_id": key[2],
                "task_id": key[3],
                "probe_role": key[4],
                "repeat_id": key[5],
                "base": outcomes["BASE"],
                "discard": outcomes["DISCARD"],
                "rejected": outcomes["REJECTED_EDIT"],
                "discard_minus_base": float(outcomes["DISCARD"]) - float(outcomes["BASE"]),
                "rejected_minus_base": float(outcomes["REJECTED_EDIT"]) - float(outcomes["BASE"]),
                "branches_agree": len(set(outcomes.values())) == 1,
                "same_initial_request": len(hashes) == 1,
            }
        )
    return out


def bias_rows(triads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in triads:
        grouped[(row["env"], row["state_id"], row["probe_role"])].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        n = len(items)
        out.append(
            {
                "env": key[0],
                "state_id": key[1],
                "probe_role": key[2],
                "n_triads": n,
                "mean_discard_minus_base": sum(float(row["discard_minus_base"]) for row in items) / n,
                "mean_rejected_minus_base": sum(float(row["rejected_minus_base"]) for row in items) / n,
                "n_disagree": sum(1 for row in items if not row["branches_agree"]),
                "disagree_rate": sum(1 for row in items if not row["branches_agree"]) / n,
                "same_initial_request": sum(1 for row in items if row["same_initial_request"]),
            }
        )
    return out


def audit_history_state(path: Path, probes: list[str]) -> dict[str, Any]:
    state = PersistentState.load(path)
    ids = [record.task_id for record in state.episodic]
    committed = set(state.committed_task_ids)
    successes = [record for record in state.episodic if record.success]
    hits = []
    for instruction in probes:
        retrieved = retrieve_episodic(state, instruction, committed_only=True)
        hits.append({"instruction": instruction[:80], "n_hits": len(retrieved)})
    return {
        "path": str(path),
        "n_episodic": len(state.episodic),
        "n_success": len(successes),
        "n_committed": len(committed),
        "intersection": len(set(ids) & committed),
        "status": "STATE_NONEMPTY_BUT_UNCONSUMED" if state.episodic and not (set(ids) & committed) else "checked",
        "retrieval_hits": hits,
        "injection_nonempty": any(format_episodic_injection(retrieve_episodic(state, text, committed_only=True)) for text in probes),
    }
