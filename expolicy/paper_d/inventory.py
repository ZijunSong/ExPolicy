"""Read-only inventory. Completion comes from a verified DONE file, not STATUS_LIVE."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from expolicy.paper_d.constants import LEGACY_VALUE_KEYS
from expolicy.r9_lab.io_util import read_json, sha256_file

DONE_NAME = "DONE.json"
LIVE_NAME = "STATUS_LIVE.json"


def collect_artifact_inventory(roots: Sequence[Path]) -> dict[str, Any]:
    """Keep complete, hash-checked artifacts and copy their legacy values unchanged."""
    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    ignored_live: list[str] = []
    missing_roots: list[str] = []
    for root in roots:
        path = Path(root)
        if not path.exists():
            missing_roots.append(str(path))
            continue
        ignored_live.extend(str(item) for item in sorted(path.rglob(LIVE_NAME)))
        for done in sorted(path.rglob(DONE_NAME)):
            item = _inspect(done)
            if item["verified"]:
                verified.append(item)
            else:
                rejected.append(item)
    return {
        "n_verified": len(verified),
        "n_rejected": len(rejected),
        "n_status_live_ignored": len(ignored_live),
        "n_reusable_task_level": sum(1 for item in verified if item["reusable_for_protocol_matrix"]),
        "missing_roots": missing_roots,
        "verified": verified,
        "rejected": rejected,
        "ignored_status_live": ignored_live,
        "inferred_from_status_live": False,
    }


def _inspect(done: Path) -> dict[str, Any]:
    try:
        payload = read_json(done)
    except (OSError, ValueError) as exc:
        return _rejected(done, f"unreadable DONE file: {exc}", {})
    if not isinstance(payload, dict):
        return _rejected(done, "DONE payload is not an object", {})
    legacy = _legacy_values(payload)
    complete = payload.get("done") is True or payload.get("complete") is True
    hash_ok, hash_reason = _hashes_match(done, payload)
    batch_id = payload.get("batch_id") or done.parent.parent.name
    action = payload.get("action") or done.parent.name
    task_level = _task_level(payload, done.parent)
    reasons: list[str] = []
    if not complete:
        reasons.append("DONE file is not marked complete")
    if not hash_ok:
        reasons.append(hash_reason)
    if not batch_id or not action:
        reasons.append("batch or action is missing")
    verified = not reasons
    return {
        "verified": verified,
        "path": str(done),
        "batch_id": batch_id,
        "action": action,
        "legacy_values": legacy,
        "task_level": task_level,
        "reusable_for_protocol_matrix": verified and task_level,
        "source_hash": payload.get("source_hash"),
        "reasons": reasons,
    }


def _legacy_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the stored numbers. Do not recompute them with the protocol utility."""
    legacy: dict[str, Any] = {}
    for key in LEGACY_VALUE_KEYS:
        if key in payload:
            legacy[key] = deepcopy(payload[key])
    return legacy


def _hashes_match(done: Path, payload: Mapping[str, Any]) -> tuple[bool, str]:
    file_hashes = payload.get("file_hashes")
    if isinstance(file_hashes, dict) and file_hashes:
        for name, digest in file_hashes.items():
            target = done.parent / str(name)
            if not target.is_file() or sha256_file(target) != str(digest):
                return False, f"hash mismatch for {name}"
        return True, ""
    source_file = payload.get("source_file")
    source_hash = payload.get("source_hash")
    if source_file and source_hash:
        target = done.parent / str(source_file)
        if target.is_file() and sha256_file(target) == str(source_hash):
            return True, ""
        return False, "source hash does not match source_file"
    return False, "missing source hash"


def _task_level(payload: Mapping[str, Any], directory: Path) -> bool:
    for key in ("task_rows", "per_task", "items"):
        if isinstance(payload.get(key), list) and payload[key]:
            return True
    return any((directory / name).is_file() for name in ("task_rows.jsonl", "per_task.jsonl"))


def _rejected(done: Path, reason: str, legacy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "verified": False,
        "path": str(done),
        "batch_id": None,
        "action": None,
        "legacy_values": dict(legacy),
        "task_level": False,
        "reusable_for_protocol_matrix": False,
        "source_hash": None,
        "reasons": [reason],
    }
