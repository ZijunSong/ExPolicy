"""Checkpoint helpers for runs and agent versions."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from expolicy.core.errors import CheckpointError
from expolicy.core.state import PersistentAgentState


def git_commit_hash(repo: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def snapshot_run(output_dir: Path, config: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.time(),
        "config": config,
        "git_commit": git_commit_hash(Path.cwd()),
        "seed": config.get("seed"),
        "extra": extra or {},
        "cwd": os.getcwd(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (output_dir / "config.yaml").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    return payload


def load_agent(path: str | Path) -> PersistentAgentState:
    return PersistentAgentState.load(path)


def bump_agent(state: PersistentAgentState, dest_root: Path, reason: str, experience_ids: list[str]) -> PersistentAgentState:
    child_ver = state.next_version()
    dest = dest_root / child_ver
    child = state.snapshot_into(dest, copy_stores=True)
    child.version = child_ver
    child.lineage = list(state.lineage)
    from expolicy.core.state import LineageRecord

    child.lineage.append(
        LineageRecord(parent=state.version, child=child_ver, reason=reason, experience_ids=experience_ids)
    )
    child.write_manifest()
    return child
