"""Watchdog helpers. Kill ONLY recorded PIDs of this round. Never pkill python."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from expolicy.r9_lab.constants import OUTPUT_ROOT, STALL_SEC
from expolicy.r9_lab.io_util import atomic_write_json


def append_recorded_pid(jobs_dir: Path, pid: int, cmd: str) -> None:
    """Append one PID to this-round PIDS.txt. Watchdog may only kill these."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    path = jobs_dir / "PIDS.txt"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{int(pid)} {cmd}\n")


def write_status_live(
    path: Path,
    *,
    pid: int | None = None,
    gpu: str | None = None,
    stage: str = "",
    batch_id: str = "",
    action: str = "",
    completed_tasks: int = 0,
    tokens: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "pid": int(pid if pid is not None else os.getpid()),
        "gpu": gpu,
        "stage": stage,
        "batch_id": batch_id,
        "action": action,
        "last_progress_time": time.time(),
        "completed_tasks": completed_tasks,
        "tokens": tokens,
        **(extra or {}),
    }
    atomic_write_json(path, payload)


def load_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_status_files(root: Path | None = None) -> list[Path]:
    base = root or OUTPUT_ROOT
    return list(base.rglob("STATUS_LIVE.json"))


def recorded_pids(files: list[Path]) -> list[int]:
    pids: list[int] = []
    for path in files:
        row = load_status(path) or {}
        pid = row.get("pid")
        if isinstance(pid, int) and pid > 1:
            pids.append(pid)
        elif isinstance(pid, str) and pid.isdigit():
            pids.append(int(pid))
    return sorted(set(pids))


def is_stalled(row: dict[str, Any], *, now: float | None = None, stall_sec: int = STALL_SEC) -> bool:
    last = float(row.get("last_progress_time") or 0.0)
    if last <= 0:
        return True
    return (now if now is not None else time.time()) - last >= stall_sec


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_recorded_pid(pid: int) -> dict[str, Any]:
    """SIGTERM one recorded PID. Never pkill / killall python."""
    if pid <= 1:
        return {"pid": pid, "killed": False, "reason": "refused_low_pid"}
    if not pid_alive(pid):
        return {"pid": pid, "killed": False, "reason": "not_alive"}
    os.kill(pid, signal.SIGTERM)
    return {"pid": pid, "killed": True, "signal": "SIGTERM"}
