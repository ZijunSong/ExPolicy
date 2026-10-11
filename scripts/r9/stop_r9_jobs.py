#!/usr/bin/env python3
"""Stop this-round ExPolicy R9 jobs only. Never pkill python. Never touch other users."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

JOBS = Path("/data/ppnm/expolicy/outputs/r9_lab/jobs/PIDS.txt")
MARKERS = (
    "scripts/r9/",
    "expolicy/scripts/r9",
    "07_run_assim_matrix",
    "05_operator_sanity",
    "supervise_phase2",
    "supervise_r9_runtime",
    "serve_vllm.py",
    "qwen3-14b-r9",
    "qwen3-32b-r9",
    "qwen3-30b-a3b-r9",
    "qwq-32b-r9",
    "/data/ppnm/models/Qwen3-14B",
    "/data/ppnm/models/Qwen3-32B",
    "/data/ppnm/models/Qwen3-30B-A3B-Instruct-2507",
    "/data/ppnm/models/QwQ-32B",
    "freebase_virtuoso",
    "virtuoso_db.zip",
    "start_lab_sparql",
)
FORBIDDEN = (
    "openpi-comet",
    "curobo",
    "/data/zyt/",
)


def cmdline(pid: int) -> str:
    p = Path(f"/proc/{pid}/cmdline")
    if not p.exists():
        return ""
    return p.read_text(errors="ignore").replace("\x00", " ")


def ppid_of(pid: int) -> int | None:
    path = Path(f"/proc/{pid}/status")
    if not path.exists():
        return None
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    return None


def recorded() -> set[int]:
    out: set[int] = set()
    if not JOBS.exists():
        return out
    for line in JOBS.read_text(encoding="utf-8").splitlines():
        for token in line.replace("=", " ").replace(",", " ").split():
            if token.isdigit():
                out.add(int(token))
    return out


def is_ours(pid: int, cmd: str) -> bool:
    if pid <= 1:
        return False
    if any(bad in cmd for bad in FORBIDDEN):
        return False
    return any(m in cmd for m in MARKERS)


def children_of(parents: set[int]) -> set[int]:
    found: set[int] = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        parent = ppid_of(pid)
        if parent in parents:
            found.add(pid)
    return found


def collect_targets() -> dict[int, str]:
    rec = recorded()
    targets: dict[int, str] = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        cmd = cmdline(pid)
        if not cmd:
            continue
        if pid in rec and is_ours(pid, cmd):
            targets[pid] = cmd
            continue
        if is_ours(pid, cmd) and any(m in cmd for m in MARKERS[:12]):
            # Live R9 workers/vLLM even if wrapper PID was not the api_server.
            targets[pid] = cmd
    # Also EngineCore / children of targeted vLLM parents.
    extra = children_of(set(targets))
    for pid in extra:
        cmd = cmdline(pid)
        if cmd and not any(bad in cmd for bad in FORBIDDEN):
            targets[pid] = cmd
    return targets


def main() -> int:
    targets = collect_targets()
    print("targets", len(targets))
    for pid, cmd in sorted(targets.items()):
        print(f"SIGTERM {pid} {cmd[:180]}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            print(f"  already dead {pid}")
    time.sleep(3)
    leftover = {pid: cmdline(pid) for pid in targets if Path(f"/proc/{pid}").exists()}
    for pid, cmd in leftover.items():
        if not cmd or any(bad in cmd for bad in FORBIDDEN):
            continue
        print(f"SIGKILL {pid} {cmd[:180]}")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(1)
    still = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        cmd = cmdline(pid)
        if is_ours(pid, cmd) and not any(bad in cmd for bad in FORBIDDEN):
            still.append((pid, cmd[:180]))
    print("still_ours", still)
    return 0 if not still else 1


if __name__ == "__main__":
    raise SystemExit(main())
