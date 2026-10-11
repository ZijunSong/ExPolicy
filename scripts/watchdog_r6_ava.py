#!/usr/bin/env python3
"""Watch GPU 4-7 R6 Stage-1 wrappers. Kill/resume on 15-minute STATUS_LIVE stall."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
OUT = ROOT / "outputs" / "r6_ava"
LOG = OUT / "logs"
STALL_SEC = 15 * 60
POLL_SEC = 60
MAX_RESTARTS = 8

WRAPPERS = {
    "4": ROOT / "scripts" / "run_r6_gpu4.sh",
    "5": ROOT / "scripts" / "run_r6_gpu5.sh",
    "6": ROOT / "scripts" / "run_r6_gpu6.sh",
    "7": ROOT / "scripts" / "run_r6_gpu7.sh",
}

GPU_JOBS = {
    "4": [("exact_ava", 0)],
    "5": [("exact_ava", 1)],
    "6": [("exact_ava", 2)],
    "7": [
        ("passive", 0),
        ("passive", 1),
        ("passive", 2),
        ("ace_skill_prioritized", 0),
        ("ace_skill_prioritized", 1),
        ("ace_skill_prioritized", 2),
        ("legacy_frontier", 0),
        ("legacy_frontier", 1),
        ("legacy_frontier", 2),
        ("rasa_full", 0),
        ("rasa_full", 1),
        ("rasa_full", 2),
    ],
}


def log(msg: str) -> None:
    line = f"[watch {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "watchdog.log").open("a", encoding="utf-8").write(line + "\n")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def gpu_finished(gpu: str) -> bool:
    for cond, seed in GPU_JOBS[gpu]:
        d = OUT / "per_seed" / cond / f"seed_{seed}"
        if not ((d / "DONE").exists() and (d / "final_metrics.json").exists()):
            return False
    return True


def youngest_status(gpu: str) -> float | None:
    ages = []
    for cond, seed in GPU_JOBS[gpu]:
        d = OUT / "per_seed" / cond / f"seed_{seed}"
        if (d / "DONE").exists():
            continue
        live = d / "STATUS_LIVE.json"
        if live.exists():
            ages.append(time.time() - live.stat().st_mtime)
    return min(ages) if ages else None


def pid_tree(pid: int) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-P", str(pid)], text=True)
        kids = [int(x) for x in out.split() if x.strip()]
    except subprocess.CalledProcessError:
        kids = []
    all_pids = list(kids)
    for k in kids:
        all_pids.extend(pid_tree(k))
    return all_pids


def kill_tree(pid: int) -> None:
    pids = [pid] + pid_tree(pid)
    for p in pids:
        if alive(p):
            try:
                os.kill(p, signal.SIGTERM)
            except OSError:
                pass
    time.sleep(5)
    for p in pids:
        if alive(p):
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass


def launch(gpu: str) -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    log_path = LOG / f"gpu{gpu}.log"
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n===== watchdog relaunch {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    handle.flush()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["EXPOLICY_RETRIEVER_URL"] = "http://127.0.0.1:18080"
    proc = subprocess.Popen(
        ["bash", str(WRAPPERS[gpu])],
        cwd=str(ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (LOG / f"gpu{gpu}.pid").write_text(str(proc.pid), encoding="utf-8")
    log(f"launch gpu{gpu} pid={proc.pid} log={log_path}")
    return proc.pid


def read_pid(gpu: str) -> int | None:
    path = LOG / f"gpu{gpu}.pid"
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    restarts = {g: 0 for g in WRAPPERS}
    launched_at = {g: 0.0 for g in WRAPPERS}
    for gpu in WRAPPERS:
        if gpu_finished(gpu):
            log(f"gpu{gpu} already finished")
            continue
        pid = read_pid(gpu)
        if pid and alive(pid):
            log(f"gpu{gpu} already running pid={pid}")
            launched_at[gpu] = time.time()
            continue
        launch(gpu)
        launched_at[gpu] = time.time()
    while True:
        if all(gpu_finished(g) for g in WRAPPERS):
            log("all Stage-1 GPU wrappers finished")
            (OUT / "STAGE1_JOBS.DONE").write_text("ok\n", encoding="utf-8")
            return
        time.sleep(POLL_SEC)
        for gpu in WRAPPERS:
            if gpu_finished(gpu):
                continue
            pid = read_pid(gpu)
            running = bool(pid and alive(pid))
            age = youngest_status(gpu)
            stalled = False
            if running and age is not None and age > STALL_SEC:
                stalled = True
            if running and age is None and time.time() - launched_at[gpu] > 900:
                stalled = True
            if not running:
                log(f"gpu{gpu} dead — resume")
                stalled = True
            if not stalled:
                continue
            if pid and alive(pid):
                log(f"gpu{gpu} stall age={age} — kill tree pid={pid}")
                kill_tree(pid)
            restarts[gpu] += 1
            if restarts[gpu] > MAX_RESTARTS:
                raise SystemExit(f"gpu{gpu} stalled too many times")
            launch(gpu)
            launched_at[gpu] = time.time()


if __name__ == "__main__":
    main()
