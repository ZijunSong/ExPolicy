#!/usr/bin/env python3
"""Watch R7 GPU jobs on 4-7. Resume stalled (model, condition, seed) only. Never delete results."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
OUT = ROOT / "outputs" / "r7_policy_generalization"
LOG = OUT / "logs"
PY = str(ROOT / ".venv" / "bin" / "python")
STALL = 15 * 60
POLL = 30
MAX_RESTARTS = 8
GPUS = ["4", "5", "6", "7"]

JOBS = [
    {
        "id": "ival:Qwen2.5-3B-Instruct",
        "gpu": "6",
        "cmd": [
            PY,
            str(ROOT / "scripts" / "r7" / "run_r7_worker.py"),
            "--phase",
            "interventional_validate",
            "--model",
            "Qwen2.5-3B-Instruct",
            "--gpu",
            "6",
            "--per-dataset",
            "40",
            "--out",
            str(OUT / "assimilation" / "interventional_validate" / "Qwen2.5-3B-Instruct"),
        ],
        "out": OUT / "assimilation" / "interventional_validate" / "Qwen2.5-3B-Instruct",
        "pid_hint": None,
    },
    {
        "id": "ival:Qwen2.5-7B-Instruct",
        "gpu": "7",
        "cmd": [
            PY,
            str(ROOT / "scripts" / "r7" / "run_r7_worker.py"),
            "--phase",
            "interventional_validate",
            "--model",
            "Qwen2.5-7B-Instruct",
            "--gpu",
            "7",
            "--per-dataset",
            "40",
            "--out",
            str(OUT / "assimilation" / "interventional_validate" / "Qwen2.5-7B-Instruct"),
        ],
        "out": OUT / "assimilation" / "interventional_validate" / "Qwen2.5-7B-Instruct",
        "pid_hint": None,
    },
    {
        "id": "ival:Qwen3-4B-Instruct-2507",
        "gpu": "4",
        "cmd": [
            PY,
            str(ROOT / "scripts" / "r7" / "run_r7_worker.py"),
            "--phase",
            "interventional_validate",
            "--model",
            "Qwen3-4B-Instruct-2507",
            "--gpu",
            "4",
            "--per-dataset",
            "40",
            "--out",
            str(OUT / "assimilation" / "interventional_validate" / "Qwen3-4B-Instruct-2507"),
        ],
        "out": OUT / "assimilation" / "interventional_validate" / "Qwen3-4B-Instruct-2507",
        "pid_hint": None,
    },
    {
        "id": "ival:Qwen3-4B-Thinking-2507",
        "gpu": "5",
        "cmd": [
            PY,
            str(ROOT / "scripts" / "r7" / "run_r7_worker.py"),
            "--phase",
            "interventional_validate",
            "--model",
            "Qwen3-4B-Thinking-2507",
            "--gpu",
            "5",
            "--per-dataset",
            "40",
            "--out",
            str(OUT / "assimilation" / "interventional_validate" / "Qwen3-4B-Thinking-2507"),
        ],
        "out": OUT / "assimilation" / "interventional_validate" / "Qwen3-4B-Thinking-2507",
        "pid_hint": None,
    },
]


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


def newest_live(out: Path) -> Path | None:
    lives = list(out.rglob("STATUS_LIVE.json"))
    if not lives:
        return None
    return max(lives, key=lambda p: p.stat().st_mtime)


def done(job: dict) -> bool:
    return (job["out"] / "DONE").exists()


def find_pid(job: dict) -> int | None:
    model = None
    cmd = job.get("cmd") or []
    if "--model" in cmd:
        model = cmd[cmd.index("--model") + 1]
    try:
        out = subprocess.check_output(["pgrep", "-af", "run_r7_worker.py"], text=True)
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        if "watchdog" in line:
            continue
        if model and f"--model {model}" not in line and f"--model {model} " not in f"{line} ":
            if f"--model {model}" not in line:
                continue
        if "run_r7_worker.py" in line:
            try:
                return int(line.split()[0])
            except Exception:
                continue
    return None


def launch(job: dict) -> int:
    log_path = LOG / f"watch_{job['id'].replace(':', '_')}.log"
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n===== relaunch {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    handle.flush()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = job["gpu"]
    env["EXPOLICY_RETRIEVER_URL"] = "http://127.0.0.1:18080"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    proc = subprocess.Popen(job["cmd"], cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    log(f"launch {job['id']} gpu={job['gpu']} pid={proc.pid}")
    return proc.pid


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    restarts = {j["id"]: 0 for j in JOBS}
    pids: dict[str, int | None] = {j["id"]: find_pid(j) for j in JOBS}
    log(f"watch start pids={pids}")
    while True:
        if all(done(j) for j in JOBS):
            log("all interventional_validate DONE")
            (OUT / "PHASE_INTERVENTIONAL.DONE").write_text("ok\n")
            return
        for job in JOBS:
            if done(job):
                continue
            pid = pids.get(job["id"]) or find_pid(job)
            pids[job["id"]] = pid
            live = newest_live(job["out"])
            age = (time.time() - live.stat().st_mtime) if live else None
            dead = pid is None or not alive(pid)
            # Fresh STATUS_LIVE means the original worker is alive even if pid lookup failed.
            stalled = False
            if age is not None and age > STALL:
                stalled = True
            elif dead and (age is None or age > STALL):
                stalled = True
            if stalled:
                n = restarts[job["id"]]
                log(f"STALL {job['id']} pid={pid} age={age} restarts={n}")
                if pid and alive(pid):
                    kill_tree(pid)
                if n >= MAX_RESTARTS:
                    log(f"GIVE UP {job['id']}")
                    continue
                restarts[job["id"]] = n + 1
                pids[job["id"]] = launch(job)
        try:
            snap = subprocess.check_output(
                ["nvidia-smi", "-i", "4,5,6,7", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
                text=True,
            )
            (LOG / "nvidia.snap").write_text(snap, encoding="utf-8")
        except Exception:
            pass
        time.sleep(POLL)


if __name__ == "__main__":
    main()
