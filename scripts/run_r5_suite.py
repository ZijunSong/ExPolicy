#!/usr/bin/env python3
"""Launch and watchdog R5 jobs on a fixed GPU list. Always --resume after a kill."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from expolicy.experiments.common import ROOT, write_json

DEFAULT_OUT = ROOT / "outputs" / "r5_acquisition_skill"
METHODS = ("passive", "reward_seeking", "uncertainty", "frontier", "failure_replay")
SEEDS = (0, 1, 2)
STALL_SEC = 12 * 60
MAX_RESTARTS = 3
POLL_SEC = 30


def _venv_python() -> str:
    cand = ROOT / ".venv" / "bin" / "python"
    return str(cand) if cand.exists() else sys.executable


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _status_age(out: Path) -> float | None:
    path = out / "STATUS_LIVE.json"
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime


def _run_dir(root: Path, method: str, seed: int) -> Path:
    if method == "base":
        return root / "base" / f"seed_{seed}"
    return root / method / f"seed_{seed}"


def _launch(job: dict, python: str) -> subprocess.Popen:
    gpu = str(job["gpu"])
    log_path = Path(job["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        python,
        str(ROOT / "scripts" / "run_r5_acquisition_skill.py"),
        "--method",
        job["method"],
        "--seed",
        str(job["seed"]),
        "--gpu",
        gpu,
        "--model",
        job["model"],
        "--output_dir",
        job["output_dir"],
        "--budget",
        str(job["budget"]),
        "--resume",
    ]
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} restart={job['restarts']} =====\n")
    handle.flush()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    job["pid"] = proc.pid
    job["handle"] = handle
    job["proc"] = proc
    job["launched_at"] = time.time()
    print(f"[launch] {job['name']} gpu={gpu} pid={proc.pid} log={log_path}", flush=True)
    return proc


def _kill(job: dict) -> None:
    proc: subprocess.Popen | None = job.get("proc")
    pid = job.get("pid")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    elif pid and _alive(pid):
        os.kill(pid, 15)
        time.sleep(2)
        if _alive(pid):
            os.kill(pid, 9)
    handle = job.get("handle")
    if handle:
        try:
            handle.close()
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument("--model", default="/data/ppnm/models/Qwen2.5-3B-Instruct")
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--budget", type=int, default=48)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--skip_prep", action="store_true")
    p.add_argument("--skip_analyze", action="store_true")
    args = p.parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    python = _venv_python()

    if not args.skip_prep:
        prep = [
            [python, "-m", "pytest", "-q", "tests/unit/test_r5_skill_assimilator.py", "tests/unit/test_r_assimilation.py", "tests/unit/test_memory_skills.py"],
            [python, str(ROOT / "scripts" / "audit_r5_causal_path.py"), "--output_dir", str(root)],
            [python, str(ROOT / "scripts" / "build_r5_splits.py"), "--output_dir", str(root), "--seeds", *[str(s) for s in args.seeds]],
        ]
        for cmd in prep:
            print("[prep]", " ".join(cmd), flush=True)
            subprocess.check_call(cmd, cwd=str(ROOT))

    pending_base: list[dict] = []
    pending_methods: list[dict] = []
    for seed in args.seeds:
        pending_base.append(
            {
                "name": f"base_s{seed}",
                "method": "base",
                "seed": seed,
                "restarts": 0,
                "gpu": None,
                "output_dir": str(root),
                "model": args.model,
                "budget": args.budget,
                "log": str(log_dir / f"r5_base_s{seed}.log"),
            }
        )
    for seed in args.seeds:
        for method in args.methods:
            pending_methods.append(
                {
                    "name": f"{method}_s{seed}",
                    "method": method,
                    "seed": seed,
                    "restarts": 0,
                    "gpu": None,
                    "output_dir": str(root),
                    "model": args.model,
                    "budget": args.budget,
                    "log": str(log_dir / f"r5_{method}_s{seed}.log"),
                }
            )
    pending = pending_base
    methods_queued = False

    running: list[dict] = []
    finished: list[dict] = []
    failed: list[dict] = []
    free = list(gpus)

    def start_ready() -> None:
        while pending and free:
            job = pending[0]
            out = _run_dir(root, job["method"], job["seed"])
            if (out / "DONE").exists():
                print(f"[skip-done] {job['name']}", flush=True)
                finished.append(pending.pop(0))
                continue
            job = pending.pop(0)
            job["gpu"] = free.pop(0)
            _launch(job, python)
            running.append(job)

    start_ready()
    if not running and not pending and not methods_queued:
        pending = pending_methods
        methods_queued = True
        start_ready()
    write_json(root / "SUITE_STATUS.json", {"running": [j["name"] for j in running], "pending": [j["name"] for j in pending]})

    while running or pending:
        time.sleep(POLL_SEC)
        still = []
        for job in running:
            proc: subprocess.Popen = job["proc"]
            code = proc.poll()
            out = _run_dir(root, job["method"], job["seed"])
            age = _status_age(out)
            stalled = False
            if code is None:
                # allow startup without STATUS for 3 minutes
                if age is None and time.time() - job["launched_at"] > 180:
                    stalled = True
                if age is not None and age > STALL_SEC:
                    stalled = True
            if stalled:
                print(f"[stall] {job['name']} pid={job.get('pid')} age={age} — kill and resume", flush=True)
                _kill(job)
                free.append(job["gpu"])
                job["gpu"] = None
                job["restarts"] += 1
                if job["restarts"] <= MAX_RESTARTS:
                    pending.insert(0, job)
                else:
                    failed.append(job)
                    print(f"[fail] {job['name']} exceeded restarts", flush=True)
                continue
            if code is None:
                still.append(job)
                continue
            handle = job.get("handle")
            if handle:
                try:
                    handle.close()
                except Exception:
                    pass
            free.append(job["gpu"])
            job["gpu"] = None
            if code == 0 and (out / "DONE").exists():
                print(f"[done] {job['name']} exit=0", flush=True)
                finished.append(job)
            else:
                print(f"[exit] {job['name']} code={code} — resume", flush=True)
                job["restarts"] += 1
                if job["restarts"] <= MAX_RESTARTS and not (out / "DONE").exists():
                    pending.insert(0, job)
                else:
                    if (out / "DONE").exists():
                        finished.append(job)
                    else:
                        failed.append(job)
        running = still
        if not running and not pending and not methods_queued:
            pending = pending_methods
            methods_queued = True
            print("[suite] base evals complete; launching method jobs", flush=True)
        start_ready()
        write_json(
            root / "SUITE_STATUS.json",
            {
                "ts": time.time(),
                "running": [{"name": j["name"], "gpu": j["gpu"], "pid": j.get("pid")} for j in running],
                "pending": [j["name"] for j in pending],
                "finished": [j["name"] for j in finished],
                "failed": [j["name"] for j in failed],
            },
        )

    if failed:
        print("[suite] failed jobs:", [j["name"] for j in failed], flush=True)
        raise SystemExit(1)
    if not args.skip_analyze:
        subprocess.check_call([python, str(ROOT / "scripts" / "analyze_r5_gate.py"), "--output_dir", str(root)], cwd=str(ROOT))
    print("[suite] all jobs complete", flush=True)


if __name__ == "__main__":
    main()
