#!/usr/bin/env python3
"""R5.2 strong-baseline collision: frontier vs Ace-Skill Prioritized vs passive.

CLI matches ExPolicy_R5_2_Strong_Baseline_Collision_Instructions.md:
  --methods --seeds --budget --gpu --resume --output-dir

Pass --gpus 4,5,6,7 to launch the 4-way watchdog. A single --gpu (or
CUDA_VISIBLE_DEVICES) runs the given seeds sequentially on that card.
Always --resume after a kill. Never writes into outputs/r5_acquisition_skill.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from expolicy.experiments.common import ROOT, write_json
from expolicy.experiments.r5_1 import R51_ROOT, R5_ROOT, ensure_seed_splits, sha256_file
from expolicy.experiments.r5_2 import (
    DEFAULT_MODEL,
    R52_ROOT,
    maybe_text_router_eval,
    run_one_seed,
    write_audit,
)
from expolicy.experiments.r_common import apply_gpu

STALL_SEC = 12 * 60
MAX_RESTARTS = 3
POLL_SEC = 30
SEED_GPU_MAP = {
    "4": (0, 4, 8),
    "5": (1, 5, 9),
    "6": (2, 6),
    "7": (3, 7),
}


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


def _cuda_visible(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
        env = dict(x.split(b"=", 1) for x in raw.split(b"\0") if b"=" in x)
        val = env.get(b"CUDA_VISIBLE_DEVICES", b"").decode()
        return val or None
    except Exception:
        return None


def _live_run(out: Path) -> dict | None:
    status = out / "STATUS_LIVE.json"
    if (out / "DONE").exists() or not status.exists():
        return None
    try:
        payload = json.loads(status.read_text(encoding="utf-8"))
    except Exception:
        return None
    pid = payload.get("pid")
    age = time.time() - status.stat().st_mtime
    if pid and _alive(int(pid)) and age < STALL_SEC:
        return {"pid": int(pid), "age": age, "phase": payload.get("phase")}
    return None


def _run_dir(root: Path, method: str, seed: int) -> Path:
    return root / method / f"seed_{seed}"


def prepare_output(root: Path, r5: Path, r51: Path, seeds: list[int]) -> None:
    if root.resolve() == r5.resolve() or r5.resolve() in root.resolve().parents:
        raise RuntimeError(f"refusing to write into original R5 output: {root}")
    root.mkdir(parents=True, exist_ok=True)
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        dest = split_dir / f"seed_{seed}.json"
        if dest.exists():
            continue
        if seed <= 2:
            src = r5 / "splits" / f"seed_{seed}.json"
        else:
            src = r51 / "seed_extension" / "splits" / f"seed_{seed}.json"
        if src.exists():
            dest.write_bytes(src.read_bytes())
    ensure_seed_splits(root, seeds, r5_root=r5)
    write_audit(root)
    hashes = {str(s): sha256_file(split_dir / f"seed_{s}.json") for s in seeds if (split_dir / f"seed_{s}.json").exists()}
    write_json(split_dir / "R5_2_SPLIT_HASHES.json", hashes)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R5.2 frontier vs Ace-Skill vs passive")
    p.add_argument("--methods", nargs="+", default=["ace_skill_prioritized"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--budget", type=int, default=48)
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES") or None)
    p.add_argument("--gpus", default=None, help="comma-separated physical GPU ids; launches watchdog")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", default=str(R52_ROOT))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--r5-root", dest="r5_root", default=str(R5_ROOT))
    p.add_argument("--r5-1-root", dest="r51_root", default=str(R51_ROOT))
    p.add_argument("--min-support", dest="min_support", type=int, default=8)
    p.add_argument("--dispatch", default="oracle")
    p.add_argument("--text-router", dest="text_router", action="store_true")
    p.add_argument("--skip-prep", dest="skip_prep", action="store_true")
    return p.parse_args(argv)


def run_worker(args: argparse.Namespace) -> None:
    apply_gpu(args.gpu)
    root = Path(args.output_dir)
    r5 = Path(args.r5_root)
    r51 = Path(args.r51_root)
    if not args.skip_prep:
        prepare_output(root, r5, r51, list(args.seeds))
    world = None
    cache: dict = {}
    for method in args.methods:
        for seed in args.seeds:
            print(f"[r5.2] method={method} seed={seed} gpu={args.gpu}", flush=True)
            rec = run_one_seed(
                method=method,
                seed=int(seed),
                output_dir=root,
                model=args.model,
                budget=args.budget,
                gpu=args.gpu,
                resume=args.resume,
                r5_root=r5,
                r51_root=r51,
                min_support=args.min_support,
                dispatch=args.dispatch,
                world=world,
                cache=cache,
            )
            print(json.dumps({k: rec.get(k) for k in ("status", "method", "seed", "future_em", "delta_future", "n_acquired") if k in rec or rec.get("status")}, default=str), flush=True)
            if args.text_router:
                maybe_text_router_eval(
                    method=method,
                    seed=int(seed),
                    output_dir=root,
                    r5_root=r5,
                    r51_root=r51,
                    world=world,
                )


def _launch(job: dict, python: str) -> subprocess.Popen:
    gpu = str(job["gpu"])
    log_path = Path(job["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        python,
        str(ROOT / "scripts" / "run_r5_2_baselines.py"),
        "--methods",
        *job["methods"],
        "--seeds",
        *[str(s) for s in job["seeds"]],
        "--budget",
        str(job["budget"]),
        "--gpu",
        gpu,
        "--resume",
        "--output-dir",
        job["output_dir"],
        "--model",
        job["model"],
        "--r5-root",
        job["r5_root"],
        "--r5-1-root",
        job["r51_root"],
        "--min-support",
        str(job["min_support"]),
        "--dispatch",
        job["dispatch"],
        "--skip-prep",
    ]
    if job.get("text_router"):
        cmd.append("--text-router")
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


def _worker_done(job: dict) -> bool:
    root = Path(job["output_dir"])
    for method in job["methods"]:
        for seed in job["seeds"]:
            if not (_run_dir(root, method, seed) / "DONE").exists():
                return False
    return True


def _worker_status_age(job: dict) -> float | None:
    ages = []
    root = Path(job["output_dir"])
    for method in job["methods"]:
        for seed in job["seeds"]:
            out = _run_dir(root, method, seed)
            if (out / "DONE").exists():
                continue
            age = _status_age(out)
            if age is not None:
                ages.append(age)
    if not ages:
        return None
    return min(ages)


def watchdog(jobs: list[dict], gpus: list[str], *, status_path: Path) -> None:
    python = _venv_python()
    pending = list(jobs)
    running: list[dict] = []
    finished: list[dict] = []
    failed: list[dict] = []
    free = list(gpus)

    def start_ready() -> None:
        while pending and free:
            job = pending[0]
            if _worker_done(job):
                print(f"[skip-done] {job['name']}", flush=True)
                finished.append(pending.pop(0))
                continue
            job = pending.pop(0)
            job["gpu"] = free.pop(0)
            _launch(job, python)
            running.append(job)

    start_ready()
    write_json(status_path, {"running": [j["name"] for j in running], "pending": [j["name"] for j in pending]})
    while running or pending:
        time.sleep(POLL_SEC)
        still = []
        for job in running:
            proc: subprocess.Popen | None = job.get("proc")
            code = proc.poll() if proc is not None else None
            age = _worker_status_age(job)
            if age is not None:
                job["saw_status"] = True
            stalled = False
            if code is None:
                if age is None and not job.get("saw_status") and time.time() - job["launched_at"] > 600:
                    stalled = True
                if age is not None and age > STALL_SEC:
                    stalled = True
            if stalled:
                print(f"[stall] {job['name']} pid={job.get('pid')} age={age} — kill and resume", flush=True)
                _kill(job)
                if job.get("gpu") is not None:
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
            if job.get("gpu") is not None:
                free.append(job["gpu"])
            job["gpu"] = None
            if code == 0 and _worker_done(job):
                print(f"[done] {job['name']} exit=0", flush=True)
                finished.append(job)
            else:
                print(f"[exit] {job['name']} code={code} — resume", flush=True)
                job["restarts"] += 1
                if job["restarts"] <= MAX_RESTARTS and not _worker_done(job):
                    pending.insert(0, job)
                elif _worker_done(job):
                    finished.append(job)
                else:
                    failed.append(job)
        running = still
        start_ready()
        write_json(
            status_path,
            {
                "ts": time.time(),
                "running": [{"name": j["name"], "gpu": j["gpu"], "pid": j.get("pid")} for j in running],
                "pending": [j["name"] for j in pending],
                "finished": [j["name"] for j in finished],
                "failed": [j["name"] for j in failed],
            },
        )
    if failed:
        raise SystemExit(f"failed jobs: {[j['name'] for j in failed]}")


def assign_seeds_to_gpus(seeds: list[int], gpus: list[str]) -> dict[str, list[int]]:
    assigned = {g: [] for g in gpus}
    if set(gpus) <= set(SEED_GPU_MAP) and set(seeds) <= set(range(10)):
        for g in gpus:
            assigned[g] = [s for s in SEED_GPU_MAP[g] if s in seeds]
        leftover = [s for s in seeds if not any(s in assigned[g] for g in gpus)]
        for i, s in enumerate(leftover):
            assigned[gpus[i % len(gpus)]].append(s)
        return assigned
    for i, s in enumerate(seeds):
        assigned[gpus[i % len(gpus)]].append(s)
    return assigned


def run_suite(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    r5 = Path(args.r5_root)
    r51 = Path(args.r51_root)
    gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip()]
    prepare_output(root, r5, r51, list(args.seeds))
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    mapping = assign_seeds_to_gpus(list(args.seeds), gpus)
    jobs = []
    for gpu, seeds in mapping.items():
        if not seeds:
            continue
        jobs.append(
            {
                "name": f"gpu{gpu}_{'_'.join(args.methods)}_{'-'.join(str(s) for s in seeds)}",
                "methods": list(args.methods),
                "seeds": seeds,
                "gpu": gpu,
                "restarts": 0,
                "budget": args.budget,
                "output_dir": str(root),
                "model": args.model,
                "r5_root": str(r5),
                "r51_root": str(r51),
                "min_support": args.min_support,
                "dispatch": args.dispatch,
                "text_router": args.text_router,
                "log": str(log_dir / f"r52_gpu{gpu}.log"),
            }
        )
    watchdog(jobs, [j["gpu"] for j in jobs], status_path=root / "SUITE_STATUS.json")
    print("[suite] all jobs complete", flush=True)


def main() -> None:
    args = parse_args()
    if args.gpus:
        run_suite(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
