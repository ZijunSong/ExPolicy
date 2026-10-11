#!/usr/bin/env python3
"""Stage-2 queue for R5.4. Only run after CASE_A_PASS."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from expolicy.experiments.common import ROOT
from expolicy.experiments.r5_4 import R54_ROOT

PY = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in parse_csv(text)]


def done(root: Path, condition: str, seed: int) -> bool:
    return (root / "per_seed" / condition / f"seed_{seed}" / "DONE").exists() and (
        root / "per_seed" / condition / f"seed_{seed}" / "final_metrics.json"
    ).exists()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(R54_ROOT))
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument("--seeds", default="3,4,5,6,7,8,9")
    p.add_argument("--conditions", default="rasa_full,ace_k8,passive_k8")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--budget", type=int, default=48)
    args = p.parse_args()
    root = Path(args.root)
    decision = root / "R5_4_GATE_DECISION.json"
    if not decision.exists():
        raise SystemExit("Stage-1 decision missing")
    payload = json.loads(decision.read_text(encoding="utf-8"))
    if payload.get("case") != "CASE_A_PASS":
        raise SystemExit(f"Stage-2 forbidden: case={payload.get('case')}")
    gpus = parse_csv(args.gpus)
    seeds = parse_ints(args.seeds)
    conditions = parse_csv(args.conditions)
    jobs = [(c, s) for c in conditions for s in seeds if not (args.resume and done(root, c, s))]
    # round-robin GPUs
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: list[dict] = []
    pending = list(jobs)
    print(f"[stage2] {len(jobs)} jobs on gpus={gpus}", flush=True)
    while pending or running:
        while pending and len(running) < len(gpus):
            used = {j["gpu"] for j in running}
            gpu = next(g for g in gpus if g not in used)
            cond, seed = pending.pop(0)
            log = log_dir / f"stage2_{cond}_seed{seed}_gpu{gpu}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONUNBUFFERED"] = "1"
            env["EXPOLICY_RETRIEVER_URL"] = env.get("EXPOLICY_RETRIEVER_URL", "http://127.0.0.1:18080")
            cmd = [
                PY,
                str(ROOT / "scripts" / "run_r5_4_condition.py"),
                "--condition",
                cond,
                "--seeds",
                str(seed),
                "--budget",
                str(args.budget),
                "--resume",
                "--root",
                str(root),
            ]
            handle = log.open("a", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
            running.append({"gpu": gpu, "cond": cond, "seed": seed, "proc": proc, "handle": handle, "log": str(log), "t0": time.time()})
            print(f"[stage2 launch] {cond} seed={seed} gpu={gpu} pid={proc.pid}", flush=True)
        time.sleep(30)
        still = []
        for job in running:
            code = job["proc"].poll()
            live = root / "per_seed" / job["cond"] / f"seed_{job['seed']}" / "STATUS_LIVE.json"
            age = time.time() - live.stat().st_mtime if live.exists() else None
            if code is None and age is not None and age > 15 * 60:
                print(f"[stage2 stall] {job['cond']} seed={job['seed']} — kill/resume", flush=True)
                job["proc"].terminate()
                try:
                    job["proc"].wait(timeout=15)
                except subprocess.TimeoutExpired:
                    job["proc"].kill()
                pending.insert(0, (job["cond"], job["seed"]))
                continue
            if code is None:
                still.append(job)
                continue
            if done(root, job["cond"], job["seed"]):
                print(f"[stage2 done] {job['cond']} seed={job['seed']}", flush=True)
            else:
                print(f"[stage2 exit {code}] {job['cond']} seed={job['seed']} — resume", flush=True)
                pending.append((job["cond"], job["seed"]))
        running = still
    print("[stage2] all queued jobs finished", flush=True)


if __name__ == "__main__":
    main()
