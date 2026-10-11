#!/usr/bin/env python3
"""Launch R7 conditions on GPUs 5/6/7. Default: one seed per GPU, sequential conditions."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from expolicy.closed_loop.constants import ROOT

DEFAULT_ORDER = [
    "expolicy",
    "passive_noop",
    "passive_episodic",
    "passive_count_skill",
    "random_joint",
    "no_feedback",
    "reward_shuffle",
    "assimilation_fixed",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", default=DEFAULT_ORDER)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--gpus", nargs="+", default=["5", "6", "7"])
    p.add_argument("--budget", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if len(args.gpus) < len(args.seeds):
        raise SystemExit("need at least one GPU per seed")
    py = sys.executable
    script = ROOT / "scripts" / "r7" / "run_r7_condition.py"
    log_dir = ROOT / "outputs" / "r7_closed_loop" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for seed, gpu in zip(args.seeds, args.gpus):
        log = log_dir / f"seed{seed}.log"
        cmd = [
            py,
            str(ROOT / "scripts" / "r7" / "_seed_queue.py"),
            "--seed",
            str(seed),
            "--gpu",
            str(gpu),
            "--conditions",
            *args.conditions,
            "--budget",
            str(args.budget),
            "--batch-size",
            str(args.batch_size),
            "--rounds",
            str(args.rounds),
        ]
        if args.resume:
            cmd.append("--resume")
        print("launch", cmd, "log", log)
        if args.dry_run:
            continue
        handle = log.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT), env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
        procs.append((seed, gpu, proc, log))
    for seed, gpu, proc, log in procs:
        print(f"seed{seed} gpu{gpu} pid={proc.pid} log={log}")
    if procs:
        print("watch: python scripts/r7/watchdog_r7.py")


if __name__ == "__main__":
    main()
