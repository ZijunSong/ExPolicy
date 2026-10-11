#!/usr/bin/env python3
"""Per-GPU sequential queue: M0 then baselines for one seed."""

from __future__ import annotations

import argparse
import subprocess
import sys

from expolicy.closed_loop.constants import ROOT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--conditions", nargs="+", required=True)
    p.add_argument("--budget", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    py = sys.executable
    script = ROOT / "scripts" / "r7" / "run_r7_condition.py"
    for cond in args.conditions:
        cmd = [
            py,
            str(script),
            "--condition",
            cond,
            "--seed",
            str(args.seed),
            "--gpu",
            str(args.gpu),
            "--budget",
            str(args.budget),
            "--batch-size",
            str(args.batch_size),
            "--rounds",
            str(args.rounds),
        ]
        if args.resume:
            cmd.append("--resume")
        print("RUN", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=str(ROOT))
    print("SEED_QUEUE_DONE", args.seed, flush=True)


if __name__ == "__main__":
    main()
