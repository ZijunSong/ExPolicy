#!/usr/bin/env python3
"""Run one R7 condition × seed with --resume hash guards."""

from __future__ import annotations

import argparse
import os

from expolicy.closed_loop.constants import DEFAULT_BATCH, DEFAULT_BUDGET, DEFAULT_RETRIEVER_URL, DEFAULT_ROUNDS
from expolicy.closed_loop.runner.config import R7Config
from expolicy.closed_loop.runner.episode import run_r7_episode
from expolicy.experiments.r_common import apply_gpu


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    p.add_argument("--retriever-url", default=os.environ.get("EXPOLICY_RETRIEVER_URL", DEFAULT_RETRIEVER_URL))
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    apply_gpu(args.gpu)
    cfg = R7Config(
        condition=args.condition,
        seed=args.seed,
        rounds=args.rounds,
        batch_size=args.batch_size,
        budget=args.budget,
        retriever_url=args.retriever_url,
        smoke=bool(args.smoke),
        full_eval_at=(0,) if args.smoke else (0, 16, 32, 48),
    )
    run_r7_episode(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
