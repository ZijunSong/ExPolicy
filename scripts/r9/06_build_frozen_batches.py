#!/usr/bin/env python3
"""Freeze 12x8 batch IDs per env BEFORE any utility evaluation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from expolicy.r9_lab.benchmark.split import load_split, records_in_bucket
from expolicy.r9_lab.constants import (
    ENVS,
    MATRIX_BATCHES_PER_ENV,
    MATRIX_EXPERIENCES_PER_BATCH,
    OUTPUT_ROOT,
    SPLIT_SEED,
)
from expolicy.r9_lab.io_util import atomic_write_json, sha256_json
from expolicy.r9_lab.resume import run_cell_once


def build_frozen_batches(split: dict, *, seed: int = SPLIT_SEED) -> dict:
    rng = random.Random(seed + 17)
    batches = []
    for env in ENVS:
        pool = [r.to_dict() for r in records_in_bucket(split, env, "acquisition_pool")]
        # stratify by difficulty / skill
        pool.sort(key=lambda r: (r.get("difficulty") or 0, "|".join(r.get("skill_list") or []), r["task_id"]))
        rng.shuffle(pool)
        need = MATRIX_BATCHES_PER_ENV * MATRIX_EXPERIENCES_PER_BATCH
        used = pool[:need]
        for i in range(MATRIX_BATCHES_PER_ENV):
            chunk = used[i * MATRIX_EXPERIENCES_PER_BATCH : (i + 1) * MATRIX_EXPERIENCES_PER_BATCH]
            batches.append(
                {
                    "batch_id": f"{env}-B{i:02d}",
                    "env": env,
                    "task_ids": [x["task_id"] for x in chunk],
                    "items": chunk,
                    "difficulty_mean": sum(int(x.get("difficulty") or 0) for x in chunk) / max(1, len(chunk)),
                    "skill_count_mean": sum(len(x.get("skill_list") or []) for x in chunk) / max(1, len(chunk)),
                }
            )
    payload = {"seed": seed, "n_batches": len(batches), "batches": batches, "frozen_before_utility": True}
    payload["sha256"] = sha256_json({k: v for k, v in payload.items() if k != "sha256"})
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--allow-synthetic", action="store_true")
    args = p.parse_args(argv)
    out = args.output_root

    def _job() -> dict:
        split = load_split(out / "split.json")
        if split.get("synthetic") and not args.allow_synthetic:
            raise RuntimeError("Refusing synthetic split unless --allow-synthetic")
        payload = build_frozen_batches(split)
        dest = out / "audit" / "FROZEN_BATCHES.json"
        atomic_write_json(dest, payload)
        atomic_write_json(out / "FROZEN_BATCHES.json", payload)
        return {"n": payload["n_batches"], "sha256": payload["sha256"]}

    print(json.dumps(run_cell_once(out / "frozen_batches", _job, resume=args.resume), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
