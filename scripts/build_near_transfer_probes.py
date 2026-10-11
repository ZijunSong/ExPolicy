#!/usr/bin/env python3
"""Build capability-matched near-transfer probe sets for R1–R4.

Writes outputs/r1_near_transfer/probes/batches.json (shared by later runs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expolicy.experiments.common import ROOT, load_world, write_json
from expolicy.experiments.patterns import BUCKET_PRIMARY_CAP, TARGET_CAPS
from expolicy.experiments.probes import (
    bucket_pool,
    build_probe_set,
    split_experiences_into_batches,
)
from expolicy.experiments.r_common import (
    add_common_args,
    apply_gpu,
    load_e0_experiences,
    write_run_header,
)

DEFAULT_OUT = str(ROOT / "outputs" / "r1_near_transfer")


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_args(p, default_out=DEFAULT_OUT)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-batches", type=int, default=3)
    p.add_argument("--n-near", type=int, default=16)
    p.add_argument("--n-far", type=int, default=16)
    p.add_argument("--n-retention", type=int, default=16)
    args = p.parse_args()
    apply_gpu(args.gpu)
    out = Path(args.output_dir)
    probe_dir = out / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    world = load_world(Path(args.world))
    exps = load_e0_experiences()
    if len(exps) < 20:
        raise SystemExit(f"need E0 experiences, found {len(exps)}")

    batches_out = []
    for bucket in TARGET_CAPS:
        pool = bucket_pool(exps, bucket)
        if len(pool) < args.batch_size:
            raise SystemExit(f"{bucket}: only {len(pool)} experiences, need {args.batch_size}")
        groups = split_experiences_into_batches(
            pool, n_batches=args.n_batches, batch_size=args.batch_size, seed=args.seed
        )
        for i, group in enumerate(groups):
            batch_id = f"{bucket}_b{i}"
            probes = build_probe_set(
                world,
                batch_id=batch_id,
                bucket=bucket,
                experiences=group,
                n_near=args.n_near,
                n_far=args.n_far,
                n_retention=args.n_retention,
                seed=args.seed + i * 17,
            )
            row = probes.as_ids()
            row["experiences"] = [
                {
                    "experience_id": e.experience_id,
                    "task_id": e.task_id,
                    "capability": e.capability,
                    "success": e.success,
                    "failure_type": e.failure_type.value if e.failure_type else None,
                }
                for e in group
            ]
            row["n_near"] = len(probes.near)
            row["n_far"] = len(probes.far)
            row["n_retention"] = len(probes.retention)
            batches_out.append(row)

    payload = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "n_batches": args.n_batches,
        "n_near": args.n_near,
        "n_far": args.n_far,
        "n_retention": args.n_retention,
        "primary_caps": BUCKET_PRIMARY_CAP,
        "n_e0": len(exps),
        "batches": batches_out,
    }
    write_json(probe_dir / "batches.json", payload)
    write_run_header(
        out,
        config={
            "script": "build_near_transfer_probes.py",
            "seed": args.seed,
            "model": args.model,
            "batch_size": args.batch_size,
            "n_batches": args.n_batches,
            "n_near": args.n_near,
            "n_far": args.n_far,
            "n_retention": args.n_retention,
        },
        extra_manifest={"n_batches": len(batches_out)},
    )
    print(json.dumps({"n_batches": len(batches_out), "path": str(probe_dir / "batches.json")}, indent=2))
    for row in batches_out:
        print(
            f"{row['batch_id']}: exp={len(row['experiences'])} "
            f"near={row['n_near']} far={row['n_far']} ret={row['n_retention']} notes={row.get('notes')}"
        )


if __name__ == "__main__":
    main()
