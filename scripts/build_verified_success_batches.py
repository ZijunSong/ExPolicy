#!/usr/bin/env python3
"""Build verified-success batches for R3 weight internalization (32–64 per capability)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expolicy.experiments.common import ROOT, load_world, write_json
from expolicy.experiments.r_common import add_common_args, apply_gpu, load_e0_experiences, write_run_header
from expolicy.assimilation.teacher import collect_verified_successes

DEFAULT_OUT = str(ROOT / "outputs" / "r3_weight_batch")
CAPS = ("multi_hop", "temporal_reasoning", "cross_document_synthesis")


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_args(p, default_out=DEFAULT_OUT)
    p.add_argument("--min-n", type=int, default=32)
    p.add_argument("--max-n", type=int, default=64)
    args = p.parse_args()
    apply_gpu(args.gpu)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_run_header(
        out,
        config={"script": "build_verified_success_batches.py", "seed": args.seed, "model": args.model, "min_n": args.min_n},
    )
    world = load_world(Path(args.world))
    e0 = load_e0_experiences()
    payload = {"capabilities": {}}
    for cap in CAPS:
        existing = [e for e in e0 if e.capability == cap and e.success]
        tasks = [t for t in world.tasks.values() if t.capability == cap]
        tasks = sorted(tasks, key=lambda t: t.task_id)
        verified = collect_verified_successes(world, tasks, existing=existing, min_n=args.max_n)
        verified = verified[: args.max_n]
        if len(verified) < args.min_n:
            print(f"WARN {cap}: only {len(verified)} verified successes (min {args.min_n})")
        cap_dir = out / "data" / cap
        cap_dir.mkdir(parents=True, exist_ok=True)
        path = cap_dir / "verified.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for exp in verified:
                handle.write(exp.model_dump_json() + "\n")
        payload["capabilities"][cap] = {
            "n": len(verified),
            "n_from_e0": sum(1 for e in verified if not e.branch_origin),
            "n_corrected": sum(1 for e in verified if e.branch_origin),
            "path": str(path),
            "task_ids": [e.task_id for e in verified],
        }
    write_json(out / "verified_index.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
