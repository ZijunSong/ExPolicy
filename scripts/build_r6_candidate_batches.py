#!/usr/bin/env python3
"""CLI around unsupervised candidate-batch construction."""

from __future__ import annotations

import argparse
import json

from expolicy.acquisition.ava import balanced_cluster_batches
from expolicy.experiments.r5_4 import records_to_tasks
from expolicy.experiments.r6_ava import load_r6_embeddings, load_split_r6
from expolicy.experiments.common import write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--num-candidates", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    from pathlib import Path

    split = json.loads(Path(args.split).read_text(encoding="utf-8")) if args.split.endswith(".json") else None
    if split is None:
        split = load_split_r6(Path(args.split))
    from expolicy.experiments.r5_4 import load_predictions, R54_ROOT

    preds = load_predictions(R54_ROOT)
    pool = records_to_tasks(split.get("candidate_pool") or [], split="candidate_pool", predictions=preds)
    emb = load_r6_embeddings(Path(args.split).parent if args.split.endswith(".json") else Path(args.split), [t.task_id for t in pool])
    batches = balanced_cluster_batches(pool, emb, n_batches=args.num_candidates, batch_size=args.batch_size, seed=args.seed, round_idx=args.round)
    write_json(Path(args.out), {"batches": batches, "n": len(batches)})
    print(json.dumps({"n_batches": len(batches), "sizes": [len(b) for b in batches]}), flush=True)


if __name__ == "__main__":
    main()
