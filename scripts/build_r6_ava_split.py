#!/usr/bin/env python3
"""Freeze R6 AVA split: 240 candidate + 60 shadow per dataset from R5.4 acquisition pool.

Does not touch R5.4 future_probe. Seed 20260816. Write-once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from expolicy.environments.public_search.capability_map import PUBLIC_CAPS
from expolicy.experiments.common import ROOT, write_json
from expolicy.experiments.r5_4 import R54_ROOT, DATASETS, sha256_file
from expolicy.experiments.r6_ava import CANDIDATE_PER_DS, R6_ROOT, SHADOW_PER_DS, SPLIT_SEED


def take_stratified(rows: list[dict], n: int, rng) -> tuple[list[dict], list[dict]]:
    import random as _random

    if not isinstance(rng, _random.Random):
        raise TypeError("rng must be random.Random")
    by = defaultdict(list)
    for rec in rows:
        by[str(rec.get("capability") or "unknown")].append(rec)
    for cap in by:
        rng.shuffle(by[cap])
    chosen: list[dict] = []
    caps = [c for c in PUBLIC_CAPS if c in by] + [c for c in sorted(by) if c not in PUBLIC_CAPS]
    i = 0
    while len(chosen) < n:
        progressed = False
        for cap in caps:
            bucket = by[cap]
            if not bucket:
                continue
            chosen.append(bucket.pop())
            progressed = True
            if len(chosen) >= n:
                break
        if not progressed:
            break
        i += 1
        if i > n * 4:
            break
    leftover = [rec for cap in by for rec in by[cap]]
    rng.shuffle(chosen)
    return chosen[:n], leftover


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-split", default=str(R54_ROOT / "split_r5_4.json"))
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--candidate-per-dataset", type=int, default=CANDIDATE_PER_DS)
    p.add_argument("--shadow-per-dataset", type=int, default=SHADOW_PER_DS)
    p.add_argument("--out", default=str(R6_ROOT / "split_r6_ava.json"))
    args = p.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        print(f"[split] already frozen at {out_path} — refusing to resample", flush=True)
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        print(json.dumps(payload.get("sizes") or {}, indent=2), flush=True)
        return

    import random

    rng = random.Random(int(args.seed))
    src = json.loads(Path(args.source_split).read_text(encoding="utf-8"))
    future = list(src.get("future_probe") or [])
    acq = list(src.get("acquisition_pool") or [])
    by_ds = defaultdict(list)
    for rec in acq:
        by_ds[str(rec["dataset"])].append(rec)

    candidate: list[dict] = []
    shadow: list[dict] = []
    fail = []
    for ds in DATASETS:
        rows = list(by_ds.get(ds) or [])
        if len(rows) < args.candidate_per_dataset + args.shadow_per_dataset:
            fail.append({"dataset": ds, "n": len(rows)})
            continue
        rng.shuffle(rows)
        cand, rest = take_stratified(rows, args.candidate_per_dataset, rng)
        sh, _ = take_stratified(rest, args.shadow_per_dataset, rng)
        candidate.extend(cand)
        shadow.extend(sh)

    if fail:
        raise SystemExit(f"split failed: {fail}")

    def ids(rows):
        return {r["task_id"] for r in rows}

    def qs(rows):
        return {str(r.get("question") or "").strip().lower() for r in rows}

    c_ids, s_ids, f_ids = ids(candidate), ids(shadow), ids(future)
    c_q, s_q, f_q = qs(candidate), qs(shadow), qs(future)
    overlaps = {
        "candidate_shadow_ids": len(c_ids & s_ids),
        "candidate_future_ids": len(c_ids & f_ids),
        "shadow_future_ids": len(s_ids & f_ids),
        "candidate_shadow_q": len(c_q & s_q),
        "candidate_future_q": len(c_q & f_q),
        "shadow_future_q": len(s_q & f_q),
    }
    if any(overlaps.values()):
        raise SystemExit(f"overlap nonzero: {overlaps}")

    payload = {
        "candidate_pool": candidate,
        "shadow_probe": shadow,
        "future_probe": future,
        "bamboogle": list(src.get("bamboogle") or []),
        "split_seed": int(args.seed),
        "source_split": str(args.source_split),
        "source_future_untouched": True,
        "sizes": {
            "candidate_pool": len(candidate),
            "shadow_probe": len(shadow),
            "future_probe": len(future),
            "per_dataset": {
                ds: {
                    "candidate_pool": sum(1 for r in candidate if r["dataset"] == ds),
                    "shadow_probe": sum(1 for r in shadow if r["dataset"] == ds),
                    "future_probe": sum(1 for r in future if r["dataset"] == ds),
                }
                for ds in DATASETS
            },
        },
        "overlaps": overlaps,
        "protocol": "r6 AVA; candidate/shadow carved from R5.4 acquisition_pool; future_probe copied byte-stable",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, payload)
    digest = sha256_file(out_path)
    (out_path.parent / "SPLIT_HASH.txt").write_text(digest + "\n", encoding="utf-8")
    md = [
        "# R6 SPLIT_AUDIT",
        "",
        f"seed = {args.seed}",
        f"sha256 = {digest}",
        f"candidate = {len(candidate)} (target {3 * args.candidate_per_dataset})",
        f"shadow = {len(shadow)} (target {3 * args.shadow_per_dataset})",
        f"future = {len(future)} (R5.4 future_probe, untouched)",
        "",
        json.dumps(payload["sizes"], indent=2),
        "",
        json.dumps(overlaps, indent=2),
        "",
        "All pairwise task-id and exact-question overlaps are 0.",
        "R5.4 final future_test was not resampled.",
    ]
    (out_path.parent / "SPLIT_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
