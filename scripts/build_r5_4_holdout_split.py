#!/usr/bin/env python3
"""Freeze R5.4 holdout split. Never resample from results. Zero overlap with R5.3."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from expolicy.environments.public_search.capability_map import PUBLIC_CAPS, apply_mapping
from expolicy.environments.public_search.datasets import example_to_record, load_raw
from expolicy.experiments.common import ROOT, write_json
from expolicy.experiments.r5_4 import (
    ACQ_PER_DS,
    DATASETS,
    FUTURE_PER_DS,
    R53_ROOT,
    R54_ROOT,
    SPLIT_SEED,
    r53_blocked,
    sha256_file,
)

sys.path.insert(0, str(ROOT / "scripts"))
from build_r5_3_public_splits import overlap_report, take_stratified  # noqa: E402


def collect(dataset: str) -> dict[str, list[dict]]:
    raw = load_raw(dataset)
    out = {"train": [], "dev": [], "test": []}
    for split, rows in raw.items():
        key = "dev" if split in {"dev", "validation", "val"} else ("test" if split == "test" else "train")
        if split not in {"train", "dev", "validation", "val", "test"}:
            key = "train"
        for i, ex in enumerate(rows):
            rec = example_to_record(dataset, split, ex, i)
            if not rec["question"]:
                continue
            out[key].append(rec)
    return {k: apply_mapping(v) for k, v in out.items()}


def unused(rows: list[dict], blocked_ids: set[str], blocked_q: set[str]) -> list[dict]:
    out = []
    for rec in rows:
        if rec.get("capability") not in PUBLIC_CAPS:
            continue
        if not rec.get("gold_answer"):
            continue
        if rec["task_id"] in blocked_ids:
            continue
        if rec["question"].strip().lower() in blocked_q:
            continue
        out.append(rec)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--acq-per-dataset", type=int, default=ACQ_PER_DS)
    p.add_argument("--future-per-dataset", type=int, default=FUTURE_PER_DS)
    p.add_argument("--output", default=str(R54_ROOT / "split_r5_4.json"))
    args = p.parse_args()

    root = Path(args.output).parent
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))
    blocked_ids, blocked_q = r53_blocked()

    frozen = {
        "acquisition_pool": [],
        "future_probe": [],
        "bamboogle": [],
        "split_seed": int(args.seed),
        "sizes": {ds: {"acquisition_pool": args.acq_per_dataset, "future_probe": args.future_per_dataset} for ds in DATASETS},
        "excludes_r53": True,
        "protocol": "r5.4 holdout; zero task-id and exact-question overlap with R5.3",
    }
    r53 = json.loads((R53_ROOT / "splits" / "split.json").read_text(encoding="utf-8"))
    frozen["bamboogle"] = list(r53.get("bamboogle") or [])

    per_ds = {}
    fail = []
    for dataset in DATASETS:
        print(f"[r5.4 split] load {dataset}", flush=True)
        buckets = collect(dataset)
        train = unused(buckets.get("train") or [], blocked_ids, blocked_q)
        dev = unused(buckets.get("dev") or [], blocked_ids, blocked_q)
        rng.shuffle(train)
        rng.shuffle(dev)
        print(
            f"[r5.4 split] {dataset} unused train={len(train)} unused dev={len(dev)} "
            f"need acq={args.acq_per_dataset} future={args.future_per_dataset}",
            flush=True,
        )
        if len(train) < args.acq_per_dataset:
            fail.append(
                {
                    "dataset": dataset,
                    "bucket": "train_for_acquisition",
                    "remaining": len(train),
                    "need": args.acq_per_dataset,
                }
            )
            continue
        if len(dev) < args.future_per_dataset:
            fail.append(
                {
                    "dataset": dataset,
                    "bucket": "dev_for_future",
                    "remaining": len(dev),
                    "need": args.future_per_dataset,
                }
            )
            continue
        acq, train_left = take_stratified(train, args.acq_per_dataset, rng)
        fut, _ = take_stratified(dev, args.future_per_dataset, rng)
        acq_q = {r["question"].strip().lower() for r in acq}
        acq_id = {r["task_id"] for r in acq}
        fut = [r for r in fut if r["question"].strip().lower() not in acq_q and r["task_id"] not in acq_id]
        if len(fut) < args.future_per_dataset:
            extra = [
                r
                for r in unused(dev, blocked_ids | acq_id, blocked_q | acq_q)
                if r["task_id"] not in {x["task_id"] for x in fut}
            ]
            fut.extend(extra[: args.future_per_dataset - len(fut)])
        fut = fut[: args.future_per_dataset]
        if len(acq) < args.acq_per_dataset or len(fut) < args.future_per_dataset:
            fail.append(
                {
                    "dataset": dataset,
                    "remaining_acq": len(acq),
                    "remaining_future": len(fut),
                    "need_acq": args.acq_per_dataset,
                    "need_future": args.future_per_dataset,
                }
            )
            continue
        frozen["acquisition_pool"].extend(acq)
        frozen["future_probe"].extend(fut)
        per_ds[dataset] = {
            "unused_train": len(train),
            "unused_dev": len(dev),
            "acquisition_pool": len(acq),
            "future_probe": len(fut),
            "cap_hist_acq": dict(Counter(r["capability"] for r in acq)),
            "cap_hist_future": dict(Counter(r["capability"] for r in fut)),
        }
        print(f"[r5.4 split] {dataset} {per_ds[dataset]}", flush=True)

    if fail:
        print("FAIL_CLOSED insufficient unused samples", json.dumps(fail, indent=2), flush=True)
        raise SystemExit(2)

    path = Path(args.output)
    write_json(path, frozen)
    digest = sha256_file(path)
    write_json(root / "split_hashes.json", {"split_r5_4.json": digest, "split_seed": int(args.seed)})

    pairs = [
        overlap_report(frozen["acquisition_pool"], frozen["future_probe"], "r54-acq-future"),
        overlap_report(frozen["acquisition_pool"], list(r53.get("calibration") or []), "r54acq-r53cal"),
        overlap_report(frozen["acquisition_pool"], list(r53.get("acquisition_pool") or []), "r54acq-r53acq"),
        overlap_report(frozen["acquisition_pool"], list(r53.get("skill_validation") or []), "r54acq-r53val"),
        overlap_report(frozen["acquisition_pool"], list(r53.get("future_probe") or []), "r54acq-r53fut"),
        overlap_report(frozen["future_probe"], list(r53.get("calibration") or []), "r54fut-r53cal"),
        overlap_report(frozen["future_probe"], list(r53.get("acquisition_pool") or []), "r54fut-r53acq"),
        overlap_report(frozen["future_probe"], list(r53.get("skill_validation") or []), "r54fut-r53val"),
        overlap_report(frozen["future_probe"], list(r53.get("future_probe") or []), "r54fut-r53fut"),
    ]
    id_ok = all(p["task_id_overlap"] == 0 for p in pairs)
    q_ok = all(p["question_exact_overlap"] == 0 for p in pairs)
    audit = [
        "# SPLIT_AUDIT",
        "",
        f"split_seed = {args.seed}",
        f"hash(split_r5_4.json) = `{digest}`",
        "",
        "## Sizes",
        json.dumps(per_ds, indent=2),
        "",
        "## Leakage (task-id and exact-question must be 0 vs R5.3 and vs R5.4 acq/future)",
        f"- task-id overlap all pairs zero: {id_ok}",
        f"- question exact overlap all pairs zero: {q_ok}",
        "",
        json.dumps(pairs, indent=2),
        "",
        "R5.3 original split is untouched. This split is frozen. Do not resample from downstream accuracy.",
    ]
    (root / "SPLIT_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    write_json(root / "overlap.json", pairs)
    if not id_ok or not q_ok:
        raise SystemExit("split leak: task-id or question overlap is non-zero")
    print(f"[r5.4 split] wrote {path} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
