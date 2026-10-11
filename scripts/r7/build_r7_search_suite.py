#!/usr/bin/env python3
"""Build frozen R7 7-benchmark search suite. Never resample from results."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from expolicy.experiments.common import ROOT, write_json
from expolicy.r7.constants import (
    ACQUISITION_PER_DATASET,
    CALIBRATION_PER_DATASET,
    FUTURE_PER_DATASET,
    R7_CAPS,
    R7_DATA,
    R7_DATASETS,
    SKILL_VALIDATION_PER_DATASET,
    SPLIT_SEED,
)
from expolicy.r7.datasets_extra import apply_mapping_r7, example_to_record_r7, load_raw_r7


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(dataset: str) -> dict[str, list[dict]]:
    raw = load_raw_r7(dataset)
    out = {"train": [], "dev": [], "test": []}
    for split, rows in raw.items():
        key = "dev" if split in {"dev", "validation", "val"} else ("test" if split == "test" else "train")
        if split not in {"train", "dev", "validation", "val", "test"}:
            key = "train"
        for i, ex in enumerate(rows):
            rec = example_to_record_r7(dataset, split, ex, i)
            if not rec["question"]:
                continue
            out[key].append(rec)
    return {k: apply_mapping_r7(v) for k, v in out.items()}


def take_stratified(rows: list[dict], n: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    if n <= 0 or not rows:
        return [], list(rows)
    by_cap: dict[str, list[dict]] = defaultdict(list)
    for rec in rows:
        by_cap[rec.get("capability") or "unmapped"].append(rec)
    for cap in by_cap:
        rng.shuffle(by_cap[cap])
    chosen: list[dict] = []
    caps = [c for c in R7_CAPS if by_cap.get(c)]
    if not caps:
        rng.shuffle(rows)
        n_use = min(n, len(rows))
        return rows[:n_use], rows[n_use:]
    while len(chosen) < n:
        progress = False
        for cap in caps:
            bucket = by_cap[cap]
            if not bucket:
                continue
            chosen.append(bucket.pop())
            progress = True
            if len(chosen) >= n:
                break
        if not progress:
            break
    leftover = [r for cap in by_cap for r in by_cap[cap]]
    rng.shuffle(leftover)
    if len(chosen) < n:
        extra, leftover = leftover[: n - len(chosen)], leftover[n - len(chosen) :]
        chosen.extend(extra)
    return chosen, leftover


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(R7_DATASETS))
    p.add_argument("--retriever", default="e5")
    p.add_argument("--wiki-index", dest="wiki_index", default="existing")
    p.add_argument("--split-seed", dest="split_seed", type=int, default=SPLIT_SEED)
    p.add_argument("--calibration-per-dataset", type=int, default=CALIBRATION_PER_DATASET)
    p.add_argument("--acquisition-per-dataset", type=int, default=ACQUISITION_PER_DATASET)
    p.add_argument("--skill-validation-per-dataset", type=int, default=SKILL_VALIDATION_PER_DATASET)
    p.add_argument("--future-per-dataset", type=int, default=FUTURE_PER_DATASET)
    p.add_argument("--bamboogle-use-full-if-insufficient", action="store_true")
    p.add_argument("--out", default=str(R7_DATA))
    args = p.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.split_seed)

    frozen = {
        "calibration": [],
        "acquisition_pool": [],
        "skill_validation": [],
        "future_probe": [],
        "split_seed": args.split_seed,
        "retriever": args.retriever,
        "wiki_index": args.wiki_index,
        "actual_n": {},
    }
    notes = {}
    for dataset in args.datasets:
        print(f"[split] load {dataset}", flush=True)
        buckets = collect(dataset)
        train = [r for r in (buckets.get("train") or []) if r.get("capability") != "unmapped" and r.get("gold_answer")]
        dev = [r for r in (buckets.get("dev") or []) if r.get("capability") != "unmapped" and r.get("gold_answer")]
        test = [r for r in (buckets.get("test") or []) if r.get("question") and r.get("gold_answer") and r.get("capability") != "unmapped"]
        print(f"[split] {dataset} train={len(train)} dev={len(dev)} test={len(test)}", flush=True)
        if dataset == "bamboogle":
            rows = [r for r in (test or dev or train) if r.get("question") and r.get("gold_answer")]
            rng.shuffle(rows)
            need = args.future_per_dataset
            if args.bamboogle_use_full_if_insufficient or len(rows) < need:
                taken = rows
                notes[dataset] = f"used_all n={len(taken)}; no replacement sampling"
            else:
                taken = rows[:need]
                notes[dataset] = f"future={len(taken)}"
            for r in taken:
                r["r7_split"] = "future_probe"
            frozen["future_probe"].extend(taken)
            frozen["actual_n"][dataset] = {"future_probe": len(taken), "source": "all_available" if len(rows) <= need else "sampled"}
            continue
        # If train/dev are empty (e.g. PopQA test-only), partition the only legal pool.
        if not train and not dev and test:
            notes[dataset] = f"test_only_partition n={len(test)}; no replacement sampling"
            rng.shuffle(test)
            calib, rest = take_stratified(test, min(args.calibration_per_dataset, max(0, len(test) // 8)), rng)
            acq, rest = take_stratified(rest, min(args.acquisition_per_dataset, max(0, int(len(rest) * 0.55))), rng)
            val, rest = take_stratified(rest, min(args.skill_validation_per_dataset, max(0, len(rest) // 3)), rng)
            fut, _ = take_stratified(rest, min(args.future_per_dataset, len(rest)), rng)
            actual = {
                "calibration": len(calib),
                "acquisition_pool": len(acq),
                "skill_validation": len(val),
                "future_probe": len(fut),
                "source": "test_only_partition",
            }
            for rec, split in (
                (calib, "calibration"),
                (acq, "acquisition_pool"),
                (val, "skill_validation"),
                (fut, "future_probe"),
            ):
                for r in rec:
                    r["r7_split"] = split
                frozen[split].extend(rec)
            frozen["actual_n"][dataset] = actual
            print(f"[split] {dataset} {actual}", flush=True)
            continue
        # calibration + acquisition from train; skill_validation + future from dev (or leftover train)
        rng.shuffle(train)
        calib, rest = take_stratified(train, args.calibration_per_dataset, rng)
        acq, rest = take_stratified(rest, args.acquisition_per_dataset, rng)
        val_src = list(dev) if dev else list(rest)
        rng.shuffle(val_src)
        val, val_left = take_stratified(val_src, args.skill_validation_per_dataset, rng)
        fut_src = list(val_left)
        if len(fut_src) < args.future_per_dataset and rest:
            fut_src.extend(rest)
        fut, _ = take_stratified(fut_src, args.future_per_dataset, rng)
        actual = {
            "calibration": len(calib),
            "acquisition_pool": len(acq),
            "skill_validation": len(val),
            "future_probe": len(fut),
        }
        if len(fut) < args.future_per_dataset:
            notes[dataset] = f"future n={len(fut)} < requested {args.future_per_dataset}; used all legal samples; no replacement"
        for rec, split in (
            (calib, "calibration"),
            (acq, "acquisition_pool"),
            (val, "skill_validation"),
            (fut, "future_probe"),
        ):
            for r in rec:
                r["r7_split"] = split
            frozen[split].extend(rec)
        frozen["actual_n"][dataset] = actual
        print(f"[split] {dataset} {actual}", flush=True)

    split_path = root / "split.json"
    write_json(split_path, frozen)
    (root / "NOTES.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    (root / "SPLIT_SEED.txt").write_text(str(args.split_seed) + "\n", encoding="utf-8")
    print(f"wrote {split_path} sha256={sha256_file(split_path)}", flush=True)


if __name__ == "__main__":
    main()
