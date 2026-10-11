#!/usr/bin/env python3
"""Freeze R5.3 public-search splits. Hash once; never resample from results."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from expolicy.environments.public_search.capability_map import PUBLIC_CAPS, apply_mapping
from expolicy.environments.public_search.datasets import DATASET_NAMES, load_raw, example_to_record
from expolicy.experiments.common import ROOT, write_json
from expolicy.experiments.r5_3 import R53_ROOT

SPLIT_SEED = 13
SIZES = {
    "hotpotqa": {"calibration": 48, "acquisition_pool": 400, "skill_validation": 60, "future_probe": 100},
    "2wiki": {"calibration": 48, "acquisition_pool": 400, "skill_validation": 60, "future_probe": 100},
    "musique": {"calibration": 48, "acquisition_pool": 400, "skill_validation": 60, "future_probe": 100},
    "bamboogle": {"bamboogle": 125},
}
CAP_SPAN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3})\b")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entities(text: str) -> set[str]:
    return {m.group(1).strip().lower() for m in CAP_SPAN.finditer(text or "")}


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
    mapped = {k: apply_mapping(v) for k, v in out.items()}
    return mapped


def take_stratified(rows: list[dict], n: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    if n <= 0 or not rows:
        return [], list(rows)
    by_cap: dict[str, list[dict]] = defaultdict(list)
    for rec in rows:
        by_cap[rec.get("capability") or "unmapped"].append(rec)
    for cap in by_cap:
        rng.shuffle(by_cap[cap])
    chosen: list[dict] = []
    caps = [c for c in PUBLIC_CAPS if by_cap.get(c)]
    if not caps:
        rng.shuffle(rows)
        return rows[:n], rows[n:]
    i = 0
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
        i += 1
        if i > n + 10:
            break
    leftover = [r for cap in by_cap for r in by_cap[cap]]
    rng.shuffle(leftover)
    if len(chosen) < n:
        extra, leftover = leftover[: n - len(chosen)], leftover[n - len(chosen) :]
        chosen.extend(extra)
    return chosen, leftover


def overlap_report(a: list[dict], b: list[dict], name: str) -> dict:
    ids_a, ids_b = {r["task_id"] for r in a}, {r["task_id"] for r in b}
    q_a, q_b = {r["question"].strip().lower() for r in a}, {r["question"].strip().lower() for r in b}
    ans_a, ans_b = {r["gold_answer"].strip().lower() for r in a if r.get("gold_answer")}, {r["gold_answer"].strip().lower() for r in b if r.get("gold_answer")}
    ne_a = set().union(*(entities(r["question"]) for r in a)) if a else set()
    ne_b = set().union(*(entities(r["question"]) for r in b)) if b else set()
    titles_a = set()
    titles_b = set()
    for r in a:
        titles_a.update(t.lower() for t in (r.get("supporting_titles") or []))
    for r in b:
        titles_b.update(t.lower() for t in (r.get("supporting_titles") or []))
    return {
        "pair": name,
        "task_id_overlap": len(ids_a & ids_b),
        "question_exact_overlap": len(q_a & q_b),
        "answer_overlap": len(ans_a & ans_b),
        "named_entity_overlap": len(ne_a & ne_b),
        "supporting_title_overlap": len(titles_a & titles_b),
        "n_a": len(a),
        "n_b": len(b),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki", "musique", "bamboogle"])
    p.add_argument("--output-dir", default=str(R53_ROOT))
    args = p.parse_args()
    root = Path(args.output_dir)
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SPLIT_SEED)

    frozen = {
        "calibration": [],
        "acquisition_pool": [],
        "skill_validation": [],
        "future_probe": [],
        "bamboogle": [],
        "split_seed": SPLIT_SEED,
        "sizes": SIZES,
    }
    per_ds = {}
    for dataset in args.datasets:
        print(f"[split] load {dataset}", flush=True)
        buckets = collect(dataset)
        print(f"[split] loaded {dataset} train={len(buckets.get('train') or [])} dev={len(buckets.get('dev') or [])}", flush=True)
        if dataset == "bamboogle":
            rows = [r for r in (buckets.get("test") or buckets.get("dev") or buckets.get("train") or []) if r.get("question")]
            rng.shuffle(rows)
            frozen["bamboogle"] = rows[: SIZES["bamboogle"]["bamboogle"]]
            per_ds[dataset] = {"n": len(frozen["bamboogle"]), "source": "all_available"}
            continue
        train = [r for r in buckets.get("train") or [] if r.get("capability") in PUBLIC_CAPS and r.get("gold_answer")]
        dev = [r for r in buckets.get("dev") or [] if r.get("capability") in PUBLIC_CAPS and r.get("gold_answer")]
        rng.shuffle(train)
        rng.shuffle(dev)
        sizes = SIZES[dataset]
        cal, train = take_stratified(train, sizes["calibration"], rng)
        val, train = take_stratified(train, sizes["skill_validation"], rng)
        acq, train = take_stratified(train, sizes["acquisition_pool"], rng)
        future_src = dev if len(dev) >= sizes["future_probe"] else (dev + train)
        fut, _ = take_stratified(future_src, sizes["future_probe"], rng)
        # hard leak check vs acq/cal/val
        blocked_q = {r["question"].strip().lower() for r in cal + val + acq}
        blocked_id = {r["task_id"] for r in cal + val + acq}
        fut = [r for r in fut if r["question"].strip().lower() not in blocked_q and r["task_id"] not in blocked_id]
        if len(fut) < sizes["future_probe"]:
            fut_ids = {r["task_id"] for r in fut}
            extra = [
                r
                for r in future_src
                if r["task_id"] not in fut_ids
                and r["question"].strip().lower() not in blocked_q
                and r["task_id"] not in blocked_id
            ]
            fut.extend(extra[: sizes["future_probe"] - len(fut)])
        fut = fut[: sizes["future_probe"]]
        frozen["calibration"].extend(cal)
        frozen["skill_validation"].extend(val)
        frozen["acquisition_pool"].extend(acq)
        frozen["future_probe"].extend(fut)
        per_ds[dataset] = {
            "calibration": len(cal),
            "skill_validation": len(val),
            "acquisition_pool": len(acq),
            "future_probe": len(fut),
            "train_available": len(buckets.get("train") or []),
            "dev_available": len(buckets.get("dev") or []),
            "cap_hist_acq": dict(Counter(r["capability"] for r in acq)),
            "cap_hist_future": dict(Counter(r["capability"] for r in fut)),
        }
        print(f"[split] {dataset} {per_ds[dataset]}", flush=True)

    path = split_dir / "split.json"
    write_json(path, frozen)
    hashes = {"split.json": sha256_file(path)}
    write_json(root / "split_hashes.json", hashes)

    pairs = [
        overlap_report(frozen["calibration"], frozen["acquisition_pool"], "cal-acq"),
        overlap_report(frozen["calibration"], frozen["skill_validation"], "cal-val"),
        overlap_report(frozen["calibration"], frozen["future_probe"], "cal-future"),
        overlap_report(frozen["acquisition_pool"], frozen["skill_validation"], "acq-val"),
        overlap_report(frozen["acquisition_pool"], frozen["future_probe"], "acq-future"),
        overlap_report(frozen["skill_validation"], frozen["future_probe"], "val-future"),
        overlap_report(frozen["bamboogle"], frozen["acquisition_pool"], "bamb-acq"),
        overlap_report(frozen["bamboogle"], frozen["future_probe"], "bamb-future"),
    ]
    id_ok = all(p["task_id_overlap"] == 0 for p in pairs)
    q_ok = all(p["question_exact_overlap"] == 0 for p in pairs)
    audit = [
        "# SPLIT_AUDIT",
        "",
        f"split_seed = {SPLIT_SEED}",
        f"hash(split.json) = `{hashes['split.json']}`",
        "",
        "## Sizes",
        json.dumps(per_ds, indent=2),
        "",
        "## Leakage (task-id and question must be 0)",
        f"- task-id overlap all pairs zero: {id_ok}",
        f"- question exact overlap all pairs zero: {q_ok}",
        "",
        json.dumps(pairs, indent=2),
        "",
        "Answer / named-entity / supporting-title overlap are reported, not required to be zero.",
        "Split is frozen. Do not resample from downstream accuracy.",
    ]
    (root / "SPLIT_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    write_json(split_dir / "overlap.json", pairs)
    if not id_ok or not q_ok:
        raise SystemExit("split leak: task-id or question overlap is non-zero")
    print(f"[split] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
