#!/usr/bin/env python3
"""Build the frozen R7 policy-learning split. Hash once; never resample from results."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from expolicy.closed_loop.constants import CAPS, DATASETS, R7_ROOT, SPLIT_SEED, TARGET_SIZES
from expolicy.closed_loop.io_utils import sha256_json, write_json
from expolicy.environments.public_search.capability_map import apply_mapping
from expolicy.environments.public_search.datasets import example_to_record, load_raw


KEEP = (
    "task_id",
    "dataset",
    "official_split",
    "question",
    "gold_answer",
    "aliases",
    "type",
    "hops",
    "supporting_titles",
    "raw_id",
    "capability",
    "capability_reason",
)


def slim(rec: dict) -> dict:
    out = {k: rec.get(k) for k in KEEP}
    out["question"] = str(out.get("question") or "").strip()
    out["gold_answer"] = str(out.get("gold_answer") or "").strip()
    return out


def collect() -> dict[str, list[dict]]:
    by_ds: dict[str, list[dict]] = {ds: [] for ds in DATASETS}
    for dataset in DATASETS:
        raw = load_raw(dataset)
        rows = []
        for split, items in raw.items():
            key = "dev" if split in {"dev", "validation", "val"} else ("test" if split == "test" else "train")
            for i, ex in enumerate(items):
                rec = example_to_record(dataset, key, ex, i)
                if not rec["question"] or not rec["gold_answer"]:
                    continue
                rows.append(rec)
        mapped = apply_mapping(rows)
        by_ds[dataset] = [slim(r) for r in mapped if r.get("capability") in CAPS]
    return by_ds


def take(rows: list[dict], n: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n], rows[n:]


def take_per_cap(rows: list[dict], n_per_cap: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    by_cap: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cap[r["capability"]].append(r)
    chosen: list[dict] = []
    leftover: list[dict] = []
    actual: dict[str, int] = {}
    for cap in CAPS:
        bucket = list(by_cap.get(cap) or [])
        rng.shuffle(bucket)
        take_n = min(n_per_cap, len(bucket))
        actual[cap] = take_n
        chosen.extend(bucket[:take_n])
        leftover.extend(bucket[take_n:])
    rng.shuffle(chosen)
    rng.shuffle(leftover)
    return chosen, leftover, actual  # type: ignore[return-value]


def overlap(a: list[dict], b: list[dict]) -> dict:
    ids_a, ids_b = {r["task_id"] for r in a}, {r["task_id"] for r in b}
    q_a, q_b = {r["question"].strip().lower() for r in a}, {r["question"].strip().lower() for r in b}
    return {
        "task_id_overlap": len(ids_a & ids_b),
        "question_overlap": len(q_a & q_b),
        "n_a": len(a),
        "n_b": len(b),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--output-dir", default=str(R7_ROOT / "audit"))
    args = p.parse_args()
    rng = random.Random(int(args.seed))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_ds = collect()
    # Prefer official dev for future_test, train for policy learning.
    future: list[dict] = []
    remaining: list[dict] = []
    future_actual: dict[str, int] = {}
    for ds in DATASETS:
        rows = by_ds[ds]
        dev = [r for r in rows if r.get("official_split") in {"dev", "validation", "val", "test"}]
        train = [r for r in rows if r not in dev]
        if len(dev) < TARGET_SIZES["future_test_per_dataset"]:
            pool = dev + train
            fut, rest = take(pool, min(TARGET_SIZES["future_test_per_dataset"], len(pool)), rng)
        else:
            fut, rest_dev = take(dev, TARGET_SIZES["future_test_per_dataset"], rng)
            rest = rest_dev + train
        future.extend(fut)
        remaining.extend(rest)
        future_actual[ds] = len(fut)

    used_q = {r["question"].strip().lower() for r in future}
    used_id = {r["task_id"] for r in future}
    remaining = [r for r in remaining if r["task_id"] not in used_id and r["question"].strip().lower() not in used_q]

    cal, remaining, cal_actual = take_per_cap(remaining, TARGET_SIZES["calibration_per_cap"], rng)
    used_q.update(r["question"].strip().lower() for r in cal)
    used_id.update(r["task_id"] for r in cal)
    remaining = [r for r in remaining if r["task_id"] not in used_id and r["question"].strip().lower() not in used_q]

    acq, remaining, acq_actual = take_per_cap(remaining, TARGET_SIZES["acquisition_pool_per_cap"], rng)
    used_q.update(r["question"].strip().lower() for r in acq)
    used_id.update(r["task_id"] for r in acq)
    remaining = [r for r in remaining if r["task_id"] not in used_id and r["question"].strip().lower() not in used_q]

    shadow, remaining, shadow_actual = take_per_cap(remaining, TARGET_SIZES["shadow_reward_probe_per_cap"], rng)
    used_q.update(r["question"].strip().lower() for r in shadow)
    used_id.update(r["task_id"] for r in shadow)
    remaining = [r for r in remaining if r["task_id"] not in used_id and r["question"].strip().lower() not in used_q]

    ret_n = min(TARGET_SIZES["retention_probe"], len(remaining))
    retention, leftover = take(remaining, ret_n, rng)

    payload = {
        "split_seed": int(args.seed),
        "target_sizes": TARGET_SIZES,
        "actual_sizes": {
            "calibration": len(cal),
            "calibration_per_cap": cal_actual,
            "acquisition_pool": len(acq),
            "acquisition_pool_per_cap": acq_actual,
            "shadow_reward_probe": len(shadow),
            "shadow_per_cap": shadow_actual,
            "retention_probe": len(retention),
            "future_test": len(future),
            "future_test_per_dataset": future_actual,
            "leftover": len(leftover),
        },
        "fail_closed_undersized": any(
            v < TARGET_SIZES["calibration_per_cap"] for v in cal_actual.values()
        )
        or any(v < TARGET_SIZES["acquisition_pool_per_cap"] for v in acq_actual.values())
        or any(v < TARGET_SIZES["shadow_reward_probe_per_cap"] for v in shadow_actual.values())
        or any(v < TARGET_SIZES["future_test_per_dataset"] for v in future_actual.values())
        or len(retention) < TARGET_SIZES["retention_probe"],
        "calibration": cal,
        "acquisition_pool": acq,
        "shadow_reward_probe": shadow,
        "retention_probe": retention,
        "future_test": future,
    }
    path = out / "SPLIT_R7.json"
    write_json(path, payload)
    payload["split_hash"] = sha256_json(
        {k: payload[k] for k in ("split_seed", "calibration", "acquisition_pool", "shadow_reward_probe", "retention_probe", "future_test")}
    )
    write_json(path, payload)
    print("wrote", path, "hash", payload["split_hash"][:16], "actual", payload["actual_sizes"])


if __name__ == "__main__":
    main()
