#!/usr/bin/env python3
"""Audit R7 split: zero task-id / exact-question overlap; no future_test in policy pools."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

from expolicy.closed_loop.constants import CAPS, DATASETS, R7_ROOT
from expolicy.closed_loop.io_utils import read_json, write_json

KEYS = ("calibration", "acquisition_pool", "shadow_reward_probe", "retention_probe", "future_test")


def ids(rows):
    return {r["task_id"] for r in rows}


def questions(rows):
    return {str(r["question"]).strip().lower() for r in rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default=str(R7_ROOT / "audit" / "SPLIT_R7.json"))
    args = p.parse_args()
    split = read_json(Path(args.split))
    buckets = {k: split.get(k) or [] for k in KEYS}
    lines = ["# R7 Split Audit", "", f"split_seed = {split.get('split_seed')}", ""]
    ok = True
    for a, b in combinations(KEYS, 2):
        ov_id = ids(buckets[a]) & ids(buckets[b])
        ov_q = questions(buckets[a]) & questions(buckets[b])
        status = "PASS" if not ov_id and not ov_q else "FAIL"
        if status == "FAIL":
            ok = False
        lines.append(f"- {a} ∩ {b}: task_id={len(ov_id)} question={len(ov_q)} **{status}**")
    lines.append("")
    lines.append("## Sizes")
    actual = split.get("actual_sizes") or {}
    for k, v in actual.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Capability mix")
    for k in KEYS:
        counts = {c: 0 for c in CAPS}
        ds = {d: 0 for d in DATASETS}
        for r in buckets[k]:
            counts[r.get("capability")] = counts.get(r.get("capability"), 0) + 1
            ds[r.get("dataset")] = ds.get(r.get("dataset"), 0) + 1
        lines.append(f"- {k}: caps={counts} datasets={ds}")
    lines.append("")
    lines.append("## Policy isolation")
    policy_ids = ids(buckets["calibration"]) | ids(buckets["acquisition_pool"]) | ids(buckets["shadow_reward_probe"]) | ids(buckets["retention_probe"])
    fut_ids = ids(buckets["future_test"])
    leak = policy_ids & fut_ids
    lines.append(f"- future_test ∩ policy-side task_id = {len(leak)} **{'PASS' if not leak else 'FAIL'}**")
    if leak:
        ok = False
    lines.append("")
    lines.append(f"## OVERALL: {'PASS' if ok else 'FAIL'}")
    text = "\n".join(lines) + "\n"
    out = Path(args.split).with_name("SPLIT_AUDIT.md")
    out.write_text(text, encoding="utf-8")
    write_json(Path(args.split).with_name("SPLIT_AUDIT.json"), {"pass": ok, "actual_sizes": actual})
    print(text)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
