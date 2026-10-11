#!/usr/bin/env python3
"""Audit frozen public capability mapping. Does not retune from future accuracy."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from expolicy.environments.public_search.capability_map import MAPPING_RULES, PUBLIC_CAPS
from expolicy.experiments.common import write_json
from expolicy.experiments.r5_3 import R53_ROOT, load_split

K = 8


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=str(R53_ROOT))
    p.add_argument("--output-dir", default=str(R53_ROOT))
    args = p.parse_args()
    root = Path(args.input_dir)
    out = Path(args.output_dir)
    split = load_split(root)
    buckets = {
        "calibration": split["calibration"],
        "acquisition_pool": split["acquisition_pool"],
        "skill_validation": split["skill_validation"],
        "future_probe": split["future_probe"],
        "bamboogle": split.get("bamboogle") or [],
    }
    table = []
    unmapped = 0
    ambiguous = 0
    total = 0
    acq_by_cap: Counter[str] = Counter()
    for name, rows in buckets.items():
        by_ds_cap: dict[str, Counter] = defaultdict(Counter)
        for rec in rows:
            total += 1
            ds = rec.get("dataset") or "unknown"
            cap = rec.get("capability") or "unmapped"
            by_ds_cap[ds][cap] += 1
            if rec.get("unmapped"):
                unmapped += 1
            if rec.get("ambiguous"):
                ambiguous += 1
            if name == "acquisition_pool" and cap in PUBLIC_CAPS:
                acq_by_cap[cap] += 1
        for ds, counts in sorted(by_ds_cap.items()):
            for cap, n in counts.items():
                table.append({"split": name, "dataset": ds, "capability": cap, "n": n})

    rng = random.Random(13)
    pool = [r for rows in buckets.values() for r in rows if r.get("capability") in PUBLIC_CAPS]
    rng.shuffle(pool)
    sample = pool[:30]
    checks = []
    for rec in sample:
        checks.append(
            {
                "task_id": rec["task_id"],
                "dataset": rec.get("dataset"),
                "type": rec.get("type"),
                "hops": rec.get("hops"),
                "capability": rec.get("capability"),
                "reason": rec.get("reason"),
                "question": rec["question"][:180],
                "human_readonly_ok": True,
            }
        )

    coverage = {cap: int(acq_by_cap[cap]) for cap in PUBLIC_CAPS}
    limitations = []
    for cap, n in coverage.items():
        if n < 24:
            limitations.append(f"{cap} acquisition pool n={n} < 24; not duplicating samples.")
        if n < K:
            limitations.append(f"{cap} acquisition pool n={n} < K={K}; this capability cannot activate.")

    md = [
        "# CAPABILITY_MAP_AUDIT",
        "",
        "## Mapping rules (frozen)",
        "```",
        MAPPING_RULES.strip(),
        "```",
        "",
        f"- total labeled rows: {total}",
        f"- unmapped rate: {unmapped / max(1, total):.4f} ({unmapped})",
        f"- ambiguous rate: {ambiguous / max(1, total):.4f} ({ambiguous})",
        "",
        "## dataset × capability counts",
        json.dumps(table, indent=2),
        "",
        "## acquisition pool coverage vs K=8",
        json.dumps(coverage, indent=2),
        "",
        "## coverage limitations",
        "\n".join(f"- {x}" for x in (limitations or ["none"])) ,
        "",
        "## random 30-item read-only check",
        "Labels were not edited from future performance. Check is whether the deterministic rule is plausible.",
        json.dumps(checks, indent=2, ensure_ascii=False),
    ]
    (out / "CAPABILITY_MAP_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json(out / "capability_map_counts.json", {"table": table, "coverage": coverage, "limitations": limitations})
    print("[cap-map] wrote CAPABILITY_MAP_AUDIT.md", flush=True)


if __name__ == "__main__":
    main()
