#!/usr/bin/env python3
"""Build isolated R5 acquisition / future / retention splits for seeds 0,1,2."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from expolicy.environments.search.world import Task, World
from expolicy.experiments.common import ROOT, load_world, write_json
from expolicy.experiments.patterns import gold_doc_key, normalized_pattern
from expolicy.experiments.r_common import add_common_args, apply_gpu

DEFAULT_OUT = str(ROOT / "outputs" / "r5_acquisition_skill")
TARGET_CAPS = {
    "s1_multi_hop": "multi_hop",
    "s2_temporal": "temporal_reasoning",
    "s4_cross_doc": "cross_document_synthesis",
}
FUTURE_PER_CAP = 12
RETENTION_N = 24


def _same_surface(a: Task, b: Task) -> bool:
    if a.task_id == b.task_id:
        return True
    if a.gold_answer and a.gold_answer == b.gold_answer and gold_doc_key(a) == gold_doc_key(b):
        return True
    if gold_doc_key(a) and gold_doc_key(a) == gold_doc_key(b):
        return True
    return False


def _split_rank(split: str, preferred: tuple[str, ...]) -> int:
    try:
        return preferred.index(split)
    except ValueError:
        return len(preferred)


def _pick_isolated(pool: list[Task], n: int, rng, *, preferred_splits: tuple[str, ...]) -> list[Task]:
    ranked = sorted(pool, key=lambda t: (_split_rank(t.split, preferred_splits), t.task_id))
    # shuffle within split rank bands
    bands: dict[int, list[Task]] = defaultdict(list)
    for t in ranked:
        bands[_split_rank(t.split, preferred_splits)].append(t)
    ordered: list[Task] = []
    for key in sorted(bands):
        items = list(bands[key])
        rng.shuffle(items)
        ordered.extend(items)
    chosen: list[Task] = []
    for task in ordered:
        if any(_same_surface(task, prev) for prev in chosen):
            continue
        chosen.append(task)
        if len(chosen) >= n:
            break
    return chosen


def build_seed_split(world: World, seed: int) -> dict:
    import random

    rng = random.Random(10_000 + seed)
    future: dict[str, list[str]] = {}
    notes: list[str] = []
    future_tasks: list[Task] = []
    for bucket, cap in TARGET_CAPS.items():
        pool = [t for t in world.tasks.values() if t.capability == cap]
        if len(pool) < FUTURE_PER_CAP:
            raise SystemExit(
                f"FAIL CLOSED: {cap} has only {len(pool)} tasks, need {FUTURE_PER_CAP} isolated future probes"
            )
        picked = _pick_isolated(pool, FUTURE_PER_CAP, rng, preferred_splits=("future", "validation", "deployment"))
        if len(picked) < FUTURE_PER_CAP:
            raise SystemExit(
                f"FAIL CLOSED: {cap} could not isolate {FUTURE_PER_CAP} future probes "
                f"(got {len(picked)}; unique gold={len({t.gold_answer for t in pool})})"
            )
        future[bucket] = [t.task_id for t in picked]
        future_tasks.extend(picked)
        notes.append(
            f"{bucket}: n={len(pool)} picked={len(picked)} unique_gold={len({t.gold_answer for t in picked})} "
            f"unique_docs={len({tuple(t.gold_doc_ids) for t in picked})} patterns={dict(Counter(normalized_pattern(t.task_id) for t in picked))}"
        )

    s0 = [t for t in world.tasks.values() if t.stage == 0]
    retention = _pick_isolated(s0, RETENTION_N, rng, preferred_splits=("retention", "validation", "future"))
    if len(retention) < RETENTION_N:
        raise SystemExit(f"FAIL CLOSED: S0 retention could not isolate {RETENTION_N} tasks (got {len(retention)})")

    blocked = {t.task_id for t in future_tasks} | {t.task_id for t in retention}
    # drop acquisition candidates that share gold+docs with future probes
    acq: list[str] = []
    dropped_surface = 0
    for task in sorted(world.tasks.values(), key=lambda t: t.task_id):
        if task.task_id in blocked:
            continue
        if any(_same_surface(task, src) for src in future_tasks):
            dropped_surface += 1
            continue
        acq.append(task.task_id)
    if dropped_surface:
        notes.append(f"dropped {dropped_surface} acquisition candidates that share gold/docs with future probes")

    by_cap = Counter(world.tasks[i].capability for i in acq)
    for bucket, cap in TARGET_CAPS.items():
        n_cap = by_cap.get(cap, 0)
        notes.append(f"acquisition_pool {cap}={n_cap}")
        if n_cap < 8:
            raise SystemExit(
                f"FAIL CLOSED: after isolating future probes, {cap} has only {n_cap} acquisition candidates (<8)"
            )

    overlap = blocked & set(acq)
    if overlap:
        raise SystemExit(f"FAIL CLOSED: split overlap {sorted(overlap)[:8]}")

    return {
        "seed": seed,
        "future_per_cap": FUTURE_PER_CAP,
        "retention_n": RETENTION_N,
        "target_caps": TARGET_CAPS,
        "future_probe": future,
        "future_probe_ids": [t.task_id for t in future_tasks],
        "retention_probe": [t.task_id for t in retention],
        "acquisition_pool": acq,
        "notes": notes,
        "counts": {
            "future": {k: len(v) for k, v in future.items()},
            "retention": len(retention),
            "acquisition_pool": len(acq),
            "acquisition_by_cap": dict(by_cap),
        },
    }


def write_audit(out: Path, splits: list[dict], world: World) -> str:
    lines = [
        "# R5 SPLIT_AUDIT",
        "",
        "Frozen before any R5 acquisition run:",
        "",
        f"- future_probe = {FUTURE_PER_CAP} × 3 capabilities = {3 * FUTURE_PER_CAP} (S4 Helion n=24 cannot isolate 16+8 strictly)",
        f"- retention_probe = {RETENTION_N} S0 tasks",
        "- acquisition_pool = remaining tasks, no replacement, no future/retention id overlap",
        "- methods in the same seed share these three sets",
        "",
    ]
    for row in splits:
        lines.append(f"## seed {row['seed']}")
        lines.append("")
        lines.append(f"- acquisition_pool n={row['counts']['acquisition_pool']}")
        lines.append(f"- future {row['counts']['future']}")
        lines.append(f"- retention n={row['counts']['retention']}")
        for note in row["notes"]:
            lines.append(f"- {note}")
        # cross-set leakage
        fut = set(row["future_probe_ids"])
        ret = set(row["retention_probe"])
        acq = set(row["acquisition_pool"])
        lines.append(f"- id overlap future∩acq={len(fut & acq)} future∩ret={len(fut & ret)} acq∩ret={len(acq & ret)}")
        gold_f = {world.tasks[i].gold_answer for i in fut}
        gold_a = {world.tasks[i].gold_answer for i in acq if world.tasks[i].capability in set(TARGET_CAPS.values())}
        lines.append(f"- target-cap gold-answer overlap future∩acq={len(gold_f & gold_a)}")
        lines.append("")
    s4_patterns = {normalized_pattern(t.task_id) for t in world.tasks.values() if t.capability == "cross_document_synthesis"}
    lines.append("## Known Helion limits")
    lines.append("")
    lines.append(f"- S4 has a single pattern family: {sorted(s4_patterns)}")
    lines.append("- Near-duplicate templates (entity swap only) cannot be fully eliminated for S4.")
    lines.append("- This is reported, not silently patched after seeing results.")
    lines.append("")
    text = "\n".join(lines) + "\n"
    (out / "SPLIT_AUDIT.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_args(p, default_out=DEFAULT_OUT)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    apply_gpu(args.gpu)
    out = Path(args.output_dir)
    split_dir = out / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    world = load_world(Path(args.world))
    rows = []
    for seed in args.seeds:
        row = build_seed_split(world, seed)
        write_json(split_dir / f"seed_{seed}.json", row)
        rows.append(row)
    audit = write_audit(out, rows, world)
    write_json(out / "splits" / "manifest.json", {"seeds": args.seeds, "future_per_cap": FUTURE_PER_CAP})
    print(audit)


if __name__ == "__main__":
    main()
