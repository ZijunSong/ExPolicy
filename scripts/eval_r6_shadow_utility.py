#!/usr/bin/env python3
"""Precompute shadow/future metrics for every CountOnly activation mask."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from expolicy.environments.public_search.retriever import RetrieverClient
from expolicy.evaluation.shadow_probe import ALL_MASKS, eval_mask_on_tasks, mask_eval_path
from expolicy.experiments.common import mean, write_json
from expolicy.experiments.r5_3 import heartbeat, macro_from_rows, world_from_tasks
from expolicy.experiments.r5_4 import R54_ROOT, load_predictions, load_router, records_to_tasks
from expolicy.experiments.r5_4 import DATASETS
from expolicy.experiments.r6_ava import R6_ROOT, merged_base_rows
from expolicy.experiments.r_common import apply_gpu
from expolicy.logging.event_logger import append_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(R6_ROOT))
    p.add_argument("--split-name", default="shadow_probe", choices=["shadow_probe", "future_probe"])
    p.add_argument("--mask", default=None)
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--model", default="/data/ppnm/models/Qwen2.5-3B-Instruct")
    p.add_argument("--retriever-url", default=os.environ.get("EXPOLICY_RETRIEVER_URL", "http://127.0.0.1:18080"))
    return p.parse_args()


def wait_retriever(url: str, timeout: int = 600) -> None:
    client = RetrieverClient(url, timeout=10.0)
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            last = client.health()
            if last.get("ok"):
                print(f"[retriever] ok {last}", flush=True)
                return
        except Exception as exc:
            last = {"error": str(exc)}
        time.sleep(5)
    raise SystemExit(f"retriever not healthy: {last}")


def merge_mask(root: Path, split_name: str, mask: str) -> dict:
    work = root / "mask_eval" / split_name / "work" / mask.replace("+", "__")
    rows = []
    for path in sorted(work.glob("shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    # de-dup
    by = {}
    for rec in rows:
        by[str(rec["task_id"])] = rec
    rows = list(by.values())
    summary = macro_from_rows(rows)
    summary.update(
        {
            "mask": mask,
            "split": split_name,
            "n_skill_invocations": sum(1 for r in rows if str(r.get("dispatch_source") or "").startswith("skill:")),
            "n_base_fallback": sum(1 for r in rows if r.get("dispatch_source") == "base_llm"),
            "n_active_skills": 0 if mask == "none" else mask.count("+") + 1,
        }
    )
    dest = mask_eval_path(root, split_name, mask)
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest, {k: v for k, v in summary.items() if k != "rows"})
    with (work / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for rec in rows:
            handle.write(json.dumps(rec) + "\n")
    print(f"[merge] {split_name} {mask} n={len(rows)} macro={summary['macro']:.4f}", flush=True)
    return summary


def none_from_base(root: Path, split_name: str, tasks) -> dict:
    base = merged_base_rows(root)
    rows = []
    missing = []
    for t in tasks:
        rec = dict(base.get(t.task_id) or {})
        if not rec:
            missing.append(t.task_id)
            continue
        rec["task_id"] = t.task_id
        rec["dataset"] = (t.metadata or {}).get("dataset")
        rec["capability"] = t.capability
        rec["dispatch_source"] = "base_llm"
        rows.append(rec)
    if missing:
        raise SystemExit(f"base rows missing for {split_name}: {len(missing)}")
    summary = macro_from_rows(rows)
    summary.update({"mask": "none", "split": split_name, "n_skill_invocations": 0, "n_base_fallback": len(rows), "n_active_skills": 0})
    dest = mask_eval_path(root, split_name, "none")
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest, {k: v for k, v in summary.items() if k != "rows"})
    return summary


def main() -> None:
    args = parse_args()
    apply_gpu(args.gpu)
    root = Path(args.root)
    split = json.loads((root / "split_r6_ava.json").read_text(encoding="utf-8"))
    preds = load_predictions(R54_ROOT)
    router = load_router(R54_ROOT)
    recs = list(split.get(args.split_name) or [])
    tasks = records_to_tasks(recs, split=args.split_name, predictions=preds)
    masks = [args.mask] if args.mask else list(ALL_MASKS)
    if "none" in masks and args.shard == 0:
        none_from_base(root, args.split_name, tasks)
        masks = [m for m in masks if m != "none"]
    if not masks:
        print("[done] only none mask", flush=True)
        return
    wait_retriever(args.retriever_url)
    from expolicy.experiments.r5_3 import build_public_agent

    world = world_from_tasks(tasks)
    work_root = root / "mask_eval" / args.split_name / "work"
    agent, state, backend, env = build_public_agent(
        world, model_path=args.model, output_dir=work_root / f"agent_gpu_{args.gpu or 'x'}", retriever_url=args.retriever_url
    )
    base_rows = merged_base_rows(root)
    for mask in masks:
        dest = mask_eval_path(root, args.split_name, mask)
        if args.resume and dest.exists():
            print(f"[skip] {mask} already merged", flush=True)
            continue
        work = work_root / mask.replace("+", "__")
        work.mkdir(parents=True, exist_ok=True)
        print(f"[mask] {args.split_name} {mask} shard={args.shard}/{args.n_shards}", flush=True)
        eval_mask_on_tasks(
            tasks,
            mask_key=mask,
            base_rows=base_rows,
            env=env,
            model=backend,
            router=router,
            out=work,
            resume_path=work / f"shard_{args.shard}.jsonl",
            shard=args.shard,
            n_shards=args.n_shards,
        )
        (work / f"DONE_shard_{args.shard}").write_text("ok\n", encoding="utf-8")
        if args.n_shards == 1:
            merge_mask(root, args.split_name, mask)
        else:
            done = all((work / f"DONE_shard_{i}").exists() for i in range(args.n_shards))
            if done:
                merge_mask(root, args.split_name, mask)


if __name__ == "__main__":
    main()
