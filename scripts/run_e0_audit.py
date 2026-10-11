#!/usr/bin/env python3
"""E0: static capability-gap audit. No training. One process per stage (or --all)."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from expolicy.evaluation.metrics import aggregate
from expolicy.experiments.common import (
    ROOT,
    build_static_agent,
    load_world,
    mean,
    sample_stage_tasks,
    score_with_retrieval,
    write_json,
)
from expolicy.logging.event_logger import append_jsonl

OUT = ROOT / "outputs" / "e0_audit"


def run_stage(stage: int, n: int, seed: int, model: str) -> dict:
    out = OUT / f"stage_{stage}"
    out.mkdir(parents=True, exist_ok=True)
    world = load_world()
    tasks = sample_stage_tasks(world, stage, n, seed=seed)
    write_json(out / "task_ids.json", [t.task_id for t in tasks])
    agent, state = build_static_agent(world, model_path=model, output_dir=out)
    rows = []
    started = time.time()
    traj = out / "trajectories.jsonl"
    exps = out / "experiences.jsonl"
    if traj.exists():
        traj.unlink()
    if exps.exists():
        exps.unlink()
    for i, task in enumerate(tasks, 1):
        exp = agent.run_task(task, state=state)
        row = score_with_retrieval(exp, task)
        rows.append(row)
        append_jsonl(traj, row)
        append_jsonl(exps, json.loads(exp.model_dump_json()))
        if i % 10 == 0 or i == len(tasks):
            live = {
                "stage": stage,
                "done": i,
                "n": len(tasks),
                "em_so_far": mean(r["em_strict"] for r in rows),
                "success_so_far": mean(r["success"] for r in rows),
            }
            write_json(out / "STATUS_LIVE.json", live)
            (out / "STATUS_LIVE.md").write_text(
                f"stage={stage} {i}/{len(tasks)} EM={live['em_so_far']:.3f}\n",
                encoding="utf-8",
            )
    summary = aggregate(rows)
    summary.update(
        {
            "stage": stage,
            "n_tasks": len(tasks),
            "em_strict": mean(r["em_strict"] for r in rows),
            "search_recall": mean(r["search_recall"] for r in rows),
            "open_recall": mean(r["open_recall"] for r in rows),
            "evidence_recall": mean(r["evidence_recall"] for r in rows),
            "avg_tool_calls": mean(r["tool_calls"] for r in rows),
            "avg_hops": mean(r["hops"] for r in rows),
            "failure_types": dict(Counter(r["failure_type"] for r in rows if r.get("failure_type"))),
            "by_split": _group(rows, "split"),
            "by_capability": _group(rows, "capability"),
            "by_hops": _group(rows, "hops"),
            "wall_seconds": time.time() - started,
            "model": model,
            "seed": seed,
            "error_rate": 1.0 - mean(r["em_strict"] for r in rows),
        }
    )
    write_json(out / "summary.json", summary)
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    return summary


def run_passk(*, k: int, temperature: float, per_stage: int, seed: int, model: str) -> dict:
    world = load_world()
    out = OUT / "passk"
    out.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[int, str]] = []
    for stage in range(5):
        traj = OUT / f"stage_{stage}" / "trajectories.jsonl"
        if not traj.exists():
            continue
        for line in traj.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if float(row.get("em_strict") or 0) < 0.5:
                failures.append((stage, row["task_id"]))
    rng = __import__("random").Random(seed)
    sampled: list[tuple[int, str]] = []
    by_stage: dict[int, list[str]] = defaultdict(list)
    for stage, tid in failures:
        by_stage[stage].append(tid)
    for stage, ids in by_stage.items():
        rng.shuffle(ids)
        sampled.extend((stage, tid) for tid in ids[:per_stage])
    write_json(out / "sampled_failures.json", [{"stage": s, "task_id": t} for s, t in sampled])
    agent, state = build_static_agent(
        world, model_path=model, output_dir=out, temperature=temperature
    )
    records = []
    for stage, tid in sampled:
        task = world.tasks[tid]
        hits = 0
        preds = []
        for trial in range(k):
            exp = agent.run_task(task, state=state)
            ok = bool(exp.success)
            hits += int(ok)
            preds.append(exp.predicted_answer)
        rec = {
            "task_id": tid,
            "stage": stage,
            "capability": task.capability,
            "pass_at_k": hits > 0,
            "k_successes": hits,
            "k": k,
            "preds": preds,
        }
        records.append(rec)
        append_jsonl(out / "passk.jsonl", rec)
        write_json(out / "STATUS_LIVE.json", {"done": len(records), "n": len(sampled)})
    n = len(records)
    summary = {
        "n_failures_sampled": n,
        "k": k,
        "temperature": temperature,
        "pass_at_k_rate": mean(float(r["pass_at_k"]) for r in records) if n else 0.0,
        "mean_k_successes": mean(float(r["k_successes"]) for r in records) if n else 0.0,
        "unstable_rate": mean(float(r["pass_at_k"] and r["k_successes"] < k) for r in records) if n else 0.0,
        "hard_gap_rate": mean(float(not r["pass_at_k"]) for r in records) if n else 0.0,
        "by_stage": {},
    }
    for stage in range(5):
        sub = [r for r in records if r["stage"] == stage]
        if not sub:
            continue
        summary["by_stage"][str(stage)] = {
            "n": len(sub),
            "pass_at_k": mean(float(r["pass_at_k"]) for r in sub),
            "hard_gap": mean(float(not r["pass_at_k"]) for r in sub),
        }
    write_json(out / "summary.json", summary)
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    return summary


def aggregate_stages() -> dict:
    stages = []
    all_rows = []
    for stage in range(5):
        p = OUT / f"stage_{stage}" / "summary.json"
        if not p.exists():
            continue
        stages.append(json.loads(p.read_text(encoding="utf-8")))
        traj = OUT / f"stage_{stage}" / "trajectories.jsonl"
        for line in traj.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_rows.append(json.loads(line))
    overall_em = mean(r["em_strict"] for r in all_rows) if all_rows else 0.0
    payload = {
        "n_total": len(all_rows),
        "overall_em": overall_em,
        "overall_success": mean(r["success"] for r in all_rows) if all_rows else 0.0,
        "frontier_error_rate": 1.0 - overall_em,
        "search_recall": mean(r["search_recall"] for r in all_rows) if all_rows else 0.0,
        "open_recall": mean(r["open_recall"] for r in all_rows) if all_rows else 0.0,
        "evidence_recall": mean(r["evidence_recall"] for r in all_rows) if all_rows else 0.0,
        "avg_tool_calls": mean(r["tool_calls"] for r in all_rows) if all_rows else 0.0,
        "failure_types": dict(Counter(r["failure_type"] for r in all_rows if r.get("failure_type"))),
        "stages": stages,
        "gate_a_band": _gate(overall_em),
    }
    write_json(OUT / "E0_SUMMARY.json", payload)
    return payload


def _gate(em: float) -> str:
    err = 1.0 - em
    if em > 0.90:
        return "TOO_EASY_NEED_HARDER_SPLIT"
    if em < 0.50:
        return "TOO_HARD_MAYBE_BEYOND_FRONTIER"
    if 0.25 <= err <= 0.45 or 0.55 <= em <= 0.75:
        return "IDEAL_FRONTIER"
    return "USABLE_FRONTIER"


def _group(rows: list[dict], key: str) -> dict:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        g[str(r.get(key))].append(r)
    return {
        k: {
            "n": len(v),
            "em": mean(x["em_strict"] for x in v),
            "search_recall": mean(x["search_recall"] for x in v),
            "avg_tools": mean(x["tool_calls"] for x in v),
        }
        for k, v in g.items()
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="/data/ppnm/models/Qwen2.5-3B-Instruct")
    p.add_argument("--passk", action="store_true")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--passk-per-stage", type=int, default=8)
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args()
    if args.aggregate:
        print(json.dumps(aggregate_stages(), indent=2))
        return
    if args.passk:
        print(json.dumps(run_passk(k=args.k, temperature=0.7, per_stage=args.passk_per_stage, seed=args.seed, model=args.model), indent=2))
        return
    if args.all:
        for stage in range(5):
            print(json.dumps(run_stage(stage, args.n, args.seed, args.model), indent=2))
        print(json.dumps(aggregate_stages(), indent=2))
        return
    if args.stage is None:
        raise SystemExit("pass --stage N, --all, --passk, or --aggregate")
    print(json.dumps(run_stage(args.stage, args.n, args.seed, args.model), indent=2))


if __name__ == "__main__":
    main()
