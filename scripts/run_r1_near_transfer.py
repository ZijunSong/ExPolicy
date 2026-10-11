#!/usr/bin/env python3
"""R1: near-transfer utility of no-memory vs legacy episodic vs capability-aware episodic."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from expolicy.agents.persistent_agent import MemoryFacade
from expolicy.assimilation.abstractor import ExperienceAbstractor
from expolicy.assimilation.carriers.episodic_memory import EpisodicMemory
from expolicy.assimilation.carriers.semantic_memory import SemanticMemory
from expolicy.assimilation.verifier import ExperienceVerifier
from expolicy.core.experience import Experience
from expolicy.core.state import empty_agent
from expolicy.experiments.common import ROOT, build_static_agent, load_world, mean, write_json
from expolicy.experiments.patterns import TARGET_CAPS
from expolicy.experiments.r_common import (
    add_common_args,
    apply_gpu,
    capability_utility,
    dump_probe_rows,
    eval_probe,
    load_e0_experiences,
    load_batches_file,
    mark_done,
    metrics_without_rows,
    resume_done,
    write_run_header,
)
from expolicy.logging.event_logger import append_jsonl
from expolicy.models.embedder import Embedder

DEFAULT_OUT = str(ROOT / "outputs" / "r1_near_transfer")
CONDITIONS = ("no_memory", "old_episodic", "new_episodic")


def _exps_for_batch(all_exps: list[Experience], row: dict) -> list[Experience]:
    wanted = {item["experience_id"] for item in row["experiences"]}
    by_id = {e.experience_id: e for e in all_exps}
    return [by_id[i] for i in wanted if i in by_id]


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_args(p, default_out=DEFAULT_OUT)
    p.add_argument("--bucket", default=None, help="s1_multi_hop|s2_temporal|s4_cross_doc")
    p.add_argument("--probes", default=None)
    args = p.parse_args()
    apply_gpu(args.gpu)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    probes_path = Path(args.probes) if args.probes else out / "probes" / "batches.json"
    if not probes_path.exists():
        raise SystemExit(f"missing probes file {probes_path}; run scripts/build_near_transfer_probes.py first")

    write_run_header(
        out,
        config={
            "script": "run_r1_near_transfer.py",
            "seed": args.seed,
            "model": args.model,
            "gpu": args.gpu,
            "bucket": args.bucket,
            "conditions": list(CONDITIONS),
        },
    )
    world = load_world(Path(args.world))
    batches = load_batches_file(probes_path, world)
    if args.bucket:
        batches = [b for b in batches if b["bucket"] == args.bucket]
        if not batches:
            raise SystemExit(f"no batches for bucket {args.bucket}")
    all_exps = load_e0_experiences()
    embedder = Embedder(args.embedder)
    verifier = ExperienceVerifier(world)
    abstractor = ExperienceAbstractor()
    agent, base_state = build_static_agent(world, model_path=args.model, output_dir=out / "base")
    t0 = time.time()
    done = resume_done(out, "completed") if args.resume else set()
    summary_rows = []

    for row in batches:
        batch_id = row["batch_id"]
        if batch_id in done:
            print(f"skip {batch_id} (resume)")
            continue
        experiences = _exps_for_batch(all_exps, row)
        probes = {
            "near": row["near_tasks"],
            "far": row["far_tasks"],
            "retention": row["retention_tasks"],
        }
        batch_dir = out / "batches" / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        write_json(batch_dir / "experiences.json", row["experiences"])

        # Shared no-memory baseline for this probe set
        agent.memory_retriever = None
        agent.skill_store = None
        agent.use_executable_skill = False
        agent.memory_mode = "hybrid"
        before = {name: eval_probe(agent, base_state, tasks) for name, tasks in probes.items()}
        write_json(batch_dir / "before_metrics.json", {k: metrics_without_rows(v) for k, v in before.items()})
        for split, payload in before.items():
            dump_probe_rows(out / "trajectories.jsonl", split=split, condition="no_memory", batch_id=batch_id, rows=payload["rows"])

        after_by_cond = {"no_memory": before}
        for cond in ("old_episodic", "new_episodic"):
            child = empty_agent(batch_dir / cond, model_name=args.model, version="agent_v001")
            child.harness = base_state.harness
            child.write_manifest()
            episodic = EpisodicMemory(child.episodic_dir(), embedder)
            semantic = SemanticMemory(child.semantic_dir())
            for exp in experiences:
                abs_ = abstractor.abstract(exp, verifier.verify(exp))
                episodic.add(exp, abs_, world=world)
            agent.memory_retriever = MemoryFacade(episodic, semantic)
            agent.memory_mode = "legacy" if cond == "old_episodic" else "hybrid"
            after = {name: eval_probe(agent, child, tasks) for name, tasks in probes.items()}
            after_by_cond[cond] = after
            write_json(batch_dir / f"after_{cond}.json", {k: metrics_without_rows(v) for k, v in after.items()})
            for split, payload in after.items():
                dump_probe_rows(out / "trajectories.jsonl", split=split, condition=cond, batch_id=batch_id, rows=payload["rows"])
            agent.memory_retriever = None

        rec = {
            "batch_id": batch_id,
            "bucket": row["bucket"],
            "n_exp": len(experiences),
            "conditions": {},
        }
        for cond, after in after_by_cond.items():
            deltas = {name: after[name]["em"] - before[name]["em"] for name in probes}
            rec["conditions"][cond] = {
                "em": {name: after[name]["em"] for name in probes},
                "tokens": {name: after[name]["tokens"] for name in probes},
                "retrieval_hit_rate": {name: after[name].get("retrieval_hit_rate", 0.0) for name in probes},
                "memory_used_rate": {name: after[name].get("memory_used_rate", 0.0) for name in probes},
                "delta": deltas,
                "near_transfer_gain": deltas["near"],
                "U_capability": capability_utility(deltas["near"], deltas["far"], deltas["retention"]),
            }
        write_json(batch_dir / "after_metrics.json", rec["conditions"])
        write_json(batch_dir / "summary.json", rec)
        append_jsonl(out / "interventions.jsonl", rec)
        summary_rows.append(rec)
        mark_done(out, "completed", batch_id, extra={"last": batch_id})
        print(json.dumps({k: rec["conditions"][k]["near_transfer_gain"] for k in rec["conditions"]} | {"batch": batch_id}, indent=2))

    # If resume, reload previous rows
    if (out / "interventions.jsonl").exists() and not summary_rows:
        for line in (out / "interventions.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                summary_rows.append(json.loads(line))
    elif (out / "interventions.jsonl").exists() and args.resume:
        loaded = []
        for line in (out / "interventions.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                loaded.append(json.loads(line))
        have = {r["batch_id"] for r in summary_rows}
        summary_rows = [r for r in loaded if r["batch_id"] not in have] + summary_rows

    bucket_gain: dict[str, dict[str, list[float]]] = {b: {c: [] for c in CONDITIONS} for b in TARGET_CAPS}
    for rec in summary_rows:
        for cond, payload in rec["conditions"].items():
            bucket_gain[rec["bucket"]][cond].append(payload["near_transfer_gain"])
    go = {}
    for bucket, conds in bucket_gain.items():
        new_gains = conds.get("new_episodic") or []
        if not new_gains:
            continue
        go[bucket] = {
            "mean_near_gain_new": mean(new_gains),
            "mean_near_gain_old": mean(conds.get("old_episodic") or []),
            "n_batches": len(new_gains),
            "direction_positive": sum(1 for g in new_gains if g > 0) / len(new_gains),
        }
    n_pos_buckets = sum(1 for v in go.values() if v["mean_near_gain_new"] > 0)
    summary = {
        "n_batches": len(summary_rows),
        "by_bucket": go,
        "go_at_least_2_of_3_positive": n_pos_buckets >= 2,
        "wall_seconds": time.time() - t0,
        "gpu": args.gpu,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
