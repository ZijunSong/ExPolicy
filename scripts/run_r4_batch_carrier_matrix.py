#!/usr/bin/env python3
"""R4: batch-level carrier utility matrix (episodic / semantic / skill / weight)."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from expolicy.agents.persistent_agent import MemoryFacade
from expolicy.assimilation.abstractor import ExperienceAbstractor
from expolicy.assimilation.carriers.episodic_memory import EpisodicMemory
from expolicy.assimilation.carriers.semantic_memory import SemanticMemory
from expolicy.assimilation.carriers.skill_store import SkillStore, builtin_executable_skills, run_skill_on_task
from expolicy.assimilation.carriers.weight_update import WeightUpdater
from expolicy.assimilation.verifier import ExperienceVerifier
from expolicy.core.experience import Experience
from expolicy.core.state import empty_agent
from expolicy.experiments.common import ROOT, build_static_agent, load_world, mean, write_json
from expolicy.experiments.patterns import BUCKET_PRIMARY_CAP, TARGET_CAPS
from expolicy.experiments.probes import bucket_pool, build_probe_set, split_experiences_into_batches
from expolicy.experiments.r_common import (
    add_common_args,
    apply_gpu,
    capability_utility,
    dump_probe_rows,
    eval_probe,
    load_e0_experiences,
    mark_done,
    metrics_without_rows,
    resume_done,
    write_run_header,
)
from expolicy.logging.event_logger import append_jsonl
from expolicy.models.embedder import Embedder

DEFAULT_OUT = str(ROOT / "outputs" / "r4_batch_carrier_matrix")
CARRIERS = ("episodic", "semantic", "skill", "weight")


def _load_r2_accepted(r2_dir: Path) -> set[str]:
    summary_path = r2_dir / "summary.json"
    if not summary_path.exists():
        return set()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {row["capability"] for row in payload.get("skills") or [] if row.get("accepted")}


def _load_verified(r3_dir: Path, cap: str) -> list[Experience]:
    index_path = r3_dir / "verified_index.json"
    if not index_path.exists():
        return []
    index = json.loads(index_path.read_text(encoding="utf-8"))
    path = Path(index["capabilities"][cap]["path"])
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(Experience.model_validate(json.loads(line)))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_args(p, default_out=DEFAULT_OUT)
    p.add_argument("--n-batches-per-cap", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--bucket", default=None, help="s1_multi_hop|s2_temporal|s4_cross_doc")
    p.add_argument("--r2-dir", default=str(ROOT / "outputs" / "r2_executable_skill"))
    p.add_argument("--r3-dir", default=str(ROOT / "outputs" / "r3_weight_batch"))
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    args = p.parse_args()
    apply_gpu(args.gpu)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_run_header(
        out,
        config={
            "script": "run_r4_batch_carrier_matrix.py",
            "seed": args.seed,
            "model": args.model,
            "gpu": args.gpu,
            "n_batches_per_cap": args.n_batches_per_cap,
            "batch_size": args.batch_size,
            "bucket": args.bucket,
        },
    )
    world = load_world(Path(args.world))
    e0 = load_e0_experiences()
    accepted_skills = _load_r2_accepted(Path(args.r2_dir))
    embedder = Embedder(args.embedder)
    verifier = ExperienceVerifier(world)
    abstractor = ExperienceAbstractor()
    agent, base_state = build_static_agent(world, model_path=args.model, output_dir=out / "base")
    t0 = time.time()
    done = resume_done(out, "completed") if args.resume else set()
    matrix = []

    buckets = [args.bucket] if args.bucket else list(TARGET_CAPS)
    for bucket in buckets:
        if bucket not in TARGET_CAPS:
            raise SystemExit(f"unknown bucket {bucket}; expected {list(TARGET_CAPS)}")
        cap = BUCKET_PRIMARY_CAP[bucket]
        pool = bucket_pool(e0, bucket)
        groups = split_experiences_into_batches(
            pool, n_batches=args.n_batches_per_cap, batch_size=args.batch_size, seed=args.seed + 100
        )
        verified_pool = _load_verified(Path(args.r3_dir), cap)
        while len(groups) < args.n_batches_per_cap and verified_pool:
            extra = verified_pool[: args.batch_size]
            verified_pool = verified_pool[args.batch_size :]
            if len(extra) == args.batch_size:
                groups.append(extra)
        for i, group in enumerate(groups[: args.n_batches_per_cap]):
            batch_id = f"{bucket}_b{i}"
            if batch_id in done:
                continue
            probeset = build_probe_set(
                world,
                batch_id=batch_id,
                bucket=bucket,
                experiences=group,
                n_near=16,
                n_far=16,
                n_retention=16,
                seed=args.seed + i,
            )
            probes = {"near": probeset.near, "far": probeset.far, "retention": probeset.retention}
            batch_dir = out / "batches" / batch_id
            batch_dir.mkdir(parents=True, exist_ok=True)
            agent.memory_retriever = None
            agent.skill_store = None
            agent.use_executable_skill = False
            if hasattr(agent.model, "unload_adapter"):
                agent.model.unload_adapter()
            before = {name: eval_probe(agent, base_state, tasks) for name, tasks in probes.items()}
            write_json(batch_dir / "before_metrics.json", {k: metrics_without_rows(v) for k, v in before.items()})
            row = {
                "batch_id": batch_id,
                "bucket": bucket,
                "capability": cap,
                "n_exp": len(group),
                "experience_task_ids": [e.task_id for e in group],
                "carriers": {},
            }
            for carrier in CARRIERS:
                child = empty_agent(batch_dir / carrier, model_name=args.model)
                child.harness = base_state.harness
                if carrier == "skill" and cap not in accepted_skills:
                    row["carriers"][carrier] = {"status": "N/A", "reason": "skill_failed_r2"}
                    continue
                t_c = time.time()
                train_tokens = 0
                if carrier == "episodic":
                    epi = EpisodicMemory(child.episodic_dir(), embedder)
                    for exp in group:
                        epi.add(exp, abstractor.abstract(exp, verifier.verify(exp)), world=world)
                    agent.memory_retriever = MemoryFacade(epi, SemanticMemory(child.semantic_dir()))
                    agent.memory_mode = "hybrid"
                    after = {name: eval_probe(agent, child, tasks) for name, tasks in probes.items()}
                    agent.memory_retriever = None
                elif carrier == "semantic":
                    sem = SemanticMemory(child.semantic_dir())
                    for exp in group:
                        sem.add_from_abstraction(abstractor.abstract(exp, verifier.verify(exp)))
                    agent.memory_retriever = MemoryFacade(EpisodicMemory(child.episodic_dir(), embedder), sem)
                    after = {name: eval_probe(agent, child, tasks) for name, tasks in probes.items()}
                    agent.memory_retriever = None
                elif carrier == "skill":
                    skill = next(s for s in builtin_executable_skills() if s.capability == cap)
                    store = SkillStore(child.skills_dir())
                    store.create(skill)
                    from expolicy.experiments.common import score_with_retrieval
                    from expolicy.experiments.r_common import compact_metrics

                    def eval_skill(tasks):
                        rows = [score_with_retrieval(run_skill_on_task(skill, agent.env, t), t) for t in tasks]
                        return compact_metrics(rows)

                    after = {name: eval_skill(tasks) for name, tasks in probes.items()}
                else:
                    successes = [e for e in group if e.success]
                    if len(successes) < 4 and verified_pool:
                        successes = (successes + verified_pool)[: max(8, len(group))]
                    if not successes:
                        row["carriers"][carrier] = {"status": "N/A", "reason": "no_verified_success"}
                        continue
                    updater = WeightUpdater(args.model, child.adapter_dir())
                    train = updater.train(successes, epochs=args.epochs, lr=args.lr, lora_rank=args.lora_rank)
                    train_tokens = int(train.get("n_rows", 0)) * 512
                    agent.model.load(train["adapter_dir"])
                    after = {name: eval_probe(agent, child, tasks) for name, tasks in probes.items()}
                    agent.model.unload_adapter()
                    agent.model.rebuild()
                deltas = {name: after[name]["em"] - before[name]["em"] for name in probes}
                u_cap = capability_utility(deltas["near"], deltas["far"], deltas["retention"])
                infer_tokens = sum(after[s].get("tokens", 0) for s in probes)
                cost_norm = (infer_tokens + train_tokens) / 1e6
                u_eff = u_cap - cost_norm
                payload = {
                    "status": "ok",
                    "em": {name: after[name]["em"] for name in probes},
                    "delta": deltas,
                    "U_capability": u_cap,
                    "U_efficiency": u_eff,
                    "tokens": infer_tokens,
                    "train_tokens": train_tokens,
                    "wall_seconds": time.time() - t_c,
                }
                row["carriers"][carrier] = payload
                for split, payload_split in after.items():
                    dump_probe_rows(
                        out / "trajectories.jsonl",
                        split=split,
                        condition=carrier,
                        batch_id=batch_id,
                        rows=payload_split.get("rows") or [],
                    )
            utils = {
                c: row["carriers"][c]["U_capability"]
                for c in CARRIERS
                if row["carriers"].get(c, {}).get("status") == "ok"
            }
            if utils:
                best = max(utils.values())
                row["best_carriers"] = sorted(k for k, v in utils.items() if v == best)
                row["best_carrier"] = row["best_carriers"][0]
                row["tied_best"] = len(row["best_carriers"]) > 1
            write_json(batch_dir / "after_metrics.json", row)
            write_json(batch_dir / "summary.json", row)
            append_jsonl(out / "interventions.jsonl", row)
            matrix.append(row)
            mark_done(out, "completed", batch_id)
            print(json.dumps({"batch": batch_id, "best": row.get("best_carrier"), "U": utils}, indent=2))

    if (out / "interventions.jsonl").exists():
        matrix = []
        for line in (out / "interventions.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                matrix.append(json.loads(line))

    write_json(out / "carrier_matrix.json", matrix)
    csv_path = out / "carrier_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["batch_id", "bucket", "capability", "carrier", "U_capability", "U_efficiency", "d_near", "d_far", "d_ret", "status"])
        for row in matrix:
            for carrier in CARRIERS:
                cell = row.get("carriers", {}).get(carrier) or {}
                writer.writerow(
                    [
                        row.get("batch_id"),
                        row.get("bucket"),
                        row.get("capability"),
                        carrier,
                        cell.get("U_capability"),
                        cell.get("U_efficiency"),
                        (cell.get("delta") or {}).get("near"),
                        (cell.get("delta") or {}).get("far"),
                        (cell.get("delta") or {}).get("retention"),
                        cell.get("status"),
                    ]
                )
    write_json(
        out / "summary.json",
        {
            "n_batches": len(matrix),
            "accepted_skills": sorted(accepted_skills),
            "wall_seconds": time.time() - t0,
        },
    )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
