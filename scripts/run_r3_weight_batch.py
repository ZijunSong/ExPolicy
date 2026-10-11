#!/usr/bin/env python3
"""R3: verified-success batch LoRA vs episodic / semantic / executable skill / no update."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from expolicy.agents.persistent_agent import MemoryFacade
from expolicy.assimilation.abstractor import ExperienceAbstractor
from expolicy.assimilation.carriers.episodic_memory import EpisodicMemory
from expolicy.assimilation.carriers.semantic_memory import SemanticMemory
from expolicy.assimilation.carriers.skill_store import builtin_executable_skills, run_skill_on_task
from expolicy.assimilation.carriers.weight_update import WeightUpdater
from expolicy.assimilation.verifier import ExperienceVerifier
from expolicy.core.experience import Experience
from expolicy.core.state import empty_agent
from expolicy.experiments.common import ROOT, build_static_agent, load_world, write_json
from expolicy.experiments.r_common import (
    add_common_args,
    apply_gpu,
    capability_utility,
    dump_probe_rows,
    eval_probe,
    file_sha256,
    load_batches_file,
    metrics_without_rows,
    write_run_header,
)
from expolicy.logging.event_logger import append_jsonl
from expolicy.models.embedder import Embedder

DEFAULT_OUT = str(ROOT / "outputs" / "r3_weight_batch")
BUCKET_BY_CAP = {
    "multi_hop": "s1_multi_hop",
    "temporal_reasoning": "s2_temporal",
    "cross_document_synthesis": "s4_cross_doc",
}


def _load_verified(path: Path) -> list[Experience]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(Experience.model_validate(json.loads(line)))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_args(p, default_out=DEFAULT_OUT)
    p.add_argument("--capability", default=None)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--probes", default=str(ROOT / "outputs" / "r1_near_transfer" / "probes" / "batches.json"))
    args = p.parse_args()
    apply_gpu(args.gpu)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_run_header(
        out,
        config={
            "script": "run_r3_weight_batch.py",
            "seed": args.seed,
            "model": args.model,
            "gpu": args.gpu,
            "lora_rank": args.lora_rank,
            "epochs": args.epochs,
            "lr": args.lr,
        },
    )
    world = load_world(Path(args.world))
    batches = load_batches_file(Path(args.probes), world)
    index = json.loads((out / "verified_index.json").read_text(encoding="utf-8"))
    caps = [args.capability] if args.capability else list(BUCKET_BY_CAP)
    embedder = Embedder(args.embedder)
    verifier = ExperienceVerifier(world)
    abstractor = ExperienceAbstractor()
    agent, base_state = build_static_agent(world, model_path=args.model, output_dir=out / "base")
    t0 = time.time()
    summaries = []

    for cap in caps:
        bucket = BUCKET_BY_CAP[cap]
        row = next(b for b in batches if b["bucket"] == bucket)
        probes = {"near": row["near_tasks"], "far": row["far_tasks"], "retention": row["retention_tasks"]}
        verified = _load_verified(Path(index["capabilities"][cap]["path"]))
        cap_dir = out / cap
        cap_dir.mkdir(parents=True, exist_ok=True)

        agent.memory_retriever = None
        agent.skill_store = None
        agent.use_executable_skill = False
        if hasattr(agent.model, "unload_adapter"):
            agent.model.unload_adapter()
        no_update = {name: eval_probe(agent, base_state, tasks) for name, tasks in probes.items()}
        write_json(cap_dir / "before_metrics.json", {k: metrics_without_rows(v) for k, v in no_update.items()})

        # episodic batch
        child = empty_agent(cap_dir / "episodic", model_name=args.model)
        child.harness = base_state.harness
        epi = EpisodicMemory(child.episodic_dir(), embedder)
        sem = SemanticMemory(child.semantic_dir())
        for exp in verified:
            epi.add(exp, abstractor.abstract(exp, verifier.verify(exp)), world=world)
        agent.memory_retriever = MemoryFacade(epi, sem)
        agent.memory_mode = "hybrid"
        episodic = {name: eval_probe(agent, child, tasks) for name, tasks in probes.items()}
        agent.memory_retriever = None

        # semantic batch
        child_s = empty_agent(cap_dir / "semantic", model_name=args.model)
        child_s.harness = base_state.harness
        sem2 = SemanticMemory(child_s.semantic_dir())
        for exp in verified:
            sem2.add_from_abstraction(abstractor.abstract(exp, verifier.verify(exp)))
        agent.memory_retriever = MemoryFacade(EpisodicMemory(child_s.episodic_dir(), embedder), sem2)
        semantic = {name: eval_probe(agent, child_s, tasks) for name, tasks in probes.items()}
        agent.memory_retriever = None

        # executable skill
        skill = next(s for s in builtin_executable_skills() if s.capability == cap)
        from expolicy.experiments.common import score_with_retrieval
        from expolicy.experiments.r_common import compact_metrics

        def eval_skill(tasks):
            rows = [score_with_retrieval(run_skill_on_task(skill, agent.env, t), t) for t in tasks]
            return compact_metrics(rows)

        executable = {name: eval_skill(tasks) for name, tasks in probes.items()}

        # LoRA batch
        train_dir = cap_dir / "lora_train"
        updater = WeightUpdater(args.model, train_dir)
        train_info = updater.train(
            verified,
            epochs=args.epochs,
            lr=args.lr,
            lora_rank=args.lora_rank,
        )
        adapter_meta = {
            "base_model_hash": file_sha256(Path(args.model) / "config.json"),
            "adapter_path": train_info["adapter_dir"],
            "training_data_ids": train_info["training_data_ids"],
            "seed": args.seed,
            "steps": train_info["steps"],
            "lr": args.lr,
            "lora_rank": args.lora_rank,
        }
        write_json(cap_dir / "adapter_meta.json", adapter_meta)
        agent.model.load(train_info["adapter_dir"])
        lora = {name: eval_probe(agent, base_state, tasks) for name, tasks in probes.items()}
        agent.model.unload_adapter()
        if hasattr(agent.model, "rebuild"):
            agent.model.rebuild()

        def pack(after):
            deltas = {name: after[name]["em"] - no_update[name]["em"] for name in probes}
            return {
                "em": {name: after[name]["em"] for name in probes},
                "tokens": {name: after[name].get("tokens", 0) for name in probes},
                "delta": deltas,
                "U_capability": capability_utility(deltas["near"], deltas["far"], deltas["retention"]),
            }

        rec = {
            "capability": cap,
            "n_verified": len(verified),
            "no_update": pack(no_update),
            "episodic_batch": pack(episodic),
            "semantic_batch": pack(semantic),
            "executable_skill": pack(executable),
            "lora_batch": pack(lora),
            "adapter": adapter_meta,
            "train": train_info,
        }
        for cond, after in (
            ("no_update", no_update),
            ("episodic_batch", episodic),
            ("semantic_batch", semantic),
            ("executable_skill", executable),
            ("lora_batch", lora),
        ):
            for split, payload in after.items():
                dump_probe_rows(out / "trajectories.jsonl", split=split, condition=cond, batch_id=cap, rows=payload.get("rows") or [])
        write_json(cap_dir / "after_metrics.json", rec)
        write_json(cap_dir / "summary.json", rec)
        append_jsonl(out / "interventions.jsonl", rec)
        summaries.append(rec)
        print(json.dumps({"cap": cap, "U": {k: rec[k]["U_capability"] for k in ("episodic_batch", "semantic_batch", "executable_skill", "lora_batch")}}, indent=2))

    write_json(out / "summary.json", {"capabilities": summaries, "wall_seconds": time.time() - t0})
    print(json.dumps({"n": len(summaries), "wall_seconds": time.time() - t0}, indent=2))


if __name__ == "__main__":
    main()
