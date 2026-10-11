#!/usr/bin/env python3
"""R2: no-skill vs old prompt skill vs executable skill on near/far/retention probes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from expolicy.assimilation.abstractor import ExperienceAbstractor
from expolicy.assimilation.carriers.skill_store import (
    SkillStore,
    builtin_executable_skills,
    run_skill_on_task,
    validate_skill_schema,
)
from expolicy.assimilation.verifier import ExperienceVerifier
from expolicy.core.experience import Experience
from expolicy.core.state import empty_agent
from expolicy.experiments.common import ROOT, build_static_agent, load_world, mean, write_json
from expolicy.experiments.r_common import (
    add_common_args,
    apply_gpu,
    dump_probe_rows,
    eval_probe,
    load_e0_experiences,
    load_batches_file,
    metrics_without_rows,
    write_run_header,
)
from expolicy.logging.event_logger import append_jsonl

DEFAULT_OUT = str(ROOT / "outputs" / "r2_executable_skill")
SKILL_BY_CAP = {
    "multi_hop": "multi_hop_bridge_search",
    "temporal_reasoning": "temporal_resolution",
    "cross_document_synthesis": "cross_document_synthesis",
}
BUCKET_BY_CAP = {
    "multi_hop": "s1_multi_hop",
    "temporal_reasoning": "s2_temporal",
    "cross_document_synthesis": "s4_cross_doc",
}


def _old_store(root: Path, experiences: list[Experience], world) -> SkillStore:
    store = SkillStore(root)
    store.create(SkillStore.default_lookup_skill())
    verifier = ExperienceVerifier(world)
    abstractor = ExperienceAbstractor()
    for exp in experiences[:8]:
        abs_ = abstractor.abstract(exp, verifier.verify(exp))
        from expolicy.assimilation.apply import _executable_skill_from_experience

        skill = _executable_skill_from_experience(exp, abs_)
        skill.executable = False
        skill.name = f"old_{exp.task_id}"
        store.create(skill)
    return store


def _eval_executable(skill, env, tasks) -> dict:
    from expolicy.experiments.r_common import compact_metrics
    from expolicy.experiments.common import score_with_retrieval

    rows = []
    compliance = []
    evidence = []
    for task in tasks:
        exp = run_skill_on_task(skill, env, task)
        row = score_with_retrieval(exp, task)
        row["predicted"] = exp.predicted_answer
        row["gold"] = task.gold_answer
        info = exp.outcome or {}
        planned = (info.get("compliance") or {}).get("planned") or []
        executed = (info.get("compliance") or {}).get("executed") or []
        row["compliance"] = float(len([x for x in planned if x in executed]) / max(len(planned), 1))
        row["evidence_complete"] = float(bool(info.get("evidence_complete")))
        compliance.append(row["compliance"])
        evidence.append(row["evidence_complete"])
        rows.append(row)
    metrics = compact_metrics(rows)
    metrics["tool_sequence_compliance"] = mean(compliance) if compliance else 0.0
    metrics["evidence_completion_rate"] = mean(evidence) if evidence else 0.0
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_args(p, default_out=DEFAULT_OUT)
    p.add_argument("--capability", default=None, choices=list(SKILL_BY_CAP))
    p.add_argument("--probes", default=str(ROOT / "outputs" / "r1_near_transfer" / "probes" / "batches.json"))
    args = p.parse_args()
    apply_gpu(args.gpu)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_run_header(
        out,
        config={
            "script": "run_r2_executable_skill.py",
            "seed": args.seed,
            "model": args.model,
            "gpu": args.gpu,
            "capability": args.capability,
        },
    )
    world = load_world(Path(args.world))
    batches = load_batches_file(Path(args.probes), world)
    all_exps = load_e0_experiences()
    caps = [args.capability] if args.capability else list(SKILL_BY_CAP)
    agent, base_state = build_static_agent(world, model_path=args.model, output_dir=out / "base")
    t0 = time.time()
    summaries = []

    for cap in caps:
        bucket = BUCKET_BY_CAP[cap]
        row = next(b for b in batches if b["bucket"] == bucket)
        experiences = [e for e in all_exps if e.experience_id in {x["experience_id"] for x in row["experiences"]}]
        probes = {"near": row["near_tasks"], "far": row["far_tasks"], "retention": row["retention_tasks"]}
        cap_dir = out / cap
        cap_dir.mkdir(parents=True, exist_ok=True)

        skill = next(s for s in builtin_executable_skills() if s.capability == cap)
        schema_errors = validate_skill_schema(skill)
        write_json(cap_dir / "schema.json", {"errors": schema_errors, "skill": skill.model_dump(mode="json")})

        agent.memory_retriever = None
        agent.skill_store = None
        agent.use_executable_skill = False
        no_skill = {name: eval_probe(agent, base_state, tasks) for name, tasks in probes.items()}
        write_json(cap_dir / "before_metrics.json", {k: metrics_without_rows(v) for k, v in no_skill.items()})
        for split, payload in no_skill.items():
            dump_probe_rows(out / "trajectories.jsonl", split=split, condition="no_skill", batch_id=cap, rows=payload["rows"])

        old_store = _old_store(cap_dir / "old_skills", experiences, world)
        agent.skill_store = old_store
        agent.use_executable_skill = False
        old = {name: eval_probe(agent, base_state, tasks) for name, tasks in probes.items()}
        for split, payload in old.items():
            dump_probe_rows(out / "trajectories.jsonl", split=split, condition="old_skill", batch_id=cap, rows=payload["rows"])
        agent.skill_store = None

        exe = {name: _eval_executable(skill, agent.env, tasks) for name, tasks in probes.items()}
        for split, payload in exe.items():
            dump_probe_rows(out / "trajectories.jsonl", split=split, condition="executable_skill", batch_id=cap, rows=payload["rows"])

        def pack(after, cond):
            deltas = {name: after[name]["em"] - no_skill[name]["em"] for name in probes}
            return {
                "em": {name: after[name]["em"] for name in probes},
                "tokens": {name: after[name].get("tokens", 0) for name in probes},
                "delta": deltas,
                "near_em_gain": deltas["near"],
                "far_em_gain": deltas["far"],
                "retention_delta": deltas["retention"],
                "tool_sequence_compliance": after.get("near", {}).get("tool_sequence_compliance"),
                "evidence_completion_rate": after.get("near", {}).get("evidence_completion_rate"),
            }

        accepted = (not schema_errors) and (exe["near"]["em"] - no_skill["near"]["em"] >= 0)
        rec = {
            "capability": cap,
            "schema_errors": schema_errors,
            "accepted": accepted,
            "no_skill": pack(no_skill, "no_skill"),
            "old_skill": pack(old, "old_skill"),
            "executable_skill": pack(exe, "executable_skill"),
        }
        rec["executable_skill"]["tool_sequence_compliance"] = exe["near"].get("tool_sequence_compliance")
        rec["executable_skill"]["evidence_completion_rate"] = exe["near"].get("evidence_completion_rate")
        rec["executable_skill"]["token_overhead"] = (
            sum(exe[s]["tokens"] for s in probes) - sum(no_skill[s]["tokens"] for s in probes)
        )
        write_json(cap_dir / "after_metrics.json", rec)
        write_json(cap_dir / "summary.json", rec)
        append_jsonl(out / "interventions.jsonl", rec)
        summaries.append(rec)
        print(json.dumps({"cap": cap, "accepted": accepted, "near_gain": rec["executable_skill"]["near_em_gain"]}, indent=2))

    summary = {"skills": summaries, "n_accepted": sum(1 for s in summaries if s["accepted"]), "wall_seconds": time.time() - t0}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
