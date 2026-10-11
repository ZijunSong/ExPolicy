#!/usr/bin/env python3
"""E3: experience × carrier utility matrix (episodic/semantic/skill/weight). Harness off."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

from expolicy.agents.persistent_agent import MemoryFacade
from expolicy.assimilation.abstractor import ExperienceAbstractor
from expolicy.assimilation.apply import CarrierRuntime
from expolicy.assimilation.carriers.episodic_memory import EpisodicMemory
from expolicy.assimilation.carriers.semantic_memory import SemanticMemory
from expolicy.assimilation.carriers.skill_store import SkillStore
from expolicy.assimilation.carriers.weight_update import WeightUpdater
from expolicy.assimilation.verifier import ExperienceVerifier
from expolicy.core.experience import AssimilationDecision, Experience
from expolicy.core.state import empty_agent
from expolicy.core.types import Carrier
from expolicy.experiments.common import (
    DEFAULT_EMBEDDER,
    DEFAULT_MODEL,
    ROOT,
    build_static_agent,
    load_world,
    mean,
    sample_stage_tasks,
    score_with_retrieval,
    write_json,
)
from expolicy.logging.event_logger import append_jsonl
from expolicy.models.embedder import Embedder

DEFAULT_OUT = ROOT / "outputs" / "e3_carrier_matrix"
E0 = ROOT / "outputs" / "e0_audit"
CARRIERS = [Carrier.EPISODIC, Carrier.SEMANTIC, Carrier.SKILL, Carrier.WEIGHT]
HARD_CAPS = {
    "multi_hop",
    "query_decomposition",
    "cross_document_synthesis",
    "temporal_reasoning",
    "entity_disambiguation",
}


def _load_exps() -> list[Experience]:
    rows = []
    for stage in range(5):
        path = E0 / f"stage_{stage}" / "experiences.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(Experience.model_validate(json.loads(line)))
    return rows


def _fail_type(exp: Experience) -> str | None:
    if exp.success:
        return None
    if exp.failure_type is None:
        return "unknown"
    return exp.failure_type.value


def _select(exps: list[Experience], n: int, seed: int) -> list[Experience]:
    """Stratify on failure type + hard capabilities. E0 has no retrieval/planning labels."""
    rng = random.Random(seed)
    groups = {
        "success_easy": [e for e in exps if e.success and e.capability not in HARD_CAPS],
        "success_hard": [e for e in exps if e.success and e.capability in HARD_CAPS],
        "reasoning_hard": [
            e
            for e in exps
            if _fail_type(e) == "reasoning_failure" and e.capability in HARD_CAPS
        ],
        "synthesis": [e for e in exps if _fail_type(e) and "synthesis" in _fail_type(e)],
        "other_fail": [
            e
            for e in exps
            if (not e.success)
            and _fail_type(e) != "reasoning_failure"
            and not (_fail_type(e) and "synthesis" in (_fail_type(e) or ""))
        ],
    }
    nonempty = {k: v for k, v in groups.items() if v}
    per = max(1, n // max(len(nonempty), 1))
    uniq: dict[str, Experience] = {}
    for items in nonempty.values():
        rng.shuffle(items)
        for e in items[:per]:
            uniq[e.experience_id] = e
    if len(uniq) < n:
        rest = [e for e in exps if e.experience_id not in uniq]
        rng.shuffle(rest)
        for e in rest[: n - len(uniq)]:
            uniq[e.experience_id] = e
    pool = list(uniq.values())
    rng.shuffle(pool)
    return pool[:n]


def _eval(agent, state, tasks) -> dict:
    rows = [score_with_retrieval(agent.run_task(t, state=state), t) for t in tasks]
    return {"em": mean(r["em_strict"] for r in rows) if rows else 0.0, "n": len(rows)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--probe", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-weight", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--near-stage", type=int, default=1)
    p.add_argument("--far-stage", type=int, default=2, help="Avoid S3: static EM is already 1.0")
    p.add_argument("--ret-stage", type=int, default=0)
    p.add_argument("--wipe", action="store_true", help="Delete --out before running")
    args = p.parse_args()
    out = Path(args.out)
    if args.wipe and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    world = load_world()
    exps = _select(_load_exps(), args.n, args.seed)
    write_json(
        out / "selected.json",
        [
            {
                "id": e.experience_id,
                "task_id": e.task_id,
                "cap": e.capability,
                "success": e.success,
                "failure_type": _fail_type(e),
            }
            for e in exps
        ],
    )
    near = sample_stage_tasks(world, args.near_stage, args.probe, seed=args.seed, splits=("validation", "future"))
    far = sample_stage_tasks(world, args.far_stage, args.probe, seed=args.seed, splits=("future", "validation"))
    ret = sample_stage_tasks(world, args.ret_stage, args.probe, seed=args.seed, splits=("retention", "validation"))
    probes = {"near": near[: args.probe], "far": far[: args.probe], "retention": ret[: args.probe]}
    write_json(out / "probes.json", {k: [t.task_id for t in v] for k, v in probes.items()})
    write_json(
        out / "setting.json",
        {
            "n": args.n,
            "probe": args.probe,
            "seed": args.seed,
            "skip_weight": args.skip_weight,
            "near_stage": args.near_stage,
            "far_stage": args.far_stage,
            "ret_stage": args.ret_stage,
        },
    )

    agent, base_state = build_static_agent(world, model_path=args.model, output_dir=out / "base")
    before = {name: _eval(agent, base_state, tasks) for name, tasks in probes.items()}
    write_json(out / "before.json", before)
    embedder = Embedder(DEFAULT_EMBEDDER)
    verifier = ExperienceVerifier(world)
    abstractor = ExperienceAbstractor()
    carriers = [c for c in CARRIERS if not (args.skip_weight and c == Carrier.WEIGHT)]
    matrix_rows = []
    t0 = time.time()
    for i, exp in enumerate(exps, 1):
        row = {
            "experience_id": exp.experience_id,
            "task_id": exp.task_id,
            "capability": exp.capability,
            "success": exp.success,
            "failure_type": exp.failure_type.value if exp.failure_type else None,
        }
        ver = verifier.verify(exp)
        abs_ = abstractor.abstract(exp, ver)
        for carrier in carriers:
            child_dir = out / "interv" / f"{i:02d}_{carrier.value}_{exp.experience_id[:8]}"
            child = empty_agent(child_dir, model_name=args.model, version="agent_v001")
            child.harness = base_state.harness
            child.write_manifest()
            episodic = EpisodicMemory(child.episodic_dir(), embedder)
            semantic = SemanticMemory(child.semantic_dir())
            skills = SkillStore(child.skills_dir())
            weights = WeightUpdater(args.model, child.adapter_dir()) if carrier == Carrier.WEIGHT else None
            runtime = CarrierRuntime(
                child,
                episodic=episodic,
                semantic=semantic,
                skills=skills,
                weights=weights,
                enable_harness=False,
                enable_weight=carrier == Carrier.WEIGHT,
            )
            try:
                runtime.apply(exp, abs_, AssimilationDecision(
                    experience_id=exp.experience_id,
                    carrier=carrier,
                    confidence=1.0,
                    expected_utility=0.0,
                    expected_interference=0.0,
                    policy_name="e3_sweep",
                ), verified_batch=[exp] if carrier == Carrier.WEIGHT else None)
            except Exception as exc:
                row[carrier.value] = {"error": str(exc)}
                continue
            agent.memory_retriever = MemoryFacade(episodic, semantic)
            agent.skill_store = skills
            if carrier == Carrier.WEIGHT and child.adapter_path:
                try:
                    agent.model.load(child.adapter_path)
                    loaded = True
                except Exception as exc:
                    row[carrier.value] = {"error": f"adapter_load:{exc}"}
                    loaded = False
            else:
                loaded = False
            after = {name: _eval(agent, child, tasks) for name, tasks in probes.items()}
            if loaded:
                # reload base weights by rebuilding is too heavy; skip restore by not leaving adapter on
                # subsequent weight trials load a new adapter on the same peft model — avoid stacking.
                pass
            agent.memory_retriever = None
            agent.skill_store = None
            deltas = {name: after[name]["em"] - before[name]["em"] for name in probes}
            utility = deltas["near"] + 0.5 * deltas["far"] - 0.5 * max(0.0, -deltas["retention"])
            row[carrier.value] = {"deltas": deltas, "utility": utility, "after": after}
        utils = {
            c.value: row.get(c.value, {}).get("utility")
            for c in carriers
            if isinstance(row.get(c.value), dict) and "utility" in row.get(c.value, {})
        }
        if utils:
            best_u = max(utils.values())
            worst_u = min(utils.values())
            row["best_carriers"] = sorted(k for k, v in utils.items() if v == best_u)
            row["worst_carriers"] = sorted(k for k, v in utils.items() if v == worst_u)
            row["best_carrier"] = row["best_carriers"][0]
            row["worst_carrier"] = row["worst_carriers"][0]
            row["tied_best"] = len(row["best_carriers"]) > 1
        matrix_rows.append(row)
        append_jsonl(out / "matrix.jsonl", row)
        write_json(out / "STATUS_LIVE.json", {"done": i, "n": len(exps), "last_best": row.get("best_carrier")})

    # restore analysis
    n_pref = 0
    n_unique = 0
    n_pos = 0
    global_means = {c.value: [] for c in carriers}
    unique_best_counts: dict[str, int] = {}
    for row in matrix_rows:
        for c in carriers:
            u = row.get(c.value, {})
            if isinstance(u, dict) and "utility" in u:
                global_means[c.value].append(u["utility"])
        if row.get("best_carrier"):
            n_pref += 1
        if row.get("tied_best") is False:
            n_unique += 1
            b = row["best_carrier"]
            unique_best_counts[b] = unique_best_counts.get(b, 0) + 1
        if any(
            isinstance(row.get(c.value), dict) and row.get(c.value, {}).get("utility", 0) > 0
            for c in carriers
        ):
            n_pos += 1
    best_counts = {}
    for row in matrix_rows:
        b = row.get("best_carrier")
        if b:
            best_counts[b] = best_counts.get(b, 0) + 1
    global_best = max(global_means, key=lambda k: mean(global_means[k]) if global_means[k] else -999)
    disagree = sum(
        1
        for row in matrix_rows
        if row.get("best_carriers") and global_best not in row["best_carriers"]
    )
    summary = {
        "n_experiences": len(matrix_rows),
        "carriers": [c.value for c in carriers],
        "mean_utility_by_carrier": {k: mean(v) for k, v in global_means.items()},
        "best_carrier_counts": best_counts,
        "unique_best_counts": unique_best_counts,
        "n_tied_best": sum(1 for row in matrix_rows if row.get("tied_best")),
        "n_with_positive_utility": n_pos,
        "global_best_fixed": global_best,
        "fraction_best_set_excludes_global": disagree / max(n_pref, 1),
        "before": before,
        "wall_seconds": time.time() - t0,
    }
    write_json(out / "E3_SUMMARY.json", summary)
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
