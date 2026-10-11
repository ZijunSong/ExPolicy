#!/usr/bin/env python3
"""E1: realized utility sanity check. Fixed probe; apply each experience via episodic+semantic."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

from expolicy.assimilation.abstractor import ExperienceAbstractor
from expolicy.assimilation.apply import CarrierRuntime
from expolicy.assimilation.carriers.episodic_memory import EpisodicMemory
from expolicy.assimilation.carriers.semantic_memory import SemanticMemory
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

OUT = ROOT / "outputs" / "e1_utility"
E0 = ROOT / "outputs" / "e0_audit"


def _load_e0_experiences() -> list[Experience]:
    rows: list[Experience] = []
    for stage in range(5):
        path = E0 / f"stage_{stage}" / "experiences.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(Experience.model_validate(json.loads(line)))
    return rows


def _bucket(exps: list[Experience], traj_by_id: dict[str, dict]) -> dict[str, list[Experience]]:
    buckets: dict[str, list[Experience]] = {
        "success": [],
        "ordinary_failure": [],
        "frontier_failure": [],
        "repeated_pattern": [],
    }
    cap_em: dict[str, list[float]] = {}
    for e in exps:
        row = traj_by_id.get(e.task_id, {})
        cap_em.setdefault(e.capability or "?", []).append(float(row.get("em_strict") or 0.0))
    cap_mean = {k: mean(v) for k, v in cap_em.items()}
    fail_by_cap: dict[str, list[Experience]] = {}
    for e in exps:
        em = float(traj_by_id.get(e.task_id, {}).get("em_strict") or 0.0)
        if em >= 0.5 or e.success:
            buckets["success"].append(e)
            continue
        cap = e.capability or "?"
        fail_by_cap.setdefault(cap, []).append(e)
        if 0.20 <= cap_mean.get(cap, 0.5) <= 0.80:
            buckets["frontier_failure"].append(e)
        else:
            buckets["ordinary_failure"].append(e)
    for cap, items in fail_by_cap.items():
        if len(items) >= 3:
            buckets["repeated_pattern"].extend(items)
    return buckets


def _load_traj() -> dict[str, dict]:
    out = {}
    for stage in range(5):
        path = E0 / f"stage_{stage}" / "trajectories.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["task_id"]] = row
    return out


def _eval_probe(agent, state, tasks, world) -> dict:
    rows = [score_with_retrieval(agent.run_task(t, state=state), t) for t in tasks]
    return {
        "n": len(rows),
        "em": mean(r["em_strict"] for r in rows) if rows else 0.0,
        "success": mean(r["success"] for r in rows) if rows else 0.0,
        "tokens": sum(int(r["tokens"]) for r in rows),
        "tool_calls": sum(int(r["tool_calls"]) for r in rows),
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-bucket", type=int, default=8)
    p.add_argument("--probe", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    world = load_world()
    traj = _load_traj()
    exps = _load_e0_experiences()
    if len(exps) < 20:
        raise SystemExit(f"E1 needs E0 experiences, found {len(exps)}")
    buckets = _bucket(exps, traj)
    rng = random.Random(args.seed)
    selected: list[tuple[str, Experience]] = []
    for name, items in buckets.items():
        uniq = {e.experience_id: e for e in items}
        pool = list(uniq.values())
        rng.shuffle(pool)
        for e in pool[: args.n_per_bucket]:
            selected.append((name, e))
    write_json(
        OUT / "selected.json",
        [{"bucket": b, "experience_id": e.experience_id, "task_id": e.task_id, "capability": e.capability} for b, e in selected],
    )

    future_probe = []
    for stage in (1, 2, 3):
        future_probe.extend(sample_stage_tasks(world, stage, args.probe, seed=args.seed, splits=("future",)))
    future_probe = future_probe[: args.probe]
    retention_probe = sample_stage_tasks(world, 0, args.probe, seed=args.seed, splits=("retention",))
    used_ids = {e.task_id for _, e in selected}
    future_probe = [t for t in future_probe if t.task_id not in used_ids][: args.probe]
    retention_probe = [t for t in retention_probe if t.task_id not in used_ids][: args.probe]
    write_json(OUT / "probes.json", {"future": [t.task_id for t in future_probe], "retention": [t.task_id for t in retention_probe]})

    agent, base_state = build_static_agent(world, model_path=args.model, output_dir=OUT / "base")
    embedder = Embedder(DEFAULT_EMBEDDER)
    t0 = time.time()
    before_future = _eval_probe(agent, base_state, future_probe, world)
    before_ret = _eval_probe(agent, base_state, retention_probe, world)
    write_json(OUT / "before.json", {"future": {k: v for k, v in before_future.items() if k != "rows"}, "retention": {k: v for k, v in before_ret.items() if k != "rows"}})

    verifier = ExperienceVerifier(world)
    abstractor = ExperienceAbstractor()
    records = []
    for i, (bucket, exp) in enumerate(selected, 1):
        child_dir = OUT / "interv" / f"{i:02d}_{bucket}_{exp.experience_id[:8]}"
        if child_dir.exists():
            shutil.rmtree(child_dir)
        child = empty_agent(child_dir, model_name=args.model, version="agent_v001")
        child.harness = base_state.harness
        child.write_manifest()
        episodic = EpisodicMemory(child.episodic_dir(), embedder)
        semantic = SemanticMemory(child.semantic_dir())
        runtime = CarrierRuntime(child, episodic=episodic, semantic=semantic, enable_harness=False, enable_weight=False)
        ver = verifier.verify(exp)
        abs_ = abstractor.abstract(exp, ver)
        runtime.apply(exp, abs_, AssimilationDecision(
            experience_id=exp.experience_id,
            carrier=Carrier.EPISODIC,
            confidence=1.0,
            expected_utility=0.0,
            expected_interference=0.0,
            policy_name="e1_fixed_episodic",
        ))
        if exp.success:
            runtime.apply(exp, abs_, AssimilationDecision(
                experience_id=exp.experience_id,
                carrier=Carrier.SEMANTIC,
                confidence=1.0,
                expected_utility=0.0,
                expected_interference=0.0,
                policy_name="e1_fixed_semantic",
            ))
        # attach memory to the same LLM agent for this eval
        from expolicy.agents.persistent_agent import MemoryFacade

        agent.memory_retriever = MemoryFacade(episodic, semantic)
        after_future = _eval_probe(agent, child, future_probe, world)
        after_ret = _eval_probe(agent, child, retention_probe, world)
        agent.memory_retriever = None
        future_gain = after_future["em"] - before_future["em"]
        ret_drop = max(0.0, before_ret["em"] - after_ret["em"])
        tokens = after_future["tokens"] + after_ret["tokens"] + exp.cost.tokens
        norm_cost = tokens / 1_000_000.0
        utility = future_gain - 0.5 * ret_drop - norm_cost
        rec = {
            "bucket": bucket,
            "experience_id": exp.experience_id,
            "task_id": exp.task_id,
            "capability": exp.capability,
            "stage": exp.stage,
            "success": exp.success,
            "failure_type": exp.failure_type.value if exp.failure_type else None,
            "future_gain": future_gain,
            "retention_drop": ret_drop,
            "retention_delta": after_ret["em"] - before_ret["em"],
            "tokens": tokens,
            "normalized_cost": norm_cost,
            "utility": utility,
            "after_future_em": after_future["em"],
            "after_retention_em": after_ret["em"],
        }
        records.append(rec)
        append_jsonl(OUT / "utilities.jsonl", rec)
        write_json(OUT / "STATUS_LIVE.json", {"done": i, "n": len(selected), "last_utility": utility})

    by_b: dict[str, list[dict]] = {}
    for r in records:
        by_b.setdefault(r["bucket"], []).append(r)
    utils = [r["utility"] for r in records]
    utils_sorted = sorted(utils, reverse=True)
    topk = max(1, int(0.2 * len(utils_sorted)))
    top_sum = sum(utils_sorted[:topk])
    pos_sum = sum(u for u in utils if u > 0)
    summary = {
        "n": len(records),
        "before_future_em": before_future["em"],
        "before_retention_em": before_ret["em"],
        "positive_utility_rate": mean(float(u > 0) for u in utils),
        "mean_utility": mean(utils),
        "median_utility": sorted(utils)[len(utils) // 2] if utils else 0.0,
        "variance": (mean(u * u for u in utils) - mean(utils) ** 2) if utils else 0.0,
        "top20_share_of_positive": (top_sum / pos_sum) if pos_sum > 1e-12 else None,
        "top20_mean": mean(utils_sorted[:topk]),
        "by_bucket": {
            k: {
                "n": len(v),
                "mean_utility": mean(x["utility"] for x in v),
                "positive_rate": mean(float(x["utility"] > 0) for x in v),
                "mean_future_gain": mean(x["future_gain"] for x in v),
            }
            for k, v in by_b.items()
        },
        "wall_seconds": time.time() - t0,
    }
    write_json(OUT / "E1_SUMMARY.json", summary)
    (OUT / "DONE").write_text("ok\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
