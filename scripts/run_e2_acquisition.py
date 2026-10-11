#!/usr/bin/env python3
"""E2: active acquisition pilot. Fixed episodic+semantic assimilation, same experience budget."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from expolicy.acquisition.base import make_acquisition
from expolicy.agents.persistent_agent import MemoryFacade
from expolicy.assimilation.abstractor import ExperienceAbstractor
from expolicy.assimilation.apply import CarrierRuntime
from expolicy.assimilation.carriers.episodic_memory import EpisodicMemory
from expolicy.assimilation.carriers.semantic_memory import SemanticMemory
from expolicy.assimilation.verifier import ExperienceVerifier
from expolicy.core.capability import CapabilityProfile
from expolicy.core.experience import AssimilationDecision, Experience
from expolicy.core.state import empty_agent
from expolicy.core.types import Carrier
from expolicy.diagnosis.capability_profile import CapabilityDiagnoser
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

OUT = ROOT / "outputs" / "e2_acquisition"
E0 = ROOT / "outputs" / "e0_audit"


def _e0_profile() -> tuple[CapabilityProfile, list[Experience]]:
    exps: list[Experience] = []
    for stage in range(5):
        path = E0 / f"stage_{stage}" / "experiences.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                exps.append(Experience.model_validate(json.loads(line)))
    return CapabilityDiagnoser().diagnose(exps), exps


def _eval(agent, state, tasks) -> dict:
    rows = [score_with_retrieval(agent.run_task(t, state=state), t) for t in tasks]
    return {
        "n": len(rows),
        "em": mean(r["em_strict"] for r in rows) if rows else 0.0,
        "tokens": sum(int(r["tokens"]) for r in rows),
        "tool_calls": sum(int(r["tool_calls"]) for r in rows),
        "immediate_reward": mean(r["success"] for r in rows) if rows else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--budget", type=int, default=24)
    p.add_argument("--probe", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()
    out = OUT / args.method
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    world = load_world()
    profile, recent = _e0_profile()
    pool = []
    for stage in (0, 1, 2, 3, 4):
        pool.extend(sample_stage_tasks(world, stage, 80, seed=args.seed, splits=("acquisition",)))
    # unique
    seen = set()
    uniq = []
    for t in pool:
        if t.task_id not in seen:
            seen.add(t.task_id)
            uniq.append(t)
    pool = uniq
    policy = make_acquisition(args.method, seed=args.seed)
    dummy_state = empty_agent(out / "dummy", model_name=args.model)
    proposals = policy.propose(dummy_state, profile, pool, args.budget, recent=recent)
    task_ids = proposals[0].task_ids[: args.budget]
    chosen = [world.tasks[i] for i in task_ids if i in world.tasks]
    write_json(out / "chosen.json", [{"task_id": t.task_id, "capability": t.capability, "stage": t.stage} for t in chosen])

    future_probe = []
    for stage in (1, 2, 3):
        future_probe.extend(sample_stage_tasks(world, stage, args.probe, seed=args.seed + 7, splits=("future",)))
    future_probe = [t for t in future_probe if t.task_id not in task_ids][: args.probe]
    retention_probe = [
        t
        for t in sample_stage_tasks(world, 0, args.probe, seed=args.seed + 7, splits=("retention",))
        if t.task_id not in task_ids
    ][: args.probe]

    agent, base_state = build_static_agent(world, model_path=args.model, output_dir=out / "base")
    before_f = _eval(agent, base_state, future_probe)
    before_r = _eval(agent, base_state, retention_probe)
    embedder = Embedder(DEFAULT_EMBEDDER)
    child = empty_agent(out / "agent_v001", model_name=args.model, version="agent_v001")
    child.harness = base_state.harness
    child.write_manifest()
    episodic = EpisodicMemory(child.episodic_dir(), embedder)
    semantic = SemanticMemory(child.semantic_dir())
    runtime = CarrierRuntime(child, episodic=episodic, semantic=semantic, enable_harness=False, enable_weight=False)
    verifier = ExperienceVerifier(world)
    abstractor = ExperienceAbstractor()
    t0 = time.time()
    acquired = []
    cost_tokens = 0
    cost_tools = 0
    for i, task in enumerate(chosen, 1):
        exp = agent.run_task(task, state=base_state)
        acquired.append(exp)
        cost_tokens += exp.cost.tokens
        cost_tools += exp.cost.tool_calls
        ver = verifier.verify(exp)
        abs_ = abstractor.abstract(exp, ver)
        runtime.apply(
            exp,
            abs_,
            AssimilationDecision(
                experience_id=exp.experience_id,
                carrier=Carrier.EPISODIC,
                confidence=1.0,
                expected_utility=0.0,
                expected_interference=0.0,
                policy_name="e2_fixed_memory",
            ),
        )
        if exp.success:
            runtime.apply(
                exp,
                abs_,
                AssimilationDecision(
                    experience_id=exp.experience_id,
                    carrier=Carrier.SEMANTIC,
                    confidence=1.0,
                    expected_utility=0.0,
                    expected_interference=0.0,
                    policy_name="e2_fixed_memory",
                ),
            )
        append_jsonl(out / "acquired.jsonl", json.loads(exp.model_dump_json()))
        write_json(out / "STATUS_LIVE.json", {"method": args.method, "acquired": i, "budget": len(chosen)})

    agent.memory_retriever = MemoryFacade(episodic, semantic)
    after_f = _eval(agent, child, future_probe)
    after_r = _eval(agent, child, retention_probe)
    future_gain = after_f["em"] - before_f["em"]
    summary = {
        "method": args.method,
        "n_acquired": len(chosen),
        "immediate_reward": mean(float(e.success) for e in acquired),
        "before_future_em": before_f["em"],
        "after_future_em": after_f["em"],
        "future_gain": future_gain,
        "retention_delta": after_r["em"] - before_r["em"],
        "gain_per_experience": future_gain / max(len(chosen), 1),
        "gain_per_1k_tools": future_gain / max(cost_tools / 1000.0, 1e-9),
        "gain_per_1m_tokens": future_gain / max(cost_tokens / 1_000_000.0, 1e-9),
        "tokens": cost_tokens,
        "tool_calls": cost_tools,
        "caps_chosen": {},
        "wall_seconds": time.time() - t0,
    }
    from collections import Counter

    summary["caps_chosen"] = dict(Counter(t.capability for t in chosen))
    write_json(out / "summary.json", summary)
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
