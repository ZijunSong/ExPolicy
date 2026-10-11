#!/usr/bin/env python3
"""Re-evaluate saved E2 agents on a larger future/retention probe (no new acquisition)."""

from __future__ import annotations

import json
from pathlib import Path

from expolicy.agents.persistent_agent import MemoryFacade
from expolicy.assimilation.carriers.episodic_memory import EpisodicMemory
from expolicy.assimilation.carriers.semantic_memory import SemanticMemory
from expolicy.core.state import PersistentAgentState
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
from expolicy.models.embedder import Embedder

E2 = ROOT / "outputs" / "e2_acquisition"
METHODS = ["passive", "frontier", "reward_seeking", "uncertainty", "failure_replay"]


def main() -> None:
    world = load_world()
    future = []
    for stage in (1, 2, 4):
        future.extend(sample_stage_tasks(world, stage, 16, seed=99, splits=("future", "validation")))
    future = future[:24]
    retention = sample_stage_tasks(world, 0, 16, seed=99, splits=("retention", "validation"))[:16]
    write_json(E2 / "large_probes.json", {"future": [t.task_id for t in future], "retention": [t.task_id for t in retention]})
    embedder = Embedder(DEFAULT_EMBEDDER)
    agent, state = build_static_agent(world, model_path=DEFAULT_MODEL, output_dir=E2 / "large_probe_base")
    base_f = [score_with_retrieval(agent.run_task(t, state=state), t) for t in future]
    base_r = [score_with_retrieval(agent.run_task(t, state=state), t) for t in retention]
    base = {"future_em": mean(r["em_strict"] for r in base_f), "retention_em": mean(r["em_strict"] for r in base_r)}
    results = {"base": base, "methods": {}}
    for method in METHODS:
        mem_root = E2 / method / "agent_v001"
        if not (mem_root / "manifest.json").exists():
            continue
        st = PersistentAgentState.load(mem_root)
        episodic = EpisodicMemory(st.episodic_dir(), embedder)
        semantic = SemanticMemory(st.semantic_dir())
        agent.memory_retriever = MemoryFacade(episodic, semantic)
        fr = [score_with_retrieval(agent.run_task(t, state=st), t) for t in future]
        rr = [score_with_retrieval(agent.run_task(t, state=st), t) for t in retention]
        agent.memory_retriever = None
        rec = {
            "n_memory": len(episodic),
            "future_em": mean(x["em_strict"] for x in fr),
            "retention_em": mean(x["em_strict"] for x in rr),
            "future_gain": mean(x["em_strict"] for x in fr) - base["future_em"],
            "retention_delta": mean(x["em_strict"] for x in rr) - base["retention_em"],
        }
        results["methods"][method] = rec
        write_json(E2 / "LARGE_PROBE_STATUS.json", {"done": method, **rec})
    write_json(E2 / "E2_LARGE_PROBE.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
