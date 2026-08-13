#!/usr/bin/env python3
"""Phase 1 acceptance: real LLM agent on 50+ Helion validation tasks."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from expolicy.agents.llm_agent import LLMSearchAgent
from expolicy.benchmarks.search_benchmark import SearchBenchmark
from expolicy.core.state import HarnessConfig, empty_agent
from expolicy.environments.search.environment import SearchEnvironment
from expolicy.environments.search.world import build_world
from expolicy.evaluation.evaluator import Evaluator
from expolicy.logging.event_logger import EventLogger
from expolicy.models.factory import build_backend
from expolicy.orchestration.checkpoint import snapshot_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase1_static"
N_TASKS = int(os.environ.get("EXPOLICY_PHASE1_TASKS", "55"))
MODEL = os.environ.get("EXPOLICY_MODEL", "/data/ppnm/models/Qwen2.5-3B-Instruct")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    world_path = ROOT / "data" / "world" / "helion.json"
    world = build_world(13)
    world.save(world_path)
    env = SearchEnvironment(world)
    bench = SearchBenchmark(world)
    tasks = bench.tasks(stage=0, split="validation", limit=N_TASKS)
    if len(tasks) < 50:
        extra = bench.tasks(stage=0, split="deployment", limit=50)
        tasks = (tasks + extra)[:N_TASKS]
    snapshot_run(
        OUT,
        {
            "phase": 1,
            "n_tasks": len(tasks),
            "model": MODEL,
            "seed": 13,
            "split": "validation+deployment_fill",
        },
    )
    logger = EventLogger(OUT / "events.jsonl", run_id="phase1")
    backend = build_backend({"backend": "transformers", "path": MODEL, "dtype": "bfloat16"})
    harness = HarnessConfig(max_steps=8, search_top_k=5, min_evidence_before_answer=1, temperature=0.0)
    agent_state = empty_agent(OUT / "agent_v000", model_name=MODEL)
    agent_state.harness = harness
    agent_state.write_manifest()
    agent = LLMSearchAgent(backend, env, harness=harness, logger=logger)
    evaluator = Evaluator(agent, OUT)
    started = time.time()
    result = evaluator.evaluate(tasks, state=agent_state, tag="phase1")
    result["summary"]["wall_seconds"] = time.time() - started
    result["summary"]["model"] = MODEL
    (OUT / "metrics.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    if int(result["summary"]["n"]) < 50:
        raise SystemExit(f"Phase 1 requires 50+ tasks, got {result['summary']['n']}")


if __name__ == "__main__":
    main()
