"""Run static / baseline / evolve experiments from YAML config."""

from __future__ import annotations

import json
from pathlib import Path

from expolicy.agents.llm_agent import LLMSearchAgent
from expolicy.agents.persistent_agent import PersistentAgent
from expolicy.benchmarks.search_benchmark import SearchBenchmark
from expolicy.core.config import ExperimentConfig, load_experiment
from expolicy.core.state import empty_agent
from expolicy.environments.search.environment import SearchEnvironment
from expolicy.environments.search.world import World, build_world
from expolicy.evaluation.evaluator import Evaluator
from expolicy.logging.event_logger import EventLogger
from expolicy.models.embedder import Embedder
from expolicy.models.factory import build_backend
from expolicy.orchestration.checkpoint import snapshot_run
from expolicy.orchestration.evolve_loop import EvolveLoop


def _load_world(cfg: ExperimentConfig) -> World:
    path = Path(cfg.environment.world_path)
    if path.exists():
        return World.load(path)
    world = build_world(cfg.environment.seed)
    world.save(path)
    return world


def run_experiment(config_path: Path, *, limit: int | None = None, collect_only: bool = False) -> dict:
    cfg = load_experiment(config_path)
    run_dir = Path(cfg.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_run(run_dir, cfg.model_dump(mode="json"))
    world = _load_world(cfg)
    env = SearchEnvironment(world)
    bench = SearchBenchmark(world)
    logger = EventLogger(run_dir / "events.jsonl")
    backend = build_backend(cfg.model.model_dump())
    agent_dir = run_dir / "agent_v000"
    state = empty_agent(agent_dir, model_name=cfg.model.path or cfg.model.name)
    state.harness = cfg.harness
    state.write_manifest()
    embedder = None
    try:
        embedder = Embedder(cfg.model.embedder_path)
    except Exception as exc:
        logger.emit("RUN_FAILED", {"embedder": str(exc), "recover": "set model.embedder_path to MiniLM"})
        # Embedder is required for memory; static B0 can continue without it.
        if cfg.variant not in {"B0", "static"}:
            raise
    search_agent = LLMSearchAgent(backend, env, harness=cfg.harness, logger=logger)
    persistent = PersistentAgent(state, backend, env, embedder=embedder, logger=logger)
    persistent.search_agent = search_agent
    persistent.search_agent.memory_retriever = persistent.search_agent.memory_retriever
    n = limit if limit is not None else cfg.budget.max_tasks
    tasks = bench.tasks(stage=cfg.stage, split=cfg.split, limit=n)
    evaluator = Evaluator(persistent.search_agent, run_dir)
    result = evaluator.evaluate(tasks, state=persistent.state, tag=cfg.variant)
    summary = result["summary"]
    summary["run_dir"] = str(run_dir)
    summary["n_tasks_requested"] = n
    summary["collect_only"] = collect_only
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_evolve(config_path: Path) -> dict:
    cfg = load_experiment(config_path)
    run_dir = Path(cfg.output_dir) / cfg.run_name
    world = _load_world(cfg)
    env = SearchEnvironment(world)
    bench = SearchBenchmark(world)
    logger = EventLogger(run_dir / "events.jsonl")
    backend = build_backend(cfg.model.model_dump())
    embedder = Embedder(cfg.model.embedder_path)
    agent = PersistentAgent.create(
        run_dir / "agent_v000",
        backend,
        env,
        embedder=embedder,
        logger=logger,
        model_name=cfg.model.path,
    )
    loop = EvolveLoop(cfg, agent, bench, run_dir)
    return loop.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--evolve", action="store_true")
    args = parser.parse_args()
    path = Path(args.config)
    if args.evolve:
        print(json.dumps(run_evolve(path), indent=2, default=str))
    else:
        print(json.dumps(run_experiment(path, limit=args.limit), indent=2, default=str))
