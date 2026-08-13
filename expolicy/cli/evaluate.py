"""Evaluate a saved agent checkpoint on a split."""

from __future__ import annotations

from pathlib import Path

from expolicy.agents.llm_agent import LLMSearchAgent
from expolicy.benchmarks.search_benchmark import SearchBenchmark
from expolicy.core.config import load_experiment
from expolicy.core.state import PersistentAgentState
from expolicy.environments.search.environment import SearchEnvironment
from expolicy.environments.search.world import World, build_world
from expolicy.evaluation.evaluator import Evaluator
from expolicy.logging.event_logger import EventLogger
from expolicy.models.factory import build_backend


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    config: Path | None,
    split: str,
    stage: int,
    limit: int,
) -> dict:
    state = PersistentAgentState.load(checkpoint)
    if config is not None:
        cfg = load_experiment(config)
        model_cfg = cfg.model.model_dump()
        seed = cfg.environment.seed
        world_path = Path(cfg.environment.world_path)
    else:
        model_cfg = {"backend": "transformers", "path": state.model_name}
        seed = 13
        world_path = Path("data/world/helion.json")
    world = World.load(world_path) if world_path.exists() else build_world(seed)
    env = SearchEnvironment(world)
    backend = build_backend(model_cfg)
    if state.adapter_path:
        backend.load(state.adapter_path)
    logger = EventLogger(Path(checkpoint) / "eval_events.jsonl")
    agent = LLMSearchAgent(backend, env, harness=state.harness, logger=logger)
    bench = SearchBenchmark(world)
    tasks = bench.tasks(stage=stage, split=split, limit=limit)
    evaluator = Evaluator(agent, Path(checkpoint) / "eval")
    result = evaluator.evaluate(tasks, state=state, tag=f"{split}_stage{stage}")
    return result["summary"]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_checkpoint(
                Path(args.checkpoint),
                config=Path(args.config) if args.config else None,
                split=args.split,
                stage=args.stage,
                limit=args.limit,
            ),
            indent=2,
            default=str,
        )
    )
