"""Shared helpers for roadmap experiments (E0–E8)."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from expolicy.agents.llm_agent import LLMSearchAgent
from expolicy.core.experience import Experience
from expolicy.core.state import HarnessConfig, empty_agent
from expolicy.environments.search.environment import SearchEnvironment
from expolicy.environments.search.world import Task, World, build_world
from expolicy.evaluation.metrics import exact_match, score_experience
from expolicy.logging.event_logger import EventLogger
from expolicy.models.factory import build_backend

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORLD = ROOT / "data" / "world" / "helion.json"
DEFAULT_MODEL = "/data/ppnm/models/Qwen2.5-3B-Instruct"
DEFAULT_EMBEDDER = "/data/ppnm/models/all-MiniLM-L6-v2"


def load_world(path: Path = DEFAULT_WORLD, seed: int = 13) -> World:
    if path.exists():
        return World.load(path)
    world = build_world(seed)
    world.save(path)
    return world


def sample_stage_tasks(
    world: World,
    stage: int,
    n: int,
    *,
    seed: int = 0,
    splits: Sequence[str] | None = None,
) -> list[Task]:
    pool = [t for t in world.tasks.values() if t.stage == stage]
    if splits is not None:
        allowed = set(splits)
        pool = [t for t in pool if t.split in allowed]
    pool = sorted(pool, key=lambda t: t.task_id)
    rng = random.Random(seed + 1000 * stage)
    if len(pool) <= n:
        rng.shuffle(pool)
        return pool
    by_split: dict[str, list[Task]] = defaultdict(list)
    for t in pool:
        by_split[t.split].append(t)
    # stratified then fill
    chosen: list[Task] = []
    keys = sorted(by_split)
    per = max(1, n // max(len(keys), 1))
    for key in keys:
        items = list(by_split[key])
        rng.shuffle(items)
        chosen.extend(items[:per])
    leftover = [t for t in pool if t not in chosen]
    rng.shuffle(leftover)
    chosen.extend(leftover)
    rng.shuffle(chosen)
    return chosen[:n]


def build_static_agent(
    world: World,
    *,
    model_path: str = DEFAULT_MODEL,
    output_dir: Path,
    temperature: float = 0.0,
    max_steps: int = 8,
) -> tuple[LLMSearchAgent, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = SearchEnvironment(world)
    backend = build_backend({"backend": "transformers", "path": model_path, "dtype": "bfloat16"})
    harness = HarnessConfig(
        max_steps=max_steps,
        search_top_k=5,
        min_evidence_before_answer=1,
        temperature=temperature,
    )
    logger = EventLogger(output_dir / "events.jsonl")
    state = empty_agent(output_dir / "agent_v000", model_name=model_path)
    state.harness = harness
    state.write_manifest()
    agent = LLMSearchAgent(backend, env, harness=harness, logger=logger)
    return agent, state


def searched_doc_ids(exp: Experience) -> set[str]:
    ids: set[str] = set()
    for obs in exp.observations:
        payload = obs.get("payload") if isinstance(obs, dict) else None
        if not isinstance(payload, dict):
            continue
        for hit in payload.get("hits") or []:
            if isinstance(hit, dict) and hit.get("document_id"):
                ids.add(str(hit["document_id"]))
    return ids


def opened_doc_ids(exp: Experience) -> set[str]:
    ids = {str(ev.get("document_id")) for ev in exp.evidence if ev.get("document_id")}
    for call in exp.tool_calls:
        if call.get("tool") == "open" and (call.get("arguments") or {}).get("document_id"):
            ids.add(str(call["arguments"]["document_id"]))
    return ids


def recall_against_gold(retrieved: set[str], gold: Sequence[str]) -> float:
    if not gold:
        return 0.0
    g = set(gold)
    return len(g & retrieved) / len(g)


def score_with_retrieval(exp: Experience, task: Task) -> dict:
    row = score_experience(exp)
    gold = list(task.gold_doc_ids)
    searched = searched_doc_ids(exp)
    opened = opened_doc_ids(exp)
    row["search_recall"] = recall_against_gold(searched, gold)
    row["open_recall"] = recall_against_gold(opened, gold)
    row["evidence_recall"] = recall_against_gold(
        {str(ev.get("document_id")) for ev in exp.evidence if ev.get("document_id")},
        gold,
    )
    row["hops"] = task.hops
    row["n_gold_docs"] = len(gold)
    row["n_distractors"] = len(task.distractor_doc_ids)
    row["q_tokens"] = len(task.question.split())
    row["n_search"] = sum(1 for c in exp.tool_calls if c.get("tool") == "search")
    row["n_open"] = sum(1 for c in exp.tool_calls if c.get("tool") == "open")
    row["em_strict"] = float(exact_match(exp.predicted_answer, task.gold_answer))
    return row


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
