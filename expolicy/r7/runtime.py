"""Shared R7 runtime: heartbeat, eval loops, GPU workers, retriever wait."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from expolicy.agents.llm_agent import LLMSearchAgent
from expolicy.assimilation.carriers.skill_store import SkillStore
from expolicy.core.state import HarnessConfig, empty_agent
from expolicy.environments.public_search.environment import PUBLIC_SYSTEM_PROMPT, PublicSearchEnvironment, PublicWorld
from expolicy.environments.public_search.evaluator import score_public
from expolicy.environments.public_search.retriever import RetrieverClient
from expolicy.environments.public_search.skills import run_public_skill_on_task
from expolicy.environments.search.world import Task
from expolicy.experiments.common import mean, write_json
from expolicy.experiments.r_common import apply_gpu
from expolicy.logging.event_logger import EventLogger
from expolicy.models.factory import build_backend
from expolicy.r7.constants import PUBLIC_CAPS, R7_DATASETS, RETRIEVER_URL
from expolicy.r7.interventional_skills import run_interventional_skill_on_task


def heartbeat(out: Path, **payload: Any) -> None:
    payload = {"updated_at": time.time(), "pid": os.getpid(), **payload}
    write_json(out / "STATUS_LIVE.json", payload)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def records_to_tasks(records: Sequence[dict[str, Any]], *, split: str) -> list[Task]:
    from expolicy.environments.public_search.datasets import to_task

    tasks = []
    for rec in records:
        cap = rec.get("capability") or "unmapped"
        if cap == "unmapped":
            continue
        tasks.append(to_task(rec, capability=str(cap), split=split))
    return tasks


def world_from_tasks(tasks: Sequence[Task]) -> PublicWorld:
    world = PublicWorld()
    for task in tasks:
        world.add_task(task)
    return world


def wait_retriever(url: str = RETRIEVER_URL, timeout: int = 3600) -> dict[str, Any]:
    client = RetrieverClient(url, timeout=10.0)
    t0 = time.time()
    last: Any = None
    while time.time() - t0 < timeout:
        try:
            last = client.health()
            if last.get("ok"):
                return last
        except Exception as exc:
            last = {"error": str(exc)}
        time.sleep(5)
    raise SystemExit(f"retriever not healthy at {url}: {last}")


def build_agent(
    world: PublicWorld,
    *,
    model_path: str,
    output_dir: Path,
    retriever_url: str,
    top_k: int = 5,
    max_steps: int = 8,
    dtype: str = "bfloat16",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    client = RetrieverClient(retriever_url)
    env = PublicSearchEnvironment(world, client, top_k=top_k)
    backend = build_backend({"backend": "transformers", "path": model_path, "dtype": dtype})
    harness = HarnessConfig(max_steps=max_steps, search_top_k=top_k, min_evidence_before_answer=1, temperature=0.0)
    logger = EventLogger(output_dir / "events.jsonl")
    state = empty_agent(output_dir / "agent_v000", model_name=model_path)
    state.harness = harness
    state.write_manifest()
    agent = LLMSearchAgent(backend, env, harness=harness, logger=logger, system_prompt=PUBLIC_SYSTEM_PROMPT)
    return agent, state, backend, env


def macro_from_rows(rows: Sequence[dict[str, Any]], *, datasets: Sequence[str] = R7_DATASETS) -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    by_cap: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        ds = str(row.get("dataset") or "unknown")
        cap = str(row.get("capability") or "unknown")
        val = float(row.get("em") or 0.0)
        by_ds[ds].append(val)
        by_cap[cap].append(val)
    per_dataset = {ds: mean(vs) for ds, vs in by_ds.items()}
    present = [per_dataset[ds] for ds in datasets if ds in per_dataset]
    return {
        "macro": mean(present),
        "per_dataset": per_dataset,
        "per_capability": {cap: mean(vs) for cap, vs in by_cap.items()},
        "n": len(rows),
    }


def eval_base(agent: LLMSearchAgent, tasks: Sequence[Task], *, out: Path | None = None) -> dict[str, Any]:
    rows = []
    done_ids: set[str] = set()
    rows_path = (out / "rows.jsonl") if out is not None else None
    if rows_path and rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append(rec)
            done_ids.add(str(rec.get("task_id")))
    for i, task in enumerate(tasks):
        if task.task_id in done_ids:
            continue
        if out is not None:
            heartbeat(out, phase="eval_base", i=i, n=len(tasks), task_id=task.task_id)
        exp = agent.run_task(task)
        row = score_public(exp, task)
        row["dispatch_source"] = "base_llm"
        row["model_tokens_after_skill"] = int(exp.cost.completion_tokens or 0)
        row["final_answer_source"] = "policy_model"
        rows.append(row)
        if rows_path is not None:
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        done_ids.add(task.task_id)
    summary = macro_from_rows(rows)
    summary["rows"] = rows
    return summary


def eval_with_backend(
    tasks: Sequence[Task],
    *,
    condition: str,
    store: SkillStore | None,
    env: PublicSearchEnvironment,
    model: Any,
    base_rows: dict[str, dict] | None = None,
    out: Path | None = None,
) -> dict[str, Any]:
    rows = []
    done_ids: set[str] = set()
    rows_path = (out / "rows.jsonl") if out is not None else None
    if rows_path and rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append(rec)
            done_ids.add(str(rec.get("task_id")))
    n_skill = 0
    n_base = 0
    for i, task in enumerate(tasks):
        if task.task_id in done_ids:
            continue
        if out is not None:
            heartbeat(out, phase=f"eval_{condition}", i=i, n=len(tasks), task_id=task.task_id)
        skill = None
        if store is not None and task.capability in PUBLIC_CAPS:
            skill = store.executable_for(task.capability)
        if skill is None or condition == "base":
            if base_rows and task.task_id in base_rows:
                row = dict(base_rows[task.task_id])
                row["dispatch_source"] = "base_llm"
                n_base += 1
            else:
                raise RuntimeError(f"missing base row for {task.task_id}")
        elif condition == "old_replacement_skill":
            exp = run_public_skill_on_task(skill, env, task, model=model)
            row = score_public(exp, task)
            row["dispatch_source"] = f"old_skill:{skill.name}"
            row["model_tokens_after_skill"] = int(exp.cost.completion_tokens or 0)
            row["final_answer_source"] = "replacement_or_synth"
            n_skill += 1
        else:
            exp = run_interventional_skill_on_task(skill, env, task, model=model)
            row = score_public(exp, task)
            row["dispatch_source"] = f"interventional:{skill.name}"
            row["model_tokens_after_skill"] = int((exp.outcome or {}).get("model_tokens_after_skill") or 0)
            row["final_answer_source"] = (exp.outcome or {}).get("final_answer_source")
            n_skill += 1
        row["task_id"] = task.task_id
        row["dataset"] = (task.metadata or {}).get("dataset")
        row["capability"] = task.capability
        row["condition"] = condition
        rows.append(row)
        if rows_path is not None:
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        done_ids.add(task.task_id)
    summary = macro_from_rows(rows)
    summary.update(
        {
            "rows": rows,
            "n_skill_invocations": n_skill,
            "n_base_fallback": n_base,
            "condition": condition,
        }
    )
    return summary


def peak_mem_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.max_memory_allocated() / (1024**3))
    except Exception:
        return 0.0
