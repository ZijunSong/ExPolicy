"""Run an agent on a task list and write metrics + trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from expolicy.agents.llm_agent import LLMSearchAgent
from expolicy.core.experience import Experience
from expolicy.core.state import PersistentAgentState
from expolicy.environments.search.world import Task
from expolicy.evaluation.metrics import aggregate, score_experience
from expolicy.logging.event_logger import append_jsonl


class Evaluator:
    def __init__(self, agent: LLMSearchAgent, output_dir: Path) -> None:
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        tasks: Iterable[Task],
        *,
        state: PersistentAgentState | None = None,
        tag: str = "eval",
    ) -> dict:
        rows = []
        experiences: list[Experience] = []
        traj_path = self.output_dir / "trajectories.jsonl"
        exp_path = self.output_dir / "experiences.jsonl"
        for task in tasks:
            experience = self.agent.run_task(task, state=state)
            experiences.append(experience)
            row = score_experience(experience)
            row["tag"] = tag
            rows.append(row)
            append_jsonl(traj_path, {"task_id": task.task_id, "experience_id": experience.experience_id, **row})
            append_jsonl(exp_path, json.loads(experience.model_dump_json()))
        summary = aggregate(rows)
        summary["tag"] = tag
        (self.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"summary": summary, "rows": rows, "experiences": experiences}
