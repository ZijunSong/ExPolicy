"""Synthetic task families for the protocol harness. Hidden transfer is not a feature."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from expolicy.r9_lab.state.persistent_state import PersistentState

FAMILIES: tuple[str, ...] = ("alpha", "beta", "gamma", "delta")
# Frozen before any score is read. Not the family with the smallest measured gain.
MISMATCH_OF = {"alpha": "gamma", "beta": "gamma", "gamma": "delta", "delta": "gamma"}


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    env: str
    source_template: str
    role: str
    public_difficulty: float = 0.5
    public_estimated_tools: float = 1.0

    def public(self) -> dict[str, float]:
        return {
            "public_difficulty": self.public_difficulty,
            "public_estimated_tools": self.public_estimated_tools,
        }


def mismatch_of(family: str) -> str:
    target = MISMATCH_OF[family]
    if target == family:
        raise RuntimeError("mismatch map points at the same family")
    return target


class SyntheticWorld:
    """Scores depend on committed family content. Counts without content do not raise the score."""

    env = "synth"
    families = FAMILIES

    def __init__(self, *, tag: str = "pool") -> None:
        self.tag = tag
        self.shadow_select = _probe("shadow_select", ("alpha", "alpha", "beta", "beta"), tag)
        self.shadow_eval = _probe("shadow_eval", ("alpha", "beta", "alpha", "beta"), tag)
        self.retention = _probe("retention", ("retain", "retain"), tag)
        self.future = _probe("future", ("alpha", "beta", "alpha", "beta"), tag)
        self._assert_disjoint()

    def probability(self, state: PersistentState, task: Task) -> float:
        counts = dict(state.extra.get("family_counts") or {})
        if task.family == "retain" or task.role == "retention":
            if int(counts.get("delta") or 0) >= 1:
                return 0.4
            return 0.8
        if task.family == "gamma":
            return 0.85
        if task.family == "delta":
            return 0.3
        if int(counts.get(task.family) or 0) >= 1:
            return 0.9
        return 0.2

    def score(self, state: PersistentState, tasks: Sequence[Task]) -> float | None:
        if not tasks:
            return None
        return sum(self.probability(state, task) for task in tasks) / len(tasks)

    def rollout(self, state: PersistentState, task: Task) -> dict[str, object]:
        probability = self.probability(state, task)
        success = probability >= 0.5
        return {
            "task_id": task.task_id,
            "env": task.env,
            "family": task.family,
            "instruction": f"{task.family}::{task.source_template}::{task.task_id}",
            "trajectory": [{"content": f"act {task.family}"}],
            "success": success,
            "verifier_outcome": "CORRECT" if success else "INCORRECT",
            "source_template": task.source_template,
            "actor_tokens": 4,
            "teacher_tokens": 0,
            "tool_calls": 1,
            "bucket": "acquisition_pool",
        }

    def pool(self, *, per_family: int, templates: Sequence[str] = ("wording_a", "wording_b")) -> dict[str, list[Task]]:
        grouped: dict[str, list[Task]] = {family: [] for family in self.families}
        for family in self.families:
            for index in range(per_family):
                template = templates[index % len(templates)]
                grouped[family].append(
                    Task(
                        task_id=f"{self.tag}:{family}:{index}",
                        family=family,
                        env=self.env,
                        source_template=template,
                        role="acquisition",
                    )
                )
        return grouped

    def passive_alpha_batch(self) -> list[dict[str, object]]:
        rows = []
        for index in range(4):
            rows.append(
                {
                    "task_id": f"{self.tag}:passive:{index}",
                    "env": self.env,
                    "family": "alpha",
                    "instruction": f"alpha content {index}",
                    "trajectory": [{"content": "alpha"}],
                    "success": True,
                    "verifier_outcome": "CORRECT",
                    "source_template": "wording_a",
                    "bucket": "acquisition_pool",
                }
            )
        return rows

    def _assert_disjoint(self) -> None:
        groups = {
            "select": {task.task_id for task in self.shadow_select},
            "eval": {task.task_id for task in self.shadow_eval},
            "retention": {task.task_id for task in self.retention},
            "future": {task.task_id for task in self.future},
        }
        names = list(groups)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                if groups[left] & groups[right]:
                    raise RuntimeError(f"probe ids overlap: {left} and {right}")


def _probe(role: str, families: Sequence[str], tag: str) -> list[Task]:
    return [
        Task(task_id=f"{tag}:{role}:{index}", family=family, env="synth", source_template=role, role=role)
        for index, family in enumerate(families)
    ]


def make_snapshot(root, counts: Mapping[str, int]) -> PersistentState:
    state = PersistentState.create(root)
    state.extra["family_counts"] = {str(key): int(value) for key, value in counts.items()}
    for family, count in counts.items():
        for index in range(int(count)):
            state.committed_task_ids.append(f"pre:{family}:{index}")
    state.save()
    return state
