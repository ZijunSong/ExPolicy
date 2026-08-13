"""Continual search benchmark over the Helion Archive."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from expolicy.core.errors import LeakageError
from expolicy.environments.search.world import Task, World, build_world, world_stats


class BenchmarkSplit(BaseModel):
    name: str
    stage: int
    split: str
    task_ids: list[str] = Field(default_factory=list)


class SearchBenchmark:
    def __init__(self, world: World) -> None:
        self.world = world
        self._by_key: dict[tuple[int, str], list[str]] = defaultdict(list)
        for task in world.tasks.values():
            self._by_key[(task.stage, task.split)].append(task.task_id)

    @classmethod
    def build(cls, seed: int = 13) -> SearchBenchmark:
        return cls(build_world(seed))

    @classmethod
    def load(cls, path: Path) -> SearchBenchmark:
        return cls(World.load(path))

    def tasks(self, *, stage: int | None = None, split: str | None = None, limit: int | None = None) -> list[Task]:
        selected: list[Task] = []
        for task in self.world.tasks.values():
            if stage is not None and task.stage != stage:
                continue
            if split is not None and task.split != split:
                continue
            selected.append(task)
        selected.sort(key=lambda t: t.task_id)
        if limit is not None:
            selected = selected[:limit]
        return selected

    def future_guard(self, used_ids: Iterable[str], eval_ids: Iterable[str]) -> None:
        used = set(used_ids)
        leaked = used.intersection(eval_ids)
        if leaked:
            raise LeakageError(f"future/eval leakage of {len(leaked)} task ids, e.g. {next(iter(leaked))}")

    def stats(self) -> dict:
        return world_stats(self.world)

    def stage_splits(self) -> list[BenchmarkSplit]:
        out = []
        for (stage, split), ids in sorted(self._by_key.items()):
            out.append(BenchmarkSplit(name=f"stage{stage}_{split}", stage=stage, split=split, task_ids=sorted(ids)))
        return out


class ContinualSuite:
    """Stage-wise streams with strict future/retention isolation."""

    def __init__(self, benchmark: SearchBenchmark, n_stages: int = 5) -> None:
        self.benchmark = benchmark
        self.n_stages = n_stages

    def deployment_pool(self, stage: int) -> list[Task]:
        return self.benchmark.tasks(stage=stage, split="deployment")

    def acquisition_pool(self, stage: int) -> list[Task]:
        return self.benchmark.tasks(stage=stage, split="acquisition")

    def validation_pool(self, stage: int) -> list[Task]:
        return self.benchmark.tasks(stage=stage, split="validation")

    def future_pool(self, stage: int) -> list[Task]:
        return self.benchmark.tasks(stage=stage, split="future")

    def retention_pool(self, up_to_stage: int) -> list[Task]:
        tasks: list[Task] = []
        for stage in range(up_to_stage + 1):
            tasks.extend(self.benchmark.tasks(stage=stage, split="retention"))
        return tasks
