"""Budget scheduler: stop when any budget axis is exhausted."""

from __future__ import annotations

from dataclasses import dataclass, field

from expolicy.core.config import BudgetConfig
from expolicy.core.experience import ExperienceCost


@dataclass
class BudgetTracker:
    config: BudgetConfig
    used: ExperienceCost = field(default_factory=ExperienceCost)
    n_tasks: int = 0

    def remaining(self) -> bool:
        return (
            self.n_tasks < self.config.max_tasks
            and self.used.environment_steps < self.config.max_env_steps
            and self.used.tokens < self.config.max_tokens
            and self.used.gpu_seconds < self.config.max_gpu_seconds
        )

    def consume(self, cost: ExperienceCost, n_tasks: int = 1) -> None:
        self.used = self.used.add(cost)
        self.n_tasks += n_tasks

    def snapshot(self) -> dict[str, float | int]:
        return {
            "n_tasks": self.n_tasks,
            "tokens": self.used.tokens,
            "steps": self.used.environment_steps,
            "latency": self.used.latency,
            "gpu_seconds": self.used.gpu_seconds,
        }
