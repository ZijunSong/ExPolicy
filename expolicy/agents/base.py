"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from expolicy.core.experience import Experience
from expolicy.core.state import PersistentAgentState
from expolicy.environments.search.world import Task


class BaseAgent(ABC):
    @abstractmethod
    def run_task(self, task: Task, state: PersistentAgentState | None = None) -> Experience: ...
