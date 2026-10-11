"""Assimilator protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from expolicy.r9_lab.state.persistent_state import PersistentState


class Assimilator(ABC):
    action: str

    @abstractmethod
    def apply(self, state: PersistentState, experiences: Sequence[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError
