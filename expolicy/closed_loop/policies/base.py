"""Policy interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from expolicy.closed_loop.experience import ExperienceBatchSummary
from expolicy.closed_loop.state import ExPolicyState


@dataclass
class AcquisitionDecision:
    capability: str
    scores: dict[str, float]
    sampled_values: dict[str, float]
    features: dict[str, list[float]]
    masked: list[str] = field(default_factory=list)
    rationale: str = "linear_thompson"


@dataclass
class AssimilationDecision:
    action: str
    scores: dict[str, float]
    sampled_values: dict[str, float]
    features: dict[str, list[float]]
    valid_actions: list[str] = field(default_factory=list)
    rationale: str = "linear_thompson"


class AcquisitionPolicy(Protocol):
    def select(
        self,
        state: ExPolicyState,
        available_capabilities: Sequence[str],
        rng: np.random.Generator,
        *,
        n_rounds: int,
    ) -> AcquisitionDecision: ...

    def update(
        self,
        state_before: ExPolicyState,
        capability: str,
        realized_utility: float,
        state_after: ExPolicyState,
        *,
        n_rounds: int,
    ) -> dict[str, Any]: ...

    def snapshot(self) -> dict[str, Any]: ...

    def load_snapshot(self, payload: dict[str, Any]) -> None: ...


class AssimilationPolicy(Protocol):
    def select(
        self,
        state: ExPolicyState,
        experience_summary: ExperienceBatchSummary,
        valid_actions: Sequence[str],
        rng: np.random.Generator,
    ) -> AssimilationDecision: ...

    def update(
        self,
        state_before: ExPolicyState,
        experience_summary: ExperienceBatchSummary,
        action: str,
        realized_utility: float,
        state_after: ExPolicyState,
    ) -> dict[str, Any]: ...

    def snapshot(self) -> dict[str, Any]: ...

    def load_snapshot(self, payload: dict[str, Any]) -> None: ...
