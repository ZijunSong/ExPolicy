"""Random acquisition / assimilation. No learned scores."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from expolicy.closed_loop.constants import ASSIM_ACTIONS, CAPS
from expolicy.closed_loop.experience import ExperienceBatchSummary
from expolicy.closed_loop.features import acquisition_features, assimilation_features
from expolicy.closed_loop.policies.base import AcquisitionDecision, AssimilationDecision
from expolicy.closed_loop.state import ExPolicyState


class RandomAcquisition:
    def select(
        self,
        state: ExPolicyState,
        available_capabilities: Sequence[str],
        rng: np.random.Generator,
        *,
        n_rounds: int,
    ) -> AcquisitionDecision:
        available = list(available_capabilities)
        chosen = str(available[int(rng.integers(0, len(available)))])
        feats = {c: acquisition_features(state, c, n_rounds=n_rounds).tolist() for c in CAPS}
        scores = {c: (1.0 / len(available) if c in available else 0.0) for c in CAPS}
        return AcquisitionDecision(
            capability=chosen,
            scores=scores,
            sampled_values=dict(scores),
            features=feats,
            masked=[c for c in CAPS if c not in available],
            rationale="random",
        )

    def update(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"parameter_delta_norm": 0.0, "n_updates": 0, "policy": "acquisition", "skipped": True}

    def snapshot(self) -> dict[str, Any]:
        return {"type": "random", "role": "acquisition"}

    def load_snapshot(self, payload: dict[str, Any]) -> None:
        return


class RandomAssimilation:
    def select(
        self,
        state: ExPolicyState,
        experience_summary: ExperienceBatchSummary,
        valid_actions: Sequence[str],
        rng: np.random.Generator,
    ) -> AssimilationDecision:
        valid = list(valid_actions)
        chosen = str(valid[int(rng.integers(0, len(valid)))])
        feats = {a: assimilation_features(state, experience_summary, a).tolist() for a in ASSIM_ACTIONS}
        scores = {a: (1.0 / len(valid) if a in valid else 0.0) for a in ASSIM_ACTIONS}
        return AssimilationDecision(
            action=chosen,
            scores=scores,
            sampled_values=dict(scores),
            features=feats,
            valid_actions=valid,
            rationale="random",
        )

    def update(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"parameter_delta_norm": 0.0, "n_updates": 0, "policy": "assimilation", "skipped": True}

    def snapshot(self) -> dict[str, Any]:
        return {"type": "random", "role": "assimilation"}

    def load_snapshot(self, payload: dict[str, Any]) -> None:
        return
