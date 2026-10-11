"""Acquisition Linear Thompson Sampling over capabilities."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from expolicy.closed_loop.constants import ACQ_FEATURE_DIM, CAPS
from expolicy.closed_loop.features import acquisition_feature_names, acquisition_features
from expolicy.closed_loop.policies.base import AcquisitionDecision
from expolicy.closed_loop.policies.linear_ts import LinearThompsonPolicy
from expolicy.closed_loop.state import ExPolicyState


class LinearThompsonAcquisition:
    def __init__(self, *, prior_precision: float = 1.0, reward_noise: float = 0.05) -> None:
        self.inner = LinearThompsonPolicy(
            CAPS, ACQ_FEATURE_DIM, prior_precision=prior_precision, reward_noise=reward_noise
        )
        self.feature_names = acquisition_feature_names()

    def _feats(self, state: ExPolicyState, n_rounds: int) -> dict[str, np.ndarray]:
        return {c: acquisition_features(state, c, n_rounds=n_rounds) for c in CAPS}

    def select(
        self,
        state: ExPolicyState,
        available_capabilities: Sequence[str],
        rng: np.random.Generator,
        *,
        n_rounds: int,
    ) -> AcquisitionDecision:
        feats = self._feats(state, n_rounds)
        chosen, scores, sampled = self.inner.select(feats, available_capabilities, rng)
        return AcquisitionDecision(
            capability=chosen,
            scores=scores,
            sampled_values=sampled,
            features={k: v.tolist() for k, v in feats.items()},
            masked=[c for c in CAPS if c not in set(available_capabilities)],
            rationale="linear_thompson",
        )

    def update(
        self,
        state_before: ExPolicyState,
        capability: str,
        realized_utility: float,
        state_after: ExPolicyState,
        *,
        n_rounds: int,
    ) -> dict[str, Any]:
        phi = acquisition_features(state_before, capability, n_rounds=n_rounds)
        info = self.inner.update(capability, phi, float(realized_utility))
        info["policy"] = "acquisition"
        info["posterior_mean"] = self.inner.mean_dict()
        info["state_after_hash"] = state_after.content_hash()
        return info

    def snapshot(self) -> dict[str, Any]:
        return self.inner.snapshot()

    def load_snapshot(self, payload: dict[str, Any]) -> None:
        self.inner.load_snapshot(payload)
