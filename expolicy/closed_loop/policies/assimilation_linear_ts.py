"""Assimilation Linear Thompson Sampling over DISCARD/EPISODIC/SEMANTIC/SKILL."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from expolicy.closed_loop.constants import ASSIM_ACTIONS, ASSIM_FEATURE_DIM
from expolicy.closed_loop.experience import ExperienceBatchSummary
from expolicy.closed_loop.features import assimilation_feature_names, assimilation_features
from expolicy.closed_loop.policies.base import AssimilationDecision
from expolicy.closed_loop.policies.linear_ts import LinearThompsonPolicy
from expolicy.closed_loop.state import ExPolicyState


class LinearThompsonAssimilation:
    def __init__(self, *, prior_precision: float = 1.0, reward_noise: float = 0.05) -> None:
        self.inner = LinearThompsonPolicy(
            ASSIM_ACTIONS, ASSIM_FEATURE_DIM, prior_precision=prior_precision, reward_noise=reward_noise
        )
        self.feature_names = assimilation_feature_names()

    def _feats(self, state: ExPolicyState, summary: ExperienceBatchSummary) -> dict[str, np.ndarray]:
        return {a: assimilation_features(state, summary, a) for a in ASSIM_ACTIONS}

    def select(
        self,
        state: ExPolicyState,
        experience_summary: ExperienceBatchSummary,
        valid_actions: Sequence[str],
        rng: np.random.Generator,
    ) -> AssimilationDecision:
        feats = self._feats(state, experience_summary)
        chosen, scores, sampled = self.inner.select(feats, valid_actions, rng)
        return AssimilationDecision(
            action=chosen,
            scores=scores,
            sampled_values=sampled,
            features={k: v.tolist() for k, v in feats.items()},
            valid_actions=list(valid_actions),
            rationale="linear_thompson",
        )

    def update(
        self,
        state_before: ExPolicyState,
        experience_summary: ExperienceBatchSummary,
        action: str,
        realized_utility: float,
        state_after: ExPolicyState,
    ) -> dict[str, Any]:
        phi = assimilation_features(state_before, experience_summary, action)
        info = self.inner.update(action, phi, float(realized_utility))
        info["policy"] = "assimilation"
        info["posterior_mean"] = self.inner.mean_dict()
        info["state_after_hash"] = state_after.content_hash()
        return info

    def snapshot(self) -> dict[str, Any]:
        return self.inner.snapshot()

    def load_snapshot(self, payload: dict[str, Any]) -> None:
        self.inner.load_snapshot(payload)
