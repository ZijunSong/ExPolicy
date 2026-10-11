"""Fixed assimilation baselines: always discard / episodic / CountOnly K=8."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from expolicy.closed_loop.constants import ACTIVATE_EXISTING_SKILL, ASSIM_ACTIONS, DISCARD, EPISODIC_MEMORY
from expolicy.closed_loop.experience import ExperienceBatchSummary
from expolicy.closed_loop.features import assimilation_features
from expolicy.closed_loop.policies.base import AssimilationDecision
from expolicy.closed_loop.state import ExPolicyState

COUNT_K = 8


class FixedAssimilation:
    def __init__(self, action: str) -> None:
        if action not in ASSIM_ACTIONS and action != "count_only":
            raise ValueError(action)
        self.fixed = action

    def select(
        self,
        state: ExPolicyState,
        experience_summary: ExperienceBatchSummary,
        valid_actions: Sequence[str],
        rng: np.random.Generator,
    ) -> AssimilationDecision:
        valid = list(valid_actions)
        cap = experience_summary.capability
        if self.fixed == "count_only":
            seen = int(state.capability_seen_count.get(cap) or 0) + int(experience_summary.n)
            if (
                ACTIVATE_EXISTING_SKILL in valid
                and not state.active_skills.get(cap)
                and seen >= COUNT_K
            ):
                chosen = ACTIVATE_EXISTING_SKILL
            else:
                chosen = DISCARD if DISCARD in valid else valid[0]
            rationale = "count_only_k8"
        else:
            chosen = self.fixed if self.fixed in valid else valid[0]
            rationale = f"fixed:{self.fixed}"
        feats = {a: assimilation_features(state, experience_summary, a).tolist() for a in ASSIM_ACTIONS}
        scores = {a: (1.0 if a == chosen else 0.0) for a in ASSIM_ACTIONS}
        return AssimilationDecision(
            action=chosen,
            scores=scores,
            sampled_values=dict(scores),
            features=feats,
            valid_actions=valid,
            rationale=rationale,
        )

    def update(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"parameter_delta_norm": 0.0, "n_updates": 0, "policy": "assimilation", "skipped": True}

    def snapshot(self) -> dict[str, Any]:
        return {"type": "fixed", "action": self.fixed}

    def load_snapshot(self, payload: dict[str, Any]) -> None:
        self.fixed = str(payload.get("action") or self.fixed)


def always_discard() -> FixedAssimilation:
    return FixedAssimilation(DISCARD)


def always_episodic() -> FixedAssimilation:
    return FixedAssimilation(EPISODIC_MEMORY)


def count_only() -> FixedAssimilation:
    return FixedAssimilation("count_only")
