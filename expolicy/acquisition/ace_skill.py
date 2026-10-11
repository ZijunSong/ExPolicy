"""Ace-Skill Prioritized Sampler (acquisition-only).

Faithful to official AMAP-ML/Ace-Skill ``eval/ace_skill/weighted_sampler.py``
``SqrtBiasSampler`` plus ``configs/prioritize.yaml``. Clustered organizer is
intentionally not ported: R5.2 compares acquisition policies behind a frozen
CountOnly skill-gating backend.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from expolicy.core.capability import CapabilityProfile
from expolicy.core.experience import AcquisitionProposal, Experience
from expolicy.core.state import PersistentAgentState
from expolicy.environments.search.world import Task

from expolicy.acquisition.base import AcquisitionPolicy

# Official configs/prioritize.yaml (paper §2.2 / implementation details).
DEFAULT_RHO = 0.95
DEFAULT_GAMMA = 0.4
DEFAULT_EPSILON = 0.1


class AceSkillPrioritizedAcquisition(AcquisitionPolicy):
    """Beta proficiency tracker + sqrt-bias weights, with-replacement across rounds.

    Official code initializes ``I_alpha = I_beta = 0`` (Algorithm 1), not the
    paper-text ``(0.5, 0.5)``. Both yield ``v=0.5`` before any observation;
    subsequent updates differ. We follow official code.
    """

    name = "ace_skill_prioritized"

    def __init__(
        self,
        seed: int = 0,
        *,
        rho: float = DEFAULT_RHO,
        gamma: float = DEFAULT_GAMMA,
        epsilon: float = DEFAULT_EPSILON,
        task_ids: Sequence[str] | None = None,
    ) -> None:
        self.seed = int(seed)
        self.rho = float(rho)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self._rng = np.random.RandomState(self.seed)
        self.task_ids: list[str] = []
        self.I_alpha: np.ndarray | None = None
        self.I_beta: np.ndarray | None = None
        self.T_last: np.ndarray | None = None
        self._index: dict[str, int] = {}
        if task_ids is not None:
            self.bind(task_ids)

    def bind(self, task_ids: Sequence[str]) -> None:
        ids = [str(t) for t in task_ids]
        if self.task_ids and self.task_ids == ids:
            return
        if self.task_ids and self.task_ids != ids:
            raise RuntimeError("Ace-Skill sampler bound to a different task id list")
        self.task_ids = ids
        n = len(ids)
        self.I_alpha = np.zeros(n, dtype=np.float64)
        self.I_beta = np.zeros(n, dtype=np.float64)
        self.T_last = np.zeros(n, dtype=np.int64)
        self._index = {tid: i for i, tid in enumerate(ids)}

    def _ensure_bound(self, candidate_pool: Sequence[Task]) -> None:
        if self.I_alpha is None:
            self.bind([t.task_id for t in candidate_pool])

    def _compute_weights(self, t: int) -> np.ndarray:
        assert self.I_alpha is not None and self.I_beta is not None and self.T_last is not None
        delta_t = np.maximum(int(t) - self.T_last, 0)
        decay = np.power(self.rho, delta_t)
        cur_alpha = self.I_alpha * decay
        cur_beta = self.I_beta * decay
        v = (1.0 + cur_alpha) / (2.0 + cur_alpha + cur_beta)
        return np.sqrt(v * (1.0 - v)) + self.gamma * (1.0 - v) + self.epsilon

    def proficiency(self, t: int) -> np.ndarray:
        assert self.I_alpha is not None and self.I_beta is not None and self.T_last is not None
        delta_t = np.maximum(int(t) - self.T_last, 0)
        decay = np.power(self.rho, delta_t)
        cur_alpha = self.I_alpha * decay
        cur_beta = self.I_beta * decay
        return (1.0 + cur_alpha) / (2.0 + cur_alpha + cur_beta)

    def propose(
        self,
        agent_state: PersistentAgentState,
        capability_profile: CapabilityProfile,
        candidate_pool: Sequence[Task],
        budget: int,
        *,
        recent: Sequence[Experience] | None = None,
        t: int = 0,
    ) -> list[AcquisitionProposal]:
        self._ensure_bound(candidate_pool)
        assert self.I_alpha is not None
        n = len(self.task_ids)
        k = min(int(budget), n)
        weights = self._compute_weights(int(t))
        w_sum = float(weights.sum())
        probs = weights / w_sum if w_sum > 0 else np.ones(n, dtype=np.float64) / n
        chosen_idx = self._rng.choice(n, size=k, replace=False, p=probs)
        chosen_ids = [self.task_ids[int(i)] for i in chosen_idx]
        scores = {tid: float(weights[self._index[tid]]) for tid in self.task_ids}
        return [
            AcquisitionProposal(
                mode=self.name,
                task_ids=chosen_ids,
                scores=scores,
                rationale="ace-skill sqrt-bias: sqrt(v(1-v))+gamma(1-v)+eps; with-replacement across rounds",
                metadata={"t": int(t), "rho": self.rho, "gamma": self.gamma, "epsilon": self.epsilon},
            )
        ]

    def observe(self, task_id: str, reward: float, t: int) -> None:
        if self.I_alpha is None or self.I_beta is None or self.T_last is None:
            raise RuntimeError("sampler not bound")
        idx = self._index.get(str(task_id))
        if idx is None:
            raise KeyError(f"task_id {task_id} not in Ace-Skill pool")
        r = float(reward)
        r = min(1.0, max(0.0, r))
        delta_t = max(int(t) - int(self.T_last[idx]), 0)
        decay = self.rho**delta_t
        self.I_alpha[idx] = self.I_alpha[idx] * decay + r
        self.I_beta[idx] = self.I_beta[idx] * decay + (1.0 - r)
        self.T_last[idx] = int(t)

    def state_dict(self) -> dict[str, Any]:
        rng_state = self._rng.get_state()
        return {
            "algorithm": "sqrt_bias",
            "name": self.name,
            "seed": self.seed,
            "rho": self.rho,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "task_ids": list(self.task_ids),
            "I_alpha": None if self.I_alpha is None else self.I_alpha.tolist(),
            "I_beta": None if self.I_beta is None else self.I_beta.tolist(),
            "T_last": None if self.T_last is None else self.T_last.tolist(),
            "rng_state": {
                "keys": rng_state[1].tolist(),
                "pos": int(rng_state[2]),
                "has_gauss": int(rng_state[3]),
                "cached_gaussian": float(rng_state[4]),
            },
        }

    def load_state(self, payload: dict[str, Any]) -> None:
        ids = list(payload.get("task_ids") or [])
        if ids:
            self.bind(ids)
        if payload.get("I_alpha") is not None:
            self.I_alpha = np.array(payload["I_alpha"], dtype=np.float64)
            self.I_beta = np.array(payload["I_beta"], dtype=np.float64)
            self.T_last = np.array(payload["T_last"], dtype=np.int64)
        rng = payload.get("rng_state")
        if rng:
            keys = np.array(rng["keys"], dtype=np.uint32)
            self._rng.set_state(("MT19937", keys, int(rng["pos"]), int(rng["has_gauss"]), float(rng["cached_gaussian"])))
