"""Bayesian linear Thompson Sampling heads. One head per discrete action."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class BayesianLinearHead:
    def __init__(self, dim: int, *, prior_precision: float = 1.0, reward_noise: float = 0.05) -> None:
        self.dim = int(dim)
        self.prior_precision = float(prior_precision)
        self.reward_noise = float(reward_noise)
        self.A = np.eye(self.dim, dtype=np.float64) * self.prior_precision
        self.b = np.zeros(self.dim, dtype=np.float64)
        self.n_updates = 0

    @property
    def mean(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    @property
    def cov(self) -> np.ndarray:
        return np.linalg.inv(self.A)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        mean = self.mean
        cov = self.cov
        # numerical jitter
        cov = cov + np.eye(self.dim) * 1e-8
        return rng.multivariate_normal(mean, cov)

    def score(self, phi: np.ndarray, theta: np.ndarray | None = None) -> float:
        vec = self.mean if theta is None else theta
        return float(np.dot(phi, vec))

    def update(self, phi: np.ndarray, reward: float) -> dict[str, float]:
        phi = np.asarray(phi, dtype=np.float64).reshape(-1)
        if phi.shape[0] != self.dim:
            raise ValueError(f"feature dim {phi.shape[0]} != {self.dim}")
        mean_before = self.mean.copy()
        a_before = self.A.copy()
        inv_var = 1.0 / max(self.reward_noise, 1e-8)
        self.A = self.A + inv_var * np.outer(phi, phi)
        self.b = self.b + inv_var * float(reward) * phi
        self.n_updates += 1
        mean_after = self.mean
        delta = float(np.linalg.norm(mean_after - mean_before) + np.linalg.norm(self.A - a_before))
        return {
            "parameter_delta_norm": delta,
            "n_updates": float(self.n_updates),
            "reward": float(reward),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "prior_precision": self.prior_precision,
            "reward_noise": self.reward_noise,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "mean": self.mean.tolist(),
            "n_updates": self.n_updates,
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "BayesianLinearHead":
        head = cls(
            int(payload["dim"]),
            prior_precision=float(payload.get("prior_precision") or 1.0),
            reward_noise=float(payload.get("reward_noise") or 0.05),
        )
        head.A = np.asarray(payload["A"], dtype=np.float64)
        head.b = np.asarray(payload["b"], dtype=np.float64)
        head.n_updates = int(payload.get("n_updates") or 0)
        return head


class LinearThompsonPolicy:
    def __init__(
        self,
        actions: Sequence[str],
        dim: int,
        *,
        prior_precision: float = 1.0,
        reward_noise: float = 0.05,
    ) -> None:
        self.actions = list(actions)
        self.dim = int(dim)
        self.prior_precision = float(prior_precision)
        self.reward_noise = float(reward_noise)
        self.heads = {
            a: BayesianLinearHead(dim, prior_precision=prior_precision, reward_noise=reward_noise)
            for a in self.actions
        }

    def select(
        self,
        features: dict[str, np.ndarray],
        valid: Sequence[str],
        rng: np.random.Generator,
    ) -> tuple[str, dict[str, float], dict[str, float]]:
        sampled: dict[str, float] = {}
        scores: dict[str, float] = {}
        for action in self.actions:
            phi = features[action]
            theta = self.heads[action].sample(rng)
            sampled[action] = float(np.dot(phi, theta))
            scores[action] = self.heads[action].score(phi)
        valid_set = list(valid)
        if not valid_set:
            raise RuntimeError("no valid actions")
        chosen = max(valid_set, key=lambda a: sampled[a])
        return chosen, scores, sampled

    def update(self, action: str, phi: np.ndarray, reward: float) -> dict[str, Any]:
        if action not in self.heads:
            raise KeyError(action)
        info = self.heads[action].update(phi, reward)
        info["action"] = action
        return info

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "linear_thompson",
            "actions": self.actions,
            "dim": self.dim,
            "prior_precision": self.prior_precision,
            "reward_noise": self.reward_noise,
            "heads": {a: h.snapshot() for a, h in self.heads.items()},
        }

    def load_snapshot(self, payload: dict[str, Any]) -> None:
        self.actions = list(payload["actions"])
        self.dim = int(payload["dim"])
        self.prior_precision = float(payload.get("prior_precision") or 1.0)
        self.reward_noise = float(payload.get("reward_noise") or 0.05)
        self.heads = {a: BayesianLinearHead.from_snapshot(blob) for a, blob in payload["heads"].items()}

    def mean_dict(self) -> dict[str, list[float]]:
        return {a: h.mean.tolist() for a, h in self.heads.items()}
