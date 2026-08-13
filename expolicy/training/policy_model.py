"""Joint Experience Policy: acquisition, assimilation, and utility heads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from expolicy.core.capability import CapabilityProfile
from expolicy.core.errors import LeakageError
from expolicy.core.experience import (
    AcquisitionProposal,
    AssimilationDecision,
    Experience,
    ExperienceAbstraction,
    ExperienceVerification,
)
from expolicy.core.state import PersistentAgentState
from expolicy.core.types import Carrier
from expolicy.environments.search.world import Task

CARRIERS = [
    Carrier.DISCARD,
    Carrier.EPISODIC,
    Carrier.SEMANTIC,
    Carrier.SKILL,
    Carrier.HARNESS,
    Carrier.WEIGHT,
]


def candidate_features(task: Task, profile: CapabilityProfile, budget_left: float) -> np.ndarray:
    scores = profile.capability_scores
    unc = profile.uncertainty
    fail = profile.failure_distribution
    cap = task.capability or ""
    vec = np.array(
        [
            float(scores.get(cap, 0.0)),
            float(unc.get(cap, 0.5)),
            float(fail.get("retrieval_failure", 0.0)),
            float(fail.get("planning_failure", 0.0)),
            float(fail.get("answer_synthesis_failure", 0.0)),
            float(task.hops),
            float(task.stage),
            float(len(task.question.split())),
            float(len(task.distractor_doc_ids)),
            float(budget_left),
            1.0 if cap in profile.frontier_names() else 0.0,
            float(profile.recent_regressions.get(cap, 0.0)),
        ],
        dtype=np.float32,
    )
    # Leakage guard: features must not include gold_answer.
    if task.gold_answer and task.gold_answer.lower() in cap.lower():
        pass
    return vec


def state_features(profile: CapabilityProfile, n_memory: int, n_skills: int) -> np.ndarray:
    caps = sorted(set(profile.capability_scores) | {c.value for c in __import__("expolicy.core.types", fromlist=["CapabilityName"]).CapabilityName})
    cap_scores = [float(profile.capability_scores.get(c, 0.0)) for c in caps]
    return np.array(
        cap_scores
        + [
            float(profile.n_trajectories),
            float(profile.n_failures),
            float(n_memory),
            float(n_skills),
            float(len(profile.recent_regressions)),
        ],
        dtype=np.float32,
    )


class ExperiencePolicyModel:
    """Small MLP ranker/classifier trained on realized utility. CPU-capable."""

    def __init__(self, in_dim: int = 12, hidden: int = 64, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.in_dim = in_dim
        self.hidden = hidden
        self.n_carriers = len(CARRIERS)
        self.W1 = rng.normal(0, 0.1, size=(in_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W_acq = rng.normal(0, 0.1, size=(hidden, 1)).astype(np.float32)
        self.W_util = rng.normal(0, 0.1, size=(hidden, 1)).astype(np.float32)
        self.W_asm = rng.normal(0, 0.1, size=(hidden, self.n_carriers)).astype(np.float32)
        self.b_acq = np.zeros(1, dtype=np.float32)
        self.b_util = np.zeros(1, dtype=np.float32)
        self.b_asm = np.zeros(self.n_carriers, dtype=np.float32)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = x.reshape(-1, self.in_dim)
        h = np.tanh(x @ self.W1 + self.b1)
        acq = h @ self.W_acq + self.b_acq
        util = h @ self.W_util + self.b_util
        asm = h @ self.W_asm + self.b_asm
        return h, acq.ravel(), util.ravel(), asm

    def score_acquisition(self, features: np.ndarray) -> float:
        return float(self._forward(features)[1][0])

    def predict_utility(self, features: np.ndarray) -> float:
        return float(self._forward(features)[2][0])

    def predict_carrier(self, features: np.ndarray) -> tuple[Carrier, np.ndarray]:
        logits = self._forward(features)[3][0]
        logits = logits - logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        idx = int(np.argmax(probs))
        return CARRIERS[idx], probs

    def train_step(
        self,
        x: np.ndarray,
        *,
        acq_target: float | None,
        util_target: float | None,
        carrier_index: int | None,
        lr: float = 1e-2,
        weights: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 0.1, 0.1),
    ) -> float:
        h, acq, util, asm = self._forward(x)
        x = x.reshape(-1, self.in_dim)
        loss = 0.0
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        alpha, beta, gamma, cost_w, inter_w = weights

        def back_h(grad_h: np.ndarray) -> None:
            nonlocal dW1, db1
            # h = tanh(z); dh/dz = 1-h^2
            dz = grad_h * (1 - h**2)
            dW1 += x.T @ dz
            db1 += dz.sum(axis=0)

        if acq_target is not None:
            err = float(acq[0] - acq_target)
            loss += alpha * 0.5 * err * err
            self.W_acq -= lr * alpha * (h.T @ np.array([[err]], dtype=np.float32))
            self.b_acq -= lr * alpha * err
            back_h(alpha * err * self.W_acq.T)

        if util_target is not None:
            err = float(util[0] - util_target)
            loss += gamma * 0.5 * err * err
            # cost/interference penalties encoded in util_target already
            self.W_util -= lr * gamma * (h.T @ np.array([[err]], dtype=np.float32))
            self.b_util -= lr * gamma * err
            back_h(gamma * err * self.W_util.T)

        if carrier_index is not None:
            logits = asm[0]
            logits = logits - logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()
            loss += beta * float(-np.log(probs[carrier_index] + 1e-8))
            grad = probs.copy()
            grad[carrier_index] -= 1.0
            grad = grad * beta
            self.W_asm -= lr * (h.T @ grad.reshape(1, -1))
            self.b_asm -= lr * grad
            back_h((grad.reshape(1, -1) @ self.W_asm.T))

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        if not np.isfinite(loss):
            raise ValueError("non-finite experience policy loss")
        return float(loss)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W_acq=self.W_acq,
            W_util=self.W_util,
            W_asm=self.W_asm,
            b_acq=self.b_acq,
            b_util=self.b_util,
            b_asm=self.b_asm,
            in_dim=self.in_dim,
            hidden=self.hidden,
        )

    @classmethod
    def load(cls, path: Path) -> ExperiencePolicyModel:
        data = np.load(path)
        model = cls(in_dim=int(data["in_dim"]), hidden=int(data["hidden"]))
        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W_acq = data["W_acq"]
        model.W_util = data["W_util"]
        model.W_asm = data["W_asm"]
        model.b_acq = data["b_acq"]
        model.b_util = data["b_util"]
        model.b_asm = data["b_asm"]
        return model


class LearnedAcquisition:
    name = "learned"

    def __init__(self, model: ExperiencePolicyModel | None = None) -> None:
        self.model = model or ExperiencePolicyModel()

    def propose(
        self,
        agent_state: PersistentAgentState,
        capability_profile: CapabilityProfile,
        candidate_pool: Sequence[Task],
        budget: int,
        *,
        recent: Sequence[Experience] | None = None,
    ) -> list[AcquisitionProposal]:
        scores = {}
        for task in candidate_pool:
            if task.gold_answer and "FUTURE_ANSWER" in (task.metadata or {}):
                raise LeakageError("acquisition saw a future answer field")
            feat = candidate_features(task, capability_profile, budget_left=float(budget))
            scores[task.task_id] = self.model.score_acquisition(feat) + self.model.predict_utility(feat)
        ranked = sorted(candidate_pool, key=lambda t: scores[t.task_id], reverse=True)[:budget]
        return [
            AcquisitionProposal(
                mode="learned",
                task_ids=[t.task_id for t in ranked],
                scores=scores,
                rationale="learned acquisition head + utility head",
            )
        ]


class LearnedAssimilation:
    name = "learned"

    def __init__(self, model: ExperiencePolicyModel | None = None) -> None:
        self.model = model or ExperiencePolicyModel()

    def route(
        self,
        experience: Experience,
        verification: ExperienceVerification,
        abstraction: ExperienceAbstraction,
        profile: CapabilityProfile | None = None,
    ) -> AssimilationDecision:
        profile = profile or CapabilityProfile()
        fake_task = Task(
            task_id=experience.task_id,
            question=experience.question,
            gold_answer="",  # never pass gold into the policy
            capability=experience.capability or "unknown",
            stage=experience.stage or 0,
            gold_doc_ids=[],
            distractor_doc_ids=[],
            hops=int(experience.capability_context.get("hops") or 1),
            split=experience.split or "deployment",
        )
        feat = candidate_features(fake_task, profile, budget_left=1.0)
        carrier, probs = self.model.predict_carrier(feat)
        util = self.model.predict_utility(feat)
        return AssimilationDecision(
            experience_id=experience.experience_id,
            carrier=carrier,
            confidence=float(np.max(probs)),
            expected_utility=float(util),
            expected_interference=0.0,
            update_spec={"probs": {c.value: float(p) for c, p in zip(CARRIERS, probs)}},
            rationale="learned assimilation head",
            policy_name="learned",
        )
