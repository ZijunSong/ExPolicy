"""Acquisition policy interfaces and heuristic implementations."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Any, Sequence

from expolicy.core.capability import CapabilityProfile
from expolicy.core.experience import AcquisitionProposal, Experience
from expolicy.core.state import PersistentAgentState
from expolicy.environments.search.world import Task


class AcquisitionPolicy(ABC):
    name: str

    @abstractmethod
    def propose(
        self,
        agent_state: PersistentAgentState,
        capability_profile: CapabilityProfile,
        candidate_pool: Sequence[Task],
        budget: int,
        *,
        recent: Sequence[Experience] | None = None,
    ) -> list[AcquisitionProposal]: ...


class PassiveAcquisition(AcquisitionPolicy):
    name = "passive"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def propose(
        self,
        agent_state: PersistentAgentState,
        capability_profile: CapabilityProfile,
        candidate_pool: Sequence[Task],
        budget: int,
        *,
        recent: Sequence[Experience] | None = None,
    ) -> list[AcquisitionProposal]:
        pool = list(candidate_pool)
        self.rng.shuffle(pool)
        chosen = pool[:budget]
        return [
            AcquisitionProposal(
                mode="passive",
                task_ids=[t.task_id for t in chosen],
                scores={t.task_id: 0.0 for t in chosen},
                rationale="natural deployment stream (shuffled)",
            )
        ]


class UncertaintyAcquisition(AcquisitionPolicy):
    name = "uncertainty"

    def propose(
        self,
        agent_state: PersistentAgentState,
        capability_profile: CapabilityProfile,
        candidate_pool: Sequence[Task],
        budget: int,
        *,
        recent: Sequence[Experience] | None = None,
    ) -> list[AcquisitionProposal]:
        recent = list(recent or [])
        unc_by_cap = dict(capability_profile.uncertainty)
        scores: dict[str, float] = {}
        for task in candidate_pool:
            u = float(unc_by_cap.get(task.capability or "", 0.5))
            # real signal from recent trajectories of the same capability
            related = [e for e in recent if e.capability == task.capability and e.uncertainty is not None]
            if related:
                u = sum(float(e.uncertainty or 0.0) for e in related) / len(related)
            scores[task.task_id] = u
        ranked = sorted(candidate_pool, key=lambda t: scores[t.task_id], reverse=True)[:budget]
        return [
            AcquisitionProposal(
                mode="uncertainty",
                task_ids=[t.task_id for t in ranked],
                scores=scores,
                rationale="highest capability uncertainty from real model/eval signals",
            )
        ]


class FailureReplayAcquisition(AcquisitionPolicy):
    name = "failure_replay"

    def propose(
        self,
        agent_state: PersistentAgentState,
        capability_profile: CapabilityProfile,
        candidate_pool: Sequence[Task],
        budget: int,
        *,
        recent: Sequence[Experience] | None = None,
    ) -> list[AcquisitionProposal]:
        failed_caps = [
            name
            for name, _ in sorted(capability_profile.failure_distribution.items(), key=lambda kv: -kv[1])
        ]
        chosen: list[Task] = []
        for cap in failed_caps:
            for task in candidate_pool:
                if task.capability == cap and task not in chosen:
                    chosen.append(task)
                if len(chosen) >= budget:
                    break
            if len(chosen) >= budget:
                break
        if len(chosen) < budget:
            for task in candidate_pool:
                if task not in chosen:
                    chosen.append(task)
                if len(chosen) >= budget:
                    break
        return [
            AcquisitionProposal(
                mode="failure_replay",
                task_ids=[t.task_id for t in chosen[:budget]],
                rationale="replay tasks from dominant recent failure capabilities",
            )
        ]


class FrontierAcquisition(AcquisitionPolicy):
    name = "frontier"

    def propose(
        self,
        agent_state: PersistentAgentState,
        capability_profile: CapabilityProfile,
        candidate_pool: Sequence[Task],
        budget: int,
        *,
        recent: Sequence[Experience] | None = None,
    ) -> list[AcquisitionProposal]:
        frontier = {item.name: item for item in capability_profile.frontier_capabilities}
        scores: dict[str, float] = {}
        for task in candidate_pool:
            item = frontier.get(task.capability or "")
            if item is None:
                learnability = 0.5
                novelty = 0.5
                transfer = 0.5
                unc = float(capability_profile.uncertainty.get(task.capability or "", 0.5))
                cost = 1.0 + 0.3 * task.hops
            else:
                learnability = item.learnability
                novelty = item.novelty
                transfer = item.transfer_potential
                unc = item.uncertainty
                cost = max(item.expected_cost, 1e-3)
            scores[task.task_id] = (learnability * novelty * transfer * unc) / cost
        ranked = sorted(candidate_pool, key=lambda t: scores[t.task_id], reverse=True)[:budget]
        return [
            AcquisitionProposal(
                mode="frontier",
                task_ids=[t.task_id for t in ranked],
                scores=scores,
                rationale="learnability*novelty*transfer*uncertainty / cost",
            )
        ]


class RewardSeekingAcquisition(AcquisitionPolicy):
    name = "reward_seeking"

    def propose(
        self,
        agent_state: PersistentAgentState,
        capability_profile: CapabilityProfile,
        candidate_pool: Sequence[Task],
        budget: int,
        *,
        recent: Sequence[Experience] | None = None,
    ) -> list[AcquisitionProposal]:
        # Prefer currently high-scoring (already mastered) capabilities — opposite of learning utility.
        scores = {
            t.task_id: float(capability_profile.capability_scores.get(t.capability or "", 0.0))
            for t in candidate_pool
        }
        ranked = sorted(candidate_pool, key=lambda t: scores[t.task_id], reverse=True)[:budget]
        return [
            AcquisitionProposal(
                mode="reward_seeking",
                task_ids=[t.task_id for t in ranked],
                scores=scores,
                rationale="maximize immediate success probability, not future learning utility",
            )
        ]


def make_acquisition(name: str, seed: int = 0) -> AcquisitionPolicy:
    mapping = {
        "passive": PassiveAcquisition,
        "uncertainty": UncertaintyAcquisition,
        "failure_replay": FailureReplayAcquisition,
        "frontier": FrontierAcquisition,
        "reward_seeking": RewardSeekingAcquisition,
        "random": PassiveAcquisition,
    }
    if name == "learned":
        from expolicy.acquisition.learned_policy import LearnedAcquisition

        return LearnedAcquisition()
    if name not in mapping:
        raise ValueError(f"unknown acquisition policy {name}")
    cls = mapping[name]
    if cls is PassiveAcquisition:
        return cls(seed=seed)
    return cls()
