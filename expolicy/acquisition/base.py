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
        cap_from_recent: list[str] = []
        if recent:
            from collections import Counter

            fail_caps = Counter(e.capability or "unknown" for e in recent if not e.success)
            cap_from_recent = [name for name, _ in fail_caps.most_common()]
        low_score = [
            name for name, _ in sorted(capability_profile.capability_scores.items(), key=lambda kv: kv[1])
        ]
        failed_caps = list(dict.fromkeys(cap_from_recent + capability_profile.frontier_names() + low_score))
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


class FrontierCapNormAcquisition(AcquisitionPolicy):
    """Frontier scores with capability-wise z-normalization at ranking time.

    Does not change the underlying frontier signal, read future probes, or
    inject a temporal bonus. Ranking uses z = (score - cap_mean) / (cap_std + eps).
    """

    name = "frontier_capnorm"

    def propose(
        self,
        agent_state: PersistentAgentState,
        capability_profile: CapabilityProfile,
        candidate_pool: Sequence[Task],
        budget: int,
        *,
        recent: Sequence[Experience] | None = None,
    ) -> list[AcquisitionProposal]:
        inner = FrontierAcquisition()
        props = inner.propose(agent_state, capability_profile, candidate_pool, budget, recent=recent)
        raw = dict(props[0].scores or {})
        by_cap: dict[str, list[float]] = {}
        for task in candidate_pool:
            by_cap.setdefault(task.capability or "unknown", []).append(float(raw.get(task.task_id, 0.0)))
        stats: dict[str, tuple[float, float]] = {}
        for cap, vals in by_cap.items():
            mu = sum(vals) / len(vals) if vals else 0.0
            var = sum((v - mu) ** 2 for v in vals) / len(vals) if vals else 0.0
            stats[cap] = (mu, math.sqrt(var))
        zscores: dict[str, float] = {}
        for task in candidate_pool:
            mu, sd = stats.get(task.capability or "unknown", (0.0, 0.0))
            zscores[task.task_id] = (float(raw.get(task.task_id, 0.0)) - mu) / (sd + 1e-8)
        ranked = sorted(candidate_pool, key=lambda t: zscores[t.task_id], reverse=True)[:budget]
        return [
            AcquisitionProposal(
                mode="frontier_capnorm",
                task_ids=[t.task_id for t in ranked],
                scores=zscores,
                rationale="frontier score z-normalized within capability; no temporal bonus",
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


def make_acquisition(name: str, seed: int = 0, **kwargs) -> AcquisitionPolicy:
    mapping = {
        "passive": PassiveAcquisition,
        "uncertainty": UncertaintyAcquisition,
        "failure_replay": FailureReplayAcquisition,
        "frontier": FrontierAcquisition,
        "frontier_capnorm": FrontierCapNormAcquisition,
        "reward_seeking": RewardSeekingAcquisition,
        "random": PassiveAcquisition,
    }
    if name in {"coverage", "coverage_balanced"}:
        from expolicy.acquisition.coverage import CoverageBalancedAcquisition

        return CoverageBalancedAcquisition(seed=seed)
    if name in {"ace_skill_prioritized", "ace_skill"}:
        from expolicy.acquisition.ace_skill import AceSkillPrioritizedAcquisition

        return AceSkillPrioritizedAcquisition(seed=seed, **kwargs)
    if name in {"exact_ava", "ava"}:
        from expolicy.acquisition.ava import ExactAVAAcquisition

        return ExactAVAAcquisition(seed=seed, **kwargs)
    if name in {"random_cluster"}:
        from expolicy.acquisition.ava import RandomClusterAcquisition

        return RandomClusterAcquisition(seed=seed, **kwargs)
    if name in {"rasa", "rasa_full", "rasa_k8", "rasa_qonly", "rasa_nodiversity"}:
        from expolicy.acquisition.rasa import RASAAcquisition

        qonly = bool(kwargs.pop("qonly", name == "rasa_qonly"))
        diversity = bool(kwargs.pop("diversity", name != "rasa_nodiversity"))
        return RASAAcquisition(seed=seed, qonly=qonly, diversity=diversity, **kwargs)
    if name == "learned":
        from expolicy.acquisition.learned_policy import LearnedAcquisition

        return LearnedAcquisition()
    if name not in mapping:
        raise ValueError(f"unknown acquisition policy {name}")
    cls = mapping[name]
    if cls is PassiveAcquisition:
        return cls(seed=seed)
    return cls()
