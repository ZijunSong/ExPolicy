"""Coverage-balanced acquisition: prevent 'uniform coverage accidentally wins'."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

from expolicy.core.capability import CapabilityProfile
from expolicy.core.experience import AcquisitionProposal, Experience
from expolicy.core.state import PersistentAgentState
from expolicy.environments.search.world import Task
from expolicy.acquisition.base import AcquisitionPolicy


class CoverageBalancedAcquisition(AcquisitionPolicy):
    name = "coverage_balanced"

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
        used = {str(e.task_id) for e in (recent or []) if e.task_id}
        pool = [t for t in candidate_pool if t.task_id not in used] or list(candidate_pool)
        buckets: dict[tuple[str, str], list[Task]] = defaultdict(list)
        for task in pool:
            ds = str((task.metadata or {}).get("dataset") or "unknown")
            cap = str(task.capability or "unknown")
            buckets[(ds, cap)].append(task)
        for key in buckets:
            self.rng.shuffle(buckets[key])
        keys = list(buckets)
        self.rng.shuffle(keys)
        chosen: list[Task] = []
        scores: dict[str, float] = {}
        while len(chosen) < budget:
            progress = False
            for key in keys:
                bucket = buckets[key]
                if not bucket:
                    continue
                task = bucket.pop()
                chosen.append(task)
                scores[task.task_id] = 1.0 / (1 + len(chosen))
                progress = True
                if len(chosen) >= budget:
                    break
            if not progress:
                break
        return [
            AcquisitionProposal(
                mode="coverage_balanced",
                task_ids=[t.task_id for t in chosen],
                scores=scores,
                rationale="round-robin over (dataset, capability) to maximize coverage",
            )
        ]
