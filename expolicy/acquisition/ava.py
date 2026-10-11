"""Exact AVA: score candidate micro-batches by post-assimilation shadow gain.

Candidate construction uses unsupervised E5 clustering only (no value scores).
The selector uses ΔShadowFuture of the frozen CountOnly operator.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from expolicy.acquisition.base import AcquisitionPolicy
from expolicy.core.capability import CapabilityProfile
from expolicy.core.experience import AcquisitionProposal, Experience
from expolicy.core.state import PersistentAgentState
from expolicy.environments.search.world import Task
from expolicy.state.isolated_clone import EvidenceState, MIN_SUPPORT


def balanced_cluster_batches(
    remaining: Sequence[Task],
    embeddings: Mapping[str, np.ndarray],
    *,
    n_batches: int = 8,
    batch_size: int = 8,
    seed: int = 0,
    round_idx: int = 0,
) -> list[list[str]]:
    """Build M candidate batches from remaining tasks via k-means / nearest-centroid.

    Batches may overlap (competition). Caller must enforce unique commit.
    """
    pool = [t for t in remaining if t.task_id in embeddings]
    if not pool:
        pool = list(remaining)
    if not pool:
        return []
    rng = np.random.RandomState(int(seed) * 10007 + int(round_idx) * 17 + 3)
    ids = [t.task_id for t in pool]
    dim = int(next(iter(embeddings.values())).shape[0]) if embeddings else 8
    mat = np.stack(
        [np.asarray(embeddings[t.task_id], dtype=np.float32) if t.task_id in embeddings else rng.normal(size=dim).astype(np.float32) for t in pool],
        axis=0,
    )
    k = min(int(n_batches), len(pool))
    if k <= 1:
        order = list(ids)
        rng.shuffle(order)
        return [order[: min(batch_size, len(order))]]
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=int(rng.randint(0, 10_000_000)), n_init=10)
        labels = km.fit_predict(mat)
        centers = km.cluster_centers_
    except Exception:
        labels = rng.randint(0, k, size=len(pool))
        centers = np.stack([mat[labels == i].mean(axis=0) if np.any(labels == i) else mat[rng.randint(0, len(pool))] for i in range(k)])
    batches: list[list[str]] = []
    used_fill = set()
    for i in range(k):
        idxs = [j for j, lab in enumerate(labels) if int(lab) == i]
        if not idxs:
            idxs = list(range(len(pool)))
        center = centers[i]
        idxs.sort(key=lambda j: float(np.linalg.norm(mat[j] - center)))
        chosen = [ids[j] for j in idxs[:batch_size]]
        if len(chosen) < batch_size:
            rest = [tid for tid in ids if tid not in chosen]
            rng.shuffle(rest)
            for tid in rest:
                if len(chosen) >= batch_size:
                    break
                chosen.append(tid)
        batches.append(chosen[:batch_size])
        used_fill.update(chosen)
    while len(batches) < int(n_batches) and pool:
        order = list(ids)
        rng.shuffle(order)
        batches.append(order[: min(batch_size, len(order))])
    return batches[: int(n_batches)]


class ExactAVAAcquisition(AcquisitionPolicy):
    name = "exact_ava"

    def __init__(
        self,
        seed: int = 0,
        *,
        predicted_cap: Mapping[str, str] | None = None,
        embeddings: Mapping[str, Sequence[float]] | None = None,
        n_batches: int = 8,
        batch_size: int = 8,
        min_support: int = MIN_SUPPORT,
    ) -> None:
        self.seed = int(seed)
        self.predicted_cap = {str(k): str(v) for k, v in dict(predicted_cap or {}).items()}
        self.embeddings = {str(k): np.asarray(v, dtype=np.float32) for k, v in dict(embeddings or {}).items()}
        self.n_batches = int(n_batches)
        self.batch_size = int(batch_size)
        self.min_support = int(min_support)
        self.round_idx = 0
        self.last_candidates: list[dict[str, Any]] = []

    def propose_candidates(self, remaining: Sequence[Task], *, round_idx: int | None = None) -> list[list[str]]:
        r = self.round_idx if round_idx is None else int(round_idx)
        return balanced_cluster_batches(
            remaining,
            self.embeddings,
            n_batches=self.n_batches,
            batch_size=self.batch_size,
            seed=self.seed,
            round_idx=r,
        )

    def score_candidates(
        self,
        state: EvidenceState,
        batches: Sequence[Sequence[str]],
        value_fn,
    ) -> list[dict[str, Any]]:
        """value_fn(before_mask, after_mask) -> float. Does not mutate ``state``."""
        before = state.mask_key(self.min_support)
        snapshot_before = state.snapshot()
        rows = []
        for i, batch in enumerate(batches):
            cloned = state.clone()
            after_state = cloned.apply_batch(batch, self.predicted_cap)
            after = after_state.mask_key(self.min_support)
            # isolation: original state must be identical
            assert state.snapshot() == snapshot_before, "shadow scoring mutated live evidence state"
            delta = float(value_fn(before, after))
            rows.append(
                {
                    "candidate_idx": i,
                    "task_ids": list(batch),
                    "before_mask": before,
                    "after_mask": after,
                    "shadow_delta": delta,
                    "after_counts": after_state.counts(),
                }
            )
        self.last_candidates = rows
        return rows

    def select(self, scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not scored:
            return {"task_ids": [], "shadow_delta": 0.0, "candidate_idx": -1}
        best = max(scored, key=lambda r: (float(r.get("shadow_delta") or 0.0), -int(r.get("candidate_idx") or 0)))
        return dict(best)

    def propose(
        self,
        agent_state: PersistentAgentState,
        capability_profile: CapabilityProfile,
        candidate_pool: Sequence[Task],
        budget: int,
        *,
        recent: Sequence[Experience] | None = None,
    ) -> list[AcquisitionProposal]:
        batches = self.propose_candidates(candidate_pool)
        # without a value_fn this is coverage-only; runner should call score+select
        flat = []
        seen = set()
        for batch in batches:
            for tid in batch:
                if tid not in seen:
                    seen.add(tid)
                    flat.append(tid)
                if len(flat) >= budget:
                    break
            if len(flat) >= budget:
                break
        return [
            AcquisitionProposal(
                mode=self.name,
                task_ids=flat[:budget],
                rationale="cluster coverage fallback (runner should use score_candidates)",
            )
        ]


class RandomClusterAcquisition(AcquisitionPolicy):
    name = "random_cluster"

    def __init__(
        self,
        seed: int = 0,
        *,
        embeddings: Mapping[str, Sequence[float]] | None = None,
        n_batches: int = 8,
        batch_size: int = 8,
    ) -> None:
        self.seed = int(seed)
        self.embeddings = {str(k): np.asarray(v, dtype=np.float32) for k, v in dict(embeddings or {}).items()}
        self.n_batches = int(n_batches)
        self.batch_size = int(batch_size)
        self.round_idx = 0
        self.rng = np.random.RandomState(self.seed)

    def propose(
        self,
        agent_state: PersistentAgentState,
        capability_profile: CapabilityProfile,
        candidate_pool: Sequence[Task],
        budget: int,
        *,
        recent: Sequence[Experience] | None = None,
    ) -> list[AcquisitionProposal]:
        batches = balanced_cluster_batches(
            candidate_pool,
            self.embeddings,
            n_batches=self.n_batches,
            batch_size=self.batch_size,
            seed=self.seed,
            round_idx=self.round_idx,
        )
        self.round_idx += 1
        if not batches:
            return [AcquisitionProposal(mode=self.name, task_ids=[], rationale="empty pool")]
        pick = batches[int(self.rng.randint(0, len(batches)))]
        return [
            AcquisitionProposal(
                mode=self.name,
                task_ids=list(pick)[:budget],
                rationale="uniform among unsupervised cluster batches",
            )
        ]
