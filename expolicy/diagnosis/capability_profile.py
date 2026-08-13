"""Failure extraction, embedding clustering, capability frontier."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans

from expolicy.agents.llm_agent import infer_failure
from expolicy.core.capability import CapabilityProfile, FrontierCapability
from expolicy.core.experience import Experience
from expolicy.core.types import CapabilityName, FailureType, FrontierBand
from expolicy.evaluation.metrics import score_experience
from expolicy.models.embedder import Embedder


def extract_failure(experience: Experience) -> FailureType:
    if experience.success:
        return FailureType.NONE
    if experience.failure_type and experience.failure_type != FailureType.UNKNOWN:
        return experience.failure_type
    return infer_failure(experience)


class FailureClusterer:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder

    def cluster(self, experiences: Sequence[Experience], k: int | None = None) -> dict[str, list[str]]:
        failures = [e for e in experiences if not e.success]
        if not failures:
            return {}
        if self.embedder is None:
            grouped: dict[str, list[str]] = defaultdict(list)
            for exp in failures:
                grouped[extract_failure(exp).value].append(exp.experience_id)
            return dict(grouped)
        texts = [f"{e.question}\n{e.predicted_answer}\n{e.transcript()[:800]}" for e in failures]
        vecs = self.embedder.encode(texts)
        k = k or min(6, len(failures))
        k = max(1, k)
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(vecs)
        grouped = defaultdict(list)
        for exp, lab in zip(failures, labels, strict=True):
            grouped[f"cluster_{int(lab)}"].append(exp.experience_id)
        return dict(grouped)


class CapabilityDiagnoser:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.clusterer = FailureClusterer(embedder)

    def diagnose(
        self,
        experiences: Sequence[Experience],
        *,
        previous: CapabilityProfile | None = None,
        agent_version: str = "agent_v000",
    ) -> CapabilityProfile:
        by_cap: dict[str, list[Experience]] = defaultdict(list)
        failures = Counter()
        for exp in experiences:
            cap = exp.capability or "unknown"
            by_cap[cap].append(exp)
            if not exp.success:
                failures[extract_failure(exp).value] += 1
        scores: dict[str, float] = {}
        uncertainty: dict[str, float] = {}
        for cap, items in by_cap.items():
            rows = [score_experience(e) for e in items]
            acc = sum(r["success"] for r in rows) / len(rows)
            scores[cap] = acc
            # real uncertainty: disagreement + error rate
            unc_vals = [e.uncertainty for e in items if e.uncertainty is not None]
            if unc_vals:
                model_u = sum(unc_vals) / len(unc_vals)
            else:
                model_u = 1.0 - acc
            uncertainty[cap] = float(min(1.0, 0.5 * model_u + 0.5 * (1.0 - acc)))
        n_fail = sum(failures.values())
        fail_dist = {k: v / n_fail for k, v in failures.items()} if n_fail else {}
        regressions: dict[str, float] = {}
        if previous is not None:
            for cap, score in scores.items():
                prev = previous.capability_scores.get(cap)
                if prev is not None and score + 1e-9 < prev:
                    regressions[cap] = prev - score
        frontier = []
        for cap, score in scores.items():
            band = _band(score, cap in regressions)
            n_eval = len(by_cap[cap])
            learnability = float(np.clip(4 * score * (1 - score), 0.05, 1.0))  # peak at 0.5
            novelty = 1.0 if n_eval < 8 else max(0.2, 8 / n_eval)
            transfer = 0.8 if cap in {
                CapabilityName.MULTI_HOP.value,
                CapabilityName.QUERY_DECOMPOSITION.value,
                CapabilityName.SOURCE_RELIABILITY.value,
            } else 0.45
            frontier.append(
                FrontierCapability(
                    name=cap,
                    band=band,
                    score=score,
                    learnability=learnability,
                    novelty=novelty,
                    transfer_potential=transfer,
                    uncertainty=uncertainty[cap],
                    expected_cost=1.0 + 0.4 * (1.0 - score),
                    n_eval=n_eval,
                )
            )
        clusters = self.clusterer.cluster(experiences)
        return CapabilityProfile(
            agent_version=agent_version,
            capability_scores=scores,
            failure_distribution=fail_dist,
            uncertainty=uncertainty,
            recent_regressions=regressions,
            frontier_capabilities=frontier,
            n_trajectories=len(experiences),
            n_failures=n_fail,
            extra={"failure_clusters": {k: len(v) for k, v in clusters.items()}},
        )


def _band(score: float, regressed: bool) -> FrontierBand:
    if regressed:
        return FrontierBand.REGRESSED
    if score >= 0.85:
        return FrontierBand.MASTERED
    if 0.55 <= score < 0.85:
        return FrontierBand.LEARNING
    if 0.2 <= score < 0.55:
        return FrontierBand.FRONTIER
    if score < 0.2:
        return FrontierBand.UNKNOWN
    return FrontierBand.FRONTIER
