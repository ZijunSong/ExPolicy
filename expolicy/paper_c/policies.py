"""Acquisition policies. None of them is a policy LLM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from expolicy.paper_c.constants import (
    BEST_STATIC,
    COVERAGE,
    CURRENT_REWARD,
    DIFFICULTY,
    FAILURE,
    LEARNING_PROGRESS,
    M_NO_STATE,
    M_SHUFFLE,
    UNCERTAINTY,
    UNIFORM,
    M,
)
from expolicy.paper_c.features import AcquisitionHistory, LeakageError
from expolicy.paper_c.predictor import GainPredictor, fit_gain_predictor


@dataclass(frozen=True)
class AcquisitionView:
    env: str
    round_index: int
    state_hash: str
    families: tuple[str, ...]
    remaining: Mapping[str, int]
    features_by_family: Mapping[str, Mapping[str, Any]]
    pool_hash: str

    def __getattr__(self, name: str) -> Any:
        if "future" in name or name in {"utility", "gain_target", "shadow_after", "y_after", "future_score_offline"}:
            raise LeakageError(f"selector requested {name}")
        raise AttributeError(name)


def _available(view: AcquisitionView) -> list[str]:
    return [family for family in view.families if int(view.remaining.get(family, 0)) > 0]


class UniformPolicy:
    name = UNIFORM

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.history = AcquisitionHistory()

    def select(self, view: AcquisitionView) -> str:
        return str(self.rng.choice(_available(view)))

    def observe_outcome(self, family: str, success: float) -> None:
        self.history.observe(family, success)

    def observe_gain(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        del record
        return None


class CoveragePolicy:
    name = COVERAGE

    def __init__(self):
        self.history = AcquisitionHistory()

    def select(self, view: AcquisitionView) -> str:
        return min(_available(view), key=lambda family: (self.history.stats(family).n, family))

    def observe_outcome(self, family: str, success: float) -> None:
        self.history.observe(family, success)

    def observe_gain(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        del record
        return None


class BestStaticPolicy:
    name = BEST_STATIC

    def __init__(self, weights: Mapping[str, float], rng: np.random.Generator):
        self.weights = {str(key): float(value) for key, value in weights.items()}
        self.rng = rng
        self.history = AcquisitionHistory()

    def select(self, view: AcquisitionView) -> str:
        available = _available(view)
        weights = [max(0.0, self.weights.get(family, 0.0)) for family in available]
        total = sum(weights)
        if total <= 0.0:
            return str(self.rng.choice(available))
        probabilities = [weight / total for weight in weights]
        return str(self.rng.choice(available, p=probabilities))

    def observe_outcome(self, family: str, success: float) -> None:
        self.history.observe(family, success)

    def observe_gain(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        del record
        return None


class _HistoryScorePolicy:
    def __init__(self, name: str, rng: np.random.Generator, *, epsilon: float, unseen: float):
        self.name = name
        self.rng = rng
        self.epsilon = float(epsilon)
        self.unseen = float(unseen)
        self.history = AcquisitionHistory()

    def _score(self, family: str, view: AcquisitionView) -> float:
        stats = self.history.by_family.get(family)
        if self.name == FAILURE:
            if stats is None or stats.n == 0:
                return self.unseen
            return float(stats.fail_rate() or 0.0)
        if self.name == CURRENT_REWARD:
            if stats is None or stats.n == 0:
                return self.unseen
            return float(stats.success_rate() or 0.0)
        if self.name == LEARNING_PROGRESS:
            if stats is None or stats.progress() is None:
                return self.unseen
            return float(stats.progress() or 0.0)
        if self.name == UNCERTAINTY:
            if stats is None or stats.variance() is None:
                return 1.0
            return float(stats.variance() or 0.0)
        if self.name == DIFFICULTY:
            return float(view.features_by_family[family]["public_difficulty"])
        raise KeyError(self.name)

    def select(self, view: AcquisitionView) -> str:
        available = _available(view)
        if self.name != DIFFICULTY and self.rng.random() < self.epsilon:
            return str(self.rng.choice(available))
        scored = [(self._score(family, view), family) for family in available]
        if self.name == LEARNING_PROGRESS and all(score == self.unseen for score, _family in scored):
            return str(self.rng.choice(available))
        best = max(score for score, _family in scored)
        winners = [family for score, family in scored if score == best]
        if len(winners) > 1 and self.name in {FAILURE, CURRENT_REWARD, LEARNING_PROGRESS, UNCERTAINTY, DIFFICULTY}:
            return str(self.rng.choice(winners)) if self.name != DIFFICULTY else sorted(winners)[0]
        return winners[0]

    def observe_outcome(self, family: str, success: float) -> None:
        self.history.observe(family, success)

    def observe_gain(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        del record
        return None


class GainPolicy:
    """Epsilon-greedy on predicted post-assimilation gain. Falls back to uniform without a model."""

    def __init__(
        self,
        name: str,
        model: GainPredictor | None,
        feature_names: Sequence[str],
        rng: np.random.Generator,
        *,
        epsilon: float = 0.1,
        rows: Sequence[Mapping[str, Any]] | None = None,
        min_rows: int = 8,
        alpha: float = 1.0,
        label_shuffle: int | None = None,
    ):
        self.name = name
        self.model = model
        self.feature_names = tuple(feature_names)
        self.rng = rng
        self.epsilon = float(epsilon)
        self.rows = [dict(row) for row in (rows or [])]
        self.min_rows = int(min_rows)
        self.alpha = float(alpha)
        self.label_shuffle = label_shuffle
        self.history = AcquisitionHistory()
        self.online_families: list[str] = []

    def select(self, view: AcquisitionView) -> str:
        available = _available(view)
        if self.model is None or self.rng.random() < self.epsilon:
            return str(self.rng.choice(available))
        scored: list[tuple[float, str]] = []
        for family in available:
            scored.append((self.model.predict(view.features_by_family[family]), family))
        best = max(score for score, _family in scored)
        winners = sorted(family for score, family in scored if abs(score - best) <= 1e-9)
        return winners[0]

    def observe_outcome(self, family: str, success: float) -> None:
        self.history.observe(family, success)

    def observe_gain(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        """Online rows are families that were actually selected. Refit does not invent the others."""
        if str(record.get("split") or "") == "future_test":
            raise LeakageError("online gain update refused a future label")
        self.rows.append(dict(record))
        self.online_families.append(str(record.get("family")))
        fitted = fit_gain_predictor(
            self.rows,
            self.feature_names,
            alpha=self.alpha,
            min_rows=self.min_rows,
            label_shuffle=self.label_shuffle,
        )
        if fitted["model"] is not None:
            self.model = fitted["model"]
        return fitted


def fit_best_static(rows: Sequence[Mapping[str, Any]], *, gain_key: str = "gain_eval") -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(gain_key)
        if value is None:
            continue
        buckets.setdefault(str(row["family"]), []).append(float(value))
    if not buckets:
        return {}
    means = {family: sum(values) / len(values) for family, values in buckets.items()}
    positive = {family: max(0.0, gain) for family, gain in means.items()}
    total = sum(positive.values())
    if total <= 0.0:
        share = 1.0 / len(means)
        return {family: share for family in means}
    return {family: positive[family] / total for family in means}


def freeze_baseline(dev_scores: Mapping[str, float], candidates: Sequence[str]) -> str:
    """Pick the comparison policy on development sequences only."""
    pool = [name for name in candidates if dev_scores.get(name) is not None]
    if not pool:
        raise RuntimeError("no development baseline score")
    return min(pool, key=lambda name: (-float(dev_scores[name]), name))


def make_policy(
    name: str,
    rng: np.random.Generator,
    *,
    epsilon: float,
    static_weights: Mapping[str, float] | None = None,
    model: GainPredictor | None = None,
    no_state_model: GainPredictor | None = None,
    shuffle_model: GainPredictor | None = None,
    feature_names: Sequence[str] = (),
    no_state_names: Sequence[str] = (),
    train_rows: Sequence[Mapping[str, Any]] | None = None,
    min_rows: int = 8,
    alpha: float = 1.0,
) -> Any:
    if name == UNIFORM:
        return UniformPolicy(rng)
    if name == COVERAGE:
        return CoveragePolicy()
    if name == BEST_STATIC:
        return BestStaticPolicy(static_weights or {}, rng)
    if name == FAILURE:
        return _HistoryScorePolicy(FAILURE, rng, epsilon=epsilon, unseen=1.0)
    if name == CURRENT_REWARD:
        return _HistoryScorePolicy(CURRENT_REWARD, rng, epsilon=epsilon, unseen=0.0)
    if name == LEARNING_PROGRESS:
        return _HistoryScorePolicy(LEARNING_PROGRESS, rng, epsilon=epsilon, unseen=0.0)
    if name == UNCERTAINTY:
        return _HistoryScorePolicy(UNCERTAINTY, rng, epsilon=epsilon, unseen=1.0)
    if name == DIFFICULTY:
        return _HistoryScorePolicy(DIFFICULTY, rng, epsilon=0.0, unseen=0.0)
    if name == M:
        return GainPolicy(
            M, model, feature_names, rng, epsilon=epsilon, rows=train_rows, min_rows=min_rows, alpha=alpha
        )
    if name == M_SHUFFLE:
        return GainPolicy(
            M_SHUFFLE,
            shuffle_model,
            feature_names,
            rng,
            epsilon=epsilon,
            rows=train_rows,
            min_rows=min_rows,
            alpha=alpha,
            label_shuffle=0,
        )
    if name == M_NO_STATE:
        return GainPolicy(
            M_NO_STATE,
            no_state_model,
            no_state_names,
            rng,
            epsilon=epsilon,
            rows=train_rows,
            min_rows=min_rows,
            alpha=alpha,
        )
    raise KeyError(name)
