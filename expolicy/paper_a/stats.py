"""Small numeric helpers. Shared probe tasks are not treated as independent batches."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("mean of empty sequence")
    return float(arr.mean())


def normal_mean_ci(values: Sequence[float], *, z: float = 1.96) -> tuple[float, float, float]:
    """Descriptive normal interval. n is the number of values, not a repeated-run multiplier.

    A zero-width interval here is not evidence that the population noise is zero.
    Use ``interval_claim`` before treating it as a population statement.
    """
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        raise ValueError("ci of empty sequence")
    center = float(arr.mean())
    if n == 1 or float(arr.std(ddof=0)) == 0.0:
        return center, center, center
    se = float(arr.std(ddof=1) / np.sqrt(n))
    return center, center - z * se, center + z * se


def interval_claim(values: Sequence[float]) -> dict[str, object]:
    """Say whether a normal interval may be read as a population-noise statement."""
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    zero_width = n <= 1 or (n > 0 and float(arr.std(ddof=0)) == 0.0)
    return {
        "estimator": "normal_mean_descriptive",
        "n": n,
        "zero_width": zero_width,
        "population_noise_claim": False if zero_width or n < 2 else True,
        "reason": "zero-width or n<2 intervals are descriptive only" if zero_width or n < 2 else "descriptive normal interval",
    }


def clopper_pearson(k: int, n: int, *, confidence: float = 0.975) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval. Default confidence is 97.5%, not 95%."""
    if n <= 0 or k < 0 or k > n:
        raise ValueError("clopper_pearson requires 0 <= k <= n and n > 0")
    from scipy.stats import beta

    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - tail, k + 1, n - k))
    return lower, upper


def paired_flip_interval(n_pos: int, n_neg: int, n: int) -> dict[str, object]:
    """Conservative interval for the mean of paired +1/0/-1 flips on one repeat.

    Positive and negative flip rates each get a 97.5% Clopper-Pearson interval.
    Their combination ``[L+ - U-, U+ - L-]`` has joint coverage at least 95% when
    trials are independent. It is a diagnostic interval, not a family-wise claim.
    """
    if n <= 1:
        return {"status": "INSUFFICIENT_PRECISION", "n": n, "estimator": "clopper_pearson_paired_flips"}
    pos_l, pos_u = clopper_pearson(n_pos, n)
    neg_l, neg_u = clopper_pearson(n_neg, n)
    return {
        "status": "ok",
        "estimator": "clopper_pearson_paired_flips",
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "ci": [pos_l - neg_u, pos_u - neg_l],
        "independence": "task-level independence is assumed and is not claimed when tasks share a template",
        "population_family_claim": False,
    }


def grouped_bootstrap_means(
    values: Sequence[float],
    groups: Sequence[str],
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray | None:
    """Cluster bootstrap. Returns None when fewer than two groups exist."""
    arr = np.asarray(list(values), dtype=float)
    labels = np.asarray([str(g) for g in groups])
    if arr.size == 0 or arr.size != labels.size:
        raise ValueError("values and groups must be the same non-empty length")
    uniq = np.unique(labels)
    if uniq.size < 2 or n_boot < 1:
        return None
    rng = np.random.default_rng(seed)
    buckets = {g: arr[labels == g] for g in uniq}
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        drawn = rng.choice(uniq, size=uniq.size, replace=True)
        sample = np.concatenate([buckets[g] for g in drawn])
        out[i] = float(sample.mean())
    return out


def percentile_interval(samples: np.ndarray | None, *, alpha: float = 0.05) -> tuple[float, float] | None:
    if samples is None or samples.size == 0:
        return None
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi


def quantile(samples: Sequence[float], q: float) -> float:
    arr = np.asarray(list(samples), dtype=float)
    if arr.size == 0:
        raise ValueError("quantile of empty sequence")
    return float(np.quantile(arr, q))
