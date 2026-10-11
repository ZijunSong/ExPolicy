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
    """Return (mean, lo, hi). n is the number of values, not a repeated-run multiplier."""
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        raise ValueError("ci of empty sequence")
    center = float(arr.mean())
    if n == 1:
        return center, center, center
    se = float(arr.std(ddof=1) / np.sqrt(n))
    return center, center - z * se, center + z * se


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
