"""Development-side fixed mixtures. Mean-gain is not an optimized mixture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from expolicy.paper_c.policies import fit_best_static


def fit_mean_gain_mixture(rows: Sequence[Mapping], *, gain_key: str = "gain_eval") -> dict[str, float]:
    """Positive-mean normalization. This is not a searched optimum."""
    return fit_best_static(rows, gain_key=gain_key)


def fixed_candidates(families: Sequence[str], mean_gain: Mapping[str, float]) -> dict[str, dict[str, float]]:
    names = list(families)
    share = 1.0 / len(names) if names else 0.0
    candidates = {"uniform-mixture": {name: share for name in names}}
    candidates["mean-gain-mixture"] = dict(mean_gain)
    for name in names:
        candidates[f"single:{name}"] = {other: (1.0 if other == name else 0.0) for other in names}
    return candidates


def freeze_best_fixed(dev_scores: Mapping[str, float]) -> str:
    """Pick among the pre-registered fixed policies. Confirmation scores are not an input."""
    if not dev_scores:
        raise RuntimeError("no development scores for the fixed policy")
    return min(dev_scores, key=lambda name: (-float(dev_scores[name]), name))
