"""Protocol utility. The historical R9 penalty is a separate column."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from expolicy.paper_d.constants import RETENTION_WEIGHT
from expolicy.r9_lab.evaluation.utility import compute_utility


def utility_from_expectations(
    *,
    future_before: float,
    future_after: float,
    retention_before: float,
    retention_after: float,
    retention_weight: float = RETENTION_WEIGHT,
) -> dict[str, float]:
    """Apply the penalty after the expectations, not inside each noisy repeat.

    u = (E J_future(S') - E J_future(S)) - λ max(0, E J_ret(S) - E J_ret(S')).
    The unpenalized gain and the retention change are always returned beside u.
    """
    gain = float(future_after) - float(future_before)
    retention_delta = float(retention_after) - float(retention_before)
    penalty = float(retention_weight) * max(0.0, -retention_delta)
    return {
        "utility": gain - penalty,
        "gain": gain,
        "retention_delta": retention_delta,
        "penalty": penalty,
    }


def legacy_utility_from_repeat(
    *,
    future_before: float,
    future_after: float,
    retention_before: float,
    retention_after: float,
    retention_weight: float = RETENTION_WEIGHT,
) -> dict[str, Any]:
    """One noisy evaluation, using the original R9 positive-part utility."""
    packed = compute_utility(
        shadow_before=future_before,
        shadow_after=future_after,
        retention_before=retention_before,
        retention_after=retention_after,
        retention_weight=retention_weight,
    )
    return {
        "utility": float(packed["U"]),
        "gain": float(packed["d_shadow"]),
        "retention_delta": float(packed["d_retention"]),
        "penalty": float(packed["retention_penalty"]),
        "estimator": "r9_compute_utility_per_repeat",
    }


def mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
