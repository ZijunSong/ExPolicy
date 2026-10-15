"""Per-batch utility, then an equal-weight average across batches.

Positive retention is not a reward. The content gap is the difference of two
utilities, not a penalty applied to the retention gap itself.
"""

from __future__ import annotations

from typing import Sequence

from expolicy.paper_d.confirm_protocol import PROTOCOL


def update_utility(future_gain: float | None, retention_gain: float | None) -> float | None:
    if future_gain is None or retention_gain is None:
        return None
    weight = float(PROTOCOL["retention_weight"])
    return float(future_gain) + weight * min(0.0, float(retention_gain))


def mean_utility(future_gains: Sequence[float | None], retention_gains: Sequence[float | None]) -> float | None:
    """Average batch utilities. A positive and a negative retention do not cancel before the penalty."""
    if len(future_gains) != len(retention_gains) or not future_gains:
        return None
    values = [update_utility(future, retention) for future, retention in zip(future_gains, retention_gains)]
    if any(value is None for value in values):
        return None
    return sum(values) / len(values)


def content_utility_gap(real: Sequence[float | None], control: Sequence[float | None]) -> float | None:
    if len(real) != len(control) or not real:
        return None
    if any(value is None for value in real) or any(value is None for value in control):
        return None
    return sum(float(left) - float(right) for left, right in zip(real, control)) / len(real)
