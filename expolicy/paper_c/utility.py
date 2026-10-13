"""Post-assimilation gain. Future-test scores are not an input."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def acquisition_gain(
    *,
    shadow_before: float,
    shadow_after: float,
    retention_before: float,
    retention_after: float,
    retention_weight: float = 0.5,
) -> dict[str, Any]:
    """g(S, B) = Δshadow − λ max(0, −Δretention).

    A negative value is not automatically damage: a no-op's positive-part penalty
    can push a noisy paired difference below zero. Callers report the raw rates too.
    """
    delta_shadow = float(shadow_after) - float(shadow_before)
    delta_retention = float(retention_after) - float(retention_before)
    penalty = float(retention_weight) * max(0.0, -delta_retention)
    return {
        "g": delta_shadow - penalty,
        "delta_shadow": delta_shadow,
        "delta_retention": delta_retention,
        "retention_penalty": penalty,
        "raw_shadow_after": float(shadow_after),
        "raw_retention_after": float(retention_after),
        "uses_future_test": False,
    }


def trapezoid_aulc(points: Sequence[tuple[float, float]], *, c_max: float | None = None) -> dict[str, Any]:
    """Area under the learning curve, normalized by the cost span that was observed.

    Segments may be cut at ``c_max``. The curve is not extended past its last cost.
    """
    ordered = sorted((float(cost), float(score)) for cost, score in points)
    if len(ordered) < 2:
        return {"aulc": None, "c_max": None, "truncated": False, "reason": "fewer than two points"}
    last = ordered[-1][0]
    limit = last if c_max is None else min(float(c_max), last)
    if limit <= ordered[0][0]:
        return {"aulc": None, "c_max": limit, "truncated": True, "reason": "cost span is empty"}
    area = 0.0
    for (c0, j0), (c1, j1) in zip(ordered, ordered[1:], strict=True):
        if c1 <= ordered[0][0]:
            continue
        left = max(c0, ordered[0][0])
        right = min(c1, limit)
        if right <= left:
            if c0 >= limit:
                break
            continue
        if c1 == c0:
            height = (j0 + j1) / 2.0
        else:
            y_left = j0 + (j1 - j0) * (left - c0) / (c1 - c0)
            y_right = j0 + (j1 - j0) * (right - c0) / (c1 - c0)
            height = (y_left + y_right) / 2.0
        area += height * (right - left)
        if c1 >= limit:
            break
    span = limit - ordered[0][0]
    return {
        "aulc": area / span,
        "c_max": limit,
        "truncated": limit < last,
        "extrapolated": False,
    }


def common_c_max(curves: Sequence[Sequence[tuple[float, float]]]) -> float | None:
    """Shared cost is the smallest observed final cost. Later points are cut, not extended."""
    finals = []
    for curve in curves:
        if len(curve) < 2:
            return None
        finals.append(max(float(cost) for cost, _score in curve))
    if not finals:
        return None
    return min(finals)
