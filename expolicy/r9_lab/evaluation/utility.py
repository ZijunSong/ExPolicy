"""Utility U = d_shadow - 0.5 * max(0, -d_retention). Never uses future_test."""

from __future__ import annotations

from typing import Any


def compute_utility(
    *,
    shadow_before: float,
    shadow_after: float,
    retention_before: float,
    retention_after: float,
    retention_weight: float = 0.5,
) -> dict[str, Any]:
    d_shadow = float(shadow_after) - float(shadow_before)
    d_retention = float(retention_after) - float(retention_before)
    penalty = retention_weight * max(0.0, -d_retention)
    u = d_shadow - penalty
    return {
        "U": float(u),
        "d_shadow": float(d_shadow),
        "d_retention": float(d_retention),
        "retention_penalty": float(penalty),
        "uses_future_test": False,
    }
