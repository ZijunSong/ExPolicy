"""Realized utility from shadow/retention deltas. Never uses future_test."""

from __future__ import annotations

from typing import Any


def compute_realized_utility(
    *,
    shadow_macro_before: float,
    shadow_macro_after: float,
    retention_before: float,
    retention_after: float,
    token_cost: int,
    tool_cost: int,
    retention_weight: float = 0.5,
    token_cost_per_million: float = 0.01,
    tool_cost_per_100: float = 0.005,
) -> dict[str, Any]:
    delta_shadow = float(shadow_macro_after) - float(shadow_macro_before)
    delta_retention = float(retention_after) - float(retention_before)
    retention_penalty = retention_weight * max(0.0, -delta_retention)
    token_penalty = token_cost_per_million * (float(token_cost) / 1_000_000.0)
    tool_penalty = tool_cost_per_100 * (float(tool_cost) / 100.0)
    reward = delta_shadow - retention_penalty - token_penalty - tool_penalty
    return {
        "reward": float(reward),
        "delta_shadow": float(delta_shadow),
        "delta_retention": float(delta_retention),
        "retention_penalty": float(retention_penalty),
        "token_penalty": float(token_penalty),
        "tool_penalty": float(tool_penalty),
        "shadow_macro_before": float(shadow_macro_before),
        "shadow_macro_after": float(shadow_macro_after),
        "retention_before": float(retention_before),
        "retention_after": float(retention_after),
        "token_cost": int(token_cost),
        "tool_cost": int(tool_cost),
        "source": "shadow_reward_probe",
        "uses_future_test": False,
    }
