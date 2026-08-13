"""Marginal learning utility from realized before/after capability deltas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from expolicy.core.experience import ExperienceCost, ExperienceUtility
from expolicy.core.types import Carrier, JSONDict


class LearningUtility(BaseModel):
    future_gain: float
    transfer_gain: float = 0.0
    retention_gain: float = 0.0
    interaction_cost: float = 0.0
    token_cost: float = 0.0
    compute_cost: float = 0.0
    regression: float = 0.0
    interference: float = 0.0
    utility: float = 0.0
    extras: JSONDict = Field(default_factory=dict)


def compute_utility(
    *,
    future_gain: float,
    transfer_gain: float = 0.0,
    retention_gain: float = 0.0,
    cost: ExperienceCost | None = None,
    regression: float = 0.0,
    interference: float = 0.0,
    lambda_interference: float = 1.0,
    token_unit: float = 1_000_000.0,
    interaction_unit: float = 1.0,
    compute_unit: float = 3600.0,
) -> LearningUtility:
    """U = (ΔC_future + transfer + retention) / (C_acq + C_learn) - λ I.

    Costs are normalized so a typical episode is O(1). All inputs must be
    realized measurements, not LLM self-scores.
    """
    cost = cost or ExperienceCost()
    interaction_cost = cost.environment_steps / max(interaction_unit, 1e-8)
    token_cost = cost.tokens / max(token_unit, 1e-8)
    compute_cost = (cost.gpu_seconds + cost.latency) / max(compute_unit, 1e-8)
    denom = interaction_cost + token_cost + compute_cost + 1e-6
    numerator = future_gain + transfer_gain + retention_gain
    utility = numerator / denom - lambda_interference * (interference + max(0.0, -regression))
    return LearningUtility(
        future_gain=future_gain,
        transfer_gain=transfer_gain,
        retention_gain=retention_gain,
        interaction_cost=interaction_cost,
        token_cost=token_cost,
        compute_cost=compute_cost,
        regression=regression,
        interference=interference,
        utility=utility,
    )


def to_experience_utility(
    experience_id: str,
    learning: LearningUtility,
    *,
    before_metrics: JSONDict | None = None,
    after_metrics: JSONDict | None = None,
    carrier: Carrier | None = None,
) -> ExperienceUtility:
    return ExperienceUtility(
        experience_id=experience_id,
        future_gain=learning.future_gain,
        transfer_gain=learning.transfer_gain,
        retention_gain=learning.retention_gain,
        interaction_cost=learning.interaction_cost,
        token_cost=learning.token_cost,
        compute_cost=learning.compute_cost,
        regression=learning.regression,
        interference=learning.interference,
        utility=learning.utility,
        before_metrics=before_metrics or {},
        after_metrics=after_metrics or {},
        carrier=carrier,
    )
