"""Prospective check. The diagnostic card and the later policy do not share batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from expolicy.paper_d.constants import (
    ACTIONABLE_HEADROOM,
    DELTA,
    ENV_FIXED_SUFFICIENT,
    MIN_CONFIGS_FOR_CORRELATION,
    ProtocolError,
)
from expolicy.r9_lab.io_util import sha256_json


def validate_prospectively(
    cards: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    delta: float = DELTA,
    min_configs_for_correlation: int = MIN_CONFIGS_FOR_CORRELATION,
) -> dict[str, Any]:
    """Compare a frozen card with a later policy on new batches and future tasks.

    False GO / false NO_GO rates ignore inconclusive cards. Correlation is omitted
    when only a few held-out configurations exist, and it is never a significance claim.
    Savings are refused when the counterfactual is present only for failures.
    """
    if any(outcome.get("savings_from_failures_only") for outcome in outcomes):
        raise ProtocolError("savings cannot be computed from failed configurations only")
    by_id = {str(card["configuration_id"]): card for card in cards}
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        card = by_id.get(str(outcome.get("configuration_id")))
        rows.append(_one(card, outcome, delta))
    usable = [row for row in rows if row["validation_status"] == "ok"]
    predicted_go = [row for row in usable if row["predicted_status"] == ACTIONABLE_HEADROOM]
    predicted_nogo = [row for row in usable if row["predicted_status"] == ENV_FIXED_SUFFICIENT]
    n_false_go = sum(1 for row in predicted_go if row["false_go"])
    n_false_nogo = sum(1 for row in predicted_nogo if row["false_nogo"])
    needs_more = [row for row in usable if row["predicted_status"] not in {ACTIONABLE_HEADROOM, ENV_FIXED_SUFFICIENT}]
    diagnostic_cost = _sum_if_complete(rows, "diagnostic_cost")
    training_cost = _sum_if_complete(rows, "training_cost")
    correlation_reported = len(usable) >= min_configs_for_correlation
    return {
        "rows": rows,
        "n_outcomes": len(rows),
        "n_usable": len(usable),
        "false_go_rate": None if not predicted_go else n_false_go / len(predicted_go),
        "false_nogo_rate": None if not predicted_nogo else n_false_nogo / len(predicted_nogo),
        "n_false_go": n_false_go,
        "n_false_nogo": n_false_nogo,
        "n_needs_more_measurement": len(needs_more),
        "diagnostic_cost": diagnostic_cost,
        "training_cost": training_cost,
        "net_savings": _net_savings(rows, diagnostic_cost),
        "savings_from_failures_only": False,
        "correlation_reported": correlation_reported,
        "correlation": _correlation(usable) if correlation_reported else None,
        "significance_claimed": False,
        "case_validation_only": len(usable) < min_configs_for_correlation,
    }


def _one(card: Mapping[str, Any] | None, outcome: Mapping[str, Any], delta: float) -> dict[str, Any]:
    reasons: list[str] = []
    if card is None or card.get("frozen") is not True:
        reasons.append("missing frozen card")
    elif sha256_json({key: value for key, value in card.items() if key != "card_hash"}) != card.get("card_hash"):
        reasons.append("card hash does not match the frozen judgment")
    if card is not None:
        overlap_batches = set(card.get("batch_ids") or []) & set(outcome.get("batch_ids") or [])
        overlap_tasks = set(card.get("task_ids") or []) & set(outcome.get("task_ids") or [])
        if overlap_batches or overlap_tasks:
            reasons.append("INVALID_OVERLAP")
    actual = outcome.get("actual_gain")
    if actual is None:
        reasons.append("missing actual gain")
    predicted = None if card is None else card.get("primary_status")
    false_go = predicted == ACTIONABLE_HEADROOM and actual is not None and float(actual) <= float(delta)
    false_nogo = predicted == ENV_FIXED_SUFFICIENT and actual is not None and float(actual) > float(delta)
    return {
        "configuration_id": outcome.get("configuration_id"),
        "card_hash": None if card is None else card.get("card_hash"),
        "predicted_status": predicted,
        "actual_gain": actual,
        "false_go": false_go,
        "false_nogo": false_nogo,
        "validation_status": "ok" if not reasons else "invalid",
        "reasons": reasons,
        "diagnostic_cost": outcome.get("diagnostic_cost"),
        "training_cost": outcome.get("training_cost"),
        "training_avoided": outcome.get("training_avoided"),
    }


def _sum_if_complete(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    total = 0.0
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        total += float(value)
    return total


def _net_savings(rows: Sequence[Mapping[str, Any]], diagnostic_cost: float | None) -> float | None:
    if diagnostic_cost is None:
        return None
    avoided = _sum_if_complete(rows, "training_avoided")
    if avoided is None:
        return None
    return avoided - diagnostic_cost


def _correlation(rows: Sequence[Mapping[str, Any]]) -> float | None:
    gains: list[float] = []
    for row in rows:
        value = row.get("actual_gain")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        gains.append(float(value))
    # Descriptive only. The caller already refused a significance claim.
    coded = [1.0 if row["predicted_status"] == ACTIONABLE_HEADROOM else 0.0 for row in rows]
    return _pearson(gains, coded)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    cov = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    var_left = sum((a - mean_left) ** 2 for a in left)
    var_right = sum((b - mean_right) ** 2 for b in right)
    if var_left <= 0.0 or var_right <= 0.0:
        return None
    return cov / ((var_left * var_right) ** 0.5)
