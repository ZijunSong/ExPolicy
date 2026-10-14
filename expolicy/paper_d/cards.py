"""Diagnostic cards. A card is frozen before any held-out policy is trained."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from expolicy.paper_d.constants import (
    ACTIONABLE_HEADROOM,
    CONTENT_UNSUPPORTED,
    DELTA,
    ENV_FIXED_SUFFICIENT,
    HETEROGENEOUS_BUT_UNPREDICTED,
    INCONCLUSIVE,
    LONG_TERM_UNCOVERED,
    MEASUREMENT_INVALID,
    PROTOCOL_VERSION,
    UPDATE_INEFFECTIVE,
    ProtocolError,
)
from expolicy.r9_lab.io_util import sha256_json


def judge_configuration(
    *,
    measurement_valid: bool,
    within_gap: float | None,
    within_lower_raw: float | None,
    within_upper: float | None,
    feature_gain: float | None,
    feature_evaluated: bool,
    crossfit_gain: float | None,
    population_table: bool,
    action_mean_upper: Mapping[str, float] | None,
    updates_rejected: bool,
    content_real: float | None,
    content_control: float | None,
    delta: float = DELTA,
) -> dict[str, Any]:
    """Assign a primary status and every dimension status that also applies.

    Failing to beat a fixed policy is not ENV_FIXED_SUFFICIENT. That status needs
    a within-environment upper bound below delta.
    """
    headroom_open = _headroom_open(
        within_gap=within_gap,
        within_lower_raw=within_lower_raw,
        crossfit_gain=crossfit_gain,
        population_table=population_table,
        delta=delta,
    )
    update_ineffective = bool(updates_rejected) or _all_updates_nonpositive(action_mean_upper)
    content_unsupported = (
        content_real is not None and content_control is not None and float(content_real) <= float(content_control)
    )
    feature_wins = feature_evaluated and feature_gain is not None and float(feature_gain) > float(delta)
    env_fixed = within_upper is not None and float(within_upper) < float(delta)
    dimensions: list[str] = []
    if not measurement_valid:
        dimensions.append(MEASUREMENT_INVALID)
    else:
        if feature_wins:
            dimensions.append(ACTIONABLE_HEADROOM)
        if headroom_open and feature_evaluated and not feature_wins:
            dimensions.append(HETEROGENEOUS_BUT_UNPREDICTED)
        if env_fixed:
            dimensions.append(ENV_FIXED_SUFFICIENT)
        if update_ineffective:
            dimensions.append(UPDATE_INEFFECTIVE)
        if content_unsupported:
            dimensions.append(CONTENT_UNSUPPORTED)
        if not any(
            item in dimensions
            for item in (
                ACTIONABLE_HEADROOM,
                HETEROGENEOUS_BUT_UNPREDICTED,
                ENV_FIXED_SUFFICIENT,
                UPDATE_INEFFECTIVE,
                CONTENT_UNSUPPORTED,
            )
        ):
            dimensions.append(INCONCLUSIVE)
    primary = _primary(dimensions)
    return {
        "primary_status": primary,
        "dimensions": dimensions,
        "dimension_detail": {
            "foundation": "not_used_as_a_filter",
            "update_efficacy": UPDATE_INEFFECTIVE if update_ineffective and measurement_valid else "not_flagged",
            "content_contribution": CONTENT_UNSUPPORTED if content_unsupported and measurement_valid else "not_flagged",
            "conditional_headroom": _headroom_label(measurement_valid, env_fixed, headroom_open),
            "predictability_realization": _prediction_label(measurement_valid, feature_evaluated, feature_wins, headroom_open),
        },
        "scientific_conclusion_allowed": measurement_valid and primary != MEASUREMENT_INVALID,
        "long_term_covered": False,
        "limitation": LONG_TERM_UNCOVERED,
        "delta": delta,
        "headroom_open": headroom_open and measurement_valid,
        "allowed_conclusion": _conclusion(primary),
    }


def freeze_diagnostic_card(
    judgment: Mapping[str, Any],
    *,
    configuration_id: str,
    batch_ids: Sequence[str],
    task_ids: Sequence[str],
    policy_trained: bool = False,
    estimates: Mapping[str, Any] | None = None,
    protocol_version: str = PROTOCOL_VERSION,
) -> dict[str, Any]:
    """Hash the judgment before held-out training. Training first is a protocol error."""
    if policy_trained:
        raise ProtocolError("cannot freeze a diagnostic card after held-out policy training")
    payload: dict[str, Any] = {
        "protocol_version": protocol_version,
        "configuration_id": configuration_id,
        "frozen": True,
        "policy_trained_before_freeze": False,
        "batch_ids": sorted(str(item) for item in batch_ids),
        "task_ids": sorted(str(item) for item in task_ids),
        "primary_status": judgment["primary_status"],
        "dimensions": list(judgment["dimensions"]),
        "dimension_detail": dict(judgment["dimension_detail"]),
        "scientific_conclusion_allowed": bool(judgment["scientific_conclusion_allowed"]),
        "long_term_covered": False,
        "limitation": LONG_TERM_UNCOVERED,
        "delta": judgment["delta"],
        "allowed_conclusion": judgment["allowed_conclusion"],
        "estimates": _plain(estimates or {}),
    }
    payload["card_hash"] = sha256_json(payload)
    return payload


def _headroom_open(
    *,
    within_gap: float | None,
    within_lower_raw: float | None,
    crossfit_gain: float | None,
    population_table: bool,
    delta: float,
) -> bool:
    if within_lower_raw is not None and float(within_lower_raw) > 0.0:
        return True
    if population_table and within_gap is not None and float(within_gap) > 0.0:
        return True
    return (not population_table) and crossfit_gain is not None and float(crossfit_gain) > float(delta)


def _all_updates_nonpositive(action_mean_upper: Mapping[str, float] | None) -> bool:
    if action_mean_upper is None:
        return False
    if not action_mean_upper:
        return False
    return all(float(value) <= 0.0 for value in action_mean_upper.values())


def _primary(dimensions: Sequence[str]) -> str:
    order = (
        MEASUREMENT_INVALID,
        ACTIONABLE_HEADROOM,
        HETEROGENEOUS_BUT_UNPREDICTED,
        ENV_FIXED_SUFFICIENT,
        UPDATE_INEFFECTIVE,
        CONTENT_UNSUPPORTED,
        INCONCLUSIVE,
    )
    for status in order:
        if status in dimensions:
            return status
    return INCONCLUSIVE


def _headroom_label(measurement_valid: bool, env_fixed: bool, headroom_open: bool) -> str:
    if not measurement_valid:
        return "not_interpreted"
    if env_fixed:
        return ENV_FIXED_SUFFICIENT
    if headroom_open:
        return "headroom_open"
    return INCONCLUSIVE


def _prediction_label(
    measurement_valid: bool,
    feature_evaluated: bool,
    feature_wins: bool,
    headroom_open: bool,
) -> str:
    if not measurement_valid or not feature_evaluated:
        return "not_tested"
    if feature_wins:
        return ACTIONABLE_HEADROOM
    if headroom_open:
        return HETEROGENEOUS_BUT_UNPREDICTED
    return "not_flagged"


def _conclusion(primary: str) -> str:
    text = {
        MEASUREMENT_INVALID: "暂无科学结论",
        UPDATE_INEFFECTIVE: "当前算子不能有效学习",
        CONTENT_UNSUPPORTED: "内容贡献尚未建立",
        ENV_FIXED_SUFFICIENT: "在当前面板和精度下，环境内额外选择价值有限",
        HETEROGENEOUS_BUT_UNPREDICTED: "动作交叉存在，当前特征没有实现这个空间",
        ACTIONABLE_HEADROOM: "有实际可用的选择空间",
        INCONCLUSIVE: "需要在限定预算下补样",
    }
    return text[primary]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)
