"""Policy features. Forbidden keys are scanned on every vector."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from expolicy.closed_loop.constants import (
    ACQ_FEATURE_DIM,
    ASSIM_ACTIONS,
    ASSIM_FEATURE_DIM,
    FORBIDDEN_FEATURE_KEYS,
)
from expolicy.closed_loop.experience import ExperienceBatchSummary
from expolicy.closed_loop.state import ExPolicyState


class LeakageError(RuntimeError):
    pass


def assert_no_leakage(named: dict[str, Any]) -> None:
    blob = " ".join(str(k).lower() for k in named)
    for bad in FORBIDDEN_FEATURE_KEYS:
        if bad in blob.split() or bad == str(named.get(bad, "")).lower():
            raise LeakageError(f"forbidden policy feature key: {bad}")
        for key in named:
            kl = str(key).lower()
            if kl == bad or kl.endswith("_" + bad) or bad in kl.split("_"):
                if bad in {"gold", "gold_answer", "future_test", "future_probe", "task_id", "dataset", "dataset_id", "dataset_name"}:
                    raise LeakageError(f"forbidden policy feature key: {key}")


def acquisition_feature_names() -> list[str]:
    return [
        "bias",
        "current_shadow_em",
        "failure_rate_so_far",
        "seen_fraction",
        "recent_utility_ema",
        "skill_active",
        "episodic_count_norm",
        "semantic_count_norm",
        "remaining_budget_fraction",
        "last_batch_was_c",
        "round_fraction",
    ]


def assimilation_feature_names() -> list[str]:
    return [
        "bias",
        "current_shadow_em",
        "skill_active",
        "batch_success_rate",
        "batch_failure_rate",
        "batch_repeat_query_rate",
        "batch_premature_answer_rate",
        "batch_mean_searches_norm",
        "batch_evidence_completion",
        "episodic_count_norm",
        "semantic_count_norm",
        "remaining_budget_fraction",
    ]


def acquisition_features(
    state: ExPolicyState,
    cap: str,
    *,
    n_rounds: int,
) -> np.ndarray:
    named = {
        "bias": 1.0,
        "current_shadow_em": float(state.shadow_em_by_capability.get(cap) or 0.0),
        "failure_rate_so_far": state.failure_rate(cap),
        "seen_fraction": state.seen_fraction(cap),
        "recent_utility_ema": float(state.utility_ema_by_capability.get(cap) or 0.0),
        "skill_active": 1.0 if state.active_skills.get(cap) else 0.0,
        "episodic_count_norm": float(state.episodic_memory_size.get(cap) or 0) / 48.0,
        "semantic_count_norm": float(state.semantic_memory_size.get(cap) or 0) / 16.0,
        "remaining_budget_fraction": state.remaining_budget_fraction(),
        "last_batch_was_c": 1.0 if state.last_batch_capability == cap else 0.0,
        "round_fraction": state.round_fraction(n_rounds),
    }
    assert_no_leakage(named)
    vec = np.array([named[k] for k in acquisition_feature_names()], dtype=np.float64)
    assert vec.shape[0] == ACQ_FEATURE_DIM
    return vec


def assimilation_features(
    state: ExPolicyState,
    summary: ExperienceBatchSummary,
    action: str,
) -> np.ndarray:
    if action not in ASSIM_ACTIONS:
        raise ValueError(action)
    cap = summary.capability
    named = {
        "bias": 1.0,
        "current_shadow_em": float(state.shadow_em_by_capability.get(cap) or 0.0),
        "skill_active": 1.0 if state.active_skills.get(cap) else 0.0,
        "batch_success_rate": float(summary.success_rate),
        "batch_failure_rate": float(summary.failure_rate),
        "batch_repeat_query_rate": float(summary.repeat_query_rate),
        "batch_premature_answer_rate": float(summary.premature_answer_rate),
        "batch_mean_searches_norm": float(summary.mean_searches) / 10.0,
        "batch_evidence_completion": float(summary.evidence_completion_rate),
        "episodic_count_norm": float(state.episodic_memory_size.get(cap) or 0) / 48.0,
        "semantic_count_norm": float(state.semantic_memory_size.get(cap) or 0) / 16.0,
        "remaining_budget_fraction": state.remaining_budget_fraction(),
    }
    assert_no_leakage(named)
    vec = np.array([named[k] for k in assimilation_feature_names()], dtype=np.float64)
    assert vec.shape[0] == ASSIM_FEATURE_DIM
    return vec


def scan_payload_for_leakage(payload: Any, *, path: str = "root") -> list[str]:
    issues: list[str] = []
    _walk(payload, path, issues)
    return issues


def _walk(node: Any, path: str, issues: list[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            kl = str(k).lower()
            if kl in FORBIDDEN_FEATURE_KEYS or kl in {"gold_answer", "future_test_em"}:
                issues.append(f"{path}.{k}")
            _walk(v, f"{path}.{k}", issues)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node[:32]):
            _walk(v, f"{path}[{i}]", issues)


def assert_feature_matrix_clean(rows: Iterable[dict[str, Any]]) -> None:
    for i, row in enumerate(rows):
        issues = scan_payload_for_leakage(row)
        if issues:
            raise LeakageError(f"leakage in row {i}: {issues[:8]}")
