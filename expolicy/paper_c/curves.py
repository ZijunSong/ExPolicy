"""Frozen curves are scored after the policies stop. Those scores do not re-enter a selector."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from expolicy.paper_c.constants import ACCOUNT_OVERHEAD
from expolicy.paper_c.ledger import BudgetLedger
from expolicy.paper_c.sequence import Checkpoint
from expolicy.paper_c.utility import common_c_max, trapezoid_aulc
from expolicy.r9_lab.state.persistent_state import PersistentState

ScoreState = Callable[[PersistentState], float | None]


def evaluate_frozen_curves(
    checkpoints: Sequence[Checkpoint],
    *,
    future_fn: ScoreState,
    shadow_fn: ScoreState,
    retention_fn: ScoreState,
    ledger: BudgetLedger,
) -> list[dict[str, Any]]:
    interaction_before = ledger.interaction_total()
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        future = future_fn(checkpoint.state)
        shadow = shadow_fn(checkpoint.state)
        retention = retention_fn(checkpoint.state)
        ledger.add(
            account=ACCOUNT_OVERHEAD,
            kind="frozen_curve",
            interaction_units=float(checkpoint.probe_calls),
            validation_calls=int(checkpoint.probe_calls),
            detail=f"{checkpoint.policy}:{checkpoint.sequence_seed}:{checkpoint.round_index}",
        )
        rows.append(
            {
                "policy": checkpoint.policy,
                "sequence_seed": checkpoint.sequence_seed,
                "round": checkpoint.round_index,
                "state_hash": checkpoint.state.content_hash(),
                "interaction_cost": checkpoint.interaction_cost,
                "episode_cost": checkpoint.episode_cost,
                "token_cost": checkpoint.token_cost,
                "future_score": future,
                "shadow": shadow,
                "retention": retention,
            }
        )
    if ledger.interaction_total() != interaction_before:
        raise RuntimeError("frozen evaluation changed the interaction budget")
    return rows


def attach_offline_scores(logs: Sequence[dict[str, Any]], curves: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy frozen future scores onto the log after selection. This does not call a policy."""
    by_policy_round = {
        (str(row["policy"]), int(row["sequence_seed"]), int(row["round"])): row.get("future_score") for row in curves
    }
    output = []
    for log in logs:
        copied = dict(log)
        lookup = (str(copied.get("policy") or ""), int(copied["sequence_seed"]), int(copied["round"]))
        if lookup in by_policy_round:
            copied["future_score_offline"] = by_policy_round[lookup]
        output.append(copied)
    return output


def _points(rows: Sequence[dict[str, Any]], cost_key: str) -> list[tuple[float, float]]:
    points = []
    for row in sorted(rows, key=lambda item: (float(item[cost_key]), int(item["round"]))):
        if row.get("future_score") is None:
            continue
        points.append((float(row[cost_key]), float(row["future_score"])))
    return points


def aulc_by_sequence(
    curves: Sequence[dict[str, Any]],
    *,
    policies: Sequence[str],
    cost_key: str,
) -> dict[str, dict[int, float | None]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in curves:
        grouped.setdefault((str(row["policy"]), int(row["sequence_seed"])), []).append(row)
    seeds = sorted({int(row["sequence_seed"]) for row in curves})
    scores: dict[str, dict[int, float | None]] = {policy: {} for policy in policies}
    for seed in seeds:
        series = []
        payload = {}
        for policy in policies:
            points = _points(grouped.get((policy, seed), []), cost_key)
            payload[policy] = points
            if len(points) >= 2:
                series.append(points)
        limit = common_c_max(series)
        for policy in policies:
            points = payload[policy]
            if len(points) < 2 or limit is None:
                scores[policy][seed] = None
            else:
                scores[policy][seed] = trapezoid_aulc(points, c_max=limit)["aulc"]
    return scores


def final_by_sequence(curves: Sequence[dict[str, Any]], *, score_key: str) -> dict[str, dict[int, float | None]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in curves:
        grouped.setdefault((str(row["policy"]), int(row["sequence_seed"])), []).append(row)
    output: dict[str, dict[int, float | None]] = {}
    for (policy, seed), rows in grouped.items():
        ordered = sorted(rows, key=lambda item: int(item["round"]))
        value = ordered[-1].get(score_key)
        output.setdefault(policy, {})[seed] = None if value is None else float(value)
    return output
