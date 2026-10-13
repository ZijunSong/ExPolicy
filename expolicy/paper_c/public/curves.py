"""Common-budget scores. The last logged round is not the comparison when costs differ."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _series(rows: Sequence[Mapping[str, Any]], cost_key: str, score_key: str) -> list[tuple[float, float]]:
    points = []
    for row in sorted(rows, key=lambda item: (float(item[cost_key]), int(item.get("round") or 0))):
        if row.get(score_key) is None:
            continue
        points.append((float(row[cost_key]), float(row[score_key])))
    return points


def score_at_common_budget(
    curves: Sequence[Mapping[str, Any]],
    *,
    policies: Sequence[str],
    cost_key: str = "interaction_cost",
    score_key: str = "future_score",
) -> dict[str, dict[int, float | None]]:
    """Step score at the shared observed cost. Later points are not extrapolated."""
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in curves:
        grouped.setdefault((str(row["policy"]), int(row["sequence_seed"])), []).append(row)
    seeds = sorted({int(row["sequence_seed"]) for row in curves})
    output: dict[str, dict[int, float | None]] = {policy: {} for policy in policies}
    for seed in seeds:
        series = {policy: _series(grouped.get((policy, seed), []), cost_key, score_key) for policy in policies}
        ends = [points[-1][0] for points in series.values() if points]
        if len(ends) < len(policies):
            for policy in policies:
                output[policy][seed] = None
            continue
        limit = min(ends)
        for policy in policies:
            points = [(cost, score) for cost, score in series[policy] if cost <= limit]
            output[policy][seed] = None if not points else points[-1][1]
    return output


def step_aulc(points: Sequence[tuple[float, float]], *, c_max: float) -> float | None:
    ordered = sorted((float(cost), float(score)) for cost, score in points)
    if len(ordered) < 1 or c_max <= ordered[0][0]:
        return None
    area = 0.0
    cursor = ordered[0][0]
    score = ordered[0][1]
    for cost, next_score in ordered[1:]:
        right = min(cost, c_max)
        if right > cursor:
            area += score * (right - cursor)
        cursor = right
        score = next_score
        if cursor >= c_max:
            break
    if cursor < c_max:
        return None
    return area / (c_max - ordered[0][0])
