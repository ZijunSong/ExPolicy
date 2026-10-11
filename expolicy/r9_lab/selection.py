"""Pre-registered model-screen selection rule. Do not post-hoc change the band."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from expolicy.r9_lab.constants import (
    FOUNDATION_MAX,
    FOUNDATION_MIN,
    INFRA_FAIL_MAX,
    PARSER_VALID_MIN,
    TARGET_SUCCESS,
    TIE_BREAK_ORDER,
    ENVS,
)

NO_FULL_PASS = "MODEL_LADDER_NO_FULL_PASS"


@dataclass
class EnvScreenStats:
    success: float
    parser_valid_rate: float
    infra_failure_rate: float
    n: int = 0


@dataclass
class ModelScreenResult:
    model_id: str
    by_env: dict[str, EnvScreenStats]
    full_pass: bool
    score: float | None
    reason: str = ""


@dataclass
class SelectionOutcome:
    status: str  # selected model_id or MODEL_LADDER_NO_FULL_PASS
    selected: ModelScreenResult | None
    ranked: list[ModelScreenResult] = field(default_factory=list)
    rule: str = "pre-registered R9 model-screen; band 0.10-0.70; target 0.40"


def env_in_band(success: float) -> bool:
    return FOUNDATION_MIN <= float(success) <= FOUNDATION_MAX


def is_full_pass(by_env: Mapping[str, EnvScreenStats]) -> tuple[bool, str]:
    missing = [e for e in ENVS if e not in by_env]
    if missing:
        return False, f"missing envs: {missing}"
    for env in ENVS:
        stats = by_env[env]
        if not env_in_band(stats.success):
            return False, f"{env} success={stats.success:.4f} outside [{FOUNDATION_MIN},{FOUNDATION_MAX}]"
        if stats.parser_valid_rate < PARSER_VALID_MIN:
            return False, f"{env} parser_valid_rate={stats.parser_valid_rate:.4f} < {PARSER_VALID_MIN}"
        if stats.infra_failure_rate > INFRA_FAIL_MAX:
            return False, f"{env} infra_failure_rate={stats.infra_failure_rate:.4f} > {INFRA_FAIL_MAX}"
    return True, "full_pass"


def mean_parser_valid(by_env: Mapping[str, EnvScreenStats]) -> float:
    return sum(by_env[e].parser_valid_rate for e in ENVS) / len(ENVS)


def selection_score(by_env: Mapping[str, EnvScreenStats]) -> float:
    """Score(m) = mean_e |s-0.40| + 0.5*max_e |s-0.40| + 0.5*(1-parser_valid_rate).

    Lowest score wins. parser_valid_rate is the unweighted mean across the three envs.
    """
    deviations = [abs(float(by_env[e].success) - TARGET_SUCCESS) for e in ENVS]
    mean_dev = sum(deviations) / len(deviations)
    max_dev = max(deviations)
    parser = mean_parser_valid(by_env)
    return mean_dev + 0.5 * max_dev + 0.5 * (1.0 - parser)


def evaluate_model(model_id: str, by_env: Mapping[str, EnvScreenStats]) -> ModelScreenResult:
    ok, reason = is_full_pass(by_env)
    score = selection_score(by_env) if ok else None
    return ModelScreenResult(
        model_id=model_id,
        by_env=dict(by_env),
        full_pass=ok,
        score=score,
        reason=reason,
    )


def _tie_break_rank(model_id: str) -> int:
    try:
        return TIE_BREAK_ORDER.index(model_id)
    except ValueError:
        return 10_000


def select_model(results: Sequence[ModelScreenResult]) -> SelectionOutcome:
    """Lowest Full-PASS score wins. Tie-break: Q0 > Q1 > Q2 > Q3.

    If no Full-PASS: MODEL_LADDER_NO_FULL_PASS. Do not widen the band.
    """
    ranked = list(results)
    full = [r for r in ranked if r.full_pass and r.score is not None]
    if not full:
        ranked.sort(key=lambda r: (_tie_break_rank(r.model_id), r.model_id))
        return SelectionOutcome(status=NO_FULL_PASS, selected=None, ranked=ranked)
    full.sort(key=lambda r: (float(r.score), _tie_break_rank(r.model_id)))
    winner = full[0]
    others = [r for r in ranked if r.model_id != winner.model_id]
    others.sort(key=lambda r: (0 if r.full_pass else 1, float(r.score) if r.score is not None else 9e9, _tie_break_rank(r.model_id)))
    return SelectionOutcome(status=winner.model_id, selected=winner, ranked=[winner] + others)


def stats_from_mapping(raw: Mapping[str, Mapping[str, Any]]) -> dict[str, EnvScreenStats]:
    out: dict[str, EnvScreenStats] = {}
    for env, row in raw.items():
        out[str(env)] = EnvScreenStats(
            success=float(row["success"]),
            parser_valid_rate=float(row["parser_valid_rate"]),
            infra_failure_rate=float(row["infra_failure_rate"]),
            n=int(row.get("n") or 0),
        )
    return out
