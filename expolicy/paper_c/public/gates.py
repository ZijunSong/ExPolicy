"""Public gates. A zero difference is not population equivalence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from expolicy.paper_c.stats import paired_summary

C0_SCREEN_SIGNAL = "C0-SCREEN-SIGNAL"
C0_PUBLIC_PASS = "C0-PUBLIC-PASS"
C_ASSIMILATION_FAIL = "C-ASSIMILATION-FAIL"
C_INCONCLUSIVE = "C-INCONCLUSIVE"
C_CONTROL_INVALID = "C-CONTROL-INVALID"
C_NOT_RUN = "NOT_RUN"
C_RUNTIME_VALID = "C_RUNTIME_VALID"
C_RUNTIME_INVALID = "INVALID_RUNTIME"
C_BLOCKED = "BLOCKED_INFRA"
C_SMALL_EFFECT = "C-SMALL-EFFECT"
C_RETENTION_ONLY = "C-RETENTION-ONLY"
C_SINGLE_ENV = "C-SINGLE-ENV-SIGNAL"


def judge_c0_screen(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    min_gain: float = 0.03,
    retention_tolerance: float = 0.01,
) -> dict[str, Any]:
    if not arms.get("counts_match", True) and arms.get("control_invalid"):
        return {"status": C_CONTROL_INVALID, "reasons": ["内容对照没有按定义写入"]}
    real = arms.get("real") or {}
    if real.get("gain") is None:
        return {"status": C_NOT_RUN, "reasons": ["Real 臂还没有配对增益"]}
    reasons = []
    gain = float(real["gain"])
    comparable = True
    for name in ("no-update", "empty-slots", "matched-wrong"):
        other = (arms.get(name) or {}).get("gain")
        if other is None:
            return {"status": C_INCONCLUSIVE, "reasons": [f"{name} 缺失，不能当成零增益"]}
        gap = gain - float(other)
        upper = (arms.get(name) or {}).get("gap_upper")
        if gap < min_gain:
            reasons.append(f"Real 相对 {name} 的点估计差是 {gap:.4f}")
        if upper is not None and float(upper) < min_gain and gap < min_gain:
            comparable = False
    retention = real.get("delta_retention")
    if retention is not None and float(retention) < -retention_tolerance:
        reasons.append("retention 点估计超过容忍")
    if not reasons:
        return {
            "status": C0_SCREEN_SIGNAL,
            "reasons": ["screen 有内容方向信号。这不是 C0-PUBLIC-PASS，确认集尚未执行"],
        }
    if not comparable:
        return {"status": C_ASSIMILATION_FAIL, "reasons": reasons + ["区间上界已经排除 3pp 实用效应"]}
    return {"status": C_INCONCLUSIVE, "reasons": reasons + ["区间仍可能容纳实用效应，screen 不能升为通过"]}


def classify_interval(summary: Mapping[str, Any], *, minimum: float) -> str:
    """Separate a bounded small effect, an excluded effect, and an inconclusive interval."""
    if summary.get("diff") is None or summary.get("zero_width") or not summary.get("population_noise_claim"):
        return C_INCONCLUSIVE
    diff = float(summary["diff"])
    lo, hi = (summary.get("ci") or [None, None])[:2]
    if lo is None or hi is None:
        return C_INCONCLUSIVE
    lo, hi = float(lo), float(hi)
    if lo > 0.0 and diff < minimum:
        return C_SMALL_EFFECT
    if hi < minimum:
        return "C-EFFECT-EXCLUDED"
    if lo < 0.0 < hi or lo < minimum < hi:
        return C_INCONCLUSIVE
    if lo > 0.0 and diff >= minimum:
        return "PASS"
    return C_INCONCLUSIVE


def judge_public_pilot(
    *,
    cost_diffs: Sequence[float],
    final_diffs: Sequence[float],
    retention_diffs: Sequence[float],
    c0_status: str,
    c1_status: str,
    c2_status: str,
    n_public_environments: int,
    wide: float = 0.05,
) -> dict[str, Any]:
    cost = paired_summary(cost_diffs, wide=wide)
    final = paired_summary(final_diffs, wide=wide)
    retention = paired_summary(retention_diffs, wide=wide)
    aulc_class = classify_interval(cost, minimum=0.02)
    final_class = classify_interval(final, minimum=-0.01)
    doors = c0_status == "C0-PUBLIC-PASS" and c1_status.startswith("C1") and "FAIL" not in c1_status and c2_status == "PASS"
    if aulc_class != "PASS" or not doors:
        status = aulc_class if aulc_class != "PASS" else C_NOT_RUN
    elif final.get("ci") and float(final["ci"][0]) < -0.01:
        status = C_INCONCLUSIVE
    elif retention.get("diff") is not None and float(retention["diff"]) < -0.01 and aulc_class != "PASS":
        status = C_RETENTION_ONLY
    else:
        status = "C-SINGLE-ENV-SIGNAL" if n_public_environments < 2 else "C-PUBLIC-GO"
    if n_public_environments < 2 and status == "C-PUBLIC-GO":
        status = C_SINGLE_ENV
    return {
        "status": status,
        "paper_claim": "NOT_RUN" if n_public_environments < 2 or status != "C-PUBLIC-GO" else "C-PUBLIC-GO",
        "cost": cost,
        "final": final,
        "retention": retention,
        "final_class": final_class,
    }
