"""B-v2 decisions. A lower bound above 0 is not a claim that the gain exceeds 3pp."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PILOT_EXTEND = "B-V2-PILOT-EXTEND"
CONFIRMED_GAIN = "B-V2-CONFIRMED-GAIN"
NO_GO_CONTENT = "B-V2-NO_GO-CONTENT"
NO_GO_MECHANISM = "B-V2-NO_GO-MECHANISM"
INCONCLUSIVE = "B-V2-INCONCLUSIVE"
BLOCKED = "B-V2-BLOCKED"
SMOKE_PASS = "B-V2-SMOKE-PASS"


def judge_v2(
    comparisons: Mapping[str, Mapping[str, Any]] | None,
    *,
    floor: float = 0.03,
    retention_delta: float | None,
    retention_ci: Sequence[float] | None,
    retention_tolerance: float = 0.01,
    trigger_rate: float | None,
    blocked_reason: str = "",
) -> dict[str, Any]:
    if blocked_reason:
        return {"status": BLOCKED, "reasons": [blocked_reason], "paper_claim_allowed": False}
    if not comparisons:
        return {"status": INCONCLUSIVE, "reasons": ["比较还没有形成"], "paper_claim_allowed": False}
    b1 = comparisons.get("M_vs_B1")
    b3 = comparisons.get("M_vs_B3")
    b5 = comparisons.get("M_vs_B5")
    if any(item is None or item.get("ci") is None or item.get("diff") is None for item in (b1, b3, b5)):
        return {"status": INCONCLUSIVE, "reasons": ["区间没有形成，不能把点估计写成结论"], "paper_claim_allowed": False}
    if trigger_rate is not None and trigger_rate < 0.05:
        return {"status": INCONCLUSIVE, "reasons": ["规则几乎不触发"], "paper_claim_allowed": False}
    reasons: list[str] = []
    # One key baseline inside the equivalence band, or stably worse, rejects the joint claim.
    for label, item in (("B1", b1), ("B3", b3)):
        if _equivalent(item, floor) or _worse(item):
            reasons.append(f"相对{label}的区间落在 ±{floor:.0%} 等效界内或上界不超过 0。一项关键基线即可否定“同时优于 B1 与 B3”")
            return {"status": NO_GO_CONTENT, "reasons": reasons, "paper_claim_allowed": False}
    if not (_point_and_positive(b1, floor) and _point_and_positive(b3, floor)):
        reasons.append("校正区间下界还没有同时大于 0，或点估计还没有达到 3 个百分点。宽区间是证据不足，不是等效")
        return {"status": INCONCLUSIVE, "reasons": reasons, "paper_claim_allowed": False}
    if _equivalent(b5, floor):
        reasons.append("相对公平的直接归纳，区间落在 ±3 个百分点以内。停止当前对齐机制主张")
        return {"status": NO_GO_MECHANISM, "reasons": reasons, "paper_claim_allowed": False}
    if not _point_and_positive(b5, floor):
        reasons.append("内容方向成立，但相对 B5 还没有正向的区间支持")
        return {"status": INCONCLUSIVE, "reasons": reasons, "paper_claim_allowed": False}
    if retention_delta is None or retention_ci is None:
        reasons.append("retention 还没有区间。观察到的 0 损害不是 1 个百分点上界")
        return {"status": INCONCLUSIVE, "reasons": reasons, "paper_claim_allowed": False}
    if float(retention_ci[0]) < -retention_tolerance:
        reasons.append("retention 区间下界超过 1 个百分点的损害容忍")
        return {"status": INCONCLUSIVE, "reasons": reasons, "paper_claim_allowed": False}
    if _lower(b1) > floor and _lower(b3) > floor and _lower(b5) > floor:
        reasons.append("三个关键比较的区间下界都大于 3 个百分点。这才是“实用增益超过 3pp”的确认标准")
        return {"status": CONFIRMED_GAIN, "reasons": reasons, "paper_claim_allowed": False}
    reasons.append("点估计达到 3 个百分点且校正下界大于 0。这只允许进入一次新确认，并不表示已经证明增益超过 3 个百分点")
    return {"status": PILOT_EXTEND, "reasons": reasons, "paper_claim_allowed": False}


def retention_upper_harm(n_independent: int, harms: int = 0, *, alpha: float = 0.05) -> float | None:
    """One-sided Clopper-Pearson upper bound on the harm rate. A point estimate of 0 is not this bound."""
    if n_independent <= 0 or harms < 0 or harms > n_independent:
        return None
    from scipy.stats import beta

    if harms == n_independent:
        return 1.0
    return float(beta.ppf(1.0 - alpha, harms + 1, n_independent - harms))


def render_v2_decision(decision: Mapping[str, Any]) -> str:
    reasons = "\n".join(f"- {reason}" for reason in decision.get("reasons") or [])
    return "\n".join(
        [
            "# B-v2 决策",
            "",
            f"状态：`{decision.get('status')}`",
            "",
            decision.get("scope_note") or "这次运行不是旧 B-v1 合成夹具，也不能直接写成论文确认。",
            "",
            f"论文主张是否允许引用：`{decision.get('paper_claim_allowed', False)}`",
            "",
            "## 理由",
            "",
            reasons or "- 没有形成理由",
            "",
        ]
    )


def _ci(item: Mapping[str, Any]) -> tuple[float, float]:
    return float(item["ci"][0]), float(item["ci"][1])


def _lower(item: Mapping[str, Any]) -> float:
    return _ci(item)[0]


def _equivalent(item: Mapping[str, Any], floor: float) -> bool:
    lo, hi = _ci(item)
    return lo >= -floor and hi <= floor


def _worse(item: Mapping[str, Any]) -> bool:
    return _ci(item)[1] <= 0


def _point_and_positive(item: Mapping[str, Any], floor: float) -> bool:
    """Point estimate reaches the floor and the corrected lower bound is above 0.

    The lower bound does not have to exceed the floor. That stronger statement is CONFIRMED_GAIN.
    """
    return float(item["diff"]) >= floor and _lower(item) > 0


def mean_ci(values: Sequence[float]) -> tuple[float, float] | None:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2 or float(arr.std(ddof=0)) == 0.0:
        return None
    center = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return center - 1.96 * se, center + 1.96 * se
