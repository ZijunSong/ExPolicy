"""Pre-registered decision. A wide interval is not a negative result."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from expolicy.paper_b.constants import (
    B_GO,
    B_INCONCLUSIVE,
    B_NO_GO_CONTENT,
    B_NO_GO_MECHANISM,
    B_NOT_RUN,
    TARGET_FAMILIES,
)


def judge(
    comparisons: Mapping[str, Mapping[str, Any]] | None,
    *,
    floor: float,
    retention_delta: float | None,
    retention_tolerance: float,
    family_diffs: Mapping[str, float | None],
    replacement_changed: bool,
    trigger_rate: float | None,
    illegal_rate: float | None,
    min_trigger: float = 0.05,
    max_illegal: float = 0.5,
) -> dict[str, Any]:
    if not comparisons or any(comparisons.get(key) is None for key in ("M_vs_B1", "M_vs_B3", "M_vs_B5")):
        return _result(B_NOT_RUN, ["确认集比较还没有跑完"])
    primary = (comparisons["M_vs_B1"], comparisons["M_vs_B3"])
    mechanism = comparisons["M_vs_B5"]
    if any(item.get("diff") is None or item.get("ci") is None for item in (*primary, mechanism)):
        return _result(B_INCONCLUSIVE, ["分组区间没有形成。样本或簇太少，不能把点估计写成结论"])
    if trigger_rate is None or trigger_rate < min_trigger:
        return _result(B_INCONCLUSIVE, ["规则几乎不触发，先区分执行器没有接到动作，还是假设失败"])
    if illegal_rate is not None and illegal_rate > max_illegal:
        return _result(B_INCONCLUSIVE, ["候选经常不合法，当前运行时还没有把动作接上"])
    reasons: list[str] = []
    if all(_no_content(item, floor) for item in primary):
        reasons.append("相对通用执行器和空经验归纳，增益区间落在实用界内或上界不超过 0。这是内容贡献不成立，不是 p>0.05")
        return _result(B_NO_GO_CONTENT, reasons)
    if not all(_clears(item, floor) for item in primary):
        reasons.append("相对 B1 和 B3 的校正区间还没有同时越过 0 和实用下限")
        return _result(B_INCONCLUSIVE, reasons)
    if _equivalent(mechanism, floor):
        reasons.append("真实经验优于空经验和通用执行器，但相对去掉对齐和反例的直接归纳落在等效界内")
        return _result(B_NO_GO_MECHANISM, reasons)
    if not _clears(mechanism, floor):
        reasons.append("内容增益成立，但对齐机制相对直接归纳的区间还没有越过实用下限")
        return _result(B_INCONCLUSIVE, reasons)
    if retention_delta is None or retention_delta < -retention_tolerance:
        reasons.append("retention 差值超过预注册容忍，或 retention 还没有测量")
        return _result(B_INCONCLUSIVE, reasons)
    if not _families_agree(family_diffs):
        reasons.append("两个预定义任务族的配对差方向不一致，或有一族没有测到")
        return _result(B_INCONCLUSIVE, reasons)
    if not replacement_changed:
        reasons.append("在开发集快照上替换规则参数后，动作和结果都没有变化")
        return _result(B_INCONCLUSIVE, reasons)
    reasons.append("相对 B1 和 B3 的校正下界越过 0 且超过实用下限；相对 B5 的机制差同样越过下限；任务族方向一致，替换探针和 retention 都符合预注册")
    return _result(B_GO, reasons)


def practical_floor(min_gain: float, n_eval: int, resolution_tasks: int) -> float:
    if n_eval <= 0:
        return min_gain
    return max(min_gain, resolution_tasks / n_eval)


def render_decision_md(decision: Mapping[str, Any]) -> str:
    reasons = "\n".join(f"- {reason}" for reason in decision.get("reasons") or [])
    scope = decision.get("scope_note") or "范围还没有填写。"
    return "\n".join(
        [
            "# B 决策",
            "",
            f"状态：`{decision['status']}`",
            "",
            "## 范围",
            "",
            scope,
            "",
            f"论文主张是否允许引用这次运行：`{decision.get('paper_claim_allowed', False)}`",
            "",
            "## 理由",
            "",
            reasons or "- 没有形成理由",
            "",
            "## 预注册比较",
            "",
            _fmt("M − B1", decision.get("M_vs_B1")),
            _fmt("M − B3", decision.get("M_vs_B3")),
            _fmt("M − B5", decision.get("M_vs_B5")),
            "",
        ]
    )


def _no_content(item: Mapping[str, Any], floor: float) -> bool:
    lo, hi = _ci(item)
    equivalent = lo >= -floor and hi <= floor
    worse = hi <= 0
    return equivalent or worse


def _clears(item: Mapping[str, Any], floor: float) -> bool:
    lo, _hi = _ci(item)
    return float(item["diff"]) >= floor and lo > 0


def _equivalent(item: Mapping[str, Any], floor: float) -> bool:
    lo, hi = _ci(item)
    return lo >= -floor and hi <= floor


def _ci(item: Mapping[str, Any]) -> tuple[float, float]:
    ci = item["ci"]
    return float(ci[0]), float(ci[1])


def _families_agree(family_diffs: Mapping[str, float | None]) -> bool:
    signs = []
    for family in TARGET_FAMILIES:
        value = family_diffs.get(family)
        if value is None:
            return False
        if abs(value) > 1e-12:
            signs.append(1 if value > 0 else -1)
    return len(set(signs)) <= 1


def _result(status: str, reasons: list[str]) -> dict[str, Any]:
    return {"status": status, "reasons": reasons}


def _fmt(label: str, item: Mapping[str, Any] | None) -> str:
    if not item or item.get("diff") is None:
        return f"- {label}：没有数值"
    ci = item.get("ci")
    interval = "区间未形成" if not ci else f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]"
    return f"- {label}：差 {float(item['diff']):.4f}，{interval}"
