"""Decision note. The harness result is not a public-environment claim."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from expolicy.paper_c.constants import C_NOT_RUN, C_PASS, PROTOCOL_ID


def build_decision(parts: Mapping[str, Any]) -> dict[str, Any]:
    c0 = parts.get("c0") or {"status": C_NOT_RUN, "reasons": ["C0 未运行"]}
    c1 = parts.get("c1") or {"status": C_NOT_RUN, "reasons": ["C1 未运行"]}
    c2 = parts.get("c2") or {"status": C_NOT_RUN, "reasons": ["C2 未运行"]}
    pilot = parts.get("pilot") or {"comparison_status": C_NOT_RUN, "paper_claim": C_NOT_RUN, "reasons": ["学习序列未运行"]}
    transfer = parts.get("transfer") or {"status": C_NOT_RUN, "reasons": ["第二个公开环境未运行"]}
    doors = bool(parts.get("front_door_open"))
    paper = C_NOT_RUN
    public = parts.get("environment_class") == "public" and int(parts.get("n_public_environments") or 0) >= 2
    if public and doors and pilot.get("paper_claim") == C_PASS and transfer.get("status") == C_PASS:
        paper = C_PASS
    return {
        "protocol_id": PROTOCOL_ID,
        "paper_claim": paper,
        "environment_class": parts.get("environment_class", "synthetic"),
        "c0_status": c0.get("status"),
        "c1_status": c1.get("status"),
        "c2_status": c2.get("status"),
        "pilot_status": pilot.get("comparison_status"),
        "pilot_interpreted": doors,
        "transfer_status": transfer.get("status"),
        "baseline": pilot.get("baseline"),
        "reasons": {
            "c0": list(c0.get("reasons") or []),
            "c1": list(c1.get("reasons") or []),
            "c2": list(c2.get("reasons") or []),
            "pilot": list(pilot.get("reasons") or []),
            "transfer": list(transfer.get("reasons") or []),
        },
    }


def render_decision_md(decision: Mapping[str, Any]) -> str:
    lines = [
        "# C-v1 决策",
        "",
        f"论文主张：`{decision['paper_claim']}`",
        "",
        "真实公开环境尚未执行。合成协议只检查采集代码的门禁、泄漏和成本记账，不能写成同化后增益采样已经成立。",
        "",
        "## 各门状态",
        "",
        f"- C0 同化器内容增益：`{decision['c0_status']}`",
        f"- C1 候选增益空间：`{decision['c1_status']}`",
        f"- C2 事前可预测性：`{decision['c2_status']}`",
        f"- 学习序列比较：`{decision['pilot_status']}`；前门允许解释该比较：`{decision['pilot_interpreted']}`",
        f"- 第二环境：`{decision['transfer_status']}`",
        f"- 开发序列冻住的对照：`{decision.get('baseline')}`",
        "",
        "## 原因",
        "",
    ]
    for gate, reasons in (decision.get("reasons") or {}).items():
        for reason in reasons:
            lines.append(f"- {gate}: {reason}")
    lines.extend(
        [
            "",
            "## 不能写成的主张",
            "",
            "- 不能把 exact shadow chooser 写成免费上界。",
            "- 不能把三序列全胜写成充分统计检验。",
            "- 不能把相关系数单独写成采集策略有效。",
            "- 不能把 no-op 的负效用全部解释成真实损伤；原始 shadow 和 retention 要一起报告。",
            "- 相关文献里的 learning-progress、主动学习和数据估值原文还没有补查完，不能写首次用未来学习增益指导采集。",
            "",
        ]
    )
    return "\n".join(lines)


def render_visibility_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# C 特征可见性审计",
        "",
        report["rule"],
        "",
        "## 增益模型可以使用",
        "",
    ]
    for name in report["model_features"]:
        lines.append(f"- `{name}`")
    lines.extend(["", "## 只在下一轮更新历史后可以使用", ""])
    for name in report["historical_features_next_round_only"]:
        lines.append(f"- `{name}`")
    lines.extend(["", "## 当前候选执行后不能进入本轮特征", ""])
    for name in report["forbidden_current_rollout"]:
        lines.append(f"- `{name}`")
    lines.append("")
    return "\n".join(lines)


def render_assimilator_audit_md(audit: Mapping[str, Any], judgment: Mapping[str, Any]) -> str:
    lines = [
        "# C0 同化器审计",
        "",
        f"判定：`{judgment['status']}`",
        "",
        "这是合成内容同化器上的机制检查。它不替代公开环境上的 C0，也不把 B 的归纳器标成已通过。",
        "",
        "| 臂 | 呈现条数 | 增益 | shadow 增量 | retention 增量 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for mode, arm in (audit.get("arms") or {}).items():
        lines.append(
            f"| {mode} | {arm.get('n_presented')} | {arm.get('gain')} | {arm.get('delta_shadow')} | {arm.get('delta_retention')} |"
        )
    lines.append("")
    for reason in judgment.get("reasons") or []:
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)
