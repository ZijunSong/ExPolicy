"""Decision note. A passing harness is not a multi-environment paper claim."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from expolicy.paper_d.constants import (
    DELTA,
    GUIDANCE_DATE,
    PROTOCOL_ID,
    PROTOCOL_VERSION,
    RETENTION_WEIGHT,
)


def build_decision(parts: Mapping[str, Any]) -> dict[str, Any]:
    controls_passed = bool(parts.get("controls_passed"))
    real_configs = int(parts.get("n_real_configurations") or 0)
    paper = "HARNESS_ONLY"
    if real_configs >= 2 and controls_passed and parts.get("prospective_passed"):
        paper = "NOT_AUTOMATIC"
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "paper_claim": paper,
        "harness_status": "PASS" if controls_passed else "FAIL",
        "controls_passed": controls_passed,
        "n_real_configurations": real_configs,
        "inventory_verified": int(parts.get("inventory_verified") or 0),
        "measurement_status": (parts.get("audit") or {}).get("status"),
        "prospective_case_only": bool((parts.get("prospective") or {}).get("case_validation_only")),
        "reasons": list(parts.get("reasons") or []),
    }


def render_protocol_md() -> str:
    return "\n".join(
        [
            f"# {PROTOCOL_ID} 协议",
            "",
            f"指导日期：{GUIDANCE_DATE}。协议版本：{PROTOCOL_VERSION}。",
            "",
            "评测单位是 `(environment, backbone, assimilator/action set, state distribution, budget, metric)`。",
            "本文件由代码生成，用来锁定诊断定义。它不是真实环境上的实验结果。",
            "",
            "## 效用",
            "",
            "协议效用先对未来成功率和 retention 取期望，再计算惩罚：",
            "",
            "`u = (E J_future(S') - E J_future(S)) - λ max(0, E J_ret(S) - E J_ret(S'))`",
            "",
            f"预注册的 λ = {RETENTION_WEIGHT}。预注册的实用选择阈值 δ = {DELTA}。",
            "δ 是项目决策尺度，不是录用线。成本和 retention 容忍度另列，不并进 δ。",
            "原 R9 指标是每个 noisy repeat 先取正部再平均，只作为 `legacy` 列保留，不能替换协议效用。",
            "无惩罚的 gain 和 retention 差值一起报告。",
            "",
            "## 三个不能混用的 headroom",
            "",
            "- `estimate_naive_gap`：同一样本上先按行取最大再与固定动作比较。它通常偏乐观，不是可信上界，也不是 oracle。",
            "- `crossfit_selection_gain`：按来源任务或模板分半，一半选择、一半评价，再交换。结果可以为负，不得裁成 0。",
            "- `bound_panel_headroom`：有限面板上 u(x,a) 的同时区间，再代入 global / within 的上下界。区间宽表示证据不足。",
            "",
            "单步协议不覆盖长期 acquisition × assimilation 最优性。诊断卡固定写明“未覆盖长期结论”。",
            "缺失动作不按 0 填入 max。不同预算、不同配置、未测到的格子不混算。",
            "外部称为 oracle 的数若低于固定动作，只触发测量核查，不证明结构上不可能。",
            "",
            "## 已知结构控制",
            "",
            "Exact tie、noisy tie、global winner、env-only winner、observable crossover、hidden crossover、delayed synergy",
            "都在离线效用表或精确小 MDP 上检查诊断。它们不是真实 agent 的能力增益，",
            "也不允许在真实环境里手工添加奖惩来制造交叉。",
            "",
            "## 主张边界",
            "",
            "LifelongAgentBench 已经评测 lifelong 能力。Faithful Self-Evolvers 已经研究经验忠实性。",
            "ASSAY 已经报告技能贡献异质性。本协议不把这些概念本身当作新发现，",
            "也不把这里使用的独立选择/评价和同时置信界宣称成首次提出的统计工具。",
            "",
        ]
    )


def render_audit_md(audit: Mapping[str, Any], inventory: Mapping[str, Any]) -> str:
    lines = [
        "# D 测量审计",
        "",
        f"夹具状态：`{audit.get('status')}`。允许科学结论：`{audit.get('scientific_conclusion_allowed')}`。",
        "",
        "下面的通过结果来自合成 no-op 夹具，用来确认审计代码。它没有读取一套完整的 R9 任务级矩阵。",
        f"已核实产物 {inventory.get('n_verified', 0)} 个，拒绝 {inventory.get('n_rejected', 0)} 个，",
        f"忽略 STATUS_LIVE {inventory.get('n_status_live_ignored', 0)} 个。完成与否只看带有来源 hash 的 DONE 文件。",
        "",
        f"惩罚负偏已记录：`{audit.get('penalty_negative_bias')}`。原始 no-op 差为 0 时，这不单独判为测量无效。",
        "",
    ]
    for reason in audit.get("reasons") or []:
        lines.append(f"- {reason}")
    if inventory.get("missing_roots"):
        lines.append("")
        lines.append("未找到的清点目录：")
        for root in inventory["missing_roots"]:
            lines.append(f"- `{root}`")
    lines.append("")
    return "\n".join(lines)


def render_decision_md(decision: Mapping[str, Any], control_reports: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# D 决策",
        "",
        f"论文主张：`{decision['paper_claim']}`。Harness：`{decision['harness_status']}`。",
        "",
        "已知结构控制通过，只说明诊断工具在离线表上的行为符合预注册。",
        "这不是跨环境的经验策略结论，也不能写成某个真实配置已经有或没有选择空间。",
        "",
        "## 控制",
        "",
    ]
    for report in control_reports:
        mark = "通过" if report["passed"] else "失败"
        lines.append(f"- {report['name']}：{mark}。{report['known_structure']}")
        for reason in report.get("reasons") or []:
            lines.append(f"  - {reason}")
    lines.extend(
        [
            "",
            "## 不能写成的主张",
            "",
            "- 不能把 naive gap 写成 oracle 或可信上界。",
            "- 不能因为一次估计没有显著胜过固定策略，就写成 ENV_FIXED_SUFFICIENT。",
            "- 不能把资源耗尽后仍然很宽的区间写成排除了实用增益。",
            "- 不能把单步诊断写成长期最优。",
            "- 不能用两三组 held-out 配置报告显著相关。",
            "- 不能把合成控制的通过写成 LifelongAgentBench 上的能力增益。",
            "",
        ]
    )
    return "\n".join(lines)
