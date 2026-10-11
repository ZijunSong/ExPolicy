"""Claim-level decision. A missing result is not a pass, and a wide interval is not a negative result."""

from __future__ import annotations

from typing import Any, Mapping

from expolicy.paper_a.constants import (
    AUDIT_PASS,
    DECISION_GO,
    DECISION_INCONCLUSIVE,
    DECISION_INSUFFICIENT,
    DECISION_NO_GO,
    DECISION_NOT_RUN,
    SPACE_GO,
)


def judge_prediction(
    primary: Mapping[str, Any] | None,
    *,
    min_gain: float,
    retention_tolerance: float,
) -> dict[str, Any]:
    if not primary or primary.get("diff") is None:
        return {"status": DECISION_NOT_RUN, "reasons": ["条件效用回归还没有在未见 batch 上评分"]}
    reasons: list[str] = []
    if not primary.get("threshold_validated"):
        reasons.append("动作阈值没有在内层验证折上冻住，外层数字不能当作确认结果")
        return {"status": DECISION_INSUFFICIENT, "reasons": reasons}
    ci = primary.get("ci")
    if not ci:
        reasons.append("分组重采样没有形成区间。共享 probe 的格子没有被当成独立样本，因此样本可能不够形成区间")
        return {"status": DECISION_INSUFFICIENT, "reasons": reasons}
    diff = float(primary["diff"])
    lo, hi = float(ci[0]), float(ci[1])
    retention = primary.get("mean_delta_retention")
    if retention is not None and float(retention) < -float(retention_tolerance):
        reasons.append(
            f"所选动作的平均 retention 差值是 {float(retention):.4f}，超过预注册 {retention_tolerance:.2f} 的遗忘容忍"
        )
        return {"status": DECISION_NO_GO, "reasons": reasons}
    if hi < min_gain:
        reasons.append(
            f"相对 env-only 的配对区间上界 {hi:.4f} 低于预注册增益 {min_gain:.2f}。这是精度足够时的否定，不是把 p>0.05 当成无效"
        )
        return {"status": DECISION_NO_GO, "reasons": reasons}
    versus = primary.get("versus_simple_rule")
    retention_ok = retention is None or float(retention) >= -float(retention_tolerance)
    clears = diff >= min_gain and lo > 0 and retention_ok
    if not clears:
        reasons.append(
            f"配对差 {diff:.4f}，区间 [{lo:.4f}, {hi:.4f}] 仍跨过 0 或 {min_gain:.2f}。证据不足，不能写成没有效果"
        )
        return {"status": DECISION_INSUFFICIENT, "reasons": reasons}
    if not versus or versus.get("diff") is None:
        reasons.append("相对 env-only 的点估计和区间过线，但预注册的简单规则还没有在同一批 test 上评分")
        return {"status": DECISION_INSUFFICIENT, "reasons": reasons}
    rule_ci = versus.get("ci")
    if versus.get("diff", 0.0) > 0 and rule_ci and float(rule_ci[0]) > 0:
        reasons.append("相对 env-only 与简单规则的配对下界都大于 0，且遗忘在容忍内")
        return {"status": DECISION_GO, "reasons": reasons}
    if rule_ci and float(rule_ci[1]) <= 0:
        reasons.append("相对简单规则的区间上界不大于 0")
        return {"status": DECISION_NO_GO, "reasons": reasons}
    reasons.append("相对简单规则的区间还跨过 0")
    return {"status": DECISION_INSUFFICIENT, "reasons": reasons}


def build_decision(
    *,
    audit: Mapping[str, Any] | None = None,
    space: Mapping[str, Any] | None = None,
    router: Mapping[str, Any] | None = None,
    sequence: Mapping[str, Any] | None = None,
    min_gain: float = 0.02,
    retention_tolerance: float = 0.01,
) -> dict[str, Any]:
    prediction = judge_prediction(
        None if router is None else router.get("primary"),
        min_gain=min_gain,
        retention_tolerance=retention_tolerance,
    )
    audit_status = None if audit is None else audit.get("status")
    space_status = None if space is None else space.get("status")
    sequence_status = _sequence_status(sequence, min_gain=min_gain)
    claim = _claim(audit_status, space_status, prediction["status"], sequence_status)
    return {
        "protocol_id": "A-v1",
        "claim": claim,
        "audit_status": audit_status or DECISION_NOT_RUN,
        "space_status": space_status or DECISION_NOT_RUN,
        "prediction_status": prediction["status"],
        "prediction_reasons": prediction["reasons"],
        "sequence_status": sequence_status,
        "r9_gates_unchanged": True,
        "skillopt_in_action_space": False,
        "hindsight_is_not_the_result": True,
        "status": _overall(claim, prediction["status"], space_status, sequence_status),
    }


def _sequence_status(sequence: Mapping[str, Any] | None, *, min_gain: float) -> str:
    if not sequence:
        return DECISION_NOT_RUN
    if sequence.get("ood_state"):
        return "OOD_NOT_CONFIRMED"
    paired = ((sequence.get("comparisons") or {}).get("m_vs_env_only")) or []
    if not paired:
        return DECISION_INCONCLUSIVE
    diffs = [float(row["paired_diff"]) for row in paired if row.get("paired_diff") is not None]
    if not diffs:
        return DECISION_INCONCLUSIVE
    if sum(diffs) / len(diffs) >= min_gain and min(diffs) > 0:
        return DECISION_GO
    return DECISION_INSUFFICIENT


def _claim(audit_status: str | None, space_status: str | None, prediction: str, sequence_status: str) -> str:
    if sequence_status == DECISION_GO:
        return "顺序确认：固定采集序列上，冻结后的选择器在 future-test 上优于 env-only。这不是 teacher-forced 矩阵收益。"
    if prediction == DECISION_GO:
        return "事前预测：决策前可见的状态和经验特征，在未见 batch 上的效用优于 env-only，并且超过冻结的简单规则。"
    if space_status == SPACE_GO:
        return "选择空间：同一环境内，独立半套 probe 上的动作选择优于环境固定动作。这还不是特征可预测性。"
    if audit_status == AUDIT_PASS:
        return "测量：no-op 的原始配对差与预注册等效区间相容。这只说明评分可以继续用，不支持同化选择的论文主张。"
    if any(status not in (None, DECISION_NOT_RUN) for status in (audit_status, space_status, prediction, sequence_status)):
        return "已有部分结果，但还支持不了测量、选择空间、事前预测或顺序确认中的任何一级主张。"
    return "实验尚未执行。现有代码和历史 R9 格子都不能支持 A-v1 的主张。"


def _overall(claim: str, prediction: str, space_status: str | None, sequence_status: str) -> str:
    if sequence_status == DECISION_GO or prediction == DECISION_GO:
        return DECISION_GO
    if prediction == DECISION_NO_GO or space_status == "A-SPACE-NO_GO":
        return DECISION_NO_GO
    if prediction == DECISION_INSUFFICIENT or space_status == DECISION_INCONCLUSIVE:
        return prediction if prediction != DECISION_NOT_RUN else DECISION_INCONCLUSIVE
    if "尚未执行" in claim:
        return DECISION_NOT_RUN
    return DECISION_INCONCLUSIVE


def render_decision_md(decision: Mapping[str, Any], *, audit: Mapping[str, Any] | None = None, space: Mapping[str, Any] | None = None, router: Mapping[str, Any] | None = None, sequence: Mapping[str, Any] | None = None) -> str:
    lines = [
        "# A-v1 决策",
        "",
        f"协议状态：`{decision['status']}`",
        "",
        "## 现有证据支持的主张",
        "",
        decision["claim"],
        "",
        "## 明确不支持的说法",
        "",
        "- R9 的 H1–H6 与 Gate J 保持原结论。删掉 SkillOpt 后的格子不会被改写成“原 R9 通过”。",
        "- SkillOpt 不在 A-v1 动作空间里。拒绝更新不能拿来解释编辑收益。",
        "- `naive_hindsight_gap` 是同一张测量矩阵上先取最大再平均的描述量，不是论文结果。",
        "- 矩阵从 S0 出发时，只能说单次选择。没有多种能力状态，就不能写成能力状态自适应。",
        "- teacher-forced 矩阵里的收益不会被抄成顺序实验收益。",
        "- 噪声重复不是新的实验 seed。同一个 batch 换 seed 也不是新的独立样本。",
        "",
        "## 测量",
        "",
        f"状态：`{decision['audit_status']}`",
        "",
    ]
    if audit:
        lines.append(f"原因：{'; '.join(audit.get('reasons') or []) or '见审计原文'}")
        penalty = audit.get("penalty") or {}
        if penalty.get("note"):
            lines.append("")
            lines.append(penalty["note"])
        lines.append("")
        lines.append("没有把 DISCARD 的效用逐行减掉。原始 `delta_shadow`、`delta_retention` 和带惩罚的 U 同时保留。")
    else:
        lines.append("no-op 审计还没有跑。历史矩阵里 DISCARD 的负 U 不会被事先改写成代码错误。")
    lines.extend(["", "## 选择空间", "", f"状态：`{decision['space_status']}`", ""])
    if space:
        lines.append(
            f"后见差距：{space.get('naive_hindsight_gap')}。独立半套选择增益：{space.get('crossfit_selector_gain')}。"
            f" 实用门槛：{space.get('delta_space')}。特征策略增益：{space.get('feature_policy_gain')}。"
        )
        lines.append(f"原因：{'; '.join(space.get('reasons') or [])}")
    else:
        lines.append("还没有逐题配对矩阵，因此不能判断环境内有没有可利用的选择空间。")
    lines.extend(
        [
            "",
            "## 事前预测",
            "",
            f"状态：`{decision['prediction_status']}`",
            "",
            *[f"- {reason}" for reason in decision.get("prediction_reasons") or []],
            "",
            "主结果是未见 batch 上的效用差，不是 argmax 分类准确率。",
            "",
            "## 顺序实验",
            "",
            f"状态：`{decision['sequence_status']}`",
            "",
        ]
    )
    if sequence and sequence.get("ood_state"):
        lines.append("选择器训练时只见过 S0。顺序运行里的后续状态是分布外，这里的数字不能确认顺序收益。")
    elif not sequence:
        lines.append("固定采集序列还没有跑。")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "上面的主张级别就是当前证据能到达的位置。状态不是 PASS / FAIL 两个字。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def render_audit_md(audit: Mapping[str, Any]) -> str:
    eq = (audit.get("penalty") or {}).get("equivalence") or {}
    lines = [
        "# A-v1 测量审计",
        "",
        f"状态：`{audit.get('status')}`",
        "",
        "比较的是同一 S0 分出的 BASE、DISCARD、REJECTED_EDIT。比较项包括记忆、skill、检索顺序、system prompt、工具配置、环境快照和模型身份，不只是一个 state 文件的 hash。",
        "",
        f"随机性来源：`{audit.get('randomness_located')}`",
        "",
        "原因：",
        "",
    ]
    for reason in audit.get("reasons") or []:
        lines.append(f"- {reason}")
    if not audit.get("reasons"):
        lines.append("- 无")
    lines.extend(
        [
            "",
            "## 配对差",
            "",
            f"原始配对差均值：{eq.get('mean_paired_delta')}",
            f"区间：{eq.get('ci')}",
            f"预注册等效界：±{eq.get('bound')}",
            f"配对数：{eq.get('n_pairs')}。重复运行只用于观察噪声，没有乘进样本量。",
            "",
            (audit.get("penalty") or {}).get("note") or "",
            "",
            "未做 DISCARD  recenter。",
            "",
        ]
    )
    return "\n".join(lines)
