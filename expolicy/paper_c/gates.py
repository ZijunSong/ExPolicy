"""Pre-registered gates. A missing result is not a pass, and three wins are not a test."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from expolicy.paper_c.constants import (
    C_ASSIMILATION_FAIL,
    C_BASELINE_SUFFICIENT,
    C_COST_FAIL,
    C_INCONCLUSIVE,
    C_NO_HEADROOM,
    C_NOT_RUN,
    C_PASS,
    C_PREDICTION_FAIL,
    C_STATIC_FIRST,
    C_THRESHOLD_MISS,
    C_TRANSFER_FAIL,
    MODE_EMPTY,
    MODE_NO_UPDATE,
    MODE_REAL,
    MODE_WRONG,
    STATUS_OK,
)
from expolicy.paper_c.stats import paired_summary


def judge_c0(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    min_gain: float,
    retention_tolerance: float,
) -> dict[str, Any]:
    real = arms.get(MODE_REAL) or {}
    if real.get("infra_status") != STATUS_OK or real.get("gain") is None:
        return {"status": C_NOT_RUN, "reasons": ["F(real) 还没有完整的 shadow 和 retention 测量"]}
    reasons: list[str] = []
    gain = float(real["gain"])
    for mode in (MODE_NO_UPDATE, MODE_EMPTY, MODE_WRONG):
        other = arms.get(mode) or {}
        if other.get("gain") is None:
            return {"status": C_INCONCLUSIVE, "reasons": [f"{mode} 的增益缺失，不能把缺失当成零"]}
        gap = gain - float(other["gain"])
        if gap < float(min_gain):
            reasons.append(f"F(real) 相对 {mode} 的增益差是 {gap:.4f}，低于预注册 {min_gain:.2f}")
    retention = real.get("delta_retention")
    if retention is not None and float(retention) < -float(retention_tolerance):
        reasons.append(f"F(real) 的 retention 差值是 {float(retention):.4f}，超过容忍 {retention_tolerance:.2f}")
    if reasons:
        return {"status": C_ASSIMILATION_FAIL, "reasons": reasons}
    return {"status": C_PASS, "reasons": ["同一被动序列上，真实内容优于 no-update、空内容和错配内容"]}


def judge_c1(summary: Mapping[str, Any]) -> dict[str, Any]:
    states = [item for item in summary.get("states") or [] if item.get("status") == STATUS_OK and item.get("best_eval_gain") is not None]
    if not states:
        return {"status": C_NOT_RUN, "reasons": ["候选增益矩阵还没有独立评价"]}
    advantages = [float(item["best_eval_gain"]) - float(item["uniform_eval_gain"]) for item in states]
    winners = [str(item["best_family"]) for item in states]
    if max(advantages) <= 0.0:
        return {
            "status": C_NO_HEADROOM,
            "reasons": ["独立评价探针上，选择探针排出的最佳族不优于均匀采样"],
            "advantages": advantages,
        }
    if len(set(winners)) == 1:
        return {
            "status": C_STATIC_FIRST,
            "reasons": ["有增益差异，但最佳族在已测状态上没有变化。先比较训练集上冻住的 best-static，不直接训练动态策略"],
            "family": winners[0],
            "advantages": advantages,
        }
    return {
        "status": C_PASS,
        "reasons": ["不同状态上的最佳族不同，而且独立评价探针上的最佳选择优于均匀均值"],
        "advantages": advantages,
        "winners": winners,
    }


def judge_c2(
    *,
    state_advantage: float | None,
    family_excess: float | None,
    env_advantage: float | None,
    failure_advantage: float | None,
    state_groups: str,
    margin: float,
    correlation_value: float | None = None,
) -> dict[str, Any]:
    del correlation_value  # reported by the caller; it cannot open this gate
    if state_groups == "insufficient_groups":
        return {"status": C_INCONCLUSIVE, "reasons": ["经验序列分组不足，留出验证还不能做"]}
    if state_advantage is None or family_excess is None:
        return {"status": C_INCONCLUSIVE, "reasons": ["未见状态或未见任务族的选择收益缺失"]}
    reasons: list[str] = []
    if float(state_advantage) <= 0.0:
        reasons.append("未见状态上的选择收益不大于均匀选择")
    if env_advantage is not None and float(env_advantage) >= float(state_advantage) - float(margin):
        reasons.append("环境 ID 就能复现选择收益，事前特征的信息优势没有建立")
    if failure_advantage is not None and float(failure_advantage) >= float(state_advantage) - float(margin):
        reasons.append("当前失败率就能复现选择收益，同化后增益的信息优势没有建立")
    if float(family_excess) <= 0.0:
        reasons.append("未见任务族上没有超过随机选择的命中收益")
    if reasons:
        return {"status": C_PREDICTION_FAIL, "reasons": reasons}
    return {"status": C_PASS, "reasons": ["未见状态和未见任务族上的选择收益不能被环境 ID 或失败率解释"]}


def _clears(summary: Mapping[str, Any], minimum: float) -> bool:
    if summary.get("diff") is None or not summary.get("population_noise_claim"):
        return False
    if summary.get("crosses_zero") or summary.get("wide") or summary.get("zero_width"):
        return False
    ci = summary.get("ci") or [None, None]
    return float(summary["diff"]) >= float(minimum) and ci[0] is not None and float(ci[0]) > 0.0


def judge_pilot(
    *,
    cost_diffs: Sequence[float],
    episode_diffs: Sequence[float],
    final_diffs: Sequence[float],
    retention_diffs: Sequence[float],
    baseline: str,
    min_aulc: float,
    min_final: float,
    retention_tolerance: float,
    wide: float,
    n_public_environments: int,
    environment_class: str,
) -> dict[str, Any]:
    cost = paired_summary(cost_diffs, wide=wide)
    episode = paired_summary(episode_diffs, wide=wide)
    final = paired_summary(final_diffs, wide=wide)
    retention = paired_summary(retention_diffs, wide=wide)
    if cost.get("diff") is None:
        comparison = C_NOT_RUN
        reasons = ["总预算匹配的序列差值还没有"]
    elif int(cost.get("n") or 0) < 3 or cost.get("zero_width"):
        comparison = C_INCONCLUSIVE
        reasons = ["序列不足三或区间宽度为零。三序列全胜也不当作总体检验"]
    elif (
        episode.get("diff") is not None
        and float(episode["diff"]) >= float(min_aulc)
        and float(cost["diff"]) < float(min_aulc)
    ):
        comparison = C_COST_FAIL
        reasons = ["名义 episode 轴上的增益没有保持到包含 probe 的交互总成本"]
    elif cost.get("wide") or cost.get("crosses_zero") or not cost.get("population_noise_claim"):
        comparison = C_INCONCLUSIVE
        reasons = ["区间过宽或跨过 0，还支持不了总体噪声陈述"]
    elif not _clears(cost, min_aulc) or not _clears(final, min_final):
        comparison = C_BASELINE_SUFFICIENT
        reasons = [f"相对开发序列上冻住的 {baseline}，总成本匹配的 AULC 或终局成绩没有越过预注册差值"]
    elif retention.get("diff") is not None and float(retention["diff"]) < -float(retention_tolerance):
        comparison = C_THRESHOLD_MISS
        reasons = ["retention 损失超过预注册容忍"]
    else:
        comparison = C_PASS
        reasons = [f"相对 {baseline} 的总成本匹配区间过线"]
    paper = comparison if environment_class == "public" and int(n_public_environments) >= 2 else C_NOT_RUN
    return {
        "comparison_status": comparison,
        "paper_claim": paper,
        "baseline": baseline,
        "reasons": reasons,
        "cost": cost,
        "episode": episode,
        "final": final,
        "retention": retention,
    }


def judge_transfer(second_environment_valid: bool | None) -> dict[str, Any]:
    if second_environment_valid is None:
        return {"status": C_NOT_RUN, "reasons": ["第二个公开环境还没有结果"]}
    if not second_environment_valid:
        return {"status": C_TRANSFER_FAIL, "reasons": ["新环境或新模板上的协议无效，不能把环境启发式写成通用策略"]}
    return {"status": C_PASS, "reasons": ["第二个预先登记的环境没有否定该协议"]}


def front_door_open(c0: Mapping[str, Any], c1: Mapping[str, Any], c2: Mapping[str, Any]) -> bool:
    return c0.get("status") == C_PASS and c1.get("status") == C_PASS and c2.get("status") == C_PASS
