"""Gate H analysis (H1–H6). Thresholds are pre-registered in constants.py."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from expolicy.r9_lab.constants import (
    A0_DISCARD,
    ACTIONS,
    H1_MIN_OPERATORS,
    H1_UNIQUE_BEST_MIN,
    H2_ORACLE_VS_FIXED,
    H2_PROBE_TASK_EQUIVALENTS,
    H3_ENVS_WITH_DIVERSITY,
    H3_ORACLE_VS_ENV_FIXED,
    H3_WITHIN_ENV_UNIQUE_BEST,
    H4_FULL_VS_ENV_ONLY_PP,
    H4_FULL_VS_MAJORITY_PP,
    H5_MIN_NON_DISCARD_OPERATORS,
    H5_POSITIVE_U_BATCHES,
)


def unique_best_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {a: 0 for a in ACTIONS}
    for row in rows:
        best = row.get("best_action")
        margin = float(row.get("best_margin") or 0.0)
        # Ignore unique-best that could be a single probe-task flip (margin tiny).
        probe_n = int(row.get("shadow_n") or 60)
        min_margin = 1.0 / max(1, probe_n)
        if best in counts and margin >= min_margin:
            # unique-best: strictly greater than second
            counts[str(best)] += 1
    return counts


def analyze_gate_h(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_report: Mapping[str, Any] | None = None,
    leakage_ok: bool = True,
) -> dict[str, Any]:
    n = len(rows) or 1
    ub = unique_best_counts(rows)
    ub_frac = {a: ub[a] / n for a in ACTIONS}
    h1_ops = [a for a in ACTIONS if a != A0_DISCARD and ub_frac[a] >= H1_UNIQUE_BEST_MIN]
    h1 = len(h1_ops) >= H1_MIN_OPERATORS

    def mean_u(action: str | None = None, picker=None) -> float:
        vals = []
        for row in rows:
            if picker:
                vals.append(float(picker(row)))
            else:
                vals.append(float(row.get(f"U_{action.lower()}") if action else 0.0))
        return sum(vals) / max(1, len(vals))

    # columns: U_discard / U_episodic / U_semantic / U_skillopt
    key = {
        "DISCARD": "U_discard",
        "EPISODIC_REPLAY": "U_episodic",
        "SEMANTIC_MEMORY": "U_semantic",
        "SKILLOPT_EDIT": "U_skillopt",
    }

    def u_of(row: Mapping[str, Any], action: str) -> float:
        if key[action] in row:
            return float(row[key[action]])
        return float((row.get("U") or {}).get(action) or row.get("U_" + action) or 0.0)

    oracle_vals = []
    fixed_means = {a: 0.0 for a in ACTIONS}
    for a in ACTIONS:
        fixed_means[a] = sum(u_of(r, a) for r in rows) / n
    best_fixed_action = max(ACTIONS, key=lambda a: fixed_means[a])
    best_fixed = fixed_means[best_fixed_action]
    for r in rows:
        oracle_vals.append(max(u_of(r, a) for a in ACTIONS))
    mean_oracle = sum(oracle_vals) / n
    probe_n = int(rows[0].get("shadow_n") or 60) if rows else 60
    h2_thresh = max(H2_ORACLE_VS_FIXED, H2_PROBE_TASK_EQUIVALENTS / max(1, probe_n))
    h2 = mean_oracle >= best_fixed + h2_thresh

    by_env: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_env[str(r.get("env") or "")].append(r)
    env_fixed = {}
    for env, erows in by_env.items():
        means = {a: sum(u_of(r, a) for r in erows) / max(1, len(erows)) for a in ACTIONS}
        env_fixed[env] = max(means.values()) if means else 0.0
    # best_per_env_fixed applied per row then mean
    per_env_fixed_vals = []
    for r in rows:
        env = str(r.get("env") or "")
        erows = by_env[env]
        means = {a: sum(u_of(x, a) for x in erows) / max(1, len(erows)) for a in ACTIONS}
        best_a = max(ACTIONS, key=lambda a: means[a])
        per_env_fixed_vals.append(u_of(r, best_a))
    best_per_env_fixed = sum(per_env_fixed_vals) / n
    h3_gap = mean_oracle >= best_per_env_fixed + H3_ORACLE_VS_ENV_FIXED
    env_div = 0
    for env, erows in by_env.items():
        counts = unique_best_counts(erows)
        frac = {a: counts[a] / max(1, len(erows)) for a in ACTIONS}
        n_ops = sum(1 for a in ACTIONS if a != A0_DISCARD and frac[a] >= H3_WITHIN_ENV_UNIQUE_BEST)
        if n_ops >= 2:
            env_div += 1
    h3 = h3_gap and env_div >= H3_ENVS_WITH_DIVERSITY

    feat = feature_report or {}
    full_acc = float(feat.get("full_feature_accuracy") or 0.0)
    maj_acc = float(feat.get("majority_accuracy") or 0.0)
    env_acc = float(feat.get("env_only_accuracy") or 0.0)
    h4 = (full_acc >= maj_acc + H4_FULL_VS_MAJORITY_PP) and (full_acc >= env_acc + H4_FULL_VS_ENV_ONLY_PP)

    pos = {a: 0 for a in ACTIONS if a != A0_DISCARD}
    for r in rows:
        for a in pos:
            if u_of(r, a) > 0:
                pos[a] += 1
    h5_ops = [a for a, c in pos.items() if c >= H5_POSITIVE_U_BATCHES]
    # at least one operator worse than another on some batches
    crossover = False
    non_disc = [a for a in ACTIONS if a != A0_DISCARD]
    for r in rows:
        us = [u_of(r, a) for a in non_disc]
        if max(us) - min(us) > 1e-9:
            crossover = True
            break
    h5 = len(h5_ops) >= H5_MIN_NON_DISCARD_OPERATORS and crossover

    h6 = bool(leakage_ok)
    passed = all([h1, h2, h3, h4, h5, h6])
    if passed:
        case = "CASE_H_PASS"
        decision = "HETEROGENEOUS_ASSIMILATION_PASS"
    elif (not h1) or (not h2):
        case = "CASE_H_FIXED_DOMINATES"
        decision = "SUBSTRATE_FIXED_DOMINATES"
    elif h1 and h2 and not h3:
        case = "CASE_H_ENV_ROUTER_ONLY"
        decision = "SUBSTRATE_ENV_ROUTER_ONLY"
    elif h1 and h2 and h3 and not h4:
        case = "CASE_H_UNLEARNABLE"
        decision = "SUBSTRATE_UNLEARNABLE"
    else:
        case = "CASE_H_FIXED_DOMINATES"
        decision = "SUBSTRATE_FIXED_DOMINATES"
    return {
        "H1": h1,
        "H2": h2,
        "H3": h3,
        "H4": h4,
        "H5": h5,
        "H6": h6,
        "pass": passed,
        "case": case,
        "decision": decision,
        "unique_best_fraction": ub_frac,
        "h1_operators": h1_ops,
        "mean_U_oracle": mean_oracle,
        "mean_U_best_fixed": best_fixed,
        "best_fixed_action": best_fixed_action,
        "h2_threshold": h2_thresh,
        "mean_U_best_per_env_fixed": best_per_env_fixed,
        "env_diversity_count": env_div,
        "h5_positive_operators": h5_ops,
        "feature_report": feat,
    }
