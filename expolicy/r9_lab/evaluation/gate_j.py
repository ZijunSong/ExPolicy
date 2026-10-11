"""Gate J analysis. Only meaningful after CASE_H_PASS."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from expolicy.r9_lab.constants import J1_O3_VS_AXIS, J2_SEED_FRACTION, J5_SPEARMAN_MIN, J5_TOP1_VS_RANDOM


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    rx = _ranks(xs[:n])
    ry = _ranks(ys[:n])
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    deny = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if denx * deny == 0:
        return 0.0
    return num / (denx * deny)


def _ranks(vals: Sequence[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    for r, i in enumerate(order, 1):
        ranks[i] = float(r)
    return ranks


def analyze_gate_j(seed_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """seed_rows: one dict per seed with O0/O1/O2/O3 future scores and commit traces."""
    if not seed_rows:
        return {"pass": False, "case": "CASE_J_NO_JOINT_GAP", "decision": "JOINT_IDENTIFIABILITY_FAIL"}

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in seed_rows) / len(seed_rows)

    o0, o1, o2, o3 = mean("O0_future"), mean("O1_future"), mean("O2_future"), mean("O3_future")
    axis_best = max(o0, o1, o2)
    j1 = o3 >= axis_best + J1_O3_VS_AXIS
    n_better = sum(1 for r in seed_rows if float(r["O3_future"]) > max(float(r["O1_future"]), float(r["O2_future"])))
    j2 = (n_better / len(seed_rows)) >= J2_SEED_FRACTION

    used_actions: set[str] = set()
    used_regions: set[str] = set()
    for r in seed_rows:
        used_actions |= set(r.get("committed_actions") or [])
        used_regions |= set(r.get("committed_regions") or [])
    j3 = len(used_actions) >= 2 and len(used_regions) >= 2
    j4 = any(float(r.get("within_env_joint_advantage") or 0.0) > 0 for r in seed_rows)
    rhos = [float(r.get("spearman") or _spearman(r.get("q_scores") or [], r.get("downstream") or [])) for r in seed_rows]
    top1 = [float(r.get("top1_vs_random") or 0.0) for r in seed_rows]
    j5 = (sum(rhos) / len(rhos) >= J5_SPEARMAN_MIN) or (sum(top1) / len(top1) >= J5_TOP1_VS_RANDOM)
    j6 = all(not r.get("future_leakage", False) for r in seed_rows)

    passed = all([j1, j2, j3, j4, j5, j6])
    if passed:
        case, decision = "CASE_J_PASS", "JOINT_IDENTIFIABILITY_PASS"
    elif abs(o1 - o3) <= 0.01 and o1 >= o2:
        case, decision = "CASE_J_ASSIM_ONLY", "JOINT_IDENTIFIABILITY_FAIL"
    elif abs(o2 - o3) <= 0.01 and o2 >= o1:
        case, decision = "CASE_J_ACQ_ONLY", "JOINT_IDENTIFIABILITY_FAIL"
    else:
        case, decision = "CASE_J_NO_JOINT_GAP", "JOINT_IDENTIFIABILITY_FAIL"
    return {
        "J1": j1,
        "J2": j2,
        "J3": j3,
        "J4": j4,
        "J5": j5,
        "J6": j6,
        "pass": passed,
        "case": case,
        "decision": decision,
        "mean_O0": o0,
        "mean_O1": o1,
        "mean_O2": o2,
        "mean_O3": o3,
    }
