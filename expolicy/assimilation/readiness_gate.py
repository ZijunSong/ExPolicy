"""Sequential Readiness Test for RASA skill activation.

Frozen thresholds. Do not retune from R5.4 results.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

DELTA = 0.10
CONFIDENCE = 0.90
N_MC = 4096
MIN_BASE_FAILURES = 4
MIN_CLUSTER_COVERAGE = 2
FUTILITY_FAILURES = 8
MC_SEED = 13
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0

STATE_ACTIVE = "ACTIVE"
STATE_PARKED = "PARKED"
STATE_UNRESOLVED = "UNRESOLVED"


def sample_z(
    alpha_q: float,
    beta_q: float,
    alpha_p: float,
    beta_p: float,
    *,
    n: int = N_MC,
    rng: np.random.RandomState | None = None,
    qonly: bool = False,
) -> np.ndarray:
    rng = rng or np.random.RandomState(MC_SEED)
    q = rng.beta(float(alpha_q), float(beta_q), size=int(n))
    if qonly:
        return q
    p = rng.beta(float(alpha_p), float(beta_p), size=int(n))
    return q * p


def p_good(
    alpha_q: float,
    beta_q: float,
    alpha_p: float,
    beta_p: float,
    *,
    delta: float = DELTA,
    n: int = N_MC,
    seed: int = MC_SEED,
) -> float:
    """P(z >= delta) with a frozen MC seed so the number is a function of the posterior only."""
    rng = np.random.RandomState(int(seed))
    z = sample_z(alpha_q, beta_q, alpha_p, beta_p, n=n, rng=rng, qonly=False)
    return float(np.mean(z >= float(delta)))


def posterior_means(alpha_q: float, beta_q: float, alpha_p: float, beta_p: float) -> dict[str, float]:
    q_mean = float(alpha_q / (alpha_q + beta_q))
    p_mean = float(alpha_p / (alpha_p + beta_p))
    return {
        "mean_q": q_mean,
        "mean_p": p_mean,
        "mean_z": q_mean * p_mean,
        "alpha_q": float(alpha_q),
        "beta_q": float(beta_q),
        "alpha_p": float(alpha_p),
        "beta_p": float(beta_p),
    }


def sequential_readiness_test(
    *,
    alpha_q: float,
    beta_q: float,
    alpha_p: float,
    beta_p: float,
    n_tested_base_failures: int,
    cluster_coverage: int,
    repair_count: int,
) -> dict[str, Any]:
    pg = p_good(alpha_q, beta_q, alpha_p, beta_p)
    n_fail = int(n_tested_base_failures)
    cov = int(cluster_coverage)
    repairs = int(repair_count)
    reason = "continue"
    state = STATE_UNRESOLVED
    if (
        pg >= CONFIDENCE
        and n_fail >= MIN_BASE_FAILURES
        and cov >= MIN_CLUSTER_COVERAGE
    ):
        state = STATE_ACTIVE
        reason = f"P_good={pg:.4f}>={CONFIDENCE} n_fail={n_fail}>={MIN_BASE_FAILURES} clusters={cov}>={MIN_CLUSTER_COVERAGE}"
    elif pg <= (1.0 - CONFIDENCE) and n_fail >= MIN_BASE_FAILURES:
        state = STATE_PARKED
        reason = f"P_good={pg:.4f}<=0.10 n_fail={n_fail}>={MIN_BASE_FAILURES}"
    elif n_fail >= FUTILITY_FAILURES and repairs == 0:
        state = STATE_PARKED
        reason = f"futility n_fail={n_fail}>={FUTILITY_FAILURES} repair_count=0"
    return {
        "state": state,
        "P_good": pg,
        "reason": reason,
        "n_tested_base_failures": n_fail,
        "cluster_coverage": cov,
        "repair_count": repairs,
        **posterior_means(alpha_q, beta_q, alpha_p, beta_p),
    }


def apply_gate(cap_state: Mapping[str, Any]) -> dict[str, Any]:
    if str(cap_state.get("state") or STATE_UNRESOLVED) in {STATE_ACTIVE, STATE_PARKED}:
        return dict(cap_state)
    decision = sequential_readiness_test(
        alpha_q=float(cap_state.get("alpha_q") or PRIOR_ALPHA),
        beta_q=float(cap_state.get("beta_q") or PRIOR_BETA),
        alpha_p=float(cap_state.get("alpha_p") or PRIOR_ALPHA),
        beta_p=float(cap_state.get("beta_p") or PRIOR_BETA),
        n_tested_base_failures=int(cap_state.get("n_tested_base_failures") or 0),
        cluster_coverage=int(cap_state.get("cluster_coverage") or 0),
        repair_count=int(cap_state.get("repair_count") or 0),
    )
    out = dict(cap_state)
    out.update(decision)
    return out
