"""Known-structure controls. Expectations are literals, not a second copy of the estimator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from expolicy.paper_d.cards import freeze_diagnostic_card, judge_configuration
from expolicy.paper_d.constants import (
    ACTIONABLE_HEADROOM,
    CONTENT_REAL,
    CONTROL_DELAYED,
    CONTROL_ENV_ONLY,
    CONTROL_EXACT_TIE,
    CONTROL_GLOBAL_WINNER,
    CONTROL_HIDDEN,
    CONTROL_NOISY_TIE,
    CONTROL_OBSERVABLE,
    ENV_FIXED_SUFFICIENT,
    HETEROGENEOUS_BUT_UNPREDICTED,
    INCONCLUSIVE,
    KNOWN_CONTROLS,
    LONG_TERM_UNCOVERED,
    MEASUREMENT_POPULATION,
    MEASUREMENT_SAMPLE,
    PROBE_FUTURE,
    PROBE_RETENTION,
    PROTOCOL_VERSION,
    STATUS_OK,
)
from expolicy.paper_d.headroom import (
    bound_panel_headroom,
    crossfit_selection_gain,
    estimate_naive_gap,
)
from expolicy.paper_d.policy import fit_simple_observable_policy
from expolicy.paper_d.schema import empty_log

# Independent of estimate_naive_gap. These are the population answers for the controls.
EXPECTED: dict[str, dict[str, float | None]] = {
    CONTROL_EXACT_TIE: {"global_gap": 0.0, "within_gap": 0.0},
    CONTROL_NOISY_TIE: {"global_gap": 0.0, "within_gap": 0.0},
    CONTROL_GLOBAL_WINNER: {"global_gap": 0.0, "within_gap": 0.0},
    CONTROL_ENV_ONLY: {"global_gap": 0.5, "within_gap": 0.0},
    CONTROL_OBSERVABLE: {"global_gap": 0.5, "within_gap": 0.5},
    CONTROL_HIDDEN: {"global_gap": 0.5, "within_gap": 0.5},
    CONTROL_DELAYED: {"global_gap": 0.0, "within_gap": 0.0, "two_step_optimum": 0.4},
}

STRUCTURE = {
    CONTROL_EXACT_TIE: "所有动作真实效用相等，无噪声",
    CONTROL_NOISY_TIE: "相同真实效用，加可调观测噪声",
    CONTROL_GLOBAL_WINNER: "同一动作处处更好",
    CONTROL_ENV_ONLY: "环境决定最佳动作，环境内不变",
    CONTROL_OBSERVABLE: "可见状态特征决定最佳动作",
    CONTROL_HIDDEN: "最佳动作取决于不可见随机变量",
    CONTROL_DELAYED: "两步更新互补、第一步无即时收益",
}


def two_step_optimum() -> float:
    """Exact two-step value. The single-step diagnostic is not given this number."""
    rewards = {("A", "A"): 0.0, ("A", "B"): 0.4, ("B", "A"): 0.4, ("B", "B"): 0.0}
    return max(rewards.values())


def run_known_structure_controls(*, prefix: str = "diag") -> dict[str, Any]:
    """Run every control. A pass checks the diagnosis against the literal expectation."""
    reports = [evaluate_control(build_control(name, prefix=prefix)) for name in KNOWN_CONTROLS]
    return {
        "reports": reports,
        "all_passed": all(report["passed"] for report in reports),
        "environment_class": "synthetic_control",
        "counts_as_real_environment_gain": False,
    }


def build_control(name: str, *, prefix: str = "diag") -> dict[str, Any]:
    if name == CONTROL_NOISY_TIE:
        return _noisy_control(prefix)
    if name == CONTROL_DELAYED:
        panel = _panel(
            name,
            prefix,
            _single_env_units(prefix, {"A": 0.0, "B": 0.0}, features={"capability_shadow_hist": 0.0}),
            ["db"],
            population=True,
        )
        panel["mdp_two_step"] = two_step_optimum()
        return panel
    if name == CONTROL_EXACT_TIE:
        return _panel(
            name,
            prefix,
            _single_env_units(prefix, {"A": 0.0, "B": 0.0}, features={"capability_shadow_hist": 0.0}),
            ["db"],
            population=True,
        )
    if name == CONTROL_GLOBAL_WINNER:
        return _panel(
            name,
            prefix,
            _single_env_units(prefix, {"A": 0.3, "B": 0.0}, features={"capability_shadow_hist": 0.2}),
            ["db"],
            population=True,
        )
    if name == CONTROL_ENV_ONLY:
        return _panel(name, prefix, _env_only_units(prefix), ["db", "os"], population=True)
    if name == CONTROL_OBSERVABLE:
        return _panel(name, prefix, _crossover_units(prefix, observable=True), ["db"], population=True)
    if name == CONTROL_HIDDEN:
        return _panel(name, prefix, _crossover_units(prefix, observable=False), ["db"], population=True)
    raise KeyError(name)


def evaluate_control(data: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(data["rows"])
    naive = estimate_naive_gap(rows)
    cross = crossfit_selection_gain(rows)
    bound = bound_panel_headroom(rows, allow_population=bool(data["population"]))
    policy = fit_simple_observable_policy(list(data["records"]), envs=list(data["envs"]))
    judgment = judge_configuration(
        measurement_valid=True,
        within_gap=naive.get("within_gap") if data["population"] else None,
        within_lower_raw=bound.get("within_lower_raw"),
        within_upper=bound.get("within_upper"),
        feature_gain=policy.get("feature_gain"),
        feature_evaluated=bool(policy.get("valid")),
        crossfit_gain=cross.get("selection_gain"),
        population_table=bool(data["population"]),
        action_mean_upper=bound.get("action_mean_upper"),
        updates_rejected=False,
        content_real=None,
        content_control=None,
    )
    card = freeze_diagnostic_card(
        judgment,
        configuration_id=str(data["configuration_id"]),
        batch_ids=sorted({str(row["batch_id"]) for row in rows}),
        task_ids=sorted({str(row["task_id"]) for row in rows}),
        policy_trained=False,
        estimates={
            "naive_global_gap": naive.get("global_gap"),
            "naive_within_gap": naive.get("within_gap"),
            "crossfit_selection_gain": cross.get("selection_gain"),
            "within_upper": bound.get("within_upper"),
            "within_lower": bound.get("within_lower"),
            "feature_gain": policy.get("feature_gain"),
        },
    )
    observed = {
        "naive_global": naive.get("global_gap"),
        "naive_within": naive.get("within_gap"),
        "crossfit": cross.get("selection_gain"),
        "within_upper": bound.get("within_upper"),
        "feature_gain": policy.get("feature_gain"),
        "primary": judgment["primary_status"],
        "limitation": judgment["limitation"],
        "long_term_covered": judgment["long_term_covered"],
        "naive_k2": data.get("naive_k2"),
        "naive_k6": data.get("naive_k6"),
        "crossfit_k6": data.get("crossfit_k6"),
        "mdp_two_step": data.get("mdp_two_step"),
        "reported_gap_is_two_step": naive.get("global_gap") == EXPECTED[CONTROL_DELAYED]["two_step_optimum"],
    }
    reasons = _check(str(data["name"]), observed)
    return {
        "name": data["name"],
        "known_structure": STRUCTURE[str(data["name"])],
        "expected": EXPECTED[str(data["name"])],
        "observed": observed,
        "passed": not reasons,
        "reasons": reasons,
        "naive": naive,
        "crossfit": cross,
        "bound": bound,
        "policy": policy,
        "judgment": judgment,
        "card": card,
        "rows": rows,
        "configuration_id": data["configuration_id"],
        "analytical_population_global": EXPECTED[str(data["name"])]["global_gap"],
        "analytical_population_within": EXPECTED[str(data["name"])]["within_gap"],
    }


def _check(name: str, observed: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    expected = EXPECTED[name]
    if name == CONTROL_NOISY_TIE:
        return _check_noisy(observed)
    if not _close(observed["naive_global"], expected["global_gap"]):
        reasons.append(f"naive global gap {observed['naive_global']} != {expected['global_gap']}")
    if not _close(observed["naive_within"], expected["within_gap"]):
        reasons.append(f"naive within gap {observed['naive_within']} != {expected['within_gap']}")
    if name in {CONTROL_EXACT_TIE, CONTROL_GLOBAL_WINNER, CONTROL_ENV_ONLY, CONTROL_DELAYED}:
        if observed["primary"] != ENV_FIXED_SUFFICIENT:
            reasons.append(f"primary status {observed['primary']}")
        if observed["feature_gain"] is None or float(observed["feature_gain"]) > 0.02:
            reasons.append("feature policy beat env-fixed on a control where it should not")
    if name == CONTROL_OBSERVABLE:
        if observed["primary"] != ACTIONABLE_HEADROOM:
            reasons.append(f"primary status {observed['primary']}")
        if observed["feature_gain"] is None or float(observed["feature_gain"]) <= 0.02:
            reasons.append("observable feature policy did not clear delta")
    if name == CONTROL_HIDDEN:
        if observed["primary"] != HETEROGENEOUS_BUT_UNPREDICTED:
            reasons.append(f"primary status {observed['primary']}")
        if observed["feature_gain"] is None or float(observed["feature_gain"]) > 0.02:
            reasons.append("hidden crossover was realized by the feature policy")
        if observed["crossfit"] is None or float(observed["crossfit"]) <= 0.02:
            reasons.append("probe split did not see the hidden crossover")
    if name == CONTROL_DELAYED:
        if observed["long_term_covered"] is not False or observed["limitation"] != LONG_TERM_UNCOVERED:
            reasons.append("single-step protocol did not mark the long-term gap as uncovered")
        if observed["mdp_two_step"] != 0.4:
            reasons.append("two-step MDP value drifted")
        if observed["reported_gap_is_two_step"]:
            reasons.append("single-step gap was replaced by the two-step value")
    if name == CONTROL_ENV_ONLY and not _close(observed["crossfit"], 0.0):
        reasons.append("env-only crossfit gain versus env-fixed was not zero")
    return reasons


def _check_noisy(observed: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    k2 = observed.get("naive_k2")
    k6 = observed.get("naive_k6")
    cross = observed.get("crossfit_k6")
    if k2 is None or k6 is None or not (float(k6) > float(k2) > 0.0):
        reasons.append("naive gap did not increase with the action count")
    if cross is None or float(cross) >= 0.02:
        reasons.append("independent evaluation showed a stable positive gain")
    if observed["primary"] == ACTIONABLE_HEADROOM:
        reasons.append("noisy tie was marked actionable")
    if observed["primary"] != INCONCLUSIVE:
        reasons.append(f"noisy tie primary status {observed['primary']}")
    if observed["naive_global"] == EXPECTED[CONTROL_NOISY_TIE]["global_gap"]:
        reasons.append("sample hindsight gap collapsed to the population tie")
    return reasons


def _noisy_control(prefix: str) -> dict[str, Any]:
    wide = _noise_rows(prefix, n_actions=6)
    narrow = _noise_rows(prefix, n_actions=2)
    wide_naive = estimate_naive_gap(wide["rows"])
    narrow_naive = estimate_naive_gap(narrow["rows"])
    wide_cross = crossfit_selection_gain(wide["rows"])
    panel = _panel(CONTROL_NOISY_TIE, prefix, wide["units"], ["db"], population=False)
    panel["rows"] = wide["rows"]
    panel["records"] = wide["records"]
    panel["naive_k2"] = narrow_naive.get("global_gap")
    panel["naive_k6"] = wide_naive.get("global_gap")
    panel["crossfit_k6"] = wide_cross.get("selection_gain")
    return panel


def _noise_rows(prefix: str, *, n_actions: int) -> dict[str, Any]:
    actions = [f"A{index}" for index in range(n_actions)]
    n_units = 6
    units = []
    for index in range(n_units):
        left: dict[str, float] = {}
        right: dict[str, float] = {}
        for action_index, action in enumerate(actions):
            spike = 0.05 * (action_index + 1)
            average = spike if index == action_index else -spike / (n_units - 1)
            left_value = 1.0 if index == action_index else -1.0
            left[action] = left_value
            right[action] = 2.0 * average - left_value
        batch = f"{prefix}-n{n_actions}-b{index}"
        units.append(
            {
                "env": "db",
                "batch_id": batch,
                "state_id": "s0",
                "source_group": batch,
                "halves": {"t0": left, "t1": right},
                "features": {"capability_shadow_hist": 0.0, "env__db": 1.0},
            }
        )
    return {
        "units": units,
        "rows": _rows(CONTROL_NOISY_TIE, units, MEASUREMENT_SAMPLE),
        "records": _records(CONTROL_NOISY_TIE, units),
    }


def _panel(
    name: str,
    prefix: str,
    units: Sequence[Mapping[str, Any]],
    envs: Sequence[str],
    *,
    population: bool,
) -> dict[str, Any]:
    kind = MEASUREMENT_POPULATION if population else MEASUREMENT_SAMPLE
    rows = _rows(name, units, kind)
    return {
        "name": name,
        "configuration_id": f"control:{name}",
        "population": population,
        "envs": list(envs),
        "units": list(units),
        "rows": rows,
        "records": _records(name, units),
        "mdp_two_step": None,
        "naive_k2": None,
        "naive_k6": None,
        "crossfit_k6": None,
    }


def _single_env_units(
    prefix: str,
    utilities: Mapping[str, float],
    *,
    features: Mapping[str, float],
) -> list[dict[str, Any]]:
    units = []
    for index in range(4):
        batch = f"{prefix}-b{index}"
        units.append(
            {
                "env": "db",
                "batch_id": batch,
                "state_id": "s0",
                "source_group": batch,
                "halves": {"t0": dict(utilities), "t1": dict(utilities)},
                "features": {**features, "env__db": 1.0},
            }
        )
    return units


def _env_only_units(prefix: str) -> list[dict[str, Any]]:
    units = []
    for env, utilities in (("db", {"A": 1.0, "B": 0.0}), ("os", {"A": 0.0, "B": 1.0})):
        for index in range(2):
            batch = f"{prefix}-{env}-b{index}"
            units.append(
                {
                    "env": env,
                    "batch_id": batch,
                    "state_id": "s0",
                    "source_group": batch,
                    "halves": {"t0": dict(utilities), "t1": dict(utilities)},
                    "features": {"env__db": 1.0 if env == "db" else 0.0, "env__os": 1.0 if env == "os" else 0.0},
                }
            )
    return units


def _crossover_units(prefix: str, *, observable: bool) -> list[dict[str, Any]]:
    units = []
    for index in range(4):
        batch = f"{prefix}-b{index}"
        for state_id, visible, utilities in (
            ("s0", 0.0, {"A": 1.0, "B": 0.0}),
            ("s1", 1.0, {"A": 0.0, "B": 1.0}),
        ):
            features: dict[str, float] = {"env__db": 1.0}
            if observable:
                features["visible_state"] = visible
            else:
                features["capability_shadow_hist"] = 0.5
            units.append(
                {
                    "env": "db",
                    "batch_id": batch,
                    "state_id": state_id,
                    "source_group": batch,
                    "halves": {"t0": dict(utilities), "t1": dict(utilities)},
                    "features": features,
                }
            )
    return units


def _rows(name: str, units: Sequence[Mapping[str, Any]], kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in units:
        uid = "|".join([f"control:{name}", str(unit["state_id"]), str(unit["batch_id"]), str(unit["env"]), CONTENT_REAL])
        for template, utilities in unit["halves"].items():
            for action, utility in utilities.items():
                before, after = _outcomes(float(utility))
                shared = {
                    "configuration_id": f"control:{name}",
                    "protocol_version": PROTOCOL_VERSION,
                    "model_hash": "synthetic-control",
                    "env_hash": unit["env"],
                    "env": unit["env"],
                    "state_distribution_id": "control",
                    "state_id": unit["state_id"],
                    "source_group": unit["source_group"],
                    "batch_id": unit["batch_id"],
                    "action": action,
                    "accepted": True,
                    "changed": True,
                    "repeat_id": "0",
                    "cost": 1.0,
                    "infra_status": STATUS_OK,
                    "measurement_kind": kind,
                    "content_mode": CONTENT_REAL,
                    "budget": "pilot",
                    "control_name": name,
                    "source_template": template,
                    "prompt_hash": "control-prompt",
                    "legacy_u": None,
                }
                rows.append(
                    empty_log(
                        **shared,
                        task_id=f"{uid}-{template}-future",
                        probe_role=PROBE_FUTURE,
                        outcome_before=before,
                        outcome_after=after,
                    )
                )
                rows.append(
                    empty_log(
                        **shared,
                        task_id=f"{uid}-{template}-retention",
                        probe_role=PROBE_RETENTION,
                        retention_before=1.0,
                        retention_after=1.0,
                    )
                )
    return rows


def _records(name: str, units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for unit in units:
        uid = "|".join([f"control:{name}", str(unit["state_id"]), str(unit["batch_id"]), str(unit["env"]), CONTENT_REAL])
        templates = list(unit["halves"])
        records.append(
            {
                "unit_id": uid,
                "env": unit["env"],
                "source_group": unit["source_group"],
                "features": dict(unit["features"]),
                "selection_utilities": dict(unit["halves"][templates[0]]),
                "evaluation_utilities": dict(unit["halves"][templates[1]]),
            }
        )
    return records


def _outcomes(utility: float) -> tuple[float, float]:
    if utility >= 0.0:
        return 0.0, utility
    return -utility, 0.0


def _close(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) <= 1e-9
