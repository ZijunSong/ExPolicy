"""Three headroom estimates. Their names and fields are not interchangeable."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from expolicy.paper_d.constants import (
    ESTIMATE_BOUND,
    ESTIMATE_CROSSFIT,
    ESTIMATE_NAIVE,
    MEASUREMENT_POPULATION,
    PANEL_ALPHA,
    RETENTION_WEIGHT,
    WEIGHT_MACRO,
)
from expolicy.paper_d.matrix import (
    build_task_level_matrix,
    cell_table,
    panel_problem,
    utilities_from_cells,
)
from expolicy.paper_d.stats import bound_gaps, interval_bounds_for_utility, population_gaps


def estimate_naive_gap(
    rows: Sequence[Mapping[str, Any]],
    *,
    retention_weight: float = RETENTION_WEIGHT,
    weight_mode: str = WEIGHT_MACRO,
    external_oracle: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Same-sample hindsight gap. Descriptive, optimistic under noise, and not an oracle bound."""
    matrix = build_task_level_matrix(rows)
    cells = cell_table(matrix, retention_weight=retention_weight)
    packed = utilities_from_cells(cells)
    problem = panel_problem(cells, packed)
    result = _base(ESTIMATE_NAIVE, weight_mode)
    result.update(
        {
            "descriptive_only": True,
            "is_credible_upper_bound": False,
            "is_oracle": False,
            "clipped": False,
            "finite_sample_guarantee": False,
            "generalization": False,
        }
    )
    if problem:
        result["valid"] = False
        result["reason"] = problem
        return result
    gaps = population_gaps(packed["utilities"], packed["envs"], weight_mode=weight_mode)
    result["valid"] = bool(gaps["valid"])
    result["reason"] = gaps["reason"]
    result["global_gap"] = gaps["global_gap"]
    result["within_gap"] = gaps["within_gap"]
    result["structural_impossibility"] = False
    below = bool(gaps.get("below_fixed_requires_audit"))
    if external_oracle is not None and gaps["valid"]:
        oracle_values = [float(external_oracle[unit]) for unit in packed["utilities"] if unit in external_oracle]
        if len(oracle_values) != len(packed["utilities"]):
            result["valid"] = False
            result["reason"] = "external oracle does not cover the panel"
            return result
        oracle_mean = sum(oracle_values) / len(oracle_values)
        if oracle_mean < float(gaps["best_fixed"]) - 1e-12:
            below = True
    result["below_fixed_requires_audit"] = below
    result["notes"] = "hindsight on the same noisy matrix; not a credible upper bound"
    return result


def crossfit_selection_gain(
    rows: Sequence[Mapping[str, Any]],
    *,
    retention_weight: float = RETENTION_WEIGHT,
    weight_mode: str = WEIGHT_MACRO,
) -> dict[str, Any]:
    """Select on one task/template half, score the other half, then swap.

    The result may be negative. It is not clipped, and it is not a population oracle.
    Splitting only repeated seeds of the same task measures run noise.
    """
    matrix = build_task_level_matrix(rows)
    result = _base(ESTIMATE_CROSSFIT, weight_mode)
    result.update(
        {
            "is_credible_upper_bound": False,
            "is_oracle": False,
            "clipped": False,
            "finite_sample_guarantee": False,
            "descriptive_only": False,
        }
    )
    folded = _frozen_folds(matrix)
    if folded is not None:
        return _crossfit_from_folds(matrix, folded, result, retention_weight, weight_mode)
    if _same_task_repeats_only(matrix):
        result["valid"] = False
        result["generalization"] = False
        result["selection_gain"] = None
        result["reason"] = "split only repeats the same task; this is run noise, not future-task generalization"
        return result
    keys = _split_keys(matrix)
    if len(keys) < 2:
        result["valid"] = False
        result["generalization"] = False
        result["reason"] = "fewer than two source tasks or templates"
        return result
    hold_a = keys[0::2]
    hold_b = keys[1::2]
    forward = _direction(matrix, set(hold_a), set(hold_b), retention_weight, weight_mode)
    swap = _direction(matrix, set(hold_b), set(hold_a), retention_weight, weight_mode)
    if not forward["valid"] or not swap["valid"]:
        result["valid"] = False
        result["generalization"] = False
        result["reason"] = forward["reason"] or swap["reason"]
        return result
    return _combine_directions(forward, swap, result, weight_mode)


def _combine_directions(
    forward: Mapping[str, Any],
    swap: Mapping[str, Any],
    result: dict[str, Any],
    weight_mode: str,
) -> dict[str, Any]:
    if not forward["valid"] or not swap["valid"]:
        result["valid"] = False
        result["generalization"] = False
        result["reason"] = forward["reason"] or swap["reason"]
        return result
    units = sorted(set(forward["gains"]) & set(swap["gains"]))
    if not units or set(units) != set(forward["gains"]) or set(units) != set(swap["gains"]):
        result["valid"] = False
        result["reason"] = "a half is missing a unit; the missing half is not filled with zero"
        return result
    combined = {unit: 0.5 * (forward["gains"][unit] + swap["gains"][unit]) for unit in units}
    envs = forward["envs"]
    gain = _weighted_mean(combined, envs, weight_mode)
    directed = list(forward["gains"].values()) + list(swap["gains"].values())
    result.update(
        {
            "valid": True,
            "generalization": True,
            "selection_gain": gain,
            "gain_vs_env_fixed": gain,
            "gain_vs_global_fixed": _weighted_mean(
                {unit: 0.5 * (forward["global_gains"][unit] + swap["global_gains"][unit]) for unit in units},
                envs,
                weight_mode,
            ),
            "min_gain": min(directed),
            "n_negative": sum(1 for value in directed if value < -1e-12),
            "clipped": False,
            "notes": "realizable gain of a finite probe split versus env-fixed; negative values are kept",
        }
    )
    return result


def bound_panel_headroom(
    rows: Sequence[Mapping[str, Any]],
    *,
    retention_weight: float = RETENTION_WEIGHT,
    weight_mode: str = WEIGHT_MACRO,
    alpha: float = PANEL_ALPHA,
    allow_population: bool = False,
    independence_declared: bool = False,
) -> dict[str, Any]:
    """Simultaneous interval for the finite panel. A wide interval is insufficient evidence.

    Population tables collapse to the point only when this call explicitly allows that
    control input. A real sample with zero observed variance does not collapse.
    """
    matrix = build_task_level_matrix(rows)
    cells = cell_table(matrix, retention_weight=retention_weight)
    packed = utilities_from_cells(cells)
    result = _base(ESTIMATE_BOUND, weight_mode)
    result.update(
        {
            "is_credible_upper_bound": False,
            "is_oracle": False,
            "clipped": False,
            "descriptive_only": False,
            "generalization": False,
            "environment_population_guarantee": False,
            "panel_only": True,
        }
    )
    problem = panel_problem(cells, packed)
    if problem:
        result["valid"] = False
        result["reason"] = problem
        return result
    usable = [cell for cell in cells if cell["unit_id"] in packed["utilities"]]
    kinds = {cell["measurement_kind"] for cell in usable}
    if kinds == {MEASUREMENT_POPULATION}:
        if not allow_population:
            result["valid"] = False
            result["reason"] = "population intervals require the known-structure control flag"
            return result
        return _population_bound(usable, packed, result, weight_mode)
    unequal_groups = any(not cell["group_sizes_equal"] for cell in usable)
    result["finite_sample_guarantee"] = bool(independence_declared) and not unequal_groups
    result["independence_declared"] = bool(independence_declared)
    n_primitive = 0
    for cell in usable:
        if cell["future_group_means"]:
            n_primitive += 1
        if cell["retention_group_means"]:
            n_primitive += 1
    if n_primitive < 1:
        result["valid"] = False
        result["reason"] = "no grouped outcomes"
        return result
    alpha_cell = float(alpha) / n_primitive
    lower: dict[str, dict[str, float]] = defaultdict(dict)
    upper: dict[str, dict[str, float]] = defaultdict(dict)
    for cell in usable:
        lo, hi = interval_bounds_for_utility(
            cell["future_group_means"],
            cell["retention_group_means"],
            alpha_cell=alpha_cell,
            retention_weight=retention_weight,
        )
        lower[str(cell["unit_id"])][str(cell["action"])] = lo
        upper[str(cell["unit_id"])][str(cell["action"])] = hi
    gaps = bound_gaps(lower, upper, packed["envs"], weight_mode=weight_mode)
    if not gaps["valid"]:
        result["valid"] = False
        result["reason"] = gaps["reason"]
        return result
    result.update(gaps)
    result["valid"] = True
    result["finite_sample_guarantee"] = bool(independence_declared) and not unequal_groups
    if unequal_groups:
        result["reason"] = "unequal source groups; interval is not a guarantee for the task-pooled mean"
    result["interval_source"] = "hoeffding_bonferroni"
    result["alpha"] = alpha
    result["notes"] = "finite-panel simultaneous bound; not an environment-population guarantee"
    return result


def _population_bound(
    cells: Sequence[Mapping[str, Any]],
    packed: Mapping[str, Any],
    result: dict[str, Any],
    weight_mode: str,
) -> dict[str, Any]:
    lower = {unit: dict(actions) for unit, actions in packed["utilities"].items()}
    upper = {unit: dict(actions) for unit, actions in packed["utilities"].items()}
    gaps = bound_gaps(lower, upper, packed["envs"], weight_mode=weight_mode)
    result.update(gaps)
    result["valid"] = bool(gaps["valid"])
    result["finite_sample_guarantee"] = False
    result["interval_source"] = "population_table"
    result["interval_equals_point"] = True
    result["notes"] = "collapsed only because the input is a declared population control table"
    result["n_cells"] = len(cells)
    return result


def _frozen_folds(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]] | None:
    present = {str(row.get("probe_fold")) for row in rows if row.get("probe_fold")}
    if not present:
        return None
    return {"A": {"A"}, "B": {"B"}} if present == {"A", "B"} else {"invalid": present}


def _crossfit_from_folds(
    rows: Sequence[Mapping[str, Any]],
    folded: Mapping[str, set[str]],
    result: dict[str, Any],
    retention_weight: float,
    weight_mode: str,
) -> dict[str, Any]:
    from expolicy.paper_d.constants import SPLIT_INSUFFICIENT

    if "invalid" in folded:
        result["valid"] = False
        result["reason"] = SPLIT_INSUFFICIENT
        return result
    for fold in ("A", "B"):
        roles = {str(row.get("probe_role")) for row in rows if str(row.get("probe_fold")) == fold}
        if "future" not in roles or "retention" not in roles:
            result["valid"] = False
            result["reason"] = SPLIT_INSUFFICIENT
            return result
    forward = _direction_fold(rows, "A", "B", retention_weight, weight_mode)
    swap = _direction_fold(rows, "B", "A", retention_weight, weight_mode)
    return _combine_directions(forward, swap, result, weight_mode)


def _direction_fold(
    rows: Sequence[Mapping[str, Any]],
    select_fold: str,
    eval_fold: str,
    retention_weight: float,
    weight_mode: str,
) -> dict[str, Any]:
    return _direction_rows(
        [dict(row) for row in rows if str(row.get("probe_fold")) == select_fold],
        [dict(row) for row in rows if str(row.get("probe_fold")) == eval_fold],
        retention_weight,
        weight_mode,
    )


def _direction(
    rows: Sequence[Mapping[str, Any]],
    select_keys: set[str],
    eval_keys: set[str],
    retention_weight: float,
    weight_mode: str,
) -> dict[str, Any]:
    return _direction_rows(
        _with_keys(rows, select_keys),
        _with_keys(rows, eval_keys),
        retention_weight,
        weight_mode,
    )


def _direction_rows(
    selected_rows: Sequence[Mapping[str, Any]],
    evaluated_rows: Sequence[Mapping[str, Any]],
    retention_weight: float,
    weight_mode: str,
) -> dict[str, Any]:
    selected = cell_table(list(selected_rows), retention_weight=retention_weight)
    evaluated = cell_table(list(evaluated_rows), retention_weight=retention_weight)
    select_pack = utilities_from_cells(selected)
    eval_pack = utilities_from_cells(evaluated)
    if panel_problem(selected, select_pack) or panel_problem(evaluated, eval_pack):
        return {"valid": False, "reason": panel_problem(selected, select_pack) or panel_problem(evaluated, eval_pack), "gains": {}}
    if set(select_pack["utilities"]) != set(eval_pack["utilities"]):
        return {"valid": False, "reason": "selection and evaluation units differ", "gains": {}}
    actions = sorted(next(iter(select_pack["utilities"].values())))
    global_choice, _tie = _argmax(
        {
            action: _weighted_mean(
                {unit: utilities[action] for unit, utilities in select_pack["utilities"].items()},
                select_pack["envs"],
                weight_mode,
            )
            for action in actions
        }
    )
    env_choice: dict[str, str] = {}
    by_env: dict[str, list[str]] = defaultdict(list)
    for unit, env in select_pack["envs"].items():
        by_env[env].append(unit)
    for env, members in by_env.items():
        env_means = {
            action: sum(select_pack["utilities"][unit][action] for unit in members) / len(members)
            for action in actions
        }
        env_choice[env], _tie = _argmax(env_means)
    gains: dict[str, float] = {}
    global_gains: dict[str, float] = {}
    for unit, utilities in select_pack["utilities"].items():
        chosen, _tie = _argmax(utilities)
        env_fixed = env_choice[select_pack["envs"][unit]]
        eval_u = eval_pack["utilities"][unit]
        gains[unit] = float(eval_u[chosen] - eval_u[env_fixed])
        global_gains[unit] = float(eval_u[chosen] - eval_u[global_choice])
    return {
        "valid": True,
        "reason": "",
        "gains": gains,
        "global_gains": global_gains,
        "envs": select_pack["envs"],
    }


def _with_keys(rows: Sequence[Mapping[str, Any]], keys: set[str]) -> list[dict[str, Any]]:
    kept = []
    for row in rows:
        if _row_key(row) in keys:
            kept.append(dict(row))
    return kept


def _split_keys(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({_row_key(row) for row in rows if row.get("probe_role")})


def _row_key(row: Mapping[str, Any]) -> str:
    if row.get("source_template"):
        return str(row["source_template"])
    return str(row.get("task_id"))


def _same_task_repeats_only(rows: Sequence[Mapping[str, Any]]) -> bool:
    future = [row for row in rows if str(row.get("probe_role")) == "future"]
    tasks = {str(row.get("task_id")) for row in future}
    return len(tasks) < 2


def _argmax(utilities: Mapping[str, float]) -> tuple[str, bool]:
    best = max(utilities.values())
    winners = sorted(action for action, value in utilities.items() if abs(value - best) <= 1e-12)
    return winners[0], len(winners) > 1


def _weighted_mean(values: Mapping[str, float], envs: Mapping[str, str], weight_mode: str) -> float:
    from expolicy.paper_d.stats import unit_weights

    units = list(values)
    weights = unit_weights([envs[unit] for unit in units], weight_mode=weight_mode)
    return float(sum(weight * float(values[unit]) for weight, unit in zip(weights, units, strict=True)))


def _base(name: str, weight_mode: str) -> dict[str, Any]:
    return {
        "estimate_name": name,
        "weight_mode": weight_mode,
        "valid": False,
        "reason": "",
        "global_gap": None,
        "within_gap": None,
        "selection_gain": None,
        "gain_vs_env_fixed": None,
        "gain_vs_global_fixed": None,
        "global_lower": None,
        "global_upper": None,
        "within_lower": None,
        "within_upper": None,
        "global_lower_raw": None,
        "within_lower_raw": None,
        "below_fixed_requires_audit": False,
        "structural_impossibility": False,
        "is_credible_upper_bound": False,
        "is_oracle": False,
        "clipped": False,
    }
