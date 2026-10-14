"""Panel weights, population gaps, and conservative simultaneous intervals."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from expolicy.paper_d.constants import (
    RETENTION_WEIGHT,
    WEIGHT_MACRO,
    WEIGHT_UNIFORM,
    ProtocolError,
)


def evaluation_episodes(
    n_state_batch: int,
    n_actions: int,
    n_probe_tasks: int,
    n_repeats: int,
) -> int:
    """N_state_batch × N_actions × N_probe_tasks × N_repeats. Repeats are extra cost."""
    counts = {
        "n_state_batch": n_state_batch,
        "n_actions": n_actions,
        "n_probe_tasks": n_probe_tasks,
        "n_repeats": n_repeats,
    }
    for name, value in counts.items():
        if int(value) < 0:
            raise ProtocolError(f"{name} must be non-negative")
    return int(n_state_batch) * int(n_actions) * int(n_probe_tasks) * int(n_repeats)


def followup_status(*, adaptive: bool, sequential_preregistered: bool) -> dict[str, Any]:
    """Adaptive top-ups are exploratory unless a sequential test was pre-registered."""
    exploratory = bool(adaptive) and not bool(sequential_preregistered)
    return {
        "formal_stop": not exploratory,
        "exploratory_only": exploratory,
        "intermediate_intervals_are_stopping_tests": False if exploratory else bool(sequential_preregistered),
    }


def unit_weights(envs: Sequence[str], *, weight_mode: str) -> list[float]:
    """Return one weight per row. Macro weight of a row is 1 / (|E| N_e)."""
    labels = [str(env) for env in envs]
    if not labels:
        raise ProtocolError("cannot weight an empty panel")
    if weight_mode == WEIGHT_UNIFORM:
        return [1.0 / len(labels)] * len(labels)
    if weight_mode == WEIGHT_MACRO:
        counts = Counter(labels)
        n_env = len(counts)
        return [1.0 / (n_env * counts[env]) for env in labels]
    raise ProtocolError(f"unknown weight mode: {weight_mode}")


def population_gaps(
    utilities: Mapping[str, Mapping[str, float]],
    envs: Mapping[str, str],
    *,
    weight_mode: str = WEIGHT_MACRO,
) -> dict[str, Any]:
    """Finite-panel G_global and G_within. Missing actions are not filled with 0.

    G_global = E_x max_a u(x,a) - max_a E_x u(x,a)
    G_within = E_x max_a u(x,a) - E_e max_a E[u(x,a) | e]
    """
    units = list(utilities)
    if not units:
        return _invalid_gap("empty panel")
    action_sets = [set(utilities[unit]) for unit in units]
    if any(not acts for acts in action_sets) or any(acts != action_sets[0] for acts in action_sets):
        return _invalid_gap("INCOMPLETE_ACTION_SET")
    actions = sorted(action_sets[0])
    for unit in units:
        if unit not in envs:
            return _invalid_gap("missing environment")
        if any(utilities[unit][action] is None for action in actions):
            return _invalid_gap("INCOMPLETE_ACTION_SET")
    weights = unit_weights([envs[unit] for unit in units], weight_mode=weight_mode)
    weight = dict(zip(units, weights, strict=True))
    mean_max = sum(weight[unit] * max(float(utilities[unit][a]) for a in actions) for unit in units)
    action_means = {
        action: sum(weight[unit] * float(utilities[unit][action]) for unit in units) for action in actions
    }
    best_fixed = max(action_means.values())
    global_gap = float(mean_max - best_fixed)
    by_env: dict[str, list[str]] = defaultdict(list)
    for unit in units:
        by_env[str(envs[unit])].append(unit)
    second = 0.0
    for members in by_env.values():
        pi = sum(weight[unit] for unit in members)
        if pi <= 0.0:
            return _invalid_gap("zero environment weight")
        env_means = {
            action: sum(weight[unit] / pi * float(utilities[unit][action]) for unit in members)
            for action in actions
        }
        second += pi * max(env_means.values())
    within_gap = float(mean_max - second)
    return {
        "valid": True,
        "reason": "",
        "weight_mode": weight_mode,
        "actions": actions,
        "global_gap": global_gap,
        "within_gap": within_gap,
        "mean_of_row_max": float(mean_max),
        "best_fixed": float(best_fixed),
        "action_means": {action: float(value) for action, value in action_means.items()},
        "below_fixed_requires_audit": global_gap < -1e-9,
        "structural_impossibility": False,
    }


def interval_bounds_for_utility(
    future_group_means: Sequence[float],
    retention_group_means: Sequence[float],
    *,
    alpha_cell: float,
    retention_weight: float = RETENTION_WEIGHT,
) -> tuple[float, float]:
    """Hoeffding intervals on group means, propagated through the penalty."""
    future_lo, future_hi = _mean_interval(future_group_means, alpha_cell)
    ret_lo, ret_hi = _mean_interval(retention_group_means, alpha_cell)
    pen_lo, pen_hi = _min0_bounds(ret_lo, ret_hi)
    utility_lo = future_lo + float(retention_weight) * pen_lo
    utility_hi = future_hi + float(retention_weight) * pen_hi
    floor = -1.0 - float(retention_weight)
    ceiling = 1.0
    return max(floor, utility_lo), min(ceiling, utility_hi)


def bound_gaps(
    lower: Mapping[str, Mapping[str, float]],
    upper: Mapping[str, Mapping[str, float]],
    envs: Mapping[str, str],
    *,
    weight_mode: str,
) -> dict[str, Any]:
    """Plug simultaneous intervals into the finite-panel gap bounds."""
    units = list(lower)
    if set(units) != set(upper):
        return _invalid_gap("interval units differ")
    point = {
        unit: {action: 0.0 for action in lower[unit]}
        for unit in units
    }
    # Action-set check only; values are replaced below.
    if not population_gaps(point, envs, weight_mode=weight_mode)["valid"] and _sets_differ(lower):
        return _invalid_gap("INCOMPLETE_ACTION_SET")
    action_sets = [set(lower[unit]) for unit in units]
    if not action_sets or any(acts != action_sets[0] for acts in action_sets):
        return _invalid_gap("INCOMPLETE_ACTION_SET")
    actions = sorted(action_sets[0])
    weights = unit_weights([envs[unit] for unit in units], weight_mode=weight_mode)
    weight = dict(zip(units, weights, strict=True))

    def row_max(table: Mapping[str, Mapping[str, float]], unit: str) -> float:
        return max(float(table[unit][action]) for action in actions)

    def action_mean(table: Mapping[str, Mapping[str, float]], action: str) -> float:
        return sum(weight[unit] * float(table[unit][action]) for unit in units)

    mean_max_h = sum(weight[unit] * row_max(upper, unit) for unit in units)
    mean_max_l = sum(weight[unit] * row_max(lower, unit) for unit in units)
    best_mean_l = max(action_mean(lower, action) for action in actions)
    best_mean_h = max(action_mean(upper, action) for action in actions)
    global_upper = float(mean_max_h - best_mean_l)
    global_lower_raw = float(mean_max_l - best_mean_h)

    by_env: dict[str, list[str]] = defaultdict(list)
    for unit in units:
        by_env[str(envs[unit])].append(unit)

    def env_second(table: Mapping[str, Mapping[str, float]]) -> float:
        total = 0.0
        for members in by_env.values():
            pi = sum(weight[unit] for unit in members)
            means = {
                action: sum(weight[unit] / pi * float(table[unit][action]) for unit in members)
                for action in actions
            }
            total += pi * max(means.values())
        return total

    within_upper = float(mean_max_h - env_second(lower))
    within_lower_raw = float(mean_max_l - env_second(upper))
    return {
        "valid": True,
        "reason": "",
        "weight_mode": weight_mode,
        "global_upper": global_upper,
        "global_lower_raw": global_lower_raw,
        "global_lower": max(0.0, global_lower_raw),
        "within_upper": within_upper,
        "within_lower_raw": within_lower_raw,
        "within_lower": max(0.0, within_lower_raw),
        "action_mean_upper": {action: float(action_mean(upper, action)) for action in actions},
        "panel_only": True,
        "environment_population_guarantee": False,
    }


def crossed_resample_means(
    values: Mapping[tuple[str, str], float],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Resample source groups and probe-task groups. Approximate, not a stopping rule."""
    if n_boot < 1 or not values:
        raise ProtocolError("crossed resample needs values and at least one draw")
    sources = sorted({source for source, _task in values})
    tasks = sorted({task for _source, task in values})
    if len(sources) < 2 or len(tasks) < 2:
        return {
            "samples": [],
            "approximate": True,
            "formal_inference": False,
            "formal_stop": False,
            "reason": "fewer than two source groups or task groups",
        }
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        drawn_sources = rng.choice(np.array(sources, dtype=object), size=len(sources), replace=True)
        drawn_tasks = rng.choice(np.array(tasks, dtype=object), size=len(tasks), replace=True)
        picked = [float(values[(str(source), str(task))]) for source in drawn_sources for task in drawn_tasks]
        samples.append(float(np.mean(picked)))
    return {
        "samples": samples,
        "mean": float(np.mean(samples)),
        "approximate": True,
        "formal_inference": False,
        "formal_stop": False,
        "reason": "crossed resample is an approximate noise check",
    }


def finite_sample_guarantee_allowed(
    *,
    equal_groups: bool,
    independence_declared: bool,
    shared_history: bool,
    shared_probe: bool,
) -> bool:
    """Equal group size does not create independence."""
    return bool(equal_groups and independence_declared and not shared_history and not shared_probe)


def refitting_bootstrap(
    *,
    n_boot: int,
    seed: int,
    select: Callable[[Mapping[str, Any]], Any],
) -> dict[str, Any]:
    """Each draw calls select again. A previously chosen action is not the resampled object."""
    if n_boot < 1:
        raise ProtocolError("bootstrap needs at least one draw")
    rng = np.random.default_rng(seed)
    actions = []
    for draw in range(n_boot):
        sample = {"draw": draw, "seed": int(rng.integers(0, 1_000_000))}
        actions.append(select(sample))
    return {
        "n_refits": n_boot,
        "actions": actions,
        "approximate": True,
        "formal_inference": False,
        "formal_stop": False,
    }


def sequence_cluster(state_id: str, sequence_id: str | None) -> str:
    """States from one history sequence share a cluster. The label is not a new x."""
    if sequence_id:
        return f"sequence:{sequence_id}"
    return f"state:{state_id}"


def _sets_differ(table: Mapping[str, Mapping[str, float]]) -> bool:
    sets = [set(actions) for actions in table.values()]
    return not sets or any(item != sets[0] for item in sets)


def _invalid_gap(reason: str) -> dict[str, Any]:
    return {
        "valid": False,
        "reason": reason,
        "global_gap": None,
        "within_gap": None,
        "below_fixed_requires_audit": False,
        "structural_impossibility": False,
    }


def _mean_interval(values: Sequence[float], alpha: float) -> tuple[float, float]:
    sample = [float(value) for value in values]
    if not sample:
        raise ProtocolError("cannot bound an empty group list")
    center = float(sum(sample) / len(sample))
    radius = _hoeffding_radius(len(sample), alpha, span=2.0)
    return center - radius, center + radius


def _hoeffding_radius(n: int, alpha: float, *, span: float) -> float:
    if n <= 0 or not 0.0 < alpha < 1.0:
        raise ProtocolError("Hoeffding radius needs n > 0 and alpha in (0, 1)")
    return float(span) * math.sqrt(math.log(2.0 / float(alpha)) / (2.0 * n))


def _min0_bounds(lower: float, upper: float) -> tuple[float, float]:
    """Range of min(0, x) for x in [lower, upper]."""
    if upper <= 0.0:
        return lower, upper
    if lower >= 0.0:
        return 0.0, 0.0
    return lower, 0.0
