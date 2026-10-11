"""Independent half-split action selection. Hindsight gap is reported and not used as the gate."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from expolicy.paper_a.constants import (
    ACTIONS,
    A0_DISCARD,
    SPACE_GO,
    SPACE_NO_GO,
    DECISION_INCONCLUSIVE,
    STATUS_OK,
)
from expolicy.paper_a.matrix import summarize_cell
from expolicy.paper_a.stats import grouped_bootstrap_means, percentile_interval


def _halves(task_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(set(task_ids))
    return ordered[0::2], ordered[1::2]


def _cells(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("legacy"):
            continue
        grouped[(str(row.get("batch_id")), str(row.get("action")))].append(row)
    return grouped


def _utility(items: Sequence[Mapping[str, Any]], task_ids: set[str] | None = None) -> dict[str, Any]:
    chosen = items
    if task_ids is not None:
        chosen = [item for item in items if str(item.get("probe_task_id")) in task_ids]
    return summarize_cell(chosen)


def _unique_best(utilities: Mapping[str, float | None], *, noise: float) -> tuple[str | None, str]:
    scored = [(action, value) for action, value in utilities.items() if value is not None]
    if len(scored) != len(utilities):
        return None, "MISSING"
    best_value = max(value for _, value in scored)
    winners = [action for action, value in scored if best_value - value <= noise]
    if len(winners) != 1:
        return None, "TIE"
    return winners[0], winners[0]


def _noise(n_shadow: int, equivalence: float) -> float:
    if n_shadow <= 0:
        return equivalence
    return max(equivalence, 1.0 / n_shadow)


def crossfit_action_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    actions: Sequence[str] = ACTIONS,
    equivalence: float = 0.01,
    delta_floor: float = 0.02,
    resolution_tasks: float = 2.0,
    min_positive_batches: int = 4,
    min_non_discard_actions: int = 2,
    min_envs_with_crossover: int = 2,
    n_boot: int = 200,
    seed: int = 20260822,
) -> dict[str, Any]:
    """Select on one probe half and score the other half, then swap.

    The environment-level action is chosen on the selection half only. Ties stay
    ties: there is no forced hard label, and a tie deploys DISCARD.
    """
    cells = _cells(rows)
    batch_ids = sorted({batch for batch, _action in cells})
    detail_rows: list[dict[str, Any]] = []
    batch_full: dict[str, dict[str, float | None]] = {}
    batch_meta: dict[str, dict[str, str]] = {}
    env_tasks: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"shadow": set(), "retention": set()})

    for batch_id in batch_ids:
        sample = next(iter(cells[(batch_id, actions[0])])) if (batch_id, actions[0]) in cells else None
        if sample is None:
            any_cell = cells[(batch_id, next(iter({a for b, a in cells if b == batch_id})))]
            sample = any_cell[0]
        env = str(sample.get("env") or "")
        batch_meta[batch_id] = {
            "env": env,
            "source_group": str(sample.get("source_group") or batch_id),
        }
        utilities: dict[str, float | None] = {}
        valid = True
        for action in actions:
            items = cells.get((batch_id, action)) or []
            summary = _utility(items)
            utilities[action] = summary["utility"] if summary["infra_status"] == STATUS_OK else None
            if utilities[action] is None:
                valid = False
            for item in items:
                split = str(item.get("eval_split") or "")
                if split == "shadow_reward_probe":
                    env_tasks[env]["shadow"].add(str(item.get("probe_task_id")))
                elif split == "retention_probe":
                    env_tasks[env]["retention"].add(str(item.get("probe_task_id")))
        if valid:
            batch_full[batch_id] = utilities

    directions = ("forward", "swap")
    per_batch_gains: list[dict[str, Any]] = []
    env_direction_gains: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    eval_winners: dict[str, list[str | None]] = defaultdict(list)
    full_winners: dict[str, list[str | None]] = defaultdict(list)

    for env, tasks in env_tasks.items():
        shadow_a, shadow_b = _halves(tasks["shadow"])
        retention_a, retention_b = _halves(tasks["retention"])
        slices = {
            "forward": (set(shadow_a) | set(retention_a), set(shadow_b) | set(retention_b), shadow_b),
            "swap": (set(shadow_b) | set(retention_b), set(shadow_a) | set(retention_a), shadow_a),
        }
        env_batches = [batch_id for batch_id, meta in batch_meta.items() if meta["env"] == env and batch_id in batch_full]
        for direction in directions:
            select_ids, eval_ids, eval_shadow = slices[direction]
            noise = _noise(len(eval_shadow), equivalence)
            select_noise = _noise(len(shadow_a if direction == "forward" else shadow_b), equivalence)
            half_u: dict[str, dict[str, float | None]] = {}
            eval_u: dict[str, dict[str, float | None]] = {}
            for batch_id in env_batches:
                half_u[batch_id] = {}
                eval_u[batch_id] = {}
                for action in actions:
                    items = cells[(batch_id, action)]
                    selected = _utility(items, select_ids)
                    evaluated = _utility(items, eval_ids)
                    half_u[batch_id][action] = selected["utility"] if selected["infra_status"] == STATUS_OK else None
                    eval_u[batch_id][action] = evaluated["utility"] if evaluated["infra_status"] == STATUS_OK else None
            env_means = {}
            for action in actions:
                vals = [half_u[batch_id][action] for batch_id in env_batches if half_u[batch_id][action] is not None]
                env_means[action] = float(np.mean(vals)) if vals and len(vals) == len(env_batches) else None
            env_action, env_label = _unique_best(env_means, noise=select_noise)
            if env_label == "TIE" or env_action is None:
                env_deploy = A0_DISCARD
                env_label = "TIE"
            else:
                env_deploy = env_action
            for batch_id in env_batches:
                chosen, label = _unique_best(half_u[batch_id], noise=select_noise)
                deploy = A0_DISCARD if label == "TIE" or chosen is None else chosen
                u_selected = eval_u[batch_id].get(deploy)
                u_fixed = eval_u[batch_id].get(env_deploy)
                gain = None if u_selected is None or u_fixed is None else float(u_selected) - float(u_fixed)
                if gain is not None:
                    env_direction_gains[env][direction].append(gain)
                eval_best, eval_label = _unique_best(eval_u[batch_id], noise=noise)
                if direction == "forward":
                    eval_winners[env].append(eval_best if eval_label not in {"TIE", "MISSING"} else None)
                detail_rows.append(
                    {
                        "env": env,
                        "batch_id": batch_id,
                        "source_group": batch_meta[batch_id]["source_group"],
                        "direction": direction,
                        "selected_action": deploy,
                        "selected_label": label,
                        "hard_label": None if label == "TIE" else chosen,
                        "env_fixed_action": env_deploy,
                        "env_fixed_label": env_label,
                        "u_selected": u_selected,
                        "u_env_fixed": u_fixed,
                        "gain": gain,
                        "n_eval_shadow": len(eval_shadow),
                        "descriptive_only": False,
                    }
                )
                per_batch_gains.append(detail_rows[-1])

    for batch_id, utilities in batch_full.items():
        env = batch_meta[batch_id]["env"]
        n_shadow = len(env_tasks[env]["shadow"])
        winner, label = _unique_best(utilities, noise=_noise(n_shadow, equivalence))
        full_winners[env].append(winner if label not in {"TIE", "MISSING"} else None)

    hindsight = _hindsight(batch_full, actions)
    by_env_gain = {}
    thresholds = {}
    for env, directions_map in env_direction_gains.items():
        both = []
        for direction in directions:
            both.extend(directions_map.get(direction) or [])
        by_env_gain[env] = float(np.mean(both)) if both else None
        n_eval = 0
        env_details = [row for row in detail_rows if row["env"] == env and row["direction"] == "forward"]
        if env_details:
            n_eval = int(env_details[0]["n_eval_shadow"])
        thresholds[env] = max(delta_floor, resolution_tasks / n_eval) if n_eval else None

    macro_gain = _macro(list(by_env_gain.values()))
    macro_threshold = _macro(list(thresholds.values()))
    direction_means = {}
    for direction in directions:
        vals = [gain for env in env_direction_gains for gain in env_direction_gains[env].get(direction) or []]
        direction_means[direction] = float(np.mean(vals)) if vals else None
    stable = (
        direction_means["forward"] is not None
        and direction_means["swap"] is not None
        and direction_means["forward"] > 0
        and direction_means["swap"] > 0
    )

    positive = {action: 0 for action in actions if action != A0_DISCARD}
    for utilities in batch_full.values():
        for action in positive:
            value = utilities.get(action)
            if value is not None and value > 0:
                positive[action] += 1
    positive_actions = [action for action, count in positive.items() if count >= min_positive_batches]

    def _crossover(winners: Mapping[str, Sequence[str | None]]) -> list[str]:
        found = []
        for env, labels in winners.items():
            distinct = {label for label in labels if label}
            if len(distinct) >= 2:
                found.append(env)
        return found

    eval_cross = _crossover(eval_winners)
    full_cross = _crossover(full_winners)
    disappeared = sorted(set(full_cross) - set(eval_cross))

    collapsed_values = []
    collapsed_groups = []
    by_batch: dict[str, list[float]] = defaultdict(list)
    group_of = {}
    for row in detail_rows:
        if row.get("gain") is None:
            continue
        by_batch[row["batch_id"]].append(float(row["gain"]))
        group_of[row["batch_id"]] = row["source_group"]
    for batch_id, vals in by_batch.items():
        collapsed_values.append(float(np.mean(vals)))
        collapsed_groups.append(group_of[batch_id])
    boot = grouped_bootstrap_means(collapsed_values, collapsed_groups, n_boot=n_boot, seed=seed)
    interval = percentile_interval(boot)
    ci_width = None if interval is None else interval[1] - interval[0]

    status, reasons = _space_status(
        n_valid=len(batch_full),
        positive_actions=positive_actions,
        min_non_discard_actions=min_non_discard_actions,
        eval_cross=eval_cross,
        disappeared=disappeared,
        min_envs=min_envs_with_crossover,
        min_positive_batches=min_positive_batches,
        macro_gain=macro_gain,
        macro_threshold=macro_threshold,
        stable=stable,
        interval=interval,
        ci_width=ci_width,
    )
    return {
        "status": status,
        "reasons": reasons,
        "naive_hindsight_gap": hindsight["gap"],
        "naive_hindsight_note": "descriptive only; same-matrix max is optimistic and is not the gate",
        "hindsight_oracle_mean": hindsight["oracle_mean"],
        "hindsight_best_fixed_mean": hindsight["best_fixed_mean"],
        "hindsight_best_fixed_action": hindsight["best_fixed_action"],
        "crossfit_selector_gain": macro_gain,
        "feature_policy_gain": None,
        "delta_space": macro_threshold,
        "delta_space_by_env": thresholds,
        "gain_by_env": by_env_gain,
        "direction_means": direction_means,
        "direction_stable": stable,
        "gain_ci": None if interval is None else [interval[0], interval[1]],
        "positive_batches": positive,
        "positive_actions": positive_actions,
        "eval_crossover_envs": eval_cross,
        "full_crossover_envs": full_cross,
        "crossover_disappeared": disappeared,
        "n_valid_batches": len(batch_full),
        "rows": detail_rows,
        "uses_future_test": False,
    }


def _macro(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def _hindsight(batch_full: Mapping[str, Mapping[str, float | None]], actions: Sequence[str]) -> dict[str, Any]:
    if not batch_full:
        return {"gap": None, "oracle_mean": None, "best_fixed_mean": None, "best_fixed_action": None}
    oracle_vals = []
    means = {action: [] for action in actions}
    for utilities in batch_full.values():
        vals = [utilities[action] for action in actions if utilities[action] is not None]
        if len(vals) != len(actions):
            continue
        oracle_vals.append(max(float(value) for value in vals))
        for action in actions:
            means[action].append(float(utilities[action]))  # type: ignore[arg-type]
    if not oracle_vals:
        return {"gap": None, "oracle_mean": None, "best_fixed_mean": None, "best_fixed_action": None}
    action_means = {action: float(np.mean(vals)) for action, vals in means.items() if vals}
    best_action = max(action_means, key=action_means.get)
    oracle_mean = float(np.mean(oracle_vals))
    best_fixed = action_means[best_action]
    return {
        "gap": oracle_mean - best_fixed,
        "oracle_mean": oracle_mean,
        "best_fixed_mean": best_fixed,
        "best_fixed_action": best_action,
    }


def _space_status(
    *,
    n_valid: int,
    positive_actions: Sequence[str],
    min_non_discard_actions: int,
    eval_cross: Sequence[str],
    disappeared: Sequence[str],
    min_envs: int,
    min_positive_batches: int,
    macro_gain: float | None,
    macro_threshold: float | None,
    stable: bool,
    interval: tuple[float, float] | None,
    ci_width: float | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if n_valid < min_positive_batches or macro_gain is None or macro_threshold is None:
        reasons.append("too few valid batches, or the eval half has no utility")
        return DECISION_INCONCLUSIVE, reasons
    if interval is None or (ci_width is not None and ci_width > 2 * macro_threshold and interval[0] <= macro_threshold <= interval[1]):
        reasons.append("the grouped interval on the selector gain is too wide to apply the practical threshold")
        return DECISION_INCONCLUSIVE, reasons
    if disappeared and len(eval_cross) < min_envs:
        reasons.append("action crossover on the full matrix disappears on the held-out half")
        return SPACE_NO_GO, reasons
    go = (
        len(positive_actions) >= min_non_discard_actions
        and len(eval_cross) >= min_envs
        and stable
        and macro_gain > macro_threshold
        and interval[0] > 0
    )
    if go:
        reasons.append("held-out selector gain clears the practical threshold in both split directions")
        return SPACE_GO, reasons
    if macro_gain <= macro_threshold and interval[1] < macro_threshold:
        reasons.append("environment-fixed selection already reaches the crossfit selector on the held-out half")
        return SPACE_NO_GO, reasons
    reasons.append("the space criteria are not all met, and the interval does not settle the decision")
    return DECISION_INCONCLUSIVE, reasons
