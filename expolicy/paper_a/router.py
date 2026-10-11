"""Per-action ridge utility model with a conservative fallback.

The quantity being estimated is held-out utility, not argmax accuracy.
Normalization, coefficients, and the switch threshold are fit on inner folds only.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from expolicy.paper_a.constants import A0_DISCARD, A1_EPISODIC, A2_SEMANTIC, ACTIONS
from expolicy.paper_a.features import (
    LeakageError,
    apply_simple_rule,
    assert_feature_keys,
    keys_for_variant,
    vectorize,
)
from expolicy.paper_a.groups import GroupSplit
from expolicy.paper_a.stats import grouped_bootstrap_means, mean, percentile_interval, quantile


class RidgeModel:
    def __init__(self, names: Sequence[str], mu: np.ndarray, sig: np.ndarray, coef: np.ndarray):
        self.names = list(names)
        self.mu = mu
        self.sig = sig
        self.coef = coef

    def predict(self, features: Mapping[str, Any]) -> float:
        values = vectorize(features, self.names)
        if values is None:
            raise LeakageError("prediction features are incomplete; refusing to impute zero")
        x = np.asarray(values, dtype=float)
        z = (x - self.mu) / self.sig
        return float(self.coef[0] + float(self.coef[1:] @ z))


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, dim = x.shape
    mu = x.mean(axis=0)
    sig = x.std(axis=0)
    sig = np.where(sig < 1e-8, 1.0, sig)
    z = (x - mu) / sig
    design = np.column_stack([np.ones(n), z])
    penalty = np.eye(dim + 1) * float(alpha)
    penalty[0, 0] = 0.0
    gram = design.T @ design + penalty
    try:
        coef = np.linalg.solve(gram, design.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(gram, design.T @ y, rcond=None)[0]
    return mu, sig, coef


def fit_shallow_tree(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_samples: int,
) -> dict[str, Any] | None:
    """A depth-1 tree. Refuses when the sample is below the pre-registered size."""
    if len(y) < min_samples or x.size == 0:
        return None
    best: dict[str, Any] | None = None
    best_sse = float("inf")
    for col in range(x.shape[1]):
        pivot = float(np.median(x[:, col]))
        left = y[x[:, col] <= pivot]
        right = y[x[:, col] > pivot]
        if left.size == 0 or right.size == 0:
            continue
        sse = float(((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum())
        if sse < best_sse:
            best_sse = sse
            best = {
                "feature": col,
                "pivot": pivot,
                "left_mean": float(left.mean()),
                "right_mean": float(right.mean()),
                "n": int(len(y)),
            }
    return best


def _with_roles(records: Sequence[Mapping[str, Any]], split: GroupSplit, test_fold: int) -> list[dict[str, Any]]:
    assigned = []
    for record in records:
        row = dict(record)
        row["role"] = split.role(str(row["batch_id"]), test_fold)
        assigned.append(row)
    return assigned


def _shuffle_supervision(records: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    copied = [dict(record) for record in records]
    by_env: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record["role"] != "test":
            by_env[str(record["env"])].append(index)
    for indexes in by_env.values():
        perm = rng.permutation(indexes)
        for src, dst in zip(indexes, perm, strict=True):
            copied[int(dst)]["U"] = dict(records[src]["U"])
            copied[int(dst)]["delta_shadow"] = dict(records[src].get("delta_shadow") or {})
            copied[int(dst)]["delta_retention"] = dict(records[src].get("delta_retention") or {})
    return copied


def _design(
    records: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    action: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs: list[list[float]] = []
    ys: list[float] = []
    kept: list[dict[str, Any]] = []
    for record in records:
        assert_feature_keys(record.get("features") or {})
        values = vectorize(record["features"], names)
        target = (record.get("U") or {}).get(action)
        if values is None or target is None:
            continue
        xs.append(values)
        ys.append(float(target))
        kept.append(dict(record))
    if not xs:
        return np.zeros((0, len(names))), np.zeros((0,)), []
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), kept


def _models_for(
    records: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    actions: Sequence[str],
    alpha: float,
) -> dict[str, RidgeModel]:
    fitted: dict[str, RidgeModel] = {}
    for action in actions:
        x, y, _kept = _design(records, names, action)
        if len(y) == 0:
            raise RuntimeError(f"no complete training rows for {action}")
        mu, sig, coef = _fit_ridge(x, y, alpha)
        fitted[action] = RidgeModel(names, mu, sig, coef)
    return fitted


def _bootstrap_models(
    records: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    actions: Sequence[str],
    *,
    alpha: float,
    n_boot: int,
    seed: int,
) -> list[dict[str, RidgeModel]]:
    groups = [str(record.get("source_group") or record["batch_id"]) for record in records]
    uniq = sorted(set(groups))
    if len(uniq) < 2 or n_boot < 1:
        return []
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record, group in zip(records, groups, strict=True):
        buckets[group].append(record)
    models = []
    for draw in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        sample = [dict(row) for group in chosen for row in buckets[group]]
        try:
            models.append(_models_for(sample, names, actions, alpha))
        except RuntimeError:
            continue
        _ = draw
    return models


def _env_means(records: Sequence[Mapping[str, Any]], actions: Sequence[str]) -> dict[str, str]:
    by_env: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_env[str(record["env"])].append(record)
    choice = {}
    for env, rows in by_env.items():
        scores = {}
        for action in actions:
            vals = [(row.get("U") or {}).get(action) for row in rows]
            clean = [float(value) for value in vals if value is not None]
            scores[action] = mean(clean) if clean else float("-inf")
        choice[env] = max(actions, key=lambda action: scores[action])
    return choice


def _reliable_positive(
    records: Sequence[Mapping[str, Any]],
    action: str,
    *,
    n_boot: int,
    seed: int,
) -> bool:
    vals = []
    groups = []
    for record in records:
        value = (record.get("U") or {}).get(action)
        if value is None:
            continue
        vals.append(float(value))
        groups.append(str(record.get("source_group") or record["batch_id"]))
    if len(vals) < 2:
        return False
    samples = grouped_bootstrap_means(vals, groups, n_boot=n_boot, seed=seed)
    interval = percentile_interval(samples)
    if interval is None:
        return False
    return interval[0] > 0.0


def _conservative_choice(
    features: Mapping[str, Any],
    models: Mapping[str, RidgeModel],
    boot: Sequence[Mapping[str, RidgeModel]],
    fallback: str,
    *,
    actions: Sequence[str],
    threshold: float,
    q: float,
    force_argmax: bool,
) -> tuple[str, str]:
    if force_argmax:
        preds = {action: models[action].predict(features) for action in actions}
        best = max(preds.values())
        winners = [action for action, pred in preds.items() if abs(pred - best) <= 1e-12]
        if len(winners) != 1:
            return A0_DISCARD, "TIE"
        return winners[0], winners[0]
    if not boot:
        return fallback, fallback
    scores: dict[str, float] = {}
    for action in actions:
        if action == fallback:
            continue
        samples = [
            draw[action].predict(features) - draw[fallback].predict(features)
            for draw in boot
            if action in draw and fallback in draw
        ]
        if not samples:
            continue
        scores[action] = quantile(samples, q)
    contenders = [action for action, gain in scores.items() if gain > threshold]
    if not contenders:
        return fallback, fallback
    best = max(scores[action] for action in contenders)
    winners = [action for action in contenders if abs(scores[action] - best) <= 1e-12]
    if len(winners) != 1:
        return fallback, "TIE"
    return winners[0], winners[0]


def _realized(record: Mapping[str, Any], action: str, field: str) -> float | None:
    value = (record.get(field) or {}).get(action)
    if value is None:
        return None
    return float(value)


def _frequency_policy(train: Sequence[Mapping[str, Any]], actions: Sequence[str], *, noise: float = 0.0) -> dict[str, float]:
    counts = {action: 0 for action in actions}
    used = 0
    for record in train:
        utilities = {action: (record.get("U") or {}).get(action) for action in actions}
        if any(value is None for value in utilities.values()):
            continue
        best = max(float(value) for value in utilities.values())
        winners = [action for action, value in utilities.items() if best - float(value) <= noise]
        if len(winners) != 1:
            continue
        counts[winners[0]] += 1
        used += 1
    if used == 0:
        return {A0_DISCARD: 1.0, **{action: 0.0 for action in actions if action != A0_DISCARD}}
    return {action: counts[action] / used for action in actions}


def _choose_threshold(
    val_records: Sequence[Mapping[str, Any]],
    models: Mapping[str, RidgeModel],
    boot: Sequence[Mapping[str, RidgeModel]],
    fallback: Mapping[str, str],
    *,
    actions: Sequence[str],
    grid: Sequence[float],
    q: float,
) -> float:
    best_threshold = float(grid[0]) if grid else 0.0
    best_score = float("-inf")
    # A tie prefers the larger threshold, which switches away from the fallback less often.
    for threshold in sorted(float(value) for value in grid):
        scores = []
        for record in val_records:
            action, _label = _conservative_choice(
                record["features"],
                models,
                boot,
                fallback.get(str(record["env"]), A0_DISCARD),
                actions=actions,
                threshold=threshold,
                q=q,
                force_argmax=False,
            )
            realized = _realized(record, action, "U")
            if realized is not None:
                scores.append(realized)
        score = mean(scores) if scores else float("-inf")
        if score >= best_score - 1e-12:
            best_score = score
            best_threshold = threshold
    return best_threshold


def _fit_fold_variant(
    records: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    actions: Sequence[str],
    alpha: float,
    n_boot: int,
    seed: int,
    q: float,
    threshold_grid: Sequence[float],
    default_threshold: float,
    rules: Sequence[Mapping[str, Any]],
    min_tree_samples: int,
) -> dict[str, Any]:
    names = keys_for_variant("M" if variant in {"B0", "B1", "B2", "B3", "B4", "B5"} else variant)
    if variant == "A3":
        names = keys_for_variant("M")
    train = [record for record in records if record["role"] == "train"]
    val = [record for record in records if record["role"] == "val"]
    test = [record for record in records if record["role"] == "test"]
    supervised = _shuffle_supervision(records, seed) if variant == "A3" else list(records)
    train_s = [record for record in supervised if record["role"] == "train"]
    val_s = [record for record in supervised if record["role"] == "val"]
    env_only = _env_means(train, actions)
    router_fallback = dict(env_only)
    validated = False
    if val:
        for env, action in list(router_fallback.items()):
            env_val = [record for record in val if str(record["env"]) == env]
            if not _reliable_positive(env_val, action, n_boot=n_boot, seed=seed):
                router_fallback[env] = A0_DISCARD
    dropped = []
    models: dict[str, RidgeModel] = {}
    boot: list[dict[str, RidgeModel]] = []
    threshold = float(default_threshold)
    tree = None
    if variant in {"M", "A1", "A2", "A3", "A4"}:
        complete_train = []
        for record in train_s:
            if vectorize(record["features"], names) is None or any((record.get("U") or {}).get(a) is None for a in actions):
                dropped.append(record["batch_id"])
            else:
                complete_train.append(record)
        if not complete_train:
            return {
                "variant": variant,
                "outcomes": [],
                "threshold": float(default_threshold),
                "threshold_validated": False,
                "fallback": router_fallback,
                "env_only": env_only,
                "dropped_batches": [str(record["batch_id"]) for record in train_s] or [str(record["batch_id"]) for record in records],
                "exploratory_tree": None,
                "frequency": None,
                "feature_names": list(names),
                "_models": {},
                "_boot": [],
                "skipped": "no_complete_train_rows",
            }
        models = _models_for(complete_train, names, actions, alpha)
        boot = _bootstrap_models(complete_train, names, actions, alpha=alpha, n_boot=n_boot, seed=seed)
        x, y, _kept = _design(complete_train, names, actions[0])
        tree = fit_shallow_tree(x, y, min_samples=min_tree_samples)
        if variant == "A4":
            threshold = float("-inf")
            validated = True
        elif val_s and boot:
            threshold = _choose_threshold(
                val_s,
                models,
                boot,
                router_fallback,
                actions=actions,
                grid=threshold_grid,
                q=q,
            )
            validated = True
        else:
            threshold = float(default_threshold)
            validated = False
    constant = { "B0": A0_DISCARD, "B1": A1_EPISODIC, "B2": A2_SEMANTIC }.get(variant)
    frequency = _frequency_policy(train, actions) if variant == "B4" else None
    outcomes = []
    for record in test:
        label = "MISSING"
        action = A0_DISCARD
        if constant is not None:
            action, label = constant, constant
        elif variant == "B3":
            action = env_only.get(str(record["env"]), A0_DISCARD)
            label = action
        elif variant == "B4":
            assert frequency is not None
            # Expected utility is the reported number. The sampled action is recorded separately.
            action = max(frequency, key=frequency.get)
            label = "FREQUENCY"
        elif variant == "B5":
            action = apply_simple_rule(record["features"], rules)
            label = action
        else:
            fallback = router_fallback.get(str(record["env"]), A0_DISCARD)
            action, label = _conservative_choice(
                record["features"],
                models,
                boot,
                fallback,
                actions=actions,
                threshold=threshold,
                q=q,
                force_argmax=variant == "A4",
            )
        utility = _realized(record, action, "U") if variant != "B4" else _expected_u(record, frequency or {})
        outcomes.append(
            {
                "variant": variant,
                "batch_id": record["batch_id"],
                "env": record["env"],
                "source_group": record.get("source_group"),
                "chosen_action": action,
                "label": label,
                "hard_label": None if label == "TIE" else action,
                "utility": utility,
                "delta_shadow": _realized(record, action, "delta_shadow") if variant != "B4" else None,
                "delta_retention": _realized(record, action, "delta_retention") if variant != "B4" else None,
                "fallback": router_fallback.get(str(record["env"]), A0_DISCARD),
                "threshold": threshold,
            }
        )
    return {
        "variant": variant,
        "outcomes": outcomes,
        "threshold": threshold,
        "threshold_validated": validated,
        "fallback": router_fallback,
        "env_only": env_only,
        "dropped_batches": dropped,
        "exploratory_tree": tree,
        "frequency": frequency,
        "feature_names": list(names),
        "_models": models,
        "_boot": boot,
    }


def _expected_u(record: Mapping[str, Any], frequency: Mapping[str, float]) -> float | None:
    total = 0.0
    for action, prob in frequency.items():
        value = _realized(record, action, "U")
        if value is None:
            return None
        total += float(prob) * value
    return total


def _paired(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {row["batch_id"]: row for row in right}
    diffs = []
    groups = []
    retentions = []
    for row in left:
        other = by_id.get(row["batch_id"])
        if other is None or row["utility"] is None or other["utility"] is None:
            continue
        diffs.append(float(row["utility"]) - float(other["utility"]))
        groups.append(str(row.get("source_group") or row["batch_id"]))
        if row.get("delta_retention") is not None:
            retentions.append(float(row["delta_retention"]))
    if not diffs:
        return {"n": 0, "diff": None, "ci": None, "mean_delta_retention": None}
    samples = grouped_bootstrap_means(diffs, groups, n_boot=200, seed=0)
    interval = percentile_interval(samples)
    return {
        "n": len(diffs),
        "diff": mean(diffs),
        "ci": None if interval is None else [interval[0], interval[1]],
        "mean_delta_retention": mean(retentions) if retentions else None,
    }


def fit_ridge_router(
    records: Sequence[Mapping[str, Any]],
    split: GroupSplit,
    *,
    variants: Sequence[str],
    actions: Sequence[str] = ACTIONS,
    alpha: float = 1.0,
    n_boot: int = 200,
    bootstrap_quantile: float = 0.10,
    threshold_grid: Sequence[float] = (0.0, 0.01, 0.02, 0.05),
    default_threshold: float = 0.0,
    rules: Sequence[Mapping[str, Any]] = (),
    seed: int = 20260822,
    min_tree_samples: int = 80,
    primary: Sequence[str] = ("M", "B3"),
) -> dict[str, Any]:
    """Fit on inner train, freeze the threshold on inner val, score the outer test fold.

    `records` carry pre-decision features and realized utilities. Test-fold utilities
    are read only after the action is chosen. Future-test fields are rejected.
    """
    for record in records:
        assert_feature_keys(record.get("features") or {})
        if record.get("future_success") is not None or record.get("future_test") is not None:
            raise LeakageError("future_test entered the router table")
    folds = []
    for test_fold in split.nonempty_folds():
        assigned = _with_roles(records, split, test_fold)
        fold_variants = {}
        for variant in variants:
            fold_variants[variant] = _fit_fold_variant(
                assigned,
                variant=variant,
                actions=actions,
                alpha=alpha,
                n_boot=n_boot,
                seed=seed + test_fold,
                q=bootstrap_quantile,
                threshold_grid=threshold_grid,
                default_threshold=default_threshold,
                rules=rules,
                min_tree_samples=min_tree_samples,
            )
        folds.append({"test_fold": test_fold, "variants": fold_variants, "n_outer": split.n_outer})
    pooled = _pool(folds, primary[0], primary[1] if len(primary) > 1 else "B3")
    return {
        "folds": folds,
        "primary": pooled,
        "feature_policy_gain": pooled.get("diff"),
        "n_outer": split.n_outer,
        "reduced_folds": bool(split.notes.get("reduced_folds")),
        "uses_future_test": False,
        "metric": "utility",
        "exploratory_note": "argmax accuracy is not the primary result",
    }


def _pool(folds: Sequence[Mapping[str, Any]], left: str, right: str) -> dict[str, Any]:
    left_rows = []
    right_rows = []
    validated = True
    for fold in folds:
        if left not in fold["variants"] or right not in fold["variants"]:
            continue
        left_rows.extend(fold["variants"][left]["outcomes"])
        right_rows.extend(fold["variants"][right]["outcomes"])
        if left in {"M", "A1", "A2", "A3"} and not fold["variants"][left]["threshold_validated"]:
            validated = False
    comparison = _paired(left_rows, right_rows)
    rule_rows = []
    for fold in folds:
        if "B5" in fold["variants"]:
            rule_rows.extend(fold["variants"]["B5"]["outcomes"])
    versus_rule = _paired(left_rows, rule_rows) if rule_rows else None
    return {
        "left": left,
        "right": right,
        "threshold_validated": validated and bool(folds) and all(not fold.get("n_outer", 3) < 3 for fold in folds),
        **comparison,
        "versus_simple_rule": versus_rule,
        "outcomes": left_rows,
    }
