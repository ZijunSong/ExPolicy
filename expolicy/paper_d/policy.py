"""Pre-decision ridge and shallow tree. Nested splits never score the fold they trained on."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from expolicy.paper_d.constants import (
    DELTA,
    ESTIMATE_POLICY,
    EXPERIENCE_FEATURES,
    FORBIDDEN_FEATURES,
    MIN_POLICY_GROUPS,
    STATE_FEATURES,
    WEIGHT_MACRO,
    ProtocolError,
)
from expolicy.paper_d.stats import unit_weights


class LeakageError(ProtocolError):
    """A feature is not on the pre-decision whitelist."""


def allowed_feature_names(envs: Sequence[str]) -> set[str]:
    return {f"env__{env}" for env in envs} | set(EXPERIENCE_FEATURES) | set(STATE_FEATURES)


def assert_predecision_features(features: Mapping[str, Any], envs: Sequence[str]) -> None:
    names = set(features)
    allowed = allowed_feature_names(envs)
    forbidden = names & FORBIDDEN_FEATURES
    extra = names - allowed
    if forbidden or extra:
        raise LeakageError(f"features are not pre-decision: {sorted(forbidden | extra)}")


def fit_simple_observable_policy(
    records: Sequence[Mapping[str, Any]],
    *,
    envs: Sequence[str],
    delta: float = DELTA,
    weight_mode: str = WEIGHT_MACRO,
) -> dict[str, Any]:
    """Fit on selection utilities and score evaluation utilities on held-out source groups.

    Inner folds choose the ridge penalty or tree depth. Outer folds estimate realized utility.
    Classification accuracy is reported and is not the decision metric.
    """
    for record in records:
        assert_predecision_features(record.get("features") or {}, envs)
        _require_split_utilities(record)
    groups = sorted({_fold_id(record) for record in records})
    result: dict[str, Any] = {
        "estimate_name": ESTIMATE_POLICY,
        "valid": False,
        "feature_gain": None,
        "negative_transfer_rate": None,
        "accuracy": None,
        "arms": {},
        "splits_disjoint": False,
        "used_evaluation_utilities_for_training": False,
        "outer_scored_with": "evaluation_utilities",
        "delta": delta,
        "weight_mode": weight_mode,
        "predicted_actions": [],
        "notes": "simple observable policy; a miss does not prove the best action is unpredictable",
    }
    if len(groups) < MIN_POLICY_GROUPS:
        result["reason"] = f"need at least {MIN_POLICY_GROUPS} source groups"
        return result
    names = _common_names(records)
    full = _cross_validated(records, names, groups, weight_mode)
    result["valid"] = True
    result["reason"] = ""
    result["feature_gain"] = full["gain_vs_env_fixed"]
    result["negative_transfer_rate"] = full["negative_transfer_rate"]
    result["accuracy"] = full["accuracy"]
    result["splits_disjoint"] = full["splits_disjoint"]
    result["predicted_actions"] = full["predicted_actions"]
    result["splits"] = full["splits"]
    result["arms"] = {
        "full": full["gain_vs_env_fixed"],
        "env_only": _arm(records, [name for name in names if name.startswith("env__")], groups, weight_mode),
        "experience_only": _arm(records, [name for name in names if name in EXPERIENCE_FEATURES], groups, weight_mode),
        "state_only": _arm(records, [name for name in names if name in STATE_FEATURES], groups, weight_mode),
        "global_fixed": full["global_fixed_vs_env_fixed"],
        "env_fixed": 0.0,
        "env_state_fixed": full["env_state_vs_env_fixed"],
    }
    result["gain_vs_env_state"] = full["gain_vs_env_state"]
    return result


def _arm(
    records: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    groups: Sequence[str],
    weight_mode: str,
) -> float | None:
    if not names:
        return None
    return _cross_validated(records, list(names), groups, weight_mode)["gain_vs_env_fixed"]


def _cross_validated(
    records: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    groups: Sequence[str],
    weight_mode: str,
) -> dict[str, Any]:
    per_unit: list[dict[str, Any]] = []
    splits: list[dict[str, list[str]]] = []
    for test_group in groups:
        train_groups = [group for group in groups if group != str(test_group)]
        if set(train_groups) & {test_group}:
            raise ProtocolError("outer split leaked the test group")
        choice = _select_hyperparams(records, train_groups, names)
        model = _fit_model(records, train_groups, names, choice)
        env_fixed = _env_fixed_actions(records, train_groups)
        global_fixed, _tie = _best_action(_mean_utilities(records, train_groups))
        env_state = _env_state_actions(records, train_groups)
        test_rows = [record for record in records if _fold_id(record) == test_group]
        for record in test_rows:
            action = model.predict(record["features"], names)
            evaluation = record["evaluation_utilities"]
            env_action = env_fixed[str(record["env"])]
            state_key = (str(record["env"]), str(record.get("state_id") or "unspecified"))
            state_action = env_state[state_key]
            per_unit.append(
                {
                    "unit_id": str(record["unit_id"]),
                    "env": str(record["env"]),
                    "action": action,
                    "utility": float(evaluation[action]),
                    "env_fixed_utility": float(evaluation[env_action]),
                    "env_state_utility": float(evaluation[state_action]),
                    "global_fixed_utility": float(evaluation[global_fixed]),
                    "oracle_action": _best_action(evaluation)[0],
                }
            )
        splits.append({"train": list(train_groups), "test": [str(test_group)]})
    gains = {row["unit_id"]: row["utility"] - row["env_fixed_utility"] for row in per_unit}
    versus_state = {row["unit_id"]: row["utility"] - row["env_state_utility"] for row in per_unit}
    state_versus_env = {row["unit_id"]: row["env_state_utility"] - row["env_fixed_utility"] for row in per_unit}
    global_vs_env = {row["unit_id"]: row["global_fixed_utility"] - row["env_fixed_utility"] for row in per_unit}
    envs = {row["unit_id"]: row["env"] for row in per_unit}
    negative = [row["utility"] < row["env_fixed_utility"] - 1e-12 for row in per_unit]
    accurate = [row["action"] == row["oracle_action"] for row in per_unit]
    return {
        "gain_vs_env_fixed": _weighted(gains, envs, weight_mode),
        "gain_vs_env_state": _weighted(versus_state, envs, weight_mode),
        "env_state_vs_env_fixed": _weighted(state_versus_env, envs, weight_mode),
        "global_fixed_vs_env_fixed": _weighted(global_vs_env, envs, weight_mode),
        "negative_transfer_rate": float(sum(negative) / len(negative)),
        "accuracy": float(sum(accurate) / len(accurate)),
        "predicted_actions": [{"unit_id": row["unit_id"], "action": row["action"]} for row in per_unit],
        "splits": splits,
        "splits_disjoint": all(set(item["train"]).isdisjoint(item["test"]) for item in splits),
    }


def _select_hyperparams(
    records: Sequence[Mapping[str, Any]],
    train_groups: Sequence[str],
    names: Sequence[str],
) -> tuple[str, float]:
    candidates: tuple[tuple[str, float], ...] = (
        ("ridge", 1.0),
        ("ridge", 0.1),
        ("ridge", 10.0),
        ("tree", 1.0),
        ("tree", 2.0),
    )
    best: tuple[str, float] = candidates[0]
    best_score = float("-inf")
    for kind, param in candidates:
        scores: list[float] = []
        for val_group in train_groups:
            inner = [group for group in train_groups if group != val_group]
            model = _fit_model(records, inner, names, (kind, param))
            for record in records:
                if _fold_id(record) != val_group:
                    continue
                action = model.predict(record["features"], names)
                scores.append(float(record["selection_utilities"][action]))
        score = float(np.mean(scores))
        if score > best_score + 1e-12:
            best = (kind, param)
            best_score = score
    return best


class _ActionModel:
    def __init__(self, kind: str, packs: Mapping[str, Any]):
        self.kind = kind
        self.packs = dict(packs)

    def predict(self, features: Mapping[str, Any], names: Sequence[str]) -> str:
        vector = _vector(features, names)
        scores = {action: _predict_one(self.kind, pack, vector) for action, pack in self.packs.items()}
        return _best_action(scores)[0]


def _fit_model(
    records: Sequence[Mapping[str, Any]],
    groups: Sequence[str],
    names: Sequence[str],
    choice: tuple[str, float],
) -> _ActionModel:
    train = [record for record in records if _fold_id(record) in set(groups)]
    if len(train) < 2:
        raise ProtocolError("inner training fold has fewer than two units")
    actions = sorted(train[0]["selection_utilities"])
    kind, param = choice
    packs: dict[str, Any] = {}
    design = np.vstack([_vector(record["features"], names) for record in train])
    for action in actions:
        target = np.array([float(record["selection_utilities"][action]) for record in train], dtype=float)
        if kind == "ridge":
            packs[action] = _fit_ridge(design, target, param)
        elif kind == "tree":
            tree = DecisionTreeRegressor(max_depth=int(param), random_state=0)
            tree.fit(design, target)
            packs[action] = tree
        else:
            raise ProtocolError(kind)
    return _ActionModel(kind, packs)


def _fit_ridge(design: np.ndarray, target: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = design.mean(axis=0)
    sig = np.where(design.std(axis=0) < 1e-8, 1.0, design.std(axis=0))
    scaled = (design - mu) / sig
    matrix = np.column_stack([np.ones(len(design)), scaled])
    penalty = np.eye(matrix.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    gram = matrix.T @ matrix + penalty
    try:
        coef = np.linalg.solve(gram, matrix.T @ target)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(gram, matrix.T @ target, rcond=None)[0]
    return mu, sig, coef


def _predict_one(kind: str, pack: Any, vector: np.ndarray) -> float:
    if kind == "ridge":
        mu, sig, coef = pack
        scaled = (vector - mu) / sig
        return float(coef[0] + coef[1:] @ scaled)
    return float(pack.predict(vector.reshape(1, -1))[0])


def _env_fixed_actions(records: Sequence[Mapping[str, Any]], groups: Sequence[str]) -> dict[str, str]:
    chosen: dict[str, str] = {}
    by_env: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    allowed = set(groups)
    for record in records:
        if _fold_id(record) in allowed:
            by_env[str(record["env"])].append(record)
    for env, members in by_env.items():
        means = _mean_utilities(members, [_fold_id(record) for record in members])
        chosen[env], _tie = _best_action(means)
    return chosen


def _mean_utilities(records: Sequence[Mapping[str, Any]], groups: Sequence[str]) -> dict[str, float]:
    """Environment macro-average. A larger environment does not dominate the fixed baseline."""
    allowed = set(groups)
    chosen = [record for record in records if _fold_id(record) in allowed]
    actions = sorted(chosen[0]["selection_utilities"])
    by_env: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in chosen:
        by_env[str(record["env"])].append(record)
    means: dict[str, float] = {}
    for action in actions:
        env_means = [
            float(np.mean([float(record["selection_utilities"][action]) for record in members]))
            for members in by_env.values()
        ]
        means[action] = float(np.mean(env_means))
    return means


def _env_state_actions(records: Sequence[Mapping[str, Any]], groups: Sequence[str]) -> dict[tuple[str, str], str]:
    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    allowed = set(groups)
    for record in records:
        if _fold_id(record) in allowed:
            buckets[(str(record["env"]), str(record.get("state_id") or "unspecified"))].append(record)
    chosen: dict[tuple[str, str], str] = {}
    for key, members in buckets.items():
        means = _mean_utilities(members, [_fold_id(record) for record in members])
        chosen[key], _tie = _best_action(means)
    return chosen


def _fold_id(record: Mapping[str, Any]) -> str:
    """Same experience batch stays in one fold across S0 and S_history."""
    if record.get("experience_source_group"):
        return str(record["experience_source_group"])
    return str(record["source_group"])


def _best_action(utilities: Mapping[str, float]) -> tuple[str, bool]:
    best = max(utilities.values())
    winners = sorted(action for action, value in utilities.items() if abs(float(value) - float(best)) <= 1e-12)
    return winners[0], len(winners) > 1


def _common_names(records: Sequence[Mapping[str, Any]]) -> list[str]:
    sets = [set(record["features"]) for record in records]
    if any(item != sets[0] for item in sets):
        raise LeakageError("feature keys differ across units; missing features are not imputed as zero")
    return sorted(sets[0])


def _vector(features: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    values = []
    for name in names:
        if name not in features or features[name] is None:
            raise LeakageError(f"missing feature {name}")
        values.append(float(features[name]))
    return np.asarray(values, dtype=float)


def _weighted(values: Mapping[str, float], envs: Mapping[str, str], weight_mode: str) -> float:
    units = list(values)
    weights = unit_weights([envs[unit] for unit in units], weight_mode=weight_mode)
    return float(sum(weight * float(values[unit]) for weight, unit in zip(weights, units, strict=True)))


def _require_split_utilities(record: Mapping[str, Any]) -> None:
    if "selection_utilities" not in record or "evaluation_utilities" not in record:
        raise ProtocolError("policy records need a selection half and an evaluation half")
    if set(record["selection_utilities"]) != set(record["evaluation_utilities"]):
        raise ProtocolError("selection and evaluation action sets differ")
