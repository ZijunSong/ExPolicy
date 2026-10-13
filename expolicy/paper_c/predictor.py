"""Ridge gain predictor. A high correlation is not a selection result."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from expolicy.paper_c.features import LeakageError, assert_feature_keys, vectorize


class GainPredictor:
    def __init__(self, names: Sequence[str], mu: np.ndarray, sig: np.ndarray, coef: np.ndarray):
        self.names = list(names)
        self.mu = mu
        self.sig = sig
        self.coef = coef

    def predict(self, features: Mapping[str, Any]) -> float:
        values = vectorize(features, self.names)
        if values is None:
            raise LeakageError("gain features are incomplete; refusing to impute zero")
        z = (np.asarray(values, dtype=float) - self.mu) / self.sig
        return float(self.coef[0] + float(self.coef[1:] @ z))


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, dim = x.shape
    mu = x.mean(axis=0)
    sig = np.where(x.std(axis=0) < 1e-8, 1.0, x.std(axis=0))
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


def _complete_rows(
    records: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("split") or "") == "future_test":
            raise LeakageError("future test rows cannot train the gain model")
        if record.get("gain") is None:
            continue
        features = dict(record.get("features") or {})
        assert_feature_keys(features.keys())
        if vectorize(features, names) is None:
            continue
        kept.append(dict(record))
    return kept


def permute_gains(records: Sequence[Mapping[str, Any]], seed: int) -> tuple[list[dict[str, Any]], bool]:
    """Permute realized gains inside each environment. Features stay put."""
    rng = np.random.default_rng(seed)
    copied = [dict(record) for record in records]
    by_env: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(copied):
        by_env[str(record.get("env") or "")].append(index)
    changed = False
    for indexes in by_env.values():
        gains = [float(copied[index]["gain"]) for index in indexes]
        if len(set(gains)) <= 1 or len(indexes) < 2:
            continue
        order = list(range(len(indexes)))
        for _ in range(8):
            draw = [int(item) for item in rng.permutation(len(indexes))]
            if draw != order:
                order = draw
                break
        for local, source in enumerate(order):
            copied[indexes[local]]["gain"] = gains[source]
            if gains[source] != gains[local]:
                changed = True
    return copied, changed


def fit_gain_predictor(
    records: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    *,
    alpha: float = 1.0,
    min_rows: int = 8,
    label_shuffle: int | None = None,
) -> dict[str, Any]:
    """Fit one ridge model, or refuse when the offline sample is below the registered size."""
    rows = _complete_rows(records, feature_names)
    shuffled = False
    if label_shuffle is not None:
        rows, shuffled = permute_gains(rows, label_shuffle)
    if len(rows) < int(min_rows):
        return {
            "status": "PILOT_ONLY",
            "model": None,
            "n": len(rows),
            "reason": "offline sample is below the pre-registered minimum",
            "shuffled": shuffled,
        }
    x = np.asarray([vectorize(row["features"], feature_names) for row in rows], dtype=float)
    y = np.asarray([float(row["gain"]) for row in rows], dtype=float)
    mu, sig, coef = fit_ridge(x, y, alpha)
    return {
        "status": "ok",
        "model": GainPredictor(feature_names, mu, sig, coef),
        "n": len(rows),
        "reason": "",
        "shuffled": shuffled,
    }


def _family_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("gain") is None:
            continue
        buckets[str(row["family"])].append(row)
    table = {}
    for family, items in buckets.items():
        table[family] = {
            "gain": sum(float(item["gain"]) for item in items) / len(items),
            "features": dict(items[0]["features"]),
        }
    return table


def _chosen(scores: Mapping[str, float]) -> str | None:
    if not scores:
        return None
    best = max(scores.values())
    winners = [family for family, score in scores.items() if abs(score - best) <= 1e-9]
    if len(winners) != 1:
        return None
    return winners[0]


def selection_advantage(model: GainPredictor | None, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean realized gain of the unique predicted family, minus the uniform family mean.

    A tie is indifference and scores as uniform, so a name-order tie cannot count as skill.
    """
    if model is None:
        return {"mean_advantage": None, "n_states": 0, "per_state": []}
    by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("gain") is None:
            continue
        by_state[str(record.get("state_id") or record.get("sequence_id"))].append(record)
    deltas: list[float] = []
    for items in by_state.values():
        table = _family_table(items)
        if len(table) < 2:
            continue
        scores = {}
        skip = False
        for family, payload in table.items():
            try:
                scores[family] = model.predict(payload["features"])
            except LeakageError:
                skip = True
                break
        if skip:
            continue
        uniform = sum(item["gain"] for item in table.values()) / len(table)
        chosen = _chosen(scores)
        deltas.append(0.0 if chosen is None else float(table[chosen]["gain"]) - uniform)
    if not deltas:
        return {"mean_advantage": None, "n_states": 0, "per_state": []}
    return {"mean_advantage": sum(deltas) / len(deltas), "n_states": len(deltas), "per_state": deltas}


def grouped_holdout(
    records: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    *,
    group_key: str,
    alpha: float,
    min_rows: int,
    label_shuffle: int | None = None,
) -> dict[str, Any]:
    groups = sorted({str(record.get(group_key)) for record in records})
    if len(groups) < 2:
        return {"status": "insufficient_groups", "mean_advantage": None, "folds": []}
    folds = []
    for held in groups:
        train = [record for record in records if str(record.get(group_key)) != held]
        test = [record for record in records if str(record.get(group_key)) == held]
        fitted = fit_gain_predictor(
            train,
            feature_names,
            alpha=alpha,
            min_rows=min_rows,
            label_shuffle=label_shuffle,
        )
        advantage = selection_advantage(fitted["model"], test)
        folds.append({"held": held, "fit_status": fitted["status"], "advantage": advantage["mean_advantage"]})
    usable = [float(item["advantage"]) for item in folds if item["advantage"] is not None]
    return {
        "status": "ok" if usable else "missing",
        "group_key": group_key,
        "mean_advantage": None if not usable else sum(usable) / len(usable),
        "folds": folds,
    }


def unseen_family_advantage(
    records: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    *,
    alpha: float,
    min_rows: int,
) -> dict[str, Any]:
    """On states where one held-out family is the unique best, how often the model selects it.

    Training rows for that family are removed. Chance is one over the number of families at the state.
    """
    families = sorted({str(record["family"]) for record in records})
    excesses: list[float] = []
    for held in families:
        train = [record for record in records if str(record["family"]) != held]
        fitted = fit_gain_predictor(train, feature_names, alpha=alpha, min_rows=min_rows)
        model = fitted["model"]
        if model is None:
            continue
        by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            if record.get("gain") is not None:
                by_state[str(record.get("state_id") or record.get("sequence_id"))].append(record)
        for items in by_state.values():
            table = _family_table(items)
            if held not in table or len(table) < 2:
                continue
            best_gain = max(float(item["gain"]) for item in table.values())
            winners = [family for family, item in table.items() if abs(float(item["gain"]) - best_gain) <= 1e-9]
            if winners != [held]:
                continue
            scores: dict[str, float] = {}
            try:
                for family, payload in table.items():
                    scores[family] = model.predict(payload["features"])
            except LeakageError:
                continue
            chance = 1.0 / len(table)
            excesses.append((1.0 if _chosen(scores) == held else 0.0) - chance)
    if not excesses:
        return {"mean_excess": None, "n": 0}
    return {"mean_excess": sum(excesses) / len(excesses), "n": len(excesses)}


def correlation(model: GainPredictor | None, records: Sequence[Mapping[str, Any]]) -> float | None:
    if model is None:
        return None
    pairs = []
    for record in records:
        if record.get("gain") is None:
            continue
        try:
            pairs.append((model.predict(record["features"]), float(record["gain"])))
        except LeakageError:
            continue
    if len(pairs) < 3:
        return None
    pred = np.asarray([item[0] for item in pairs], dtype=float)
    truth = np.asarray([item[1] for item in pairs], dtype=float)
    if float(pred.std()) < 1e-12 or float(truth.std()) < 1e-12:
        return None
    return float(np.corrcoef(pred, truth)[0, 1])
