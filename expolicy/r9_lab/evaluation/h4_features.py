"""H4: leave-one-batch-out predictors of best assimilation action.

Training-side / current-state features only. No future_test, no U labels as inputs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from expolicy.r9_lab.constants import ACTIONS, ENVS

FEATURE_KEYS = (
    "difficulty",
    "skill_count",
    "batch_success_rate",
    "mean_steps",
    "invalid_action_rate",
    "tokens",
    "memory_size",
    "skill_size",
)


def _env_index(env: str) -> list[float]:
    return [1.0 if env == e else 0.0 for e in ENVS]


def feature_vector(row: Mapping[str, Any]) -> list[float]:
    vec = _env_index(str(row.get("env") or ""))
    for key in FEATURE_KEYS:
        try:
            vec.append(float(row.get(key) or 0.0))
        except (TypeError, ValueError):
            vec.append(0.0)
    return vec


def _majority(labels: Sequence[str]) -> str:
    if not labels:
        return str(ACTIONS[0])
    return Counter(labels).most_common(1)[0][0]


def _loo_majority(labels: Sequence[str]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    hit = 0
    for i in range(n):
        pred = _majority([labels[j] for j in range(n) if j != i])
        hit += int(pred == labels[i])
    return hit / n


def _loo_env_only(rows: Sequence[Mapping[str, Any]]) -> float:
    n = len(rows)
    if n == 0:
        return 0.0
    hit = 0
    for i, row in enumerate(rows):
        by_env: dict[str, list[str]] = defaultdict(list)
        for j, other in enumerate(rows):
            if j == i:
                continue
            by_env[str(other.get("env") or "")].append(str(other.get("best_action") or ""))
        env = str(row.get("env") or "")
        pred = _majority(by_env.get(env) or [str(r.get("best_action") or "") for k, r in enumerate(rows) if k != i])
        hit += int(pred == str(row.get("best_action") or ""))
    return hit / n


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _softmax(logits: Sequence[float]) -> list[float]:
    m = max(logits)
    exps = [pow(2.718281828, z - m) for z in logits]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _fit_multinomial(xs: Sequence[Sequence[float]], ys: Sequence[int], n_class: int, *, steps: int = 80, lr: float = 0.15):
    dim = len(xs[0]) if xs else 1
    w = [[0.0] * dim for _ in range(n_class)]
    b = [0.0] * n_class
    n = max(1, len(xs))
    for _ in range(steps):
        gw = [[0.0] * dim for _ in range(n_class)]
        gb = [0.0] * n_class
        for x, y in zip(xs, ys):
            logits = [_dot(w[c], x) + b[c] for c in range(n_class)]
            probs = _softmax(logits)
            for c in range(n_class):
                err = probs[c] - (1.0 if c == y else 0.0)
                gb[c] += err
                for d in range(dim):
                    gw[c][d] += err * x[d]
        for c in range(n_class):
            b[c] -= lr * gb[c] / n
            for d in range(dim):
                w[c][d] -= lr * (gw[c][d] / n + 1e-4 * w[c][d])
    return w, b


def _predict(x: Sequence[float], w, b) -> int:
    logits = [_dot(w[c], x) + b[c] for c in range(len(b))]
    return max(range(len(b)), key=lambda c: logits[c])


def _loo_full(rows: Sequence[Mapping[str, Any]]) -> float:
    xs = [feature_vector(r) for r in rows]
    label_names = sorted({str(r.get("best_action") or "") for r in rows} | set(ACTIONS))
    ys = [label_names.index(str(r.get("best_action") or label_names[0])) for r in rows]
    n = len(rows)
    if n < 3:
        return 0.0
    hit = 0
    for i in range(n):
        x_tr = [xs[j] for j in range(n) if j != i]
        y_tr = [ys[j] for j in range(n) if j != i]
        w, b = _fit_multinomial(x_tr, y_tr, len(label_names))
        hit += int(_predict(xs[i], w, b) == ys[i])
    return hit / n


def analyze_h4(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [str(r.get("best_action") or "") for r in rows]
    majority = _loo_majority(labels)
    env_only = _loo_env_only(rows)
    full = _loo_full(rows)
    return {
        "full_feature_accuracy": full,
        "majority_accuracy": majority,
        "env_only_accuracy": env_only,
        "n": len(rows),
        "features": list(ENVS) + list(FEATURE_KEYS),
        "protocol": "leave-one-batch-out",
    }
