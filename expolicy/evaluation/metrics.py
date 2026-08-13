"""Answer matching and cost metrics. Failures stay failures."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from expolicy.core.experience import Experience

_PUNCT = re.compile(r"[^a-z0-9\s]+")


def normalize(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", (text or "").lower()).split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize(pred).split()
    gold_toks = normalize(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def contains_match(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    return bool(p) and (g in p or p in g)


def score_experience(exp: Experience) -> dict[str, Any]:
    pred = exp.predicted_answer
    gold = exp.gold_answer or ""
    em = exact_match(pred, gold)
    f1 = token_f1(pred, gold)
    contained = contains_match(pred, gold)
    success = bool(exp.success) or em or (f1 >= 0.85) or contained
    return {
        "task_id": exp.task_id,
        "capability": exp.capability,
        "stage": exp.stage,
        "split": exp.split,
        "em": float(em),
        "f1": f1,
        "contains": float(contained),
        "success": float(success),
        "reward": float(exp.reward or 0.0),
        "tokens": exp.cost.tokens,
        "steps": exp.cost.environment_steps,
        "latency": exp.cost.latency,
        "tool_calls": exp.cost.tool_calls,
        "failure_type": None if success else (exp.failure_type.value if exp.failure_type else "unknown"),
    }


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "f1": 0.0, "tokens": 0, "steps": 0}
    def mean(key: str) -> float:
        return sum(float(r.get(key) or 0.0) for r in rows) / n

    by_cap: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cap.setdefault(str(row.get("capability") or "unknown"), []).append(row)
    return {
        "n": n,
        "accuracy": mean("success"),
        "em": mean("em"),
        "f1": mean("f1"),
        "tokens": sum(int(r.get("tokens") or 0) for r in rows),
        "steps": sum(int(r.get("steps") or 0) for r in rows),
        "latency": sum(float(r.get("latency") or 0.0) for r in rows),
        "tool_calls": sum(int(r.get("tool_calls") or 0) for r in rows),
        "by_capability": {
            cap: {"n": len(items), "accuracy": sum(float(i.get("success") or 0) for i in items) / len(items)}
            for cap, items in by_cap.items()
        },
    }
