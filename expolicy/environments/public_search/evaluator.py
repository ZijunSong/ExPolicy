"""Public-search metrics: EM, token F1, supporting-title recall, tool diagnostics."""

from __future__ import annotations

import re
from typing import Any, Sequence

from expolicy.core.experience import Experience
from expolicy.environments.search.world import Task
from expolicy.evaluation.metrics import exact_match, normalize, token_f1

_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")


def supporting_recall(opened_titles: Sequence[str], gold_titles: Sequence[str]) -> float:
    if not gold_titles:
        return float("nan")
    opened = {normalize(t) for t in opened_titles if t}
    gold = [normalize(t) for t in gold_titles if t]
    if not gold:
        return float("nan")
    hit = sum(1 for t in gold if t and t in opened)
    return hit / len(gold)


def opened_titles_from_exp(exp: Experience) -> list[str]:
    titles = []
    for ev in exp.evidence:
        if ev.get("title"):
            titles.append(str(ev["title"]))
    for obs in exp.observations:
        payload = obs.get("payload") if isinstance(obs, dict) else None
        if isinstance(payload, dict) and payload.get("title"):
            titles.append(str(payload["title"]))
    return titles


def score_public(exp: Experience, task: Task) -> dict[str, Any]:
    pred = exp.predicted_answer or ""
    gold = task.gold_answer or ""
    aliases = list(task.aliases or [])
    em = exact_match(pred, gold) or any(exact_match(pred, a) for a in aliases)
    f1 = max([token_f1(pred, gold)] + [token_f1(pred, a) for a in aliases] or [0.0])
    titles = opened_titles_from_exp(exp)
    gold_titles = list((task.metadata or {}).get("supporting_titles") or [])
    n_search = sum(1 for c in exp.tool_calls if c.get("tool") == "search")
    n_open = sum(1 for c in exp.tool_calls if c.get("tool") == "open")
    n_invalid = sum(1 for a in exp.actions if a.get("tool") in {None, ""} or a.get("error"))
    loop = 0.0
    queries = [str((c.get("arguments") or {}).get("query") or "") for c in exp.tool_calls if c.get("tool") == "search"]
    if queries:
        uniq = len(set(q.lower() for q in queries if q))
        loop = float(uniq < len(queries) and len(queries) >= 3)
    return {
        "task_id": task.task_id,
        "dataset": (task.metadata or {}).get("dataset"),
        "capability": task.capability,
        "em": float(em),
        "em_strict": float(em),
        "f1": float(f1),
        "success": float(em or f1 >= 0.85),
        "tokens": exp.cost.tokens,
        "tool_calls": exp.cost.tool_calls,
        "n_search": n_search,
        "n_open": n_open,
        "invalid_tool_rate": float(n_invalid / max(1, len(exp.actions))),
        "loop_rate": loop,
        "trajectory_length": len(exp.actions),
        "supporting_recall": supporting_recall(titles, gold_titles),
        "predicted": pred,
        "gold": gold,
        "opened_titles": titles[:8],
    }
