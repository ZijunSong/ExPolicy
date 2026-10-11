"""Verifier helpers. Gold / reference never enter the LLM prompt."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

from expolicy.r9_lab.constants import GOLD_FIELD_NAMES


def extract_gold_fields(obj: Any, *, prefix: str = "") -> dict[str, Any]:
    """Pull gold/reference fields out of a nested mapping for the verifier only."""
    found: dict[str, Any] = {}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key = str(k)
            path = f"{prefix}.{key}" if prefix else key
            if key.lower() in GOLD_FIELD_NAMES or any(g in key.lower() for g in ("gold", "ground_truth", "reference")):
                found[path] = v
            else:
                found.update(extract_gold_fields(v, prefix=path))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found.update(extract_gold_fields(v, prefix=f"{prefix}[{i}]"))
    return found


def strip_gold(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            key = str(k)
            if key.lower() in GOLD_FIELD_NAMES:
                continue
            if any(g in key.lower() for g in ("gold", "ground_truth", "reference_action", "reference_sql")):
                continue
            out[k] = strip_gold(v)
        return out
    if isinstance(obj, list):
        return [strip_gold(x) for x in obj]
    return obj


def _flatten_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except TypeError:
        return str(obj)


def gold_strings(task_or_gold: Any) -> list[str]:
    if task_or_gold is None:
        return []
    if isinstance(task_or_gold, str):
        text = task_or_gold.strip()
        return [text] if text else []
    fields = extract_gold_fields(task_or_gold) if isinstance(task_or_gold, Mapping) else {}
    if not fields and hasattr(task_or_gold, "__dict__"):
        fields = extract_gold_fields(vars(task_or_gold))
    out: list[str] = []
    for v in fields.values():
        text = _flatten_text(v).strip()
        if text and text not in ("null", "[]", "{}", "0", "false"):
            out.append(text)
            if isinstance(v, str) and len(v.strip()) >= 8:
                out.append(v.strip())
    return out


def prompt_contains_gold(messages: Sequence[Mapping[str, Any]], gold: Any) -> list[str]:
    blob = "\n".join(_flatten_text(m) for m in messages)
    hits: list[str] = []
    for g in gold_strings(gold):
        if len(g) < 6:
            continue
        if g in blob:
            hits.append(g[:120])
    return hits


def assert_no_gold_in_prompt(messages: Sequence[Mapping[str, Any]], gold: Any) -> None:
    hits = prompt_contains_gold(messages, gold)
    if hits:
        raise AssertionError(f"gold/reference leaked into LLM prompt: {hits[:3]}")
