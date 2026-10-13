"""C0 content controls. The assimilator algorithm is not rewritten per arm."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from expolicy.paper_c.constants import (
    CONTENT_MODES,
    MODE_EMPTY,
    MODE_NO_UPDATE,
    MODE_REAL,
    MODE_WRONG,
)

MismatchFn = Callable[[str], str]


def rewrite_experiences(
    experiences: Sequence[Mapping[str, Any]],
    mode: str,
    *,
    mismatch_of: MismatchFn,
) -> list[dict[str, Any]]:
    """Keep the record count for empty and mismatched arms. No-update presents nothing.

    ``mismatch_of`` is a pre-registered map. It is not chosen from probe scores.
    """
    if mode not in CONTENT_MODES:
        raise ValueError(f"unknown content mode: {mode}")
    if mode == MODE_NO_UPDATE:
        return []
    if mode == MODE_REAL:
        return [dict(item) for item in experiences]
    rewritten: list[dict[str, Any]] = []
    for item in experiences:
        if mode == MODE_EMPTY:
            rewritten.append(
                {
                    "task_id": item.get("task_id"),
                    "env": item.get("env"),
                    "family": "",
                    "instruction": "",
                    "trajectory": [],
                    "success": False,
                    "verifier_outcome": "EMPTY",
                    "content_control": MODE_EMPTY,
                    "bucket": "acquisition_pool",
                    "source_template": item.get("source_template"),
                }
            )
            continue
        if mode == MODE_WRONG:
            family = str(item.get("family") or "")
            target = mismatch_of(family)
            if target == family:
                raise ValueError("matched-wrong target must be a different family")
            copied = dict(item)
            copied["family"] = target
            copied["instruction"] = f"mismatched content {target}"
            copied["content_control"] = MODE_WRONG
            copied["trajectory"] = [{"content": f"mismatched {target}"}]
            rewritten.append(copied)
    return rewritten
