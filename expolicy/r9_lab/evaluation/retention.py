"""Retention evaluation. Read-only: must not write persistent state."""

from __future__ import annotations

from typing import Any, Sequence

from expolicy.r9_lab.evaluation.shadow import evaluate_read_only
from expolicy.r9_lab.state.persistent_state import PersistentState


def evaluate_retention(tasks: Sequence[Any], client: Any, state: PersistentState, **kwargs: Any) -> dict[str, Any]:
    payload = evaluate_read_only(tasks, client, state, **kwargs)
    payload["split"] = "retention_probe"
    payload["used_for_update"] = False
    return payload
