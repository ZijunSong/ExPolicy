"""Held-out future_test evaluation. NEVER used for operator / oracle selection."""

from __future__ import annotations

from typing import Any, Sequence

from expolicy.r9_lab.evaluation.shadow import evaluate_read_only
from expolicy.r9_lab.state.persistent_state import PersistentState


def evaluate_future(tasks: Sequence[Any], client: Any, state: PersistentState, **kwargs: Any) -> dict[str, Any]:
    payload = evaluate_read_only(tasks, client, state, **kwargs)
    payload["split"] = "future_test"
    payload["used_for_update"] = False
    payload["used_for_oracle_selection"] = False
    return payload
