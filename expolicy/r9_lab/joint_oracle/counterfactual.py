"""Temporary counterfactual branches. Must not pollute live state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from expolicy.r9_lab.assimilators.discard import apply_discard
from expolicy.r9_lab.assimilators.episodic import apply_episodic
from expolicy.r9_lab.assimilators.semantic import apply_semantic
from expolicy.r9_lab.assimilators.skillopt import apply_skillopt
from expolicy.r9_lab.constants import A0_DISCARD, A1_EPISODIC, A2_SEMANTIC, A3_SKILLOPT
from expolicy.r9_lab.evaluation.retention import evaluate_retention
from expolicy.r9_lab.evaluation.shadow import evaluate_shadow
from expolicy.r9_lab.evaluation.utility import compute_utility
from expolicy.r9_lab.state.clone import clone_state
from expolicy.r9_lab.state.persistent_state import PersistentState


def apply_action(
    action: str,
    state: PersistentState,
    experiences: Sequence[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    if action == A0_DISCARD:
        return apply_discard(state, experiences)
    if action == A1_EPISODIC:
        return apply_episodic(state, experiences)
    if action == A2_SEMANTIC:
        return apply_semantic(state, experiences, client=kwargs.get("client"))
    if action == A3_SKILLOPT:
        validate_fn = kwargs.get("validate_fn") or (lambda _s: 0.0)
        return apply_skillopt(state, experiences, validate_fn=validate_fn, client=kwargs.get("client"))
    raise ValueError(action)


def run_counterfactual(
    live: PersistentState,
    experiences: Sequence[dict[str, Any]],
    action: str,
    *,
    branch_dir: Path,
    shadow_eval: Callable[[PersistentState], dict[str, Any]] | None = None,
    retention_eval: Callable[[PersistentState], dict[str, Any]] | None = None,
    shadow_before: float = 0.0,
    retention_before: float = 0.0,
    **kwargs: Any,
) -> dict[str, Any]:
    live_hash = live.content_hash()
    branch = clone_state(live, branch_dir)
    apply_payload = apply_action(action, branch, experiences, **kwargs)
    shadow = shadow_eval(branch) if shadow_eval else {"success_rate": 0.0}
    retention = retention_eval(branch) if retention_eval else {"success_rate": 0.0}
    util = compute_utility(
        shadow_before=shadow_before,
        shadow_after=float(shadow.get("success_rate") or 0.0),
        retention_before=retention_before,
        retention_after=float(retention.get("success_rate") or 0.0),
    )
    if live.content_hash() != live_hash:
        raise RuntimeError("counterfactual branch polluted live state")
    return {
        "action": action,
        "branch_dir": str(branch_dir),
        "apply": apply_payload,
        "shadow": {k: v for k, v in shadow.items() if k != "results"},
        "retention": {k: v for k, v in retention.items() if k != "results"},
        "utility": util,
        "live_hash": live_hash,
        "branch_hash": branch.content_hash(),
    }
