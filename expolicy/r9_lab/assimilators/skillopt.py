"""A3 SKILLOPT_EDIT assimilator wrapping the bounded adapter."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from expolicy.r9_lab.assimilators.base import Assimilator
from expolicy.r9_lab.constants import A3_SKILLOPT
from expolicy.r9_lab.skillopt_adapter.adapter import SkillOptDecision, apply_skillopt_edit
from expolicy.r9_lab.state.persistent_state import PersistentState


class SkillOptAssimilator(Assimilator):
    action = A3_SKILLOPT

    def apply(
        self,
        state: PersistentState,
        experiences: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        validate_fn: Callable[[str], float] = kwargs.get("validate_fn") or (lambda _skill: 0.0)
        client = kwargs.get("client")
        decision: SkillOptDecision = apply_skillopt_edit(
            state, experiences, validate_fn=validate_fn, client=client
        )
        return {
            "action": A3_SKILLOPT,
            "changed": decision.accepted,
            "accepted": decision.accepted,
            "reason": decision.reason,
            "before_score": decision.before_score,
            "after_score": decision.after_score,
            "before_hash": decision.before_hash,
            "after_hash": decision.after_hash,
            "backend": decision.backend,
        }


def apply_skillopt(
    state: PersistentState,
    experiences: Sequence[dict[str, Any]],
    *,
    validate_fn: Callable[[str], float],
    client: Any = None,
) -> dict[str, Any]:
    return SkillOptAssimilator().apply(state, experiences, validate_fn=validate_fn, client=client)
