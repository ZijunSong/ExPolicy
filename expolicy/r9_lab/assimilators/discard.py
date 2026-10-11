"""A0 DISCARD: true no-op. Experience is observed; persistent state hash unchanged."""

from __future__ import annotations

from typing import Any, Sequence

from expolicy.r9_lab.assimilators.base import Assimilator
from expolicy.r9_lab.constants import A0_DISCARD
from expolicy.r9_lab.state.persistent_state import PersistentState


class DiscardAssimilator(Assimilator):
    action = A0_DISCARD

    def apply(self, state: PersistentState, experiences: Sequence[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        before = state.content_hash()
        after = state.content_hash()
        if before != after:
            raise RuntimeError("DISCARD mutated state hash")
        return {"action": A0_DISCARD, "changed": False, "n": len(experiences), "hash": after}


def apply_discard(state: PersistentState, experiences: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return DiscardAssimilator().apply(state, experiences)
