"""DISCARD: no persistent capability change."""

from __future__ import annotations

from typing import Any, Sequence

from expolicy.closed_loop.constants import DISCARD
from expolicy.core.experience import Experience
from expolicy.closed_loop.state import ExPolicyState


def apply_discard(state: ExPolicyState, experiences: Sequence[Experience]) -> dict[str, Any]:
    return {"action": DISCARD, "changed": False, "n": len(experiences)}
