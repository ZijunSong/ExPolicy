"""ACTIVATE_EXISTING_SKILL: turn on the frozen executable skill for this capability."""

from __future__ import annotations

from typing import Any

from expolicy.closed_loop.constants import ACTIVATE_EXISTING_SKILL, CAPS
from expolicy.closed_loop.state import ExPolicyState


def apply_activate(state: ExPolicyState, capability: str) -> dict[str, Any]:
    if capability not in CAPS:
        raise ValueError(capability)
    already = bool(state.active_skills.get(capability))
    state.active_skills[capability] = True
    return {
        "action": ACTIVATE_EXISTING_SKILL,
        "capability": capability,
        "changed": not already,
        "already_active": already,
    }
