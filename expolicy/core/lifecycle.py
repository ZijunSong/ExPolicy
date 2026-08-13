"""Lifecycle state machine for an experience object."""

from __future__ import annotations

from expolicy.core.errors import InvalidExperienceError
from expolicy.core.types import ExperienceStatus

_ALLOWED: dict[ExperienceStatus, set[ExperienceStatus]] = {
    ExperienceStatus.ACQUIRED: {
        ExperienceStatus.VERIFIED,
        ExperienceStatus.REJECTED,
    },
    ExperienceStatus.VERIFIED: {
        ExperienceStatus.ABSTRACTED,
        ExperienceStatus.REJECTED,
    },
    ExperienceStatus.ABSTRACTED: {
        ExperienceStatus.EVALUATED,
        ExperienceStatus.ROUTED,
    },
    ExperienceStatus.EVALUATED: {ExperienceStatus.ROUTED},
    ExperienceStatus.ROUTED: {ExperienceStatus.USED, ExperienceStatus.DEPRECATED},
    ExperienceStatus.USED: {
        ExperienceStatus.REEVALUATED,
        ExperienceStatus.MERGED,
        ExperienceStatus.SUPERSEDED,
        ExperienceStatus.DEPRECATED,
    },
    ExperienceStatus.REEVALUATED: {
        ExperienceStatus.MERGED,
        ExperienceStatus.SUPERSEDED,
        ExperienceStatus.DEPRECATED,
        ExperienceStatus.USED,
    },
    ExperienceStatus.MERGED: {ExperienceStatus.DEPRECATED, ExperienceStatus.SUPERSEDED},
    ExperienceStatus.SUPERSEDED: {ExperienceStatus.DEPRECATED},
    ExperienceStatus.DEPRECATED: set(),
    ExperienceStatus.REJECTED: set(),
}


def can_transition(src: ExperienceStatus, dst: ExperienceStatus) -> bool:
    return dst in _ALLOWED.get(src, set())


def transition(src: ExperienceStatus, dst: ExperienceStatus) -> ExperienceStatus:
    if not can_transition(src, dst):
        raise InvalidExperienceError(f"illegal experience transition {src.value} -> {dst.value}")
    return dst
