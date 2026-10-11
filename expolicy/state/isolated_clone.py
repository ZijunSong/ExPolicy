"""Isolated clone of CountOnly evidence state.

R6.1 production assimilator A is frozen CountOnly unique-evidence activation.
The live agent object (LLM weights, caches, SkillStore on disk) must never be
mutated by a shadow candidate evaluation. This module is the only state that
shadow scoring is allowed to touch.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from expolicy.environments.public_search.capability_map import PUBLIC_CAPS

MIN_SUPPORT = 8


def _empty_unique() -> dict[str, list[str]]:
    return {cap: [] for cap in PUBLIC_CAPS}


@dataclass
class EvidenceState:
    """Ordered unique task ids per predicted capability, plus global acquisition order."""

    unique_ids: dict[str, list[str]] = field(default_factory=_empty_unique)
    acquired_ids: list[str] = field(default_factory=list)
    predicted_cap: dict[str, str] = field(default_factory=dict)

    def clone(self) -> "EvidenceState":
        return EvidenceState(
            unique_ids={k: list(v) for k, v in self.unique_ids.items()},
            acquired_ids=list(self.acquired_ids),
            predicted_cap=dict(self.predicted_cap),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "unique_ids": {k: list(v) for k, v in self.unique_ids.items()},
            "acquired_ids": list(self.acquired_ids),
            "predicted_cap": dict(self.predicted_cap),
            "active_caps": list(self.active_caps()),
            "counts": {k: len(v) for k, v in self.unique_ids.items()},
        }

    @classmethod
    def from_snapshot(cls, payload: Mapping[str, Any]) -> "EvidenceState":
        unique = _empty_unique()
        raw = dict(payload.get("unique_ids") or {})
        for cap in PUBLIC_CAPS:
            unique[cap] = [str(x) for x in (raw.get(cap) or [])]
        return cls(
            unique_ids=unique,
            acquired_ids=[str(x) for x in (payload.get("acquired_ids") or [])],
            predicted_cap={str(k): str(v) for k, v in dict(payload.get("predicted_cap") or {}).items()},
        )

    def counts(self) -> dict[str, int]:
        return {cap: len(self.unique_ids.get(cap) or []) for cap in PUBLIC_CAPS}

    def active_caps(self, min_support: int = MIN_SUPPORT) -> tuple[str, ...]:
        return tuple(cap for cap in PUBLIC_CAPS if len(self.unique_ids.get(cap) or []) >= int(min_support))

    def mask_key(self, min_support: int = MIN_SUPPORT) -> str:
        active = self.active_caps(min_support)
        return mask_key_from_caps(active)

    def apply_batch(
        self,
        task_ids: Sequence[str],
        predicted_cap: Mapping[str, str] | None = None,
    ) -> "EvidenceState":
        """Return a NEW state with task_ids assimilated. Self is unchanged."""
        other = self.clone()
        preds = dict(other.predicted_cap)
        if predicted_cap:
            preds.update({str(k): str(v) for k, v in dict(predicted_cap).items()})
            other.predicted_cap = preds
        acquired = set(other.acquired_ids)
        for tid in task_ids:
            tid = str(tid)
            if tid in acquired:
                continue
            other.acquired_ids.append(tid)
            acquired.add(tid)
            cap = preds.get(tid) or ""
            if cap in PUBLIC_CAPS and tid not in other.unique_ids[cap]:
                other.unique_ids[cap].append(tid)
        return other


def mask_key_from_caps(caps: Iterable[str], min_support: int = MIN_SUPPORT) -> str:
    active = [c for c in PUBLIC_CAPS if c in set(caps)]
    return "+".join(active) if active else "none"


def caps_from_mask_key(key: str) -> tuple[str, ...]:
    if not key or key == "none":
        return ()
    return tuple(c for c in key.split("+") if c in PUBLIC_CAPS)


def clone_mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    return copy.deepcopy(dict(payload or {}))


def clone_rng_state(rng: random.Random) -> tuple:
    return rng.getstate()


def restore_rng_state(rng: random.Random, state: tuple) -> None:
    rng.setstate(state)
