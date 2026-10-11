"""Persistent ExPolicyState: capability, intervention, and policy posteriors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from expolicy.closed_loop.constants import ASSIM_ACTIONS, CAPS
from expolicy.closed_loop.io_utils import sha256_json


def _cap_float() -> dict[str, float]:
    return {c: 0.0 for c in CAPS}


def _cap_int() -> dict[str, int]:
    return {c: 0 for c in CAPS}


def _cap_bool() -> dict[str, bool]:
    return {c: False for c in CAPS}


def _action_float() -> dict[str, float]:
    return {a: 0.0 for a in ASSIM_ACTIONS}


@dataclass
class ExPolicyState:
    round_id: int = 0
    budget_used: int = 0
    budget_total: int = 48

    capability_seen_count: dict[str, int] = field(default_factory=_cap_int)
    capability_success_count: dict[str, int] = field(default_factory=_cap_int)
    capability_failure_count: dict[str, int] = field(default_factory=_cap_int)
    capability_pool_size: dict[str, int] = field(default_factory=_cap_int)

    shadow_em_by_capability: dict[str, float] = field(default_factory=_cap_float)
    shadow_macro: float = 0.0
    retention_score: float = 0.0

    active_skills: dict[str, bool] = field(default_factory=_cap_bool)
    episodic_memory_size: dict[str, int] = field(default_factory=_cap_int)
    semantic_memory_size: dict[str, int] = field(default_factory=_cap_int)

    last_batch_capability: str | None = None
    last_batch_success_rate: float | None = None
    last_batch_mean_searches: float | None = None
    last_batch_repeat_rate: float | None = None
    last_batch_premature_answer_rate: float | None = None

    last_realized_utility: float | None = None
    utility_ema_by_capability: dict[str, float] = field(default_factory=_cap_float)
    utility_ema_by_assim_action: dict[str, float] = field(default_factory=_action_float)

    acquisition_posterior: dict[str, Any] = field(default_factory=dict)
    assimilation_posterior: dict[str, Any] = field(default_factory=dict)

    used_task_ids: list[str] = field(default_factory=list)
    total_tokens: int = 0
    total_tool_calls: int = 0
    total_wall_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExPolicyState":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})

    def content_hash(self) -> str:
        blob = self.to_dict()
        blob.pop("total_wall_seconds", None)
        return sha256_json(blob)

    def remaining_budget_fraction(self) -> float:
        if self.budget_total <= 0:
            return 0.0
        return max(0.0, 1.0 - self.budget_used / self.budget_total)

    def round_fraction(self, n_rounds: int) -> float:
        if n_rounds <= 0:
            return 1.0
        return min(1.0, self.round_id / n_rounds)

    def seen_fraction(self, cap: str) -> float:
        denom = max(1, int(self.capability_pool_size.get(cap) or 0))
        return min(1.0, float(self.capability_seen_count.get(cap) or 0) / denom)

    def failure_rate(self, cap: str) -> float:
        seen = float(self.capability_seen_count.get(cap) or 0)
        if seen <= 0:
            return 0.0
        return float(self.capability_failure_count.get(cap) or 0) / seen

    def copy(self) -> "ExPolicyState":
        return ExPolicyState.from_dict(self.to_dict())
