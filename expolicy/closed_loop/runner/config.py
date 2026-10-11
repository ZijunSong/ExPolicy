"""Frozen R7 experiment config. Do not tune from final test."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from expolicy.closed_loop.constants import (
    DEFAULT_BATCH,
    DEFAULT_BUDGET,
    DEFAULT_EMBEDDER,
    DEFAULT_MODEL,
    DEFAULT_RETRIEVER_URL,
    DEFAULT_ROUNDS,
    FULL_EVAL_AT,
    SPLIT_SEED,
)
from expolicy.closed_loop.io_utils import sha256_json


@dataclass
class R7Config:
    condition: str = "expolicy"
    seed: int = 0
    split_seed: int = SPLIT_SEED
    rounds: int = DEFAULT_ROUNDS
    batch_size: int = DEFAULT_BATCH
    budget: int = DEFAULT_BUDGET
    model: str = DEFAULT_MODEL
    embedder: str = DEFAULT_EMBEDDER
    retriever_url: str = DEFAULT_RETRIEVER_URL
    acquisition: str = "linear_thompson"
    assimilation: str = "linear_thompson"
    feedback: str = "true"  # true | none | shuffle
    all_skills_upfront: bool = False
    prior_precision: float = 1.0
    reward_noise: float = 0.05
    retention_weight: float = 0.5
    token_cost_per_million: float = 0.01
    tool_cost_per_100: float = 0.005
    full_eval_at: tuple[int, ...] = FULL_EVAL_AT
    max_steps: int = 8
    smoke: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["full_eval_at"] = list(self.full_eval_at)
        return d

    def content_hash(self) -> str:
        blob = self.to_dict()
        blob.pop("extra", None)
        return sha256_json(blob)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "R7Config":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        data = {k: v for k, v in payload.items() if k in known}
        if "full_eval_at" in data:
            data["full_eval_at"] = tuple(data["full_eval_at"])
        return cls(**data)
