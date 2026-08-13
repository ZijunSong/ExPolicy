"""Capability profile produced by real diagnosis over trajectories."""

from __future__ import annotations

from pydantic import BaseModel, Field

from expolicy.core.types import CapabilityName, FrontierBand, JSONDict


class FrontierCapability(BaseModel):
    name: str
    band: FrontierBand
    score: float
    learnability: float
    novelty: float
    transfer_potential: float
    uncertainty: float
    expected_cost: float
    n_eval: int = 0


class CapabilityProfile(BaseModel):
    version: str = "v0"
    agent_version: str = "agent_v000"
    capability_scores: dict[str, float] = Field(default_factory=dict)
    failure_distribution: dict[str, float] = Field(default_factory=dict)
    uncertainty: dict[str, float] = Field(default_factory=dict)
    recent_regressions: dict[str, float] = Field(default_factory=dict)
    frontier_capabilities: list[FrontierCapability] = Field(default_factory=list)
    n_trajectories: int = 0
    n_failures: int = 0
    extra: JSONDict = Field(default_factory=dict)

    def score(self, name: str | CapabilityName, default: float = 0.0) -> float:
        key = name.value if isinstance(name, CapabilityName) else name
        return float(self.capability_scores.get(key, default))

    def frontier_names(self) -> list[str]:
        return [
            item.name
            for item in self.frontier_capabilities
            if item.band in {FrontierBand.FRONTIER, FrontierBand.LEARNING}
        ]

    def vector(self, names: list[str] | None = None) -> list[float]:
        keys = names or sorted(self.capability_scores.keys())
        return [float(self.capability_scores.get(k, 0.0)) for k in keys]
