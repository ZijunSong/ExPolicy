"""Unified experience schema. Every real interaction becomes an Experience."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from expolicy.core.types import Carrier, ExperienceStatus, FailureType, JSONDict


class ExperienceCost(BaseModel):
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    environment_steps: int = 0
    api_cost: float = 0.0
    gpu_seconds: float = 0.0
    training_tokens: int = 0
    tool_calls: int = 0

    def add(self, other: ExperienceCost) -> ExperienceCost:
        return ExperienceCost(
            tokens=self.tokens + other.tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            latency=self.latency + other.latency,
            environment_steps=self.environment_steps + other.environment_steps,
            api_cost=self.api_cost + other.api_cost,
            gpu_seconds=self.gpu_seconds + other.gpu_seconds,
            training_tokens=self.training_tokens + other.training_tokens,
            tool_calls=self.tool_calls + other.tool_calls,
        )


class Experience(BaseModel):
    experience_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str
    timestamp: float = Field(default_factory=time.time)

    capability_context: JSONDict = Field(default_factory=dict)

    initial_state: JSONDict = Field(default_factory=dict)
    observations: list[JSONDict] = Field(default_factory=list)
    actions: list[JSONDict] = Field(default_factory=list)
    tool_calls: list[JSONDict] = Field(default_factory=list)

    outcome: JSONDict = Field(default_factory=dict)
    reward: float | None = None

    success: bool | None = None
    failure_type: FailureType | None = None

    evidence: list[JSONDict] = Field(default_factory=list)
    provenance: list[JSONDict] = Field(default_factory=list)

    uncertainty: float | None = None
    cost: ExperienceCost = Field(default_factory=ExperienceCost)

    parent_experience_id: str | None = None
    branch_origin: JSONDict | None = None

    status: ExperienceStatus = ExperienceStatus.ACQUIRED
    question: str = ""
    predicted_answer: str = ""
    gold_answer: str | None = None
    split: str | None = None
    capability: str | None = None
    stage: int | None = None
    agent_version: str | None = None
    run_id: str | None = None

    def mark(self, status: ExperienceStatus) -> None:
        self.status = status

    def transcript(self) -> str:
        parts: list[str] = [f"Q: {self.question}"]
        for action, obs in zip(self.actions, self.observations, strict=False):
            parts.append(f"A: {action}")
            parts.append(f"O: {obs}")
        parts.append(f"Answer: {self.predicted_answer}")
        return "\n".join(parts)


class ExperienceAnnotation(BaseModel):
    experience_id: str
    annotator: str
    labels: JSONDict = Field(default_factory=dict)
    notes: str = ""
    timestamp: float = Field(default_factory=time.time)


class ExperienceVerification(BaseModel):
    experience_id: str
    valid: bool
    evidence_score: float
    consistency_score: float
    provenance_score: float
    confidence: float
    rejection_reasons: list[str] = Field(default_factory=list)
    details: JSONDict = Field(default_factory=dict)


class ExperienceUtility(BaseModel):
    experience_id: str
    future_gain: float
    transfer_gain: float
    retention_gain: float
    interaction_cost: float
    token_cost: float
    compute_cost: float
    regression: float
    interference: float
    utility: float
    before_metrics: JSONDict = Field(default_factory=dict)
    after_metrics: JSONDict = Field(default_factory=dict)
    carrier: Carrier | None = None
    probe_split: str = "validation"


class ExperienceAbstraction(BaseModel):
    experience_id: str
    preconditions: list[str] = Field(default_factory=list)
    situation: str = ""
    decision_point: str = ""
    action_pattern: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    outcome: str = ""
    failure_mode: str | None = None
    correction: str | None = None
    reusable_principle: str = ""
    applicability_scope: list[str] = Field(default_factory=list)
    not_applicable_when: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    provenance: list[str] = Field(default_factory=list)
    recurrence: int = 1
    procedural: bool = False
    declarative: bool = True
    runtime_control: bool = False
    broad_transfer: bool = False


class AssimilationDecision(BaseModel):
    experience_id: str
    carrier: Carrier
    confidence: float
    expected_utility: float
    expected_interference: float
    update_spec: JSONDict = Field(default_factory=dict)
    rationale: str = ""
    policy_name: str = "heuristic"


class AcquisitionProposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    mode: str
    task_ids: list[str] = Field(default_factory=list)
    generated_specs: list[JSONDict] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    metadata: JSONDict = Field(default_factory=dict)


class UtilityRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = Field(default_factory=time.time)
    agent_version_before: str
    agent_version_after: str
    experience_id: str
    carrier: Carrier
    state_features: list[float] = Field(default_factory=list)
    candidate_features: list[float] = Field(default_factory=list)
    realized_utility: float
    future_gain: float
    retention_delta: float
    cost: ExperienceCost = Field(default_factory=ExperienceCost)
    split_guard: str = "utility_train"
    extras: JSONDict = Field(default_factory=dict)


class ModelCallRecord(BaseModel):
    call_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    backend: str
    model_name: str
    messages: list[JSONDict]
    output_text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    logprobs_summary: JSONDict = Field(default_factory=dict)
    stop_reason: str = ""
    extra: JSONDict = Field(default_factory=dict)
