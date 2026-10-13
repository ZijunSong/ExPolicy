"""Records for rules, tasks, traces, and episodes. Execution never reads gold."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from expolicy.paper_b.constants import FALLBACK_BASE


def _atoms(trigger: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = trigger.get("all") or []
    return [dict(atom) for atom in raw]


@dataclass
class Budget:
    max_tool_calls: int = 4
    max_steps: int = 6
    max_tokens: int = 2048
    max_interventions: int = 2

    def to_dict(self) -> dict[str, int]:
        return {
            "max_tool_calls": self.max_tool_calls,
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "max_interventions": self.max_interventions,
        }


@dataclass(frozen=True)
class Document:
    terms: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Task:
    task_id: str
    split: str
    source_group: str
    entity_key: str
    db_object: str
    family: str
    question: str
    gold_answer: str
    corpus: tuple[Document, ...]

    def public_dict(self) -> dict[str, Any]:
        """Fields a learner may see. The answer and the corpus stay out."""
        return {
            "task_id": self.task_id,
            "split": self.split,
            "source_group": self.source_group,
            "entity_key": self.entity_key,
            "db_object": self.db_object,
            "question": self.question,
        }


@dataclass
class Rule:
    rule_id: str
    family: str
    trigger: dict[str, Any]
    action_template: str
    parameters: dict[str, Any] = field(default_factory=dict)
    fallback: str = FALLBACK_BASE
    support_trace_ids: list[str] = field(default_factory=list)
    counterexample_trace_ids: list[str] = field(default_factory=list)
    validation_record: str = ""
    revision_count: int = 0

    def executable(self) -> dict[str, Any]:
        return {
            "trigger": {"all": _atoms(self.trigger)},
            "action_template": self.action_template,
            "parameters": dict(self.parameters),
            "fallback": self.fallback,
        }

    def to_dict(self) -> dict[str, Any]:
        body = self.executable()
        body.update(
            {
                "rule_id": self.rule_id,
                "family": self.family,
                "support_trace_ids": list(self.support_trace_ids),
                "counterexample_trace_ids": list(self.counterexample_trace_ids),
                "validation_record": self.validation_record,
                "revision_count": self.revision_count,
            }
        )
        return body


@dataclass
class TraceStep:
    features_before: dict[str, Any]
    action_template: str
    parameters: dict[str, Any]
    observation_redacted: str
    history_hash: str = ""
    snapshot_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "features_before": dict(self.features_before),
            "action_template": self.action_template,
            "parameters": dict(self.parameters),
            "observation_redacted": self.observation_redacted,
            "history_hash": self.history_hash,
            "snapshot_hash": self.snapshot_hash,
        }


@dataclass
class Trace:
    trace_id: str
    task_id: str
    source_group: str
    split: str
    family: str
    continuation: str
    success: bool
    steps: list[TraceStep]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "source_group": self.source_group,
            "split": self.split,
            "family": self.family,
            "continuation": self.continuation,
            "success": self.success,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class TracePair:
    pair_id: str
    alignment: str
    quality: float
    task_id: str
    source_group: str
    family: str
    diverge_index: int
    features_before: dict[str, Any]
    success_action: str
    success_params: dict[str, Any]
    failure_action: str
    success_trace_id: str
    failure_trace_id: str
    observation_redacted: str = ""
    swapped_from: str = ""
    prefix_hash: str = ""
    alignment_basis: str = "features"

    def inducer_view(self) -> dict[str, Any]:
        """Alignment features and actions only. No answer, question, family, or task id."""
        return {
            "pair_id": self.pair_id,
            "alignment": self.alignment,
            "quality": self.quality,
            "diverge_index": self.diverge_index,
            "features_before": dict(self.features_before),
            "success_action": self.success_action,
            "success_params": dict(self.success_params),
            "failure_action": self.failure_action,
            "success_trace_id": self.success_trace_id,
            "failure_trace_id": self.failure_trace_id,
            "prefix_hash": self.prefix_hash,
            "alignment_basis": self.alignment_basis,
        }

    def to_dict(self) -> dict[str, Any]:
        body = self.inducer_view()
        body.update(
            {
                "task_id": self.task_id,
                "source_group": self.source_group,
                "family": self.family,
                "observation_redacted": self.observation_redacted,
                "swapped_from": self.swapped_from,
            }
        )
        return body


@dataclass
class Episode:
    task_id: str
    answer: str
    success: bool
    steps: list[TraceStep]
    triggered_rules: list[str]
    matched_rules: list[str]
    modified_actions: int
    tool_calls: int
    tokens: int
    interventions: int
    fallbacks: int
    infra_status: str
    action_templates: list[str]
    text_context_sha256: str = ""
    fire_features: list[dict[str, Any]] = field(default_factory=list)
