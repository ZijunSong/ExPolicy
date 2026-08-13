"""Shared enumerations and lightweight typed aliases."""

from __future__ import annotations

from enum import Enum
from typing import Any, TypeAlias

JSONDict: TypeAlias = dict[str, Any]


class Carrier(str, Enum):
    DISCARD = "discard"
    EPISODIC = "episodic_memory"
    SEMANTIC = "semantic_memory"
    SKILL = "skill"
    HARNESS = "harness"
    WEIGHT = "weight"


class ExperienceStatus(str, Enum):
    ACQUIRED = "acquired"
    VERIFIED = "verified"
    ABSTRACTED = "abstracted"
    EVALUATED = "evaluated"
    ROUTED = "routed"
    USED = "used"
    REEVALUATED = "re_evaluated"
    MERGED = "merged"
    SUPERSEDED = "superseded"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


class FailureType(str, Enum):
    RETRIEVAL = "retrieval_failure"
    PLANNING = "planning_failure"
    EVIDENCE = "evidence_failure"
    REASONING = "reasoning_failure"
    TOOL = "tool_failure"
    VERIFICATION = "verification_failure"
    ANSWER_SYNTHESIS = "answer_synthesis_failure"
    NONE = "none"
    UNKNOWN = "unknown"


class CapabilityName(str, Enum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    ENTITY_DISAMBIGUATION = "entity_disambiguation"
    TEMPORAL = "temporal_reasoning"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    SOURCE_RELIABILITY = "source_reliability"
    LONG_CONTEXT = "long_context"
    QUERY_DECOMPOSITION = "query_decomposition"
    CROSS_DOCUMENT = "cross_document_synthesis"
    FAILURE_RECOVERY = "failure_recovery"


class FrontierBand(str, Enum):
    MASTERED = "mastered"
    LEARNING = "learning"
    FRONTIER = "frontier"
    UNKNOWN = "unknown"
    REGRESSED = "regressed"


class DataSplit(str, Enum):
    DEPLOYMENT = "deployment"
    ACQUISITION = "acquisition"
    VALIDATION = "validation"
    FUTURE = "future"
    RETENTION = "retention"


class EventType(str, Enum):
    TASK_STARTED = "TASK_STARTED"
    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"
    TOOL_CALL = "TOOL_CALL"
    TASK_FINISHED = "TASK_FINISHED"
    EXPERIENCE_CREATED = "EXPERIENCE_CREATED"
    EXPERIENCE_VERIFIED = "EXPERIENCE_VERIFIED"
    EXPERIENCE_ABSTRACTED = "EXPERIENCE_ABSTRACTED"
    ACQUISITION_DECISION = "ACQUISITION_DECISION"
    ASSIMILATION_DECISION = "ASSIMILATION_DECISION"
    MEMORY_UPDATED = "MEMORY_UPDATED"
    SKILL_CREATED = "SKILL_CREATED"
    HARNESS_PATCHED = "HARNESS_PATCHED"
    WEIGHT_UPDATED = "WEIGHT_UPDATED"
    UTILITY_MEASURED = "UTILITY_MEASURED"
    AGENT_VERSION_CREATED = "AGENT_VERSION_CREATED"
    CAPABILITY_DIAGNOSED = "CAPABILITY_DIAGNOSED"
    FAILURE_RECORDED = "FAILURE_RECORDED"
    CHECKPOINT_SAVED = "CHECKPOINT_SAVED"
    RUN_FAILED = "RUN_FAILED"
