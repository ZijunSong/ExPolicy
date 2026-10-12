"""Frozen B-v1 constants. Thresholds are copied from the pre-registered guidance."""

from __future__ import annotations

from pathlib import Path

PROTOCOL_ID = "B-v1"
INDUCER_VERSION = "b-v1-contrast-1"
REPO_ROOT = Path(__file__).resolve().parents[2]

B0 = "B0"
B1 = "B1"
B2 = "B2"
B3 = "B3"
B4 = "B4"
B5 = "B5"
M = "M"
CONDITIONS: tuple[str, ...] = (B0, B1, B2, B3, B4, B5, M)
EXPERIENCE_CONDITIONS: frozenset[str] = frozenset({B2, B4, B5, M})

FAMILY_COMPARE = "entity_time_compare"
FAMILY_RECOVERY = "empty_retrieval_recovery"
FAMILY_RETENTION = "retention_other"
TARGET_FAMILIES: tuple[str, ...] = (FAMILY_COMPARE, FAMILY_RECOVERY)

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_NEAR = "near"
SPLIT_CROSS = "cross"
SPLIT_RETENTION = "retention"
SPLIT_UNRELATED = "unrelated"
CONFIRMATION_SPLITS: tuple[str, ...] = (SPLIT_NEAR, SPLIT_CROSS, SPLIT_RETENTION)
TARGET_SPLITS: tuple[str, ...] = (SPLIT_NEAR, SPLIT_CROSS)

ALIGNMENT_SAME = "same_prefix"
ALIGNMENT_MATCHED = "matched_observation"
ALIGNMENT_MISMATCHED = "mismatched"
ALIGNMENT_UNRELATED = "unrelated_real"
ALIGNED: frozenset[str] = frozenset({ALIGNMENT_SAME, ALIGNMENT_MATCHED})

TEMPLATE_SEARCH = "search_entity_attribute"
TEMPLATE_REWRITE = "rewrite_empty_query"
TEMPLATE_BASE_SEARCH = "base_search"
TEMPLATE_BASE_ANSWER = "base_answer"
REGISTERED_TEMPLATES: tuple[str, ...] = (TEMPLATE_SEARCH, TEMPLATE_REWRITE)

REWRITE_TEMPLATES: tuple[str, ...] = (
    "{entity}",
    "{entity} {attribute}",
    "{entity} biography",
    "{entity} filmography",
)
LEARNED_REWRITE = "{entity} biography"
GENERIC_REWRITE = "{entity}"

TRIGGER_FEATURES: tuple[str, ...] = (
    "last_observation_empty",
    "has_comparison_cue",
    "attribute_kind",
    "n_entities",
    "evidence_filled",
    "search_count",
)
OBSERVABLE_FEATURES: tuple[str, ...] = TRIGGER_FEATURES + (
    "has_directed_by_span",
    "step_count",
)
TRIGGER_OPS: tuple[str, ...] = ("eq", "neq", "gte", "lte", "in")

FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {
        "gold_answer",
        "gold",
        "task_id",
        "won",
        "success",
        "family",
        "hidden_capability",
        "future_observation",
        "test_label",
        "correct_answer",
        "question",
    }
)
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "gold_answer",
    "hidden_capability",
    "future_observation",
    "test_label",
    "correct_answer",
    "os.system",
    "subprocess",
    "eval(",
    "exec(",
)

FALLBACK_BASE = "base_agent"
ROW_FIELDS: tuple[str, ...] = (
    "experience_pool_hash",
    "inducer_hash",
    "rule_hash",
    "runtime_hash",
    "condition",
    "source_group",
    "task_id",
    "split",
    "family",
    "triggered_rules",
    "modified_actions",
    "success",
    "retention_flag",
    "tool_calls",
    "tokens",
    "generation_cost",
    "validation_cost",
    "infra_status",
)

STATUS_OK = "ok"
STATUS_INFRA = "infra_failure"

B_GO = "B-GO"
B_NO_GO_CONTENT = "B-NO_GO-CONTENT"
B_NO_GO_MECHANISM = "B-NO_GO-MECHANISM"
B_INCONCLUSIVE = "B-INCONCLUSIVE"
B_NOT_RUN = "NOT_RUN"

HARNESS_LOCAL = "local_search"
ESI_OPERATORS = {
    TEMPLATE_SEARCH: ("FORM_QUERY", "SEARCH", "EXTRACT"),
    TEMPLATE_REWRITE: ("FORM_QUERY", "SEARCH", "EXTRACT"),
}
