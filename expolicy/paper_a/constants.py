"""Frozen A-v1 constants. Thresholds live in configs/paper_a_v1.yaml."""

from __future__ import annotations

from pathlib import Path

PROTOCOL_ID = "A-v1"

REPO_ROOT = Path(__file__).resolve().parents[2]

A0_DISCARD = "DISCARD"
A1_EPISODIC = "EPISODIC_REPLAY"
A2_SEMANTIC = "SEMANTIC_MEMORY"
ACTIONS: tuple[str, ...] = (A0_DISCARD, A1_EPISODIC, A2_SEMANTIC)
EXCLUDED_ACTIONS: tuple[str, ...] = ("SKILLOPT_EDIT",)

BRANCH_BASE = "BASE"
BRANCH_DISCARD = "DISCARD"
BRANCH_REJECTED = "REJECTED_EDIT"
NOOP_BRANCHES: tuple[str, ...] = (BRANCH_BASE, BRANCH_DISCARD, BRANCH_REJECTED)
BRANCH_SENTINEL_PREFIX = "A_V1_BRANCH::"

SHADOW = "shadow_reward_probe"
RETENTION = "retention_probe"
FUTURE = "future_test"
PROBE_SPLITS: tuple[str, ...] = (SHADOW, RETENTION)
FORBIDDEN_FEATURE_SPLITS: frozenset[str] = frozenset(
    {SHADOW, RETENTION, FUTURE, "future", "shadow", "calibration_as_label"}
)

ROW_FIELDS: tuple[str, ...] = (
    "protocol_id",
    "model_hash",
    "env",
    "source_group",
    "batch_id",
    "state_hash_before",
    "action",
    "state_hash_after",
    "changed",
    "accepted",
    "probe_task_id",
    "eval_split",
    "y_before",
    "y_after",
    "delta_shadow",
    "delta_retention",
    "utility",
    "tokens",
    "tool_calls",
    "elapsed_s",
    "infra_status",
)

EXTRA_ROW_FIELDS: tuple[str, ...] = (
    "retention_penalty",
    "utility_scope",
    "prompt_hash",
    "cache_key",
    "config_hash",
    "legacy",
)

EXPERIENCE_FEATURES: tuple[str, ...] = (
    "batch_success_rate",
    "error_rate",
    "parser_invalid_rate",
    "infra_fail_rate",
    "repetition_rate",
    "memory_similarity",
    "conflict_rate",
)
STATE_FEATURES: tuple[str, ...] = (
    "memory_size",
    "n_episodic",
    "n_semantic",
    "skill_size",
    "capability_shadow_hist",
    "capability_retention_hist",
)
CONTEXT_FEATURES: tuple[str, ...] = ("env_db", "env_os", "env_kg", "mean_steps")

FORBIDDEN_FEATURE_KEYS: frozenset[str] = frozenset(
    {
        "batch_id",
        "gold",
        "gold_sql",
        "gold_answer",
        "answer",
        "changed",
        "accepted",
        "shadow_after",
        "shadow_before",
        "shadow_score",
        "retention_after",
        "retention_before",
        "utility",
        "U",
        "delta_shadow",
        "delta_retention",
        "y_before",
        "y_after",
        "future_test",
        "future_success",
        "probe_label",
    }
)

PILOT_VARIANTS: tuple[str, ...] = ("B0", "B1", "B2", "B3", "M")
ATTRIBUTION_VARIANTS: tuple[str, ...] = ("B4", "B5", "A1", "A2", "A3", "A4")

STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_INFRA = "infra_failure"

DECISION_NOT_RUN = "NOT_RUN"
DECISION_GO = "GO"
DECISION_NO_GO = "NO_GO"
DECISION_INCONCLUSIVE = "INCONCLUSIVE"
DECISION_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
SPACE_GO = "A-SPACE-GO"
SPACE_NO_GO = "A-SPACE-NO_GO"
AUDIT_PASS = "PASS"
AUDIT_FAIL = "FAIL_SYSTEMATIC"
AUDIT_INCONCLUSIVE = "MEASUREMENT_INCONCLUSIVE"
