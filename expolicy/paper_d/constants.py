"""Frozen D-v1 constants. Thresholds live in configs/paper_d.yaml."""

from __future__ import annotations

from pathlib import Path

PROTOCOL_ID = "D-v1"
GUIDANCE_DATE = "2026-10-09"
PROTOCOL_VERSION = "D-v1"
IMPLEMENTATION_DATE = "2026-10-13"

REPO_ROOT = Path(__file__).resolve().parents[2]

# Practical selection advantage. A project decision scale, not a venue bar.
DELTA = 0.02
RETENTION_WEIGHT = 0.5
PANEL_ALPHA = 0.05
NOOP_OFFSET = 0.02
MAX_INFRA_RETRIES = 2
MIN_POLICY_GROUPS = 4
MIN_CONFIGS_FOR_CORRELATION = 4

WEIGHT_UNIFORM = "uniform"
WEIGHT_MACRO = "macro"

MEASUREMENT_POPULATION = "population"
MEASUREMENT_SAMPLE = "sample"

PROBE_FUTURE = "future"
PROBE_RETENTION = "retention"

STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_INFRA = "infra_failure"
STATUS_TIMEOUT = "timeout"
STATUS_REJECTED = "rejected"
STATUS_NOT_STARTED = "not_started"
INFRA_STATUSES: tuple[str, ...] = (
    STATUS_OK,
    STATUS_MISSING,
    STATUS_INFRA,
    STATUS_TIMEOUT,
    STATUS_REJECTED,
    STATUS_NOT_STARTED,
)

BRANCH_BASE = "BASE"
BRANCH_DISCARD = "DISCARD"
BRANCH_REJECTED = "REJECTED_EDIT"
NOOP_BRANCHES: tuple[str, ...] = (BRANCH_BASE, BRANCH_DISCARD, BRANCH_REJECTED)
NOOP_ACTIONS: frozenset[str] = frozenset({BRANCH_BASE, BRANCH_DISCARD, "NO_UPDATE", "no-update"})

CONTENT_REAL = "real"
CONTENT_EMPTY = "empty"
CONTENT_WRONG = "matched-wrong"
CONTENT_NO_UPDATE = "no-update"

# Diagnostic states. More than one may apply to the same configuration.
MEASUREMENT_INVALID = "MEASUREMENT_INVALID"
MEASUREMENT_INCONCLUSIVE = "MEASUREMENT_INCONCLUSIVE"
OPERATOR_INVALID = "OPERATOR_INVALID"
SPLIT_INSUFFICIENT = "SPLIT_INSUFFICIENT"
NOT_TESTED = "NOT_TESTED"
RETENTION_UNCERTAIN = "RETENTION_UNCERTAIN"
UPDATE_INEFFECTIVE = "UPDATE_INEFFECTIVE"
CONTENT_UNSUPPORTED = "CONTENT_UNSUPPORTED"
ENV_FIXED_SUFFICIENT = "ENV_FIXED_SUFFICIENT"
HETEROGENEOUS_BUT_UNPREDICTED = "HETEROGENEOUS_BUT_UNPREDICTED"
ACTIONABLE_HEADROOM = "ACTIONABLE_HEADROOM"
INCONCLUSIVE = "INCONCLUSIVE"

ESTIMATE_NAIVE = "estimate_naive_gap"
ESTIMATE_CROSSFIT = "crossfit_selection_gain"
ESTIMATE_BOUND = "bound_panel_headroom"
ESTIMATE_POLICY = "fit_simple_observable_policy"
HEADROOM_ESTIMATES: tuple[str, ...] = (ESTIMATE_NAIVE, ESTIMATE_CROSSFIT, ESTIMATE_BOUND)

CONTROL_EXACT_TIE = "exact_tie"
CONTROL_NOISY_TIE = "noisy_tie"
CONTROL_GLOBAL_WINNER = "global_winner"
CONTROL_ENV_ONLY = "env_only_winner"
CONTROL_OBSERVABLE = "observable_crossover"
CONTROL_HIDDEN = "hidden_crossover"
CONTROL_DELAYED = "delayed_synergy"
KNOWN_CONTROLS: tuple[str, ...] = (
    CONTROL_EXACT_TIE,
    CONTROL_NOISY_TIE,
    CONTROL_GLOBAL_WINNER,
    CONTROL_ENV_ONLY,
    CONTROL_OBSERVABLE,
    CONTROL_HIDDEN,
    CONTROL_DELAYED,
)

LONG_TERM_UNCOVERED = "未覆盖长期结论"

ENV_FEATURES: tuple[str, ...] = ()
EXPERIENCE_FEATURES: tuple[str, ...] = (
    "experience_success_rate",
    "experience_count",
    "experience_error_rate",
    "experience_conflict_rate",
)
STATE_FEATURES: tuple[str, ...] = (
    "memory_size",
    "n_episodic",
    "n_semantic",
    "skill_size",
    "capability_shadow_hist",
    "capability_retention_hist",
    "visible_state",
)
FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {
        "hidden_state",
        "outcome_before",
        "outcome_after",
        "retention_before",
        "retention_after",
        "utility",
        "gain",
        "gold",
        "future",
        "future_score",
        "label",
        "best_action",
    }
)

CLONE_RESOURCES: tuple[str, ...] = (
    "state_files",
    "external_db",
    "filesystem",
    "kg",
    "memory_index",
    "retrieval_cache",
)
PROVENANCE_FIELDS: tuple[str, ...] = (
    "prompt_hash",
    "tool_schema_hash",
    "task_ids",
    "env_reset_id",
    "model_hash",
    "tokenizer_hash",
    "concurrency",
    "inference_config_hash",
)

LOG_FIELDS: tuple[str, ...] = (
    "configuration_id",
    "protocol_version",
    "model_hash",
    "env_hash",
    "state_distribution_id",
    "source_group",
    "batch_id",
    "action",
    "accepted",
    "changed",
    "task_id",
    "probe_role",
    "outcome_before",
    "outcome_after",
    "retention_before",
    "retention_after",
    "repeat_id",
    "cost",
    "infra_status",
)
EXTRA_FIELDS: tuple[str, ...] = (
    "env",
    "state_id",
    "source_template",
    "content_mode",
    "budget",
    "sequence_id",
    "measurement_kind",
    "prompt_hash",
    "cached_baseline",
    "control_name",
    "legacy_u",
)
MATRIX_FIELDS: tuple[str, ...] = LOG_FIELDS + EXTRA_FIELDS

LEGACY_VALUE_KEYS: tuple[str, ...] = (
    "U",
    "utility",
    "d_shadow",
    "d_retention",
    "retention_penalty",
    "shadow",
    "retention",
    "legacy",
)


class ProtocolError(ValueError):
    """A D-v1 panel, split, or measurement contract was violated."""
