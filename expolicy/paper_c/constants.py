"""Frozen C-v1 constants. Thresholds live in configs/paper_c.yaml."""

from __future__ import annotations

from pathlib import Path

PROTOCOL_ID = "C-v1"
GUIDANCE_DATE = "2026-10-09"
IMPLEMENTATION_DATE = "2026-10-13"

REPO_ROOT = Path(__file__).resolve().parents[2]

UNIFORM = "Uniform"
COVERAGE = "Coverage"
BEST_STATIC = "Best-static"
FAILURE = "Failure"
DIFFICULTY = "Difficulty"
UNCERTAINTY = "Uncertainty"
CURRENT_REWARD = "Current-reward"
LEARNING_PROGRESS = "Learning-progress"
M = "M"
M_SHUFFLE = "M-shuffle"
M_NO_STATE = "M-no-state"

ALL_POLICIES: tuple[str, ...] = (
    UNIFORM,
    COVERAGE,
    BEST_STATIC,
    FAILURE,
    DIFFICULTY,
    UNCERTAINTY,
    CURRENT_REWARD,
    LEARNING_PROGRESS,
    M,
    M_SHUFFLE,
    M_NO_STATE,
)
PILOT_POLICIES: tuple[str, ...] = (UNIFORM, COVERAGE, BEST_STATIC, CURRENT_REWARD, M)
BASELINE_POLICIES: tuple[str, ...] = (UNIFORM, COVERAGE, BEST_STATIC, CURRENT_REWARD)

MODE_REAL = "real"
MODE_EMPTY = "empty"
MODE_WRONG = "matched-wrong"
MODE_NO_UPDATE = "no-update"
CONTENT_MODES: tuple[str, ...] = (MODE_NO_UPDATE, MODE_REAL, MODE_EMPTY, MODE_WRONG)

C_ASSIMILATION_FAIL = "C-ASSIMILATION-FAIL"
C_NO_HEADROOM = "C-NO-HEADROOM"
C_PREDICTION_FAIL = "C-PREDICTION-FAIL"
C_BASELINE_SUFFICIENT = "C-BASELINE-SUFFICIENT"
C_COST_FAIL = "C-COST-FAIL"
C_TRANSFER_FAIL = "C-TRANSFER-FAIL"
C_INCONCLUSIVE = "C-INCONCLUSIVE"
C_STATIC_FIRST = "C1-BEST-STATIC-FIRST"
C_PASS = "PASS"
C_NOT_RUN = "NOT_RUN"
C_HARNESS = "HARNESS_ONLY"
C_THRESHOLD_MISS = "THRESHOLD_MISS"

ENVS: tuple[str, ...] = ("db", "os", "kg", "synth")
HARNESS_ENV = "synth"

# Closed set. Post-rollout fields of the candidate being chosen are not features.
PUBLIC_CANDIDATE_KEYS: frozenset[str] = frozenset({"public_difficulty", "public_estimated_tools"})
HISTORICAL_FEATURES: tuple[str, ...] = ("hist_fail", "hist_progress", "hist_n")
FORBIDDEN_FEATURE_KEYS: frozenset[str] = frozenset(
    {
        "gold",
        "gold_sql",
        "gold_answer",
        "answer",
        "correct_answer",
        "success",
        "failure",
        "tool_error",
        "trajectory_length",
        "trajectory",
        "gain",
        "gain_target",
        "utility",
        "U",
        "shadow",
        "shadow_after",
        "shadow_before",
        "retention",
        "retention_after",
        "future",
        "future_score",
        "future_score_offline",
        "future_test",
        "y_before",
        "y_after",
        "delta_shadow",
        "delta_retention",
        "learnability",
        "hidden_capability",
    }
)
FORBIDDEN_BUCKETS: frozenset[str] = frozenset(
    {"future_test", "shadow_reward_probe", "retention_probe", "shadow_select", "shadow_eval"}
)

LOG_FIELDS: tuple[str, ...] = (
    "run_id",
    "sequence_seed",
    "round",
    "state_hash",
    "candidate_pool_hash",
    "observable_features",
    "selected_family",
    "selected_task_ids",
    "acquisition_episodes",
    "assimilation_accepted",
    "shadow_feedback_available",
    "gain_target",
    "actor_tokens",
    "teacher_tokens",
    "validation_calls",
    "cumulative_cost",
    "future_score_offline",
    "infra_status",
)

ACCOUNT_INTERACTION = "interaction"
ACCOUNT_COMPUTE = "compute"
ACCOUNT_OVERHEAD = "evaluation_overhead"

STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_INFRA = "infra_failure"

EXPLORATION_RULE = "epsilon"
LABEL_ROUTE = "offline_matrix_plus_online_calibration"
