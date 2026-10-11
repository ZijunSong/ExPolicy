"""Frozen R9 constants. Do not retune after split / model-screen hashes are written."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
LAB_ROOT = Path("/data/ppnm/third_party/LifelongAgentBench")
SKILLOPT_ROOT = Path("/data/ppnm/third_party/skillopt")
MEMRL_ROOT = Path("/data/ppnm/third_party/memrl")
OUTPUT_ROOT = ROOT / "outputs" / "r9_lab"

SPLIT_SEED = 20260822
ENVS: tuple[str, ...] = ("db", "os", "kg")

MODEL_SCREEN_N = 24
FOUNDATION_N = 80
FOUNDATION_MIN = 0.10
FOUNDATION_MAX = 0.70
PARSER_VALID_MIN = 0.90
INFRA_FAIL_MAX = 0.05
TARGET_SUCCESS = 0.40

VLLM_PORT = 18100
DTYPE = "bfloat16"
MAX_MODEL_LEN = 32768
TEMPERATURE = 0.0
TOP_P = 1.0
GPU_MEMORY_UTILIZATION = 0.72

ALLOWED_GPUS: tuple[str, ...] = ("4", "5", "6", "7")
STALL_SEC = 20 * 60
NEVER_PKILL_PYTHON = True

A0_DISCARD = "DISCARD"
A1_EPISODIC = "EPISODIC_REPLAY"
A2_SEMANTIC = "SEMANTIC_MEMORY"
A3_SKILLOPT = "SKILLOPT_EDIT"
ACTIONS: tuple[str, ...] = (A0_DISCARD, A1_EPISODIC, A2_SEMANTIC, A3_SKILLOPT)

EPISODIC_TOP_K = 4
SEMANTIC_TOP_K = 4

MATRIX_BATCHES_PER_ENV = 12
MATRIX_EXPERIENCES_PER_BATCH = 8

UTILITY_RETENTION_WEIGHT = 0.5  # U = d_shadow - 0.5 * max(0, -d_retention)

# Official LAB data paths (from configs/components/tasks/*.yaml). Do not change.
LAB_DATA_REL: dict[str, str] = {
    "db": "data/v0303/db_bench/processed/v0317_first500/entry_dict.json",
    "os": "data/v0303/os_interaction/processed/v0409_tcc_9_to_12_first500/entry_dict.json",
    "kg": (
        "data/v0303/knowledge_graph/processed/grailqa/"
        "v0417_tl2sc50_tl3sc50_tl4sc50_tl5sc50_tl6sc50_tl7sc50_tl8sc50_tl9sc46/entry_dict.json"
    ),
}

# Official max_round. Do NOT change.
MAX_ROUND: dict[str, int] = {"db": 3, "os": 5, "kg": 15}

LAB_CHAT_HISTORY_REL: dict[str, str] = {
    "db": "chat_history_items/standard/db_bench.json",
    "os": "chat_history_items/standard/os_interaction.json",
    "kg": "chat_history_items/standard/knowledge_graph.json",
}
LAB_TASK_NAME: dict[str, str] = {
    "db": "db_bench",
    "os": "os_interaction",
    "kg": "knowledge_graph",
}
LAB_SPARQL_URL = "http://127.0.0.1:3001/sparql"
LAB_KG_ONTOLOGY_REL = "data/v0121/knowledge_graph/ontology"
LAB_OS_COMMAND_TIMEOUT = 20
LAB_SYSTEM_PROMPT = "You are a helpful assistant."

HF_LAB_DATASET = "csyq/LifelongAgentBench"
HF_LAB_SUBSET: dict[str, str] = {
    "db": "db_bench",
    "os": "os_interaction",
    "kg": "knowledge_graph",
}
HF_ENDPOINTS: tuple[str, ...] = (
    "https://huggingface.co",
    "https://hf-mirror.com",
)

BUCKETS: tuple[str, ...] = (
    "model_screen",
    "calibration",
    "acquisition_pool",
    "assim_validation",
    "shadow_reward_probe",
    "retention_probe",
    "future_test",
)

FORMAL_BUCKETS: tuple[str, ...] = tuple(b for b in BUCKETS if b != "model_screen")

BUCKET_TARGETS: dict[str, int] = {
    "model_screen": MODEL_SCREEN_N,
    "calibration": FOUNDATION_N,
    "acquisition_pool": 200,
    "assim_validation": 60,
    "shadow_reward_probe": 60,
    "retention_probe": 40,
    "future_test": 80,
}

BUCKET_MIN: dict[str, int] = {
    "shadow_reward_probe": 40,
    "future_test": 60,
}

# Gate H (pre-registered; do not retune from results).
H1_UNIQUE_BEST_MIN = 0.15
H1_MIN_OPERATORS = 2
H2_ORACLE_VS_FIXED = 0.03
H2_PROBE_TASK_EQUIVALENTS = 2
H3_ORACLE_VS_ENV_FIXED = 0.02
H3_ENVS_WITH_DIVERSITY = 2
H3_WITHIN_ENV_UNIQUE_BEST = 0.20
H4_FULL_VS_MAJORITY_PP = 0.10
H4_FULL_VS_ENV_ONLY_PP = 0.05
H5_POSITIVE_U_BATCHES = 4
H5_MIN_NON_DISCARD_OPERATORS = 2

# Gate J (pre-registered).
J1_O3_VS_AXIS = 0.03
J2_SEED_FRACTION = 2 / 3
J5_SPEARMAN_MIN = 0.35
J5_TOP1_VS_RANDOM = 0.03

JOINT_SEEDS: tuple[int, ...] = (0, 1, 2)
JOINT_ROUNDS = 4
JOINT_M_CANDIDATES = 4
JOINT_MICRO_BATCH = 4
JOINT_FUTURE_BUDGETS: tuple[int, ...] = (0, 8, 16)

ALLOWED_DECISIONS: tuple[str, ...] = (
    "NOVELTY_HARD_STOP",
    "MODEL_LADDER_NO_FULL_PASS",
    "FOUNDATION_FAIL",
    "OPERATOR_SANITY_FAIL",
    "SUBSTRATE_FIXED_DOMINATES",
    "SUBSTRATE_ENV_ROUTER_ONLY",
    "SUBSTRATE_UNLEARNABLE",
    "HETEROGENEOUS_ASSIMILATION_PASS",
    "JOINT_IDENTIFIABILITY_FAIL",
    "JOINT_IDENTIFIABILITY_PASS",
)

GOLD_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "gold",
        "gold_sql",
        "gold_answer",
        "gold_action_plan",
        "gold_skill_ids",
        "gold_skill_id",
        "ground_truth",
        "ground_truth_sql",
        "ground_truth_command",
        "ground_truth_command_item",
        "answer_info",
        "answer_set",
        "answer_list",
        "answer_md5",
        "answer_direct",
        "evaluation_info",
        "reference_action",
        "reference_sql",
    }
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    hf_repo: str
    served_name: str
    thinking: str
    local_path: Path
    extra_thinking_kwarg: bool  # True => send enable_thinking=False


MODEL_LADDER: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="Q0",
        hf_repo="Qwen/Qwen3-30B-A3B-Instruct-2507",
        served_name="qwen3-30b-a3b-r9",
        thinking="non-thinking-only",
        local_path=Path("/data/ppnm/models/Qwen3-30B-A3B-Instruct-2507"),
        extra_thinking_kwarg=False,
    ),
    ModelSpec(
        model_id="Q1",
        hf_repo="Qwen/QwQ-32B",
        served_name="qwq-32b-r9",
        thinking="native-reasoning",
        local_path=Path("/data/ppnm/models/QwQ-32B"),
        extra_thinking_kwarg=False,
    ),
    ModelSpec(
        model_id="Q2",
        hf_repo="Qwen/Qwen3-14B",
        served_name="qwen3-14b-r9",
        thinking="enable_thinking False",
        local_path=Path("/data/ppnm/models/Qwen3-14B"),
        extra_thinking_kwarg=True,
    ),
    ModelSpec(
        model_id="Q3",
        hf_repo="Qwen/Qwen3-32B",
        served_name="qwen3-32b-r9",
        thinking="enable_thinking False",
        local_path=Path("/data/ppnm/models/Qwen3-32B"),
        extra_thinking_kwarg=True,
    ),
)

TIE_BREAK_ORDER: tuple[str, ...] = tuple(m.model_id for m in MODEL_LADDER)


def model_by_id(model_id: str) -> ModelSpec:
    for spec in MODEL_LADDER:
        if spec.model_id == model_id:
            return spec
    raise KeyError(model_id)


def vllm_port_for_model(model_id: str) -> int:
    try:
        return VLLM_PORT + TIE_BREAK_ORDER.index(model_id)
    except ValueError:
        return VLLM_PORT
