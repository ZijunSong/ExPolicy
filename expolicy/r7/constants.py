"""Frozen R7 protocol constants. Thresholds are not tunable after freeze."""

from __future__ import annotations

from pathlib import Path

from expolicy.experiments.common import ROOT

R7_ROOT = ROOT / "outputs" / "r7_policy_generalization"
R7_DATA = ROOT / "data" / "r7_search_suite"
MODEL_ROOT = Path("/data/ppnm/models")

C0 = "single_hop_lookup"
C1 = "bridge_decomposition"
C2 = "comparison_alignment"
C3 = "long_chain_search"
PUBLIC_CAPS = (C1, C2, C3)
R7_CAPS = (C0, C1, C2, C3)

SINGLE_HOP = ("nq", "triviaqa", "popqa")
MULTI_HOP = ("hotpotqa", "2wiki", "musique", "bamboogle")
R7_DATASETS = SINGLE_HOP + MULTI_HOP

SPLIT_SEED = 20260815
CALIBRATION_PER_DATASET = 64
ACQUISITION_PER_DATASET = 300
SKILL_VALIDATION_PER_DATASET = 80
FUTURE_PER_DATASET = 150

MIN_SUPPORT = 8
BUDGET_DEFAULT = 48
CHECKPOINTS = (0, 8, 16, 24, 32, 40, 48)
ROUND_SIZE = 8
EDIT_BUDGET = 8
N_CONTENT_BATCHES = 12
CONTENT_STD_GATE = 0.03
CONTENT_RANGE_GATE = 0.08
H_SKILL_GATE = 0.05
H_POLICY_GATE = 0.03
H_CARRIER_GATE = 0.03
ORACLE_HEADROOM_GATE = 0.05
FRONTIER_MACRO_GATE = 0.03

COMPILER_INTERVENTIONAL = "r7.interventional_skill.v1"
COMPILER_SKILLOPT = "r7.skillopt_assimilator.v1"
OLD_COMPILER = "r5.3.public_skill.v1"

LEAK_KEYS = {
    "gold_answer",
    "gold_doc_ids",
    "gold",
    "probe_task_ids",
    "future_probe",
    "task_id",
    "dataset",
    "capability_oracle",
}

PRIMARY_MODELS = [
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen3-4B-Instruct-2507",
    "Qwen3-4B-Thinking-2507",
    "Phi-4-reasoning",
    "gpt-oss-20b",
]
SECONDARY_MODELS = [
    "Qwen3-1.7B",
    "gemma-4-E4B-it",
    "Qwen3-30B-A3B-Thinking-2507",
    "harness-1",
]
ALL_PROBE_MODELS = PRIMARY_MODELS + SECONDARY_MODELS
EMBEDDING_ONLY = {
    "all-MiniLM-L6-v2",
    "cross-encoder-ms-marco-MiniLM-L-6-v2",
    "e5-base-v2",
}

DEFAULT_GPUS = "4,5,6,7"
RETRIEVER_PORT = 18080
RETRIEVER_URL = f"http://127.0.0.1:{RETRIEVER_PORT}"
STALL_SEC = 15 * 60
MAX_RESTARTS = 8
POLL_SEC = 30

GPU_MODEL_HINT = {
    "4": ["Qwen2.5-3B-Instruct", "Qwen3-1.7B", "Qwen3-4B-Instruct-2507"],
    "5": ["Qwen2.5-7B-Instruct", "Qwen3-4B-Thinking-2507"],
    "6": ["Phi-4-reasoning", "gemma-4-E4B-it"],
    "7": ["gpt-oss-20b"],
}
