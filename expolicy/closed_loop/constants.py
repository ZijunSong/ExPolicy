"""Frozen R7 closed-loop constants. Do not tune from final test."""

from __future__ import annotations

from pathlib import Path

C1 = "bridge_decomposition"
C2 = "comparison_alignment"
C3 = "long_chain_search"
CAPS = (C1, C2, C3)

DISCARD = "DISCARD"
EPISODIC_MEMORY = "EPISODIC_MEMORY"
SEMANTIC_MEMORY = "SEMANTIC_MEMORY"
ACTIVATE_EXISTING_SKILL = "ACTIVATE_EXISTING_SKILL"
ASSIM_ACTIONS = (DISCARD, EPISODIC_MEMORY, SEMANTIC_MEMORY, ACTIVATE_EXISTING_SKILL)

DATASETS = ("hotpotqa", "2wiki", "musique")
SPLIT_SEED = 20260819

ROOT = Path(__file__).resolve().parents[2]
R7_ROOT = ROOT / "outputs" / "r7_closed_loop"
DEFAULT_MODEL = "/data/ppnm/models/Qwen2.5-3B-Instruct"
DEFAULT_EMBEDDER = "/data/ppnm/models/all-MiniLM-L6-v2"
DEFAULT_RETRIEVER_URL = "http://127.0.0.1:18080"
INDEX_PATH = ROOT / "data" / "wiki18" / "e5_HNSW64.index"
CORPUS_PATH = ROOT / "data" / "wiki18" / "corpus" / "wiki-18.unpacked.jsonl"
RETRIEVER_MODEL = "/data/ppnm/models/e5-base-v2"

TARGET_SIZES = {
    "calibration_per_cap": 60,
    "acquisition_pool_per_cap": 240,
    "shadow_reward_probe_per_cap": 60,
    "retention_probe": 60,
    "future_test_per_dataset": 150,
}

FULL_EVAL_AT = (0, 16, 32, 48)
DEFAULT_ROUNDS = 6
DEFAULT_BATCH = 8
DEFAULT_BUDGET = 48

ACQ_FEATURE_DIM = 11
ASSIM_FEATURE_DIM = 12

FORBIDDEN_FEATURE_KEYS = {
    "gold",
    "gold_answer",
    "future_test",
    "future_probe",
    "task_id",
    "dataset",
    "dataset_id",
    "dataset_name",
    "frontier",
    "q*p",
    "rasa",
    "ava",
}

CONDITION_SPECS = {
    "expolicy": {
        "acquisition": "linear_thompson",
        "assimilation": "linear_thompson",
        "feedback": "true",
        "all_skills_upfront": False,
    },
    "passive_noop": {
        "acquisition": "random",
        "assimilation": "fixed_discard",
        "feedback": "true",
        "all_skills_upfront": False,
    },
    "passive_episodic": {
        "acquisition": "random",
        "assimilation": "fixed_episodic",
        "feedback": "true",
        "all_skills_upfront": False,
    },
    "passive_count_skill": {
        "acquisition": "random",
        "assimilation": "count_only",
        "feedback": "true",
        "all_skills_upfront": False,
    },
    "random_joint": {
        "acquisition": "random",
        "assimilation": "random",
        "feedback": "true",
        "all_skills_upfront": False,
    },
    "all_skills": {
        "acquisition": "random",
        "assimilation": "fixed_discard",
        "feedback": "none",
        "all_skills_upfront": True,
    },
    "no_feedback": {
        "acquisition": "linear_thompson",
        "assimilation": "linear_thompson",
        "feedback": "none",
        "all_skills_upfront": False,
    },
    "reward_shuffle": {
        "acquisition": "linear_thompson",
        "assimilation": "linear_thompson",
        "feedback": "shuffle",
        "all_skills_upfront": False,
    },
    "assimilation_fixed": {
        "acquisition": "linear_thompson",
        "assimilation": "count_only",
        "feedback": "true",
        "all_skills_upfront": False,
    },
}
