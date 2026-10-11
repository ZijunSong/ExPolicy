"""Frozen R8-B constants. Do not retune after split hash is written."""

from __future__ import annotations

from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
ALFWORLD_DATA = Path("/data/ppnm/data/alfworld")
MODEL_PATH = Path("/data/ppnm/models/Qwen2.5-3B-Instruct")
MEMRL_ROOT = Path("/data/ppnm/third_party/memrl")
SKILLOPT_ROOT = Path("/data/ppnm/third_party/skillopt")

SPLIT_SEED = 20260821
MODEL_NAME = "Qwen2.5-3B-Instruct"

# Official ALFWorld TextWorld families (AlfredTWEnv task_types 1–6).
FAMILIES: tuple[str, ...] = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)

EASIER_FAMILIES: tuple[str, ...] = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
)

PER_FAMILY_QUOTA = {
    "calibration": 20,
    "acquisition_pool": 60,
    "assim_validation": 20,
    "shadow_reward_probe": 20,
    "future_test": 30,
}

RETENTION_TOTAL = 20

BUCKETS: tuple[str, ...] = (
    "calibration",
    "acquisition_pool",
    "assim_validation",
    "shadow_reward_probe",
    "retention_probe",
    "future_test",
)

# Foundation no-learning success band.
FOUNDATION_MIN = 0.10
FOUNDATION_MAX = 0.70

# Gate H / backend validation.
FAMILY_GAIN_MIN = 0.05
ORACLE_VS_FIXED_MIN = 0.03

GPUS = ("4", "5", "6", "7")
VLLM_PORT = 18090
EMBED_PORT = 18091
