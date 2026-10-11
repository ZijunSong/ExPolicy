"""Launch official SkillOpt train.py / eval_only.py against the frozen R8 split."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from expolicy.r8.constants import ALFWORLD_DATA, MODEL_NAME, SKILLOPT_ROOT


def skillopt_env(llm_base: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SKILLOPT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["ALFWORLD_DATA"] = str(ALFWORLD_DATA)
    env["OPENAI_COMPATIBLE_BASE_URL"] = llm_base
    env["OPENAI_COMPATIBLE_API_KEY"] = "EMPTY"
    env["OPENAI_COMPATIBLE_MODEL"] = MODEL_NAME
    env["OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL"] = llm_base
    env["TARGET_OPENAI_COMPATIBLE_BASE_URL"] = llm_base
    env["OPTIMIZER_OPENAI_COMPATIBLE_MODEL"] = MODEL_NAME
    env["TARGET_OPENAI_COMPATIBLE_MODEL"] = MODEL_NAME
    # vLLM --max-model-len 4096 rejects max_tokens > 4096 (SkillOpt default 16384/8000).
    # 3B vLLM max_model_len=4096; SkillOpt analyst prompts are ~3k tokens.
    env["OPENAI_COMPATIBLE_MAX_TOKENS"] = "256"
    env["OPTIMIZER_OPENAI_COMPATIBLE_MAX_TOKENS"] = "256"
    env["TARGET_OPENAI_COMPATIBLE_MAX_TOKENS"] = "256"
    return env


def run_skillopt_train(
    *,
    split_dir: Path,
    out_root: Path,
    llm_base: str,
    python_bin: str,
    num_epochs: int,
    batch_size: int,
    workers: int,
    seed: int,
    log_path: Path,
) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        str(SKILLOPT_ROOT / "scripts" / "train.py"),
        "--config",
        str(SKILLOPT_ROOT / "configs" / "alfworld" / "default.yaml"),
        "--optimizer_backend",
        "openai_compatible",
        "--target_backend",
        "openai_compatible",
        "--env",
        "alfworld",
        "--skill_init",
        str(SKILLOPT_ROOT / "skillopt" / "envs" / "alfworld" / "skills" / "initial.md"),
        "--cfg-options",
        "model.backend=openai_compatible",
        "--optimizer_model",
        MODEL_NAME,
        "--target_model",
        MODEL_NAME,
        "--split_dir",
        str(split_dir),
        "--out_root",
        str(out_root),
        "--num_epochs",
        str(num_epochs),
        "--batch_size",
        str(batch_size),
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--max_api_workers",
        str(workers),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(SKILLOPT_ROOT),
            env=skillopt_env(llm_base),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(proc.returncode)


def run_skillopt_eval(
    *,
    split_dir: Path,
    skill_path: Path,
    out_root: Path,
    llm_base: str,
    python_bin: str,
    workers: int,
    seed: int,
    log_path: Path,
) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        str(SKILLOPT_ROOT / "scripts" / "eval_only.py"),
        "--config",
        str(SKILLOPT_ROOT / "configs" / "alfworld" / "default.yaml"),
        "--target_backend",
        "openai_compatible",
        "--env",
        "alfworld",
        "--cfg-options",
        "model.backend=openai_compatible",
        "--target_model",
        MODEL_NAME,
        "--skill",
        str(skill_path),
        "--split",
        "test",
        "--split_dir",
        str(split_dir),
        "--out_root",
        str(out_root),
        "--seed",
        str(seed),
        "--workers",
        str(workers),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(SKILLOPT_ROOT),
            env=skillopt_env(llm_base),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(proc.returncode)


def find_best_skill(out_root: Path) -> Path | None:
    hits = list(out_root.rglob("best_skill.md"))
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]
