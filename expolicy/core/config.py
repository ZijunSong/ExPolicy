"""YAML config loading with OmegaConf + pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from expolicy.core.errors import ConfigError
from expolicy.core.state import HarnessConfig


class ModelConfig(BaseModel):
    backend: str = "transformers"
    path: str = "/data/ppnm/models/Qwen2.5-3B-Instruct"
    name: str = ""
    base_url: str = ""
    api_key: str = "EMPTY"
    dtype: str = "bfloat16"
    device: str | None = None
    max_new_tokens: int = 256
    embedder_path: str = "/data/ppnm/models/all-MiniLM-L6-v2"


class EnvironmentConfig(BaseModel):
    kind: str = "search"
    seed: int = 13
    world_path: str = "data/world/helion.json"
    top_k: int = 5


class BudgetConfig(BaseModel):
    max_tasks: int = 50
    max_env_steps: int = 10_000
    max_tokens: int = 5_000_000
    max_gpu_seconds: float = 86_400.0
    acquisition_batch: int = 8


class ExperimentConfig(BaseModel):
    run_name: str = "base"
    seed: int = 13
    output_dir: str = "outputs"
    model: ModelConfig = Field(default_factory=ModelConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    acquisition: str = "passive"
    assimilation: str = "heuristic"
    enable_harness_evolution: bool = True
    enable_weight_update: bool = True
    probe_limit: int = 12
    stage: int | None = 0
    split: str = "validation"
    variant: str = "B0"
    extra: dict[str, Any] = Field(default_factory=dict)


def load_yaml(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigError(f"config not found: {cfg_path}")
    loaded = OmegaConf.load(str(cfg_path))
    return OmegaConf.to_container(loaded, resolve=True)  # type: ignore[return-value]


def load_experiment(path: str | Path) -> ExperimentConfig:
    raw = load_yaml(path)
    try:
        return ExperimentConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"invalid experiment config {path}: {exc}") from exc
