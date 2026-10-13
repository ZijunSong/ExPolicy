"""Public protocol loader. The original C-v1 load_config() stays locked."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from expolicy.paper_c.constants import REPO_ROOT


class PublicConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class PublicConfig:
    raw: Mapping[str, Any]
    root: Path
    config_path: Path

    def stage_cap(self, stage: str) -> int:
        stages = self.raw["stages"]
        if stage in {"c0-screen", "c0_screen"}:
            return int(stages["c0_screen"]["cap_episodes"])
        if stage in {"c0-confirm", "c0_confirm"}:
            return int(stages["c0_confirm"]["cap_episodes"])
        key = {"runtime": "runtime", "c1": "c1", "c2": "c2", "pilot": "pilot"}[stage]
        block = stages[key]
        return int(block["cap_episodes"] if isinstance(block, dict) else block)

    def output_dir(self, env: str, run_id: str) -> Path:
        path = self.root / str(self.raw["outputs"]["root"]) / env / f"run_{run_id}"
        return path


def load_public_config(path: Path | None = None) -> PublicConfig:
    config_path = path or (REPO_ROOT / "configs" / "paper_c_public.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PublicConfigError("public config must be a mapping")
    validate_public_config(raw)
    return PublicConfig(raw=raw, root=REPO_ROOT, config_path=config_path)


def validate_public_config(raw: Mapping[str, Any]) -> None:
    if raw.get("protocol_id") != "C-v1.1-public":
        raise PublicConfigError("protocol_id must be C-v1.1-public")
    if raw.get("parent_protocol") != "C-v1":
        raise PublicConfigError("public protocol must keep C-v1 as its parent")
    mode = str(raw.get("mode"))
    if mode not in {"harness", "public_pilot", "public_confirm"}:
        raise PublicConfigError(f"unknown mode {mode}")
    envs = [str(item) for item in raw.get("registered_environments") or []]
    if "kg" not in envs or "os" not in envs:
        raise PublicConfigError("kg and os must both be pre-registered")
    if mode == "public_confirm" and len(set(envs)) < 2:
        raise PublicConfigError("confirmation mode needs two registered environments")
    metric = raw.get("metric") or {}
    if float(metric.get("retention_weight")) != 0.5:
        raise PublicConfigError("retention weight stays 0.5")
    if bool(metric.get("future_test_in_selection")):
        raise PublicConfigError("future scores cannot enter selection")
    if metric.get("primary_integral") != "trapezoid" or metric.get("sensitivity_integral") != "step":
        raise PublicConfigError("integral rules are frozen")
    exploration = raw.get("exploration") or {}
    if exploration.get("rule") != "epsilon" or float(exploration.get("epsilon")) != 0.1:
        raise PublicConfigError("exploration stays epsilon 0.1")
    if bool(exploration.get("ucb_enabled")):
        raise PublicConfigError("UCB is not registered")
    actor = raw.get("actor") or {}
    if actor.get("model_id") != "Q2" or float(actor.get("temperature")) != 0.0:
        raise PublicConfigError("actor is frozen to Qwen3-14B at temperature 0")
    assim = raw.get("assimilator") or {}
    if bool(assim.get("online_algorithm_edit")) or bool(assim.get("char_cap_is_exact_tokens")):
        raise PublicConfigError("E0 cannot edit itself online or call the character cap an exact token count")
    if int(assim.get("max_entries")) != 256 or int(assim.get("top_k")) != 4:
        raise PublicConfigError("E0 capacity and top_k are frozen")
