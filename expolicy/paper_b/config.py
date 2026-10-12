"""Load the locked B-v1 config. Thresholds are not retuned from outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from expolicy.paper_b.constants import CONDITIONS, HARNESS_LOCAL, PROTOCOL_ID, REPO_ROOT
from expolicy.paper_b.models import Budget
from expolicy.paper_b.registry import RuntimeRegistry, build_runtime_registry
from expolicy.paper_b.world import feasibility_counts, pilot_counts


class ConfigPinError(RuntimeError):
    pass


class PaperBConfig:
    def __init__(self, raw: dict[str, Any], path: Path):
        self.raw = raw
        self.path = path

    @property
    def protocol_id(self) -> str:
        return str(self.raw["protocol_id"])

    @property
    def harness(self) -> str:
        return str(self.raw["harness"])

    @property
    def split_seed(self) -> int:
        return int(self.raw["split_seed"])

    @property
    def min_gain(self) -> float:
        return float(self.raw["decision"]["min_gain"])

    @property
    def retention_tolerance(self) -> float:
        return float(self.raw["decision"]["retention_tolerance"])

    @property
    def resolution_tasks(self) -> int:
        return int(self.raw["decision"]["resolution_tasks"])

    @property
    def alpha(self) -> float:
        return float(self.raw["decision"]["alpha"])

    @property
    def bootstrap_replicates(self) -> int:
        return int(self.raw["decision"]["bootstrap_replicates"])

    @property
    def bootstrap_seed(self) -> int:
        return int(self.raw["decision"]["bootstrap_seed"])

    @property
    def min_trigger(self) -> float:
        return float(self.raw["decision"]["min_trigger"])

    @property
    def max_illegal(self) -> float:
        return float(self.raw["decision"]["max_illegal"])

    @property
    def inducer_temperature(self) -> float:
        return float(self.raw["inducer"]["temperature"])

    def budget(self) -> Budget:
        raw = self.raw["budget"]
        return Budget(
            max_tool_calls=int(raw["max_tool_calls"]),
            max_steps=int(raw["max_steps"]),
            max_tokens=int(raw["max_tokens"]),
            max_interventions=int(raw["max_interventions"]),
        )

    def registry(self) -> RuntimeRegistry:
        limits = self.raw["limits"]
        return build_runtime_registry(
            budget=self.budget(),
            max_atoms=int(limits["max_atoms"]),
            max_rules=int(limits["max_rules"]),
            max_rules_per_family=int(limits["max_rules_per_family"]),
            max_candidates_per_batch=int(limits["max_candidates_per_batch"]),
            max_revisions=int(limits["max_revisions"]),
            harness=self.harness,
        )

    def counts(self, profile: str) -> dict[str, dict[str, int]]:
        if profile == "pilot":
            return pilot_counts()
        if profile == "feasibility":
            return feasibility_counts()
        raise ConfigPinError(f"unknown profile {profile}")

    def resolved(self, profile: str) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "harness": self.harness,
            "profile": profile,
            "split_seed": self.split_seed,
            "conditions": list(CONDITIONS),
            "budget": self.budget().to_dict(),
            "limits": dict(self.raw["limits"]),
            "decision": dict(self.raw["decision"]),
            "counts": self.counts(profile),
            "model": dict(self.raw["model"]),
            "inducer": dict(self.raw["inducer"]),
            "note": "阈值来自实验指导，不在看结果之后下调。本地 harness 不是官方测试集。",
        }


def load_config(path: Path | None = None) -> PaperBConfig:
    config_path = path or (REPO_ROOT / "configs" / "paper_b.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigPinError("paper_b config must be a mapping")
    _validate(raw)
    return PaperBConfig(raw, config_path)


def _validate(raw: dict[str, Any]) -> None:
    if raw.get("protocol_id") != PROTOCOL_ID:
        raise ConfigPinError(f"protocol_id must be {PROTOCOL_ID}")
    if raw.get("harness") != HARNESS_LOCAL:
        raise ConfigPinError("this config locks the local search harness")
    if list(raw.get("conditions") or []) != list(CONDITIONS):
        raise ConfigPinError("conditions are locked to B0 B1 B2 B3 B4 B5 M")
    decision = raw["decision"]
    if float(decision["min_gain"]) != 0.03 or float(decision["retention_tolerance"]) != 0.01:
        raise ConfigPinError("practical floor and retention tolerance are locked")
    if int(decision["resolution_tasks"]) != 2:
        raise ConfigPinError("resolution is locked at two evaluation items")
    if bool(decision.get("retune_after_outcomes")):
        raise ConfigPinError("thresholds cannot be retuned after outcomes")
    limits = raw["limits"]
    if int(limits["max_atoms"]) != 3 or int(limits["max_rules"]) != 8:
        raise ConfigPinError("complexity caps are locked")
    if int(limits["max_rules_per_family"]) != 2 or int(limits["max_candidates_per_batch"]) != 4:
        raise ConfigPinError("family and candidate caps are locked")
    if int(limits["max_revisions"]) != 1:
        raise ConfigPinError("revision cap is locked at one")
    budget = raw["budget"]
    if int(budget["max_interventions"]) > int(budget["max_tool_calls"]):
        raise ConfigPinError("rule interventions cannot exceed the tool budget")
    profiles = raw["profiles"]
    if profiles["pilot"] != _plain(pilot_counts()) or profiles["feasibility"] != _plain(feasibility_counts()):
        raise ConfigPinError("profile sizes do not match the locked generators")
    if float(raw["inducer"]["temperature"]) != 0.0:
        raise ConfigPinError("the locked inducer temperature is 0; extra continuations are preregistered")
    if bool(raw["model"]["thinking"]):
        raise ConfigPinError("thinking stays off")


def _plain(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {split: {family: int(value) for family, value in families.items()} for split, families in counts.items()}
