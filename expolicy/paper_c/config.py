"""Load the locked C-v1 config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from expolicy.paper_c.constants import (
    BASELINE_POLICIES,
    EXPLORATION_RULE,
    GUIDANCE_DATE,
    LABEL_ROUTE,
    PILOT_POLICIES,
    PROTOCOL_ID,
    REPO_ROOT,
)


class ConfigPinError(RuntimeError):
    pass


@dataclass(frozen=True)
class PaperCConfig:
    raw: Mapping[str, Any]
    root: Path
    config_path: Path

    @property
    def protocol_id(self) -> str:
        return str(self.raw["protocol_id"])

    def output_root(self) -> Path:
        path = Path(str(self.raw["outputs"]["root"]))
        if not path.is_absolute():
            path = self.root / path
        return path


def load_config(path: Path | None = None) -> PaperCConfig:
    config_path = path or (REPO_ROOT / "configs" / "paper_c.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigPinError("paper_c config must be a mapping")
    cfg = PaperCConfig(raw=raw, root=REPO_ROOT, config_path=config_path)
    validate_config(cfg.raw)
    return cfg


def validate_config(raw: Mapping[str, Any]) -> None:
    if raw.get("protocol_id") != PROTOCOL_ID:
        raise ConfigPinError(f"protocol_id must be {PROTOCOL_ID}")
    if str(raw.get("guidance_date")) != GUIDANCE_DATE:
        raise ConfigPinError("guidance date is locked to the C protocol note")
    metric = raw.get("metric") or {}
    if float(metric.get("retention_weight")) != 0.5:
        raise ConfigPinError("retention weight is locked at 0.5")
    if bool(metric.get("future_test_in_selection")):
        raise ConfigPinError("future test scores cannot enter selection")
    if metric.get("primary_cost_axis") != "interaction":
        raise ConfigPinError("primary cost axis is interaction")
    exploration = raw.get("exploration") or {}
    if exploration.get("rule") != EXPLORATION_RULE or float(exploration.get("epsilon")) != 0.1:
        raise ConfigPinError("exploration is locked to epsilon 0.1")
    if bool(exploration.get("ucb_enabled")) or "ucb" not in set(raw.get("excluded_explorations") or []):
        raise ConfigPinError("UCB is not the pre-registered exploration rule")
    if raw.get("label_route") != LABEL_ROUTE:
        raise ConfigPinError("label route is locked to offline matrix plus online calibration")
    if tuple(raw.get("pilot_policies") or []) != PILOT_POLICIES:
        raise ConfigPinError(f"pilot policies are locked to {PILOT_POLICIES}")
    if set(PILOT_POLICIES) - {PILOT_POLICIES[-1]} != set(BASELINE_POLICIES):
        raise ConfigPinError("baseline set drifted from the pilot policies")
    if bool((raw.get("assimilator") or {}).get("online_algorithm_edit")):
        raise ConfigPinError("the assimilator algorithm stays frozen during acquisition")
    if list(raw.get("public_environments") or []):
        envs = [str(item) for item in raw["public_environments"]]
        if len(envs) < 2 or len(set(envs)) < 2:
            raise ConfigPinError("a paper run needs two distinct public environments")
    c0 = raw.get("c0") or {}
    gates = raw.get("gates") or {}
    if float(c0.get("min_gain")) != 0.03 or float(c0.get("retention_tolerance")) != 0.01:
        raise ConfigPinError("C0 thresholds are locked")
    if float(gates.get("min_aulc")) != 0.02 or float(gates.get("min_final_success")) != 0.02:
        raise ConfigPinError("pilot gain thresholds are locked")
    if float(gates.get("retention_tolerance")) != 0.01:
        raise ConfigPinError("pilot retention tolerance is locked")
    if int(gates.get("resample_budget")) != 1:
        raise ConfigPinError("resample budget is one pre-registered draw")
