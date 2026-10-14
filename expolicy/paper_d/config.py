"""Load the locked D-v1 config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from expolicy.paper_d.constants import (
    DELTA,
    GUIDANCE_DATE,
    PROTOCOL_ID,
    PROTOCOL_VERSION,
    REPO_ROOT,
    RETENTION_WEIGHT,
    WEIGHT_MACRO,
)


class ConfigPinError(RuntimeError):
    pass


@dataclass(frozen=True)
class PaperDConfig:
    raw: Mapping[str, Any]
    root: Path

    def inventory_roots(self) -> list[Path]:
        roots = []
        for item in self.raw.get("inventory_roots") or []:
            path = Path(str(item))
            if not path.is_absolute():
                path = self.root / path
            roots.append(path)
        return roots


def load_config(path: Path | None = None) -> PaperDConfig:
    config_path = path or (REPO_ROOT / "configs" / "paper_d.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigPinError("paper_d config must be a mapping")
    validate_config(raw)
    return PaperDConfig(raw=raw, root=REPO_ROOT)


def validate_config(raw: Mapping[str, Any]) -> None:
    if raw.get("protocol_id") != PROTOCOL_ID:
        raise ConfigPinError(f"protocol_id must be {PROTOCOL_ID}")
    if str(raw.get("guidance_date")) != GUIDANCE_DATE:
        raise ConfigPinError("guidance date is locked")
    if raw.get("protocol_version") != PROTOCOL_VERSION:
        raise ConfigPinError("protocol version is locked")
    if not _locked_float(raw.get("delta"), DELTA):
        raise ConfigPinError("delta is locked at 0.02")
    if not _locked_float(raw.get("retention_weight"), RETENTION_WEIGHT):
        raise ConfigPinError("retention weight is locked at 0.5")
    if raw.get("weight_mode") != WEIGHT_MACRO:
        raise ConfigPinError("primary weight mode is the environment macro average")
    if bool(raw.get("count_repeats_as_new_tasks")):
        raise ConfigPinError("repeats are noise, not new tasks")
    if bool(raw.get("clip_negative_selection_gain")):
        raise ConfigPinError("negative selection gains stay negative")


def _locked_float(value: Any, expected: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return float(value) == expected
