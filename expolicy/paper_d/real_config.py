"""D-v1.1 real-pilot config. It does not loosen the locked D-v1 file."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from expolicy.paper_d.constants import (
    DELTA,
    PROTOCOL_ID,
    REPO_ROOT,
    RETENTION_WEIGHT,
    ProtocolError,
)
from expolicy.r9_lab.io_util import sha256_json

REAL_PROTOCOL = "D-v1.1"


class RealConfigError(ProtocolError):
    pass


def load_real_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or (REPO_ROOT / "configs" / "paper_d_real_v11.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RealConfigError("real config must be a mapping")
    validate_real_config(raw)
    loaded = dict(raw)
    loaded["config_path"] = str(config_path)
    loaded["config_hash"] = sha256_json(raw)
    return loaded


def validate_real_config(raw: Mapping[str, Any]) -> None:
    if raw.get("protocol_version") != REAL_PROTOCOL:
        raise RealConfigError("real protocol version must be D-v1.1")
    if raw.get("protocol_version") == PROTOCOL_ID:
        raise RealConfigError("D-v1.1 cannot reuse the D-v1 protocol id as its version")
    if not _is_number(raw.get("delta"), DELTA) or not _is_number(raw.get("retention_weight"), RETENTION_WEIGHT):
        raise RealConfigError("delta and retention weight stay at the D-v1 values")
    if not _is_number(raw.get("retention_tolerance"), 0.0):
        raise RealConfigError("retention tolerance stays 0 unless a new registration says otherwise")
    if raw.get("measurement_kind") != "sample" or raw.get("allow_population") is not False:
        raise RealConfigError("real rows are samples")
    cap = raw.get("pilot_episode_cap_all_inclusive")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap != 6000:
        raise RealConfigError("the first-stage cap is 6000 episodes")
    if raw.get("formal_claim_from_pilot") is not False or raw.get("report_correlation") is not False:
        raise RealConfigError("the first real pilot does not make a formal claim or a correlation")
    if list(raw.get("primary_actions") or []) != ["DISCARD", "EPISODIC_REPLAY", "SEMANTIC_MEMORY"]:
        raise RealConfigError("primary actions are frozen")
    model = raw.get("model") or {}
    if not model.get("local_path") or not model.get("served_name"):
        raise RealConfigError("model path and served name must come from the server inventory")


def _is_number(value: Any, expected: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return float(value) == expected
