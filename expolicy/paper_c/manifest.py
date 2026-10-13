"""Freeze the assimilator manifest: version, prompt, validation data, and resource caps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from expolicy.paper_c.assimilator import FrozenAssimilator, paper_b_candidate
from expolicy.paper_c.constants import PROTOCOL_ID
from expolicy.r9_lab.io_util import read_json, sha256_json


def build_manifest(
    assimilator: FrozenAssimilator,
    *,
    validation_ids: Sequence[str],
    resource_limits: Mapping[str, Any],
    capacity: Mapping[str, Any],
    hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "protocol_id": PROTOCOL_ID,
        "frozen": True,
        "online_algorithm_edit": False,
        "assimilator": {
            "name": assimilator.name,
            "version": assimilator.version,
            "prompt": assimilator.prompt,
            "algorithm_hash": assimilator.algorithm_hash(),
            "c0_status": "NOT_RUN",
            "paper_environment": False,
        },
        "validation_data": {
            "ids": list(validation_ids),
            "hash": sha256_json(list(validation_ids)),
        },
        "capacity": dict(capacity),
        "hyperparameters": dict(hyperparameters),
        "resource_limits": dict(resource_limits),
        "not_certified": {
            "paper_b": paper_b_candidate(),
            "episodic_replay": {"c0_status": "NOT_RUN", "usable_as_c_assimilator": False},
        },
    }
    body["content_hash"] = sha256_json({key: value for key, value in body.items() if key != "content_hash"})
    return body


def write_manifest(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    from expolicy.r9_lab.io_util import atomic_write_json

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return dict(payload)


def assert_manifest_intact(path: Path) -> dict[str, Any]:
    stored = read_json(path)
    expected = stored.get("content_hash")
    body = {key: value for key, value in stored.items() if key != "content_hash"}
    if sha256_json(body) != expected:
        raise RuntimeError("assimilator manifest hash mismatch")
    if stored.get("online_algorithm_edit"):
        raise RuntimeError("manifest allows online edits to F")
    return stored
