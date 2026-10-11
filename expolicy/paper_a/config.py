"""Load the locked A-v1 config and build the matrix cache key from the files on disk."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from expolicy.paper_a.constants import ACTIONS, EXCLUDED_ACTIONS, PROTOCOL_ID, REPO_ROOT
from expolicy.r9_lab.io_util import read_json, sha256_file, sha256_json
from expolicy.r9_lab.manifest import assert_manifest_unmodified


class ConfigPinError(RuntimeError):
    pass


@dataclass(frozen=True)
class PaperAConfig:
    raw: Mapping[str, Any]
    root: Path
    config_path: Path

    @property
    def protocol_id(self) -> str:
        return str(self.raw["protocol_id"])

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(self.raw["actions"])

    @property
    def environments(self) -> tuple[str, ...]:
        return tuple(self.raw["environments"])

    def path(self, key_path: str) -> Path:
        cursor: Any = self.raw
        for part in key_path.split("."):
            cursor = cursor[part]
        path = Path(str(cursor))
        if not path.is_absolute():
            path = self.root / path
        return path

    def output_root(self) -> Path:
        return self.path("outputs.root")


def load_config(path: Path | None = None, *, verify_files: bool = True) -> PaperAConfig:
    config_path = path or (REPO_ROOT / "configs" / "paper_a_v1.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigPinError("paper_a config must be a mapping")
    root = REPO_ROOT
    cfg = PaperAConfig(raw=raw, root=root, config_path=config_path)
    _validate_static(cfg)
    if verify_files:
        verify_assets(cfg)
    return cfg


def _validate_static(cfg: PaperAConfig) -> None:
    if cfg.protocol_id != PROTOCOL_ID:
        raise ConfigPinError(f"protocol_id must be {PROTOCOL_ID}")
    actions = list(cfg.actions)
    if tuple(actions) != ACTIONS:
        raise ConfigPinError(f"A-v1 actions are locked to {ACTIONS}")
    excluded = set(cfg.raw.get("excluded_actions") or [])
    if not set(EXCLUDED_ACTIONS).issubset(excluded):
        raise ConfigPinError("SKILLOPT_EDIT must stay outside the A-v1 action space")
    if set(actions) & excluded:
        raise ConfigPinError("an excluded action is also listed as an action")
    metric = cfg.raw["metric"]
    if float(metric["retention_weight"]) != 0.5:
        raise ConfigPinError("retention weight is locked at 0.5")
    if bool(metric["subtract_discard"]):
        raise ConfigPinError("discard recentering is not part of A-v1")
    if bool(metric["future_test_in_selection"]):
        raise ConfigPinError("future_test cannot enter selection")
    if bool(cfg.raw["measurement"]["repeats_are_independent_seeds"]):
        raise ConfigPinError("noise repeats are not independent seeds")
    if bool(cfg.raw["legacy"]["splice_into_matrix"]):
        raise ConfigPinError("legacy R9 cells must not be spliced into the A-v1 matrix")
    model = cfg.raw["model"]
    if float(model["temperature"]) != 0.0 or str(model["dtype"]) != "bfloat16":
        raise ConfigPinError("model dtype and temperature are locked")
    if bool(model["thinking"]):
        raise ConfigPinError("thinking must stay off")


def verify_assets(cfg: PaperAConfig) -> dict[str, str]:
    """Recompute hashes from the files. The yaml pins are the expected result of that check."""
    split_path = cfg.path("split.path")
    split = read_json(split_path)
    if bool(split.get("synthetic")):
        raise ConfigPinError("refusing a synthetic split for A-v1")
    content_hash = sha256_json({k: v for k, v in split.items() if k != "sha256"})
    file_hash = sha256_file(split_path)
    _expect(split.get("seed"), cfg.raw["split"]["expect_seed"], "split seed")
    _expect(split.get("sha256"), cfg.raw["split"]["expect_content_sha256"], "split content sha256")
    _expect(content_hash, split.get("sha256"), "recomputed split content sha256")
    _expect(file_hash, cfg.raw["split"]["expect_file_sha256"], "split file sha256")

    frozen_path = cfg.path("frozen_batches.path")
    frozen = read_json(frozen_path)
    frozen_hash = sha256_json({k: v for k, v in frozen.items() if k != "sha256"})
    _expect(frozen.get("sha256"), cfg.raw["frozen_batches"]["expect_sha256"], "frozen batch sha256")
    _expect(frozen_hash, frozen.get("sha256"), "recomputed frozen batch sha256")
    _expect(frozen.get("n_batches"), cfg.raw["frozen_batches"]["expect_n_batches"], "frozen batch count")
    if not frozen.get("frozen_before_utility"):
        raise ConfigPinError("frozen batches were not marked frozen_before_utility")

    manifest_path = cfg.path("model.manifest_path")
    assert_manifest_unmodified(manifest_path)
    manifest = read_json(manifest_path)
    _expect(manifest.get("content_hash"), cfg.raw["model"]["expect_manifest_content_hash"], "manifest hash")
    _expect(manifest.get("selected_model_id"), cfg.raw["model"]["model_id"], "selected model")
    _expect(manifest.get("dtype"), cfg.raw["model"]["dtype"], "manifest dtype")
    _expect(manifest.get("thinking_mode"), cfg.raw["model"]["thinking_mode"], "thinking mode")
    _expect(
        manifest.get("screen_split_sha256"),
        cfg.raw["model"]["expect_screen_split_sha256"],
        "manifest split file hash",
    )
    _expect(manifest.get("screen_split_sha256"), file_hash, "manifest split hash versus split file")
    return {
        "split_content_sha256": content_hash,
        "split_file_sha256": file_hash,
        "frozen_batches_sha256": frozen_hash,
        "model_hash": str(manifest["content_hash"]),
    }


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ConfigPinError(f"{label} mismatch: file has {actual!r}, pin is {expected!r}")


def operator_source_hash() -> str:
    paths = [
        REPO_ROOT / "expolicy/r9_lab/assimilators/discard.py",
        REPO_ROOT / "expolicy/r9_lab/assimilators/episodic.py",
        REPO_ROOT / "expolicy/r9_lab/assimilators/semantic.py",
        REPO_ROOT / "expolicy/r9_lab/evaluation/utility.py",
        REPO_ROOT / "expolicy/r9_lab/runner.py",
        REPO_ROOT / "expolicy/r9_lab/state/persistent_state.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_key(cfg: PaperAConfig, asset_hashes: Mapping[str, str] | None = None) -> str:
    """Full config identity. A semantic change must not reuse an older cell."""
    body = {
        "protocol_id": cfg.protocol_id,
        "protocol_version": cfg.raw.get("protocol_version"),
        "actions": list(cfg.actions),
        "metric": dict(cfg.raw["metric"]),
        "model": {
            "model_id": cfg.raw["model"]["model_id"],
            "dtype": cfg.raw["model"]["dtype"],
            "thinking": cfg.raw["model"]["thinking"],
            "temperature": cfg.raw["model"]["temperature"],
            "top_p": cfg.raw["model"]["top_p"],
            "max_tokens": cfg.raw["model"]["max_tokens"],
            "inference_seed": cfg.raw["model"]["inference_seed"],
        },
        "measurement": {
            "equivalence_success": cfg.raw["measurement"]["equivalence_success"],
            "formal_concurrency": cfg.raw["measurement"]["formal_concurrency"],
        },
        "assets": dict(asset_hashes or {}),
        "operator_source_hash": operator_source_hash(),
    }
    return sha256_json(body)
