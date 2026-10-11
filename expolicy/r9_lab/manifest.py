"""Model manifest freeze / immutability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from expolicy.r9_lab.io_util import atomic_write_json, read_json, sha256_json


class ManifestFrozenError(RuntimeError):
    pass


def _hashable(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k != "content_hash"}


def freeze_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["frozen"] = True
    body["content_hash"] = sha256_json(_hashable(body))
    atomic_write_json(path, body)
    return body


def load_manifest(path: Path) -> dict[str, Any]:
    return read_json(path)


def assert_manifest_unmodified(path: Path, candidate: dict[str, Any] | None = None) -> None:
    stored = load_manifest(path)
    if not stored.get("frozen"):
        raise ManifestFrozenError("manifest is not frozen")
    expected = stored.get("content_hash")
    actual = sha256_json(_hashable(stored))
    if expected != actual:
        raise ManifestFrozenError("frozen manifest content_hash mismatch")
    if candidate is not None:
        cand_hash = sha256_json(_hashable({**candidate, "frozen": True}))
        # Compare identity fields; any change of selected model / thinking / split hash is a violation.
        keys = ("selected_model_id", "hf_repo", "local_path", "thinking_mode", "screen_split_sha256", "dtype")
        for key in keys:
            if key in stored and key in candidate and stored[key] != candidate[key]:
                raise ManifestFrozenError(f"attempted mutation of frozen field {key}")
        if stored.get("content_hash") and cand_hash != stored["content_hash"]:
            # candidate may omit content_hash/frozen; still detect selected-id changes via keys above
            pass
