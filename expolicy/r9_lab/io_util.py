"""JSON, hashing, atomic writes, flock, DONE.json for R9."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from filelock import FileLock


def _default(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=_default, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(payload: Any) -> str:
    return sha256_text(canonical_json(payload))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default, ensure_ascii=False) + "\n", encoding="utf-8")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_default, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=_default, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@contextmanager
def flock(path: Path, timeout: float = 600.0) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path), timeout=timeout)
    with lock:
        yield


def is_done(cell_dir: Path) -> bool:
    return (cell_dir / "DONE.json").exists()


def mark_done(cell_dir: Path, payload: dict[str, Any] | None = None) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    body = {"done": True, "ts": time.time(), **(payload or {})}
    atomic_write_json(cell_dir / "DONE.json", body)


def skip_if_done(cell_dir: Path, *, resume: bool) -> bool:
    return bool(resume and is_done(cell_dir))


class ResumeHashError(RuntimeError):
    pass


def write_hashes(out: Path, hashes: dict[str, str]) -> None:
    atomic_write_json(out / "RESUME_HASHES.json", hashes)


def load_hashes(out: Path) -> dict[str, str] | None:
    path = out / "RESUME_HASHES.json"
    if not path.exists():
        return None
    return read_json(path)


def assert_resume_hashes(
    out: Path,
    *,
    split_hash: str,
    model_hash: str,
    config_hash: str,
) -> None:
    prev = load_hashes(out)
    if prev is None:
        return
    if prev.get("split_hash") != split_hash:
        raise ResumeHashError("split hash mismatch; refuse resume")
    if prev.get("model_hash") != model_hash:
        raise ResumeHashError("model hash mismatch; refuse resume")
    if prev.get("config_hash") != config_hash:
        raise ResumeHashError("config hash mismatch; refuse resume")
