"""Load frozen R7 split into Task objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from expolicy.closed_loop.constants import CAPS, R7_ROOT
from expolicy.closed_loop.io_utils import read_json, sha256_file
from expolicy.environments.public_search.datasets import to_task
from expolicy.environments.search.world import Task

SPLIT_PATH = R7_ROOT / "audit" / "SPLIT_R7.json"


def load_split(path: Path | None = None) -> dict[str, Any]:
    path = path or SPLIT_PATH
    if not path.exists():
        raise FileNotFoundError(f"missing split: {path}")
    return read_json(path)


def split_hash(path: Path | None = None) -> str:
    path = path or SPLIT_PATH
    return sha256_file(path)


def records_to_tasks(records: list[dict[str, Any]], *, split: str) -> list[Task]:
    tasks = []
    for rec in records:
        cap = rec.get("capability")
        if cap not in CAPS:
            continue
        tasks.append(to_task(rec, capability=str(cap), split=split))
    return tasks


def split_tasks(split: dict[str, Any], key: str) -> list[Task]:
    return records_to_tasks(split.get(key) or [], split=key)


def tasks_by_cap(tasks: list[Task]) -> dict[str, list[Task]]:
    out = {c: [] for c in CAPS}
    for t in tasks:
        if t.capability in out:
            out[t.capability].append(t)
    return out
