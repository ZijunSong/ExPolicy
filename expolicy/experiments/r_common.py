"""Shared helpers for R1–R4 assimilation-repair runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from expolicy.core.experience import Experience
from expolicy.environments.search.world import Task, World
from expolicy.experiments.common import (
    DEFAULT_EMBEDDER,
    DEFAULT_MODEL,
    ROOT,
    load_world,
    mean,
    score_with_retrieval,
    write_json,
)
from expolicy.logging.event_logger import append_jsonl

E0 = ROOT / "outputs" / "e0_audit"


def add_common_args(parser: argparse.ArgumentParser, *, default_out: str) -> argparse.ArgumentParser:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES, e.g. 6")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=default_out)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--world", default=str(ROOT / "data" / "world" / "helion.json"))
    return parser


def apply_gpu(gpu: str | None) -> None:
    if gpu is None or gpu == "":
        return
    # Do not clobber an already-narrowed CUDA_VISIBLE_DEVICES from the launcher.
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def load_e0_experiences(root: Path = E0) -> list[Experience]:
    rows: list[Experience] = []
    for stage in range(5):
        path = root / f"stage_{stage}" / "experiences.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(Experience.model_validate(json.loads(line)))
    return rows


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_run_header(out: Path, *, config: dict[str, Any], extra_manifest: dict[str, Any] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_yaml(out / "config.yaml", config)
    manifest = {
        "created_at": time.time(),
        "model": config.get("model"),
        "seed": config.get("seed"),
        "output_dir": str(out),
        **(extra_manifest or {}),
    }
    write_json(out / "manifest.json", manifest)


def eval_probe(agent: Any, state: Any, tasks: Sequence[Task]) -> dict[str, Any]:
    rows = []
    retrieval_hits = []
    used_rates = []
    for task in tasks:
        exp = agent.run_task(task, state=state)
        row = score_with_retrieval(exp, task)
        row["experience_id"] = exp.experience_id
        row["predicted"] = exp.predicted_answer
        row["gold"] = task.gold_answer
        info = getattr(agent, "last_retrieval", None) or {}
        row["retrieval_n"] = info.get("n", 0)
        row["memory_used_rate"] = info.get("memory_used_rate", 0.0)
        if info.get("n"):
            retrieval_hits.append(1.0)
            used_rates.append(float(info.get("memory_used_rate") or 0.0))
        else:
            retrieval_hits.append(0.0)
        debug = info.get("debug") or []
        if debug:
            cap_hit = mean(1.0 if d.get("capability") == task.capability else 0.0 for d in debug)
            row["retrieval_cap_hit"] = cap_hit
        rows.append(row)
    metrics = compact_metrics(rows)
    metrics["retrieval_hit_rate"] = mean(retrieval_hits) if retrieval_hits else 0.0
    metrics["memory_used_rate"] = mean(used_rates) if used_rates else 0.0
    return metrics


def compact_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    return {
        "n": len(rows),
        "em": mean(float(r.get("em_strict") or r.get("em") or 0.0) for r in rows) if rows else 0.0,
        "success": mean(float(r.get("success") or 0.0) for r in rows) if rows else 0.0,
        "tokens": int(sum(int(r.get("tokens") or 0) for r in rows)),
        "tool_calls": int(sum(int(r.get("tool_calls") or 0) for r in rows)),
        "rows": rows,
    }


def metrics_without_rows(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k != "rows"}


def capability_utility(delta_near: float, delta_far: float, delta_retention: float) -> float:
    return delta_near + 0.5 * delta_far - 0.5 * max(0.0, -delta_retention)


def dump_probe_rows(path: Path, *, split: str, condition: str, batch_id: str, rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        append_jsonl(
            path,
            {
                "split": split,
                "condition": condition,
                "batch_id": batch_id,
                **row,
            },
        )


def resume_done(out: Path, key: str) -> set[str]:
    path = out / "STATUS_LIVE.json"
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload.get(key) or [])


def mark_done(out: Path, key: str, value: str, extra: dict[str, Any] | None = None) -> None:
    path = out / "STATUS_LIVE.json"
    payload: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    items = list(payload.get(key) or [])
    if value not in items:
        items.append(value)
    payload[key] = items
    payload["updated_at"] = time.time()
    if extra:
        payload.update(extra)
    write_json(path, payload)


def load_batches_file(path: Path, world: World) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    batches = payload.get("batches") or payload
    out = []
    for row in batches:
        row = dict(row)
        for split in ("near", "far", "retention"):
            ids = row.get(split) or []
            if ids and isinstance(ids[0], str):
                row[f"{split}_tasks"] = [world.tasks[i] for i in ids if i in world.tasks]
        out.append(row)
    return out
