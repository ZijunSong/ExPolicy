"""Disjoint source, donor and confirmation-task pools. Scores are not an input."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

from expolicy.paper_d.confirm_protocol import PROTOCOL
from expolicy.paper_d.content_tasks import family_id, jaccard, normalize_instruction, token_set
from expolicy.r9_lab.io_util import sha256_json


def too_close(instruction: str, chosen: Sequence[str], limit: float) -> bool:
    tokens = token_set(instruction)
    if not tokens:
        return True
    return any(jaccard(tokens, token_set(other)) >= limit for other in chosen)


def draw(records: Sequence[dict[str, Any]], n: int, blocked: list[str], seed: int) -> list[dict[str, Any]]:
    pool = [dict(record) for record in records]
    pool.sort(key=lambda record: int(record["sample_index"]))
    random.Random(seed).shuffle(pool)
    chosen = []
    texts = list(blocked)
    limit = float(PROTOCOL["near_copy_jaccard"])
    for record in pool:
        instruction = str(record.get("instruction") or "")
        if too_close(instruction, texts, limit):
            continue
        chosen.append(record)
        texts.append(instruction)
        if len(chosen) == n:
            break
    return chosen


def task_row(record: dict[str, Any], role: str) -> dict[str, Any]:
    instruction = str(record.get("instruction") or "")
    return {
        "task_id": str(record["task_id"]),
        "sample_index": str(record["sample_index"]),
        "instruction": instruction,
        "raw_task_hash": sha256_json({"instruction": normalize_instruction(instruction)}),
        "probe_role": role,
        "task_family": family_id(record.get("skill_list") or []),
        "skill_list": [str(item) for item in (record.get("skill_list") or [])],
        "development_or_new": "confirm",
        "prior_exposure_unknown": True,
    }


def freeze_pools(buckets: dict[str, list[dict[str, Any]]], excluded: set[str], blocked_text: list[str]) -> dict[str, Any]:
    """Draw confirmation tasks first, then six source batches, then the donor order."""
    variant = "32-16"
    future_n = int(PROTOCOL["n_future"])
    retention_n = int(PROTOCOL["n_retention"])
    future = draw(
        [row for row in buckets["future_test"] if row["task_id"] not in excluded],
        future_n,
        blocked_text,
        int(PROTOCOL["task_seed"]),
    )
    retention = draw(
        [row for row in buckets["retention_probe"] if row["task_id"] not in excluded],
        retention_n,
        blocked_text + [row["instruction"] for row in future],
        int(PROTOCOL["task_seed"]) + 1,
    )
    if len(future) < future_n or len(retention) < retention_n:
        variant = "24-12"
        future_n = int(PROTOCOL["shrink_future"])
        retention_n = int(PROTOCOL["shrink_retention"])
        future = draw(
            [row for row in buckets["future_test"] if row["task_id"] not in excluded],
            future_n,
            blocked_text,
            int(PROTOCOL["task_seed"]),
        )
        retention = draw(
            [row for row in buckets["retention_probe"] if row["task_id"] not in excluded],
            retention_n,
            blocked_text + [row["instruction"] for row in future],
            int(PROTOCOL["task_seed"]) + 1,
        )
    if len(future) < future_n or len(retention) < retention_n:
        raise RuntimeError(f"confirmation pool short: future {len(future)} retention {len(retention)}")
    eval_tasks = [task_row(row, "future") for row in future] + [task_row(row, "retention") for row in retention]
    eval_ids = {row["task_id"] for row in eval_tasks}
    eval_text = blocked_text + [row["instruction"] for row in eval_tasks]
    acquisition = [row for row in buckets["acquisition_pool"] if row["task_id"] not in excluded and row["task_id"] not in eval_ids]
    needed = int(PROTOCOL["batches"]) * int(PROTOCOL["sources_per_batch"])
    sources = draw(acquisition, needed, eval_text, int(PROTOCOL["task_seed"]) + 2)
    if len(sources) < needed:
        raise RuntimeError(f"only {len(sources)} source tasks, needed {needed}")
    source_ids = {row["task_id"] for row in sources}
    donor_records = [row for row in acquisition if row["task_id"] not in source_ids]
    donors = draw(donor_records, int(PROTOCOL["donor_cap"]), eval_text + [row["instruction"] for row in sources], int(PROTOCOL["task_seed"]) + 3)
    batches = []
    width = int(PROTOCOL["sources_per_batch"])
    for index in range(int(PROTOCOL["batches"])):
        chunk = sources[index * width : (index + 1) * width]
        batches.append(
            {
                "batch_id": f"C{index + 1}",
                "sources": [task_row(row, "source") for row in chunk],
            }
        )
    return {
        "variant": variant,
        "n_future": future_n,
        "n_retention": retention_n,
        "evaluation": eval_tasks,
        "batches": batches,
        "donor_order": [task_row(row, "donor") for row in donors],
        "prior_exposure_unknown": True,
    }


def load_excluded(repo: Path) -> tuple[set[str], list[str]]:
    excluded: set[str] = set()
    texts: list[str] = []
    split = json.loads((repo / "outputs/r9_lab/split.json").read_text(encoding="utf-8"))["envs"]["os"]
    for path in repo.glob("outputs/**/executions/*/*/result.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8")).get("job") or {}
        except (OSError, json.JSONDecodeError):
            continue
        if job.get("env") == "os" and job.get("task_id"):
            excluded.add(str(job["task_id"]))
    manifest_path = repo / "outputs/paper_d_content_os/20261014-content-os/D_CONTENT_TASK_MANIFEST.json"
    bundle_path = repo / "outputs/paper_d_content_os/20261014-content-os/D_CONTENT_SOURCE_BUNDLES.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for group in ("main", "history_subset", "cases"):
            for task in manifest.get(group) or []:
                excluded.add(str(task["task_id"]))
    if bundle_path.is_file():
        bundles = json.loads(bundle_path.read_text(encoding="utf-8"))
        for bundle in (bundles.get("sources") or {}).values():
            excluded.update(str(task_id) for task_id in bundle.get("all_source_ids") or [])
        for order in (bundles.get("donor_order") or {}).values():
            for item in order:
                excluded.add(str(item["task_id"]))
    for rows in split.values():
        for record in rows:
            if str(record["task_id"]) in excluded:
                texts.append(str(record.get("instruction") or ""))
    return excluded, texts
