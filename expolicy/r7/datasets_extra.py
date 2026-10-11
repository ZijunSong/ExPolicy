"""Load NQ / TriviaQA / PopQA without gold-as-retrieval. Extends R5.3 loaders."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from expolicy.environments.public_search.capability_map import C1, C2, C3, apply_mapping, map_capability
from expolicy.environments.public_search.datasets import (
    DATA_DIR,
    aliases_of,
    answer_of,
    example_to_record,
    hops_of,
    id_of,
    load_raw as load_raw_r53,
    question_of,
    supporting_titles,
    to_task,
    type_of,
)
from expolicy.r7.constants import C0, MULTI_HOP, R7_DATASETS, SINGLE_HOP

FLASHRAG = DATA_DIR / "flashrag"


def map_capability_r7(record: dict[str, Any]) -> dict[str, Any]:
    dataset = str(record.get("dataset") or "")
    if dataset in SINGLE_HOP:
        q = str(record.get("question") or "").strip()
        a = str(record.get("gold_answer") or "").strip()
        if not q or not a:
            return {"capability": "unmapped", "reason": "empty", "ambiguous": False, "unmapped": True}
        return {
            "capability": C0,
            "reason": f"{dataset}.single_hop",
            "ambiguous": False,
            "unmapped": False,
        }
    return map_capability(record)


def apply_mapping_r7(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for rec in records:
        mapped = dict(rec)
        mapped.update(map_capability_r7(rec))
        out.append(mapped)
    return out


def _flashrag_files(name: str) -> dict[str, Path]:
    mapping = {
        "nq": "nq",
        "triviaqa": "triviaqa",
        "popqa": "popqa",
        "hotpotqa": "hotpotqa",
        "2wiki": "2wikimultihopqa",
        "musique": "musique",
        "bamboogle": "bamboogle",
    }
    folder = FLASHRAG / mapping[name]
    out: dict[str, Path] = {}
    if not folder.exists():
        return out
    for split in ("train", "dev", "test", "val"):
        for ext in (".jsonl", ".json", ".parquet"):
            path = folder / f"{split}{ext}"
            if path.exists():
                out[split] = path
                break
    return out


def _load_table(path: Path) -> list[dict[str, Any]]:
    from expolicy.environments.public_search.datasets import _load_table as _lt

    return _lt(path)


def _hf_flashrag(name: str) -> dict[str, list[dict[str, Any]]]:
    from datasets import load_dataset

    cache = DATA_DIR / "hf_flashrag"
    cache.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for repo, cfg in (
        ("RUC-NLPIR/FlashRAG_datasets", name),
        ("RUC-NLPIR/FlashRAG_datasets", name.replace("2wiki", "2wikimultihopqa")),
    ):
        try:
            ds = load_dataset(repo, cfg, cache_dir=str(cache))
            return {k: list(v) for k, v in ds.items()}
        except Exception as exc:
            last_err = exc
            continue
    # Search-R1 / FlashRAG jsonl mirrors
    for repo in (
        f"hlzhang/FlashRAG_{name}",
        "TIGER-Lab/LongRAG" if name == "nq" else "",
    ):
        if not repo:
            continue
        try:
            ds = load_dataset(repo, cache_dir=str(cache))
            return {k: list(v) for k, v in ds.items()}
        except Exception as exc:
            last_err = exc
            continue
    raise FileNotFoundError(f"{name} not found locally or on HF: {last_err}")


def load_single_hop(dataset: str) -> dict[str, list[dict[str, Any]]]:
    files = _flashrag_files(dataset)
    if files:
        return {split: _load_table(path) for split, path in files.items()}
    local = DATA_DIR / dataset
    out: dict[str, list[dict[str, Any]]] = {}
    if local.exists():
        for split in ("train", "dev", "test"):
            for ext in (".jsonl", ".json", ".parquet"):
                path = local / f"{split}{ext}"
                if path.exists():
                    out[split] = _load_table(path)
                    break
        if out:
            return out
    return _hf_flashrag(dataset)


def load_raw_r7(dataset: str) -> dict[str, list[dict[str, Any]]]:
    if dataset in MULTI_HOP:
        return load_raw_r53(dataset)
    if dataset in SINGLE_HOP:
        return load_single_hop(dataset)
    raise ValueError(dataset)


def example_to_record_r7(dataset: str, split: str, example: dict[str, Any], index: int) -> dict[str, Any]:
    rec = example_to_record(dataset, split, example, index)
    if dataset in SINGLE_HOP:
        rec["hops"] = 1
        rec["type"] = rec.get("type") or "single_hop"
    return rec
