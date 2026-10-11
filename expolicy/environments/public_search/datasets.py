"""Load HotpotQA / 2WikiMultiHopQA / MuSiQue / Bamboogle without gold-as-retrieval."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from expolicy.environments.search.world import Task
from expolicy.experiments.common import ROOT

HOTPOT_DIR = Path("/data/ppnm/tmp/HotpotQA_raw")
DATA_DIR = ROOT / "data" / "public_search"
FLASHRAG_DIR = DATA_DIR / "flashrag"

DATASET_NAMES = ("hotpotqa", "2wiki", "musique", "bamboogle")
HOP_RE = re.compile(r"^(\d+)hop", re.I)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        import orjson
        loads = orjson.loads
    except Exception:
        loads = json.loads
    rows = []
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            rec = loads(raw)
            rec.pop("context", None)
            meta = rec.get("metadata")
            if isinstance(meta, dict):
                meta.pop("context", None)
            rows.append(rec)
    return rows


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def supporting_titles(example: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    meta = example.get("metadata") or {}
    facts = example.get("supporting_facts") or example.get("supporting_facts_raw") or meta.get("supporting_facts") or []
    if isinstance(facts, dict):
        titles.extend(str(t) for t in (facts.get("title") or []))
    else:
        for item in facts:
            if isinstance(item, (list, tuple)) and item:
                titles.append(str(item[0]))
            elif isinstance(item, dict) and item.get("title"):
                titles.append(str(item["title"]))
    for item in example.get("evidences") or []:
        if isinstance(item, (list, tuple)) and item:
            titles.append(str(item[0]))
    for para in example.get("paragraphs") or []:
        if isinstance(para, dict) and para.get("is_supporting") and para.get("title"):
            titles.append(str(para["title"]))
    seen: set[str] = set()
    out = []
    for t in titles:
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(t.strip())
    return out


def hops_of(dataset: str, example: dict[str, Any]) -> int:
    meta = example.get("metadata") or {}
    decomp = example.get("question_decomposition") or meta.get("question_decomposition") or []
    if dataset == "musique":
        eid = str(example.get("id") or example.get("_id") or "")
        m = HOP_RE.match(eid)
        if m:
            return int(m.group(1))
        if decomp:
            return max(2, len(decomp))
    if dataset == "hotpotqa":
        return 2
    if dataset == "2wiki":
        titles = supporting_titles(example)
        if titles:
            return max(2, min(4, len(set(t.lower() for t in titles))))
        ev = example.get("evidences") or meta.get("evidences") or []
        if isinstance(ev, list) and ev:
            return max(2, min(4, len(ev)))
        return 2
    if dataset == "bamboogle":
        return 2
    return int(example.get("hops") or meta.get("hops") or 2)


def answer_of(example: dict[str, Any]) -> str:
    for key in ("answer", "golden_answers", "answers"):
        val = example.get(key)
        if isinstance(val, list) and val:
            return str(val[0])
        if isinstance(val, str) and val.strip():
            return val.strip()
    meta = example.get("metadata") or {}
    if meta.get("answer"):
        return str(meta["answer"])
    return ""


def question_of(example: dict[str, Any]) -> str:
    return str(example.get("question") or example.get("query") or "").strip()


def id_of(dataset: str, example: dict[str, Any], fallback: int) -> str:
    raw = example.get("id") or example.get("_id") or example.get("qid") or fallback
    return f"{dataset}:{raw}"


def type_of(example: dict[str, Any]) -> str:
    t = example.get("type") or example.get("question_type") or ""
    if not t:
        meta = example.get("metadata") or {}
        t = meta.get("type") or meta.get("question_type") or ""
    return str(t).strip().lower()


def aliases_of(example: dict[str, Any]) -> list[str]:
    aliases = []
    for key in ("answer_aliases", "aliases"):
        val = example.get(key) or []
        if isinstance(val, list):
            aliases.extend(str(x) for x in val if x)
    gold = answer_of(example)
    if gold:
        aliases.append(gold)
    # unique preserve
    seen = set()
    out = []
    for a in aliases:
        k = a.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(a.strip())
    return out


def _flashrag_files(name: str) -> dict[str, Path]:
    mapping = {
        "hotpotqa": "hotpotqa",
        "2wiki": "2wikimultihopqa",
        "musique": "musique",
        "bamboogle": "bamboogle",
    }
    folder = FLASHRAG_DIR / mapping[name]
    out = {}
    for split in ("train", "dev", "test", "val"):
        for ext in (".jsonl", ".json", ".parquet"):
            path = folder / f"{split}{ext}"
            if path.exists():
                out[split] = path
                break
    return out


def _load_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    if path.suffix == ".json":
        payload = _read_json(path)
        if isinstance(payload, list):
            return payload
        for key in ("data", "examples", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    if path.suffix == ".parquet":
        from datasets import Dataset

        return list(Dataset.from_parquet(str(path)))
    raise ValueError(f"unsupported table {path}")


def load_hotpotqa() -> dict[str, list[dict[str, Any]]]:
    files = _flashrag_files("hotpotqa")
    if files:
        return {split: _load_table(path) for split, path in files.items()}
    train = _first_existing(
        [
            HOTPOT_DIR / "hotpot_train_v1.1.json",
            DATA_DIR / "hotpotqa" / "hotpot_train_v1.1.json",
        ]
    )
    dev = _first_existing(
        [
            HOTPOT_DIR / "hotpot_dev_fullwiki_v1.json",
            HOTPOT_DIR / "hotpot_dev_distractor_v1.json",
            DATA_DIR / "hotpotqa" / "hotpot_dev_fullwiki_v1.json",
        ]
    )
    out: dict[str, list[dict[str, Any]]] = {}
    if train:
        out["train"] = _read_json(train)
    if dev:
        out["dev"] = _read_json(dev)
    if not out:
        raise FileNotFoundError("HotpotQA not found locally or in FlashRAG dump")
    return out


def load_2wiki() -> dict[str, list[dict[str, Any]]]:
    files = _flashrag_files("2wiki")
    if files:
        return {split: _load_table(path) for split, path in files.items()}
    local = DATA_DIR / "2wiki"
    out = {}
    for split, name in (("train", "train.json"), ("dev", "dev.json")):
        path = local / name
        if path.exists():
            out[split] = _read_json(path)
    if out:
        return out
    hf_dir = DATA_DIR / "hf_2wiki"
    if hf_dir.exists():
        found = list(hf_dir.rglob("*.json")) + list(hf_dir.rglob("*.parquet")) + list(hf_dir.rglob("*.jsonl"))
        if found:
            grouped: dict[str, list] = {}
            for path in found:
                split = "dev" if "dev" in path.name or "val" in path.name else ("test" if "test" in path.name else "train")
                grouped.setdefault(split, []).extend(_load_table(path) if path.suffix != ".json" or True else _read_json(path))
            if grouped:
                return grouped
    from datasets import load_dataset

    for repo in ("voidful/2WikiMultihopQA", "xanhho/2WikiMultihopQA"):
        try:
            ds = load_dataset(repo)
            return {k: list(v) for k, v in ds.items()}
        except Exception:
            continue
    raise FileNotFoundError("2WikiMultiHopQA not found")


def load_musique() -> dict[str, list[dict[str, Any]]]:
    files = _flashrag_files("musique")
    if files:
        return {split: _load_table(path) for split, path in files.items()}
    local = DATA_DIR / "musique"
    out = {}
    for split in ("train", "dev"):
        for name in (f"{split}.jsonl", f"{split}.json", f"musique_ans_{split}.jsonl"):
            path = local / name
            if path.exists():
                out[split] = _load_json_or_jsonl(path)
                break
    if out:
        return out
    from datasets import load_dataset

    for repo, cfg in (("dgslibisey/MuSiQue", None), ("bdsaglam/musique", "full")):
        try:
            ds = load_dataset(repo, cfg) if cfg else load_dataset(repo)
            mapped = {}
            for split, table in ds.items():
                rows = [r for r in table if r.get("answerable", True) is not False]
                mapped[split] = rows
            if mapped:
                return mapped
        except Exception:
            continue
    raise FileNotFoundError("MuSiQue not found")


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    return _read_json(path)


def load_bamboogle() -> dict[str, list[dict[str, Any]]]:
    files = _flashrag_files("bamboogle")
    if files:
        rows = []
        for path in files.values():
            rows.extend(_load_table(path))
        return {"test": rows}
    local = DATA_DIR / "bamboogle"
    for name in ("bamboogle.json", "test.json", "Bamboogle.csv"):
        path = local / name
        if path.exists():
            if path.suffix.lower() == ".csv":
                return {"test": _load_csv(path)}
            payload = _load_json_or_jsonl(path)
            return {"test": payload if isinstance(payload, list) else [payload]}
    # last resort: official sheet
    url = "https://raw.githubusercontent.com/ofirpress/Bamboogle/master/Bamboogle%20Prerelease%20-%20Sheet1.csv"
    dest = local / "Bamboogle.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request

        urllib.request.urlretrieve(url, dest)
        return {"test": _load_csv(dest)}
    except Exception as exc:
        raise FileNotFoundError(f"Bamboogle not found: {exc}") from exc


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for i, row in enumerate(reader):
            q = row.get("Question") or row.get("question") or ""
            a = row.get("Answer") or row.get("answer") or ""
            if q:
                rows.append({"id": f"bamb_{i}", "question": q, "answer": a, "type": "bridge"})
        return rows


def load_raw(dataset: str) -> dict[str, list[dict[str, Any]]]:
    if dataset == "hotpotqa":
        return load_hotpotqa()
    if dataset == "2wiki":
        return load_2wiki()
    if dataset == "musique":
        return load_musique()
    if dataset == "bamboogle":
        return load_bamboogle()
    raise ValueError(dataset)


def example_to_record(dataset: str, split: str, example: dict[str, Any], index: int) -> dict[str, Any]:
    q = question_of(example)
    a = answer_of(example)
    hops = hops_of(dataset, example)
    titles = supporting_titles(example)
    return {
        "task_id": id_of(dataset, example, index),
        "dataset": dataset,
        "official_split": split,
        "question": q,
        "gold_answer": a,
        "aliases": aliases_of(example),
        "type": type_of(example),
        "hops": hops,
        "supporting_titles": titles,
        "raw_id": str(example.get("id") or example.get("_id") or index),
    }


def to_task(record: dict[str, Any], *, capability: str, split: str) -> Task:
    return Task(
        task_id=record["task_id"],
        question=record["question"],
        gold_answer=record["gold_answer"],
        capability=capability,
        stage=0,
        gold_doc_ids=[],  # never pass gold docs into retrieval
        distractor_doc_ids=[],
        hops=int(record.get("hops") or 2),
        split=split,
        aliases=list(record.get("aliases") or []),
        metadata={
            "dataset": record.get("dataset"),
            "type": record.get("type"),
            "supporting_titles": list(record.get("supporting_titles") or []),
            "official_split": record.get("official_split"),
            "raw_id": record.get("raw_id"),
        },
    )
