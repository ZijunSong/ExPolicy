"""HotpotQA manifest for B-v2. Confirmation ids are recorded and not scored here."""

from __future__ import annotations

import re
from typing import Any

from expolicy.paper_b.public_exec import PublicTask
from expolicy.r9_lab.io_util import sha256_json

_PUNCT = re.compile(r"[^a-z0-9\s]+")


def normalize_question(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", (text or "").lower()).split())


def task_from_hotpot(row: dict[str, Any], split: str) -> PublicTask | None:
    question = str(row.get("question") or "").strip()
    answers = row.get("golden_answers") or row.get("answers") or []
    if isinstance(answers, str):
        answers = [answers]
    answer = str(row.get("answer") or (answers[0] if answers else "")).strip()
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    kind = str(row.get("type") or meta.get("type") or row.get("question_type") or "unknown").strip().lower()
    if not question or not answer or kind not in {"bridge", "comparison"}:
        return None
    family = "recovery" if kind == "bridge" else "retention"
    level = str(meta.get("level") or "").strip().lower()
    cluster = f"{kind}:{level}" if level in {"easy", "medium", "hard"} else kind
    return PublicTask(
        task_id=str(row.get("_id") or row.get("id") or sha256_json(question)[:12]),
        question=question,
        gold_answer=answer,
        split=split,
        source_group=kind,
        family=family,
        entity_key=normalize_question(question),
        dataset="hotpotqa",
        cluster=cluster,
    )


def assign_manifest(rows: list[dict[str, Any]], counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Stable assignment. The same question text cannot land in two splits."""
    unique: dict[str, PublicTask] = {}
    for row in rows:
        task = task_from_hotpot(row, "pool")
        if task is None:
            continue
        unique.setdefault(task.entity_key, task)
    ordered = sorted(unique.values(), key=lambda task: sha256_json({"id": task.task_id, "q": task.entity_key}))
    buckets = {"recovery": [task for task in ordered if task.family == "recovery"], "retention": [task for task in ordered if task.family == "retention"]}
    taken: set[str] = set()
    assigned: list[PublicTask] = []
    short: list[str] = []
    for split, needs in counts.items():
        for family, count in needs.items():
            chosen = _take(buckets.get(family, []), taken, int(count))
            if len(chosen) < int(count):
                short.append(f"{split}:{family}:{len(chosen)}/{count}")
            for task in chosen:
                assigned.append(
                    PublicTask(
                        task_id=task.task_id,
                        question=task.question,
                        gold_answer=task.gold_answer,
                        split=split,
                        source_group=task.source_group,
                        family=family,
                        entity_key=task.entity_key,
                        dataset=task.dataset,
                        cluster=task.cluster or task.source_group,
                    )
                )
                taken.add(task.entity_key)
    groups = sorted({task.source_group for task in assigned})
    return {
        "protocol_id": "B-v2",
        "dataset": "hotpotqa",
        "structural_available": False,
        "structural_note": "HotpotQA 只有 bridge 与 comparison 两种结构。没有把同一种桥接题改写措辞后称作结构迁移。",
        "source_groups": groups,
        "n_source_groups": len(groups),
        "clusters": sorted({task.cluster or task.source_group for task in assigned}),
        "shortfalls": short,
        "confirmation_opened": False,
        "tasks": [
            {
                "task_id": task.task_id,
                "split": task.split,
                "source_group": task.source_group,
                "family": task.family,
                "entity_key": task.entity_key,
                "cluster": task.cluster or task.source_group,
                "question": task.question,
            }
            for task in assigned
        ],
        "answers_for_scorer_only": {task.task_id: task.gold_answer for task in assigned},
        "assignment_sha256": sha256_json([(task.task_id, task.split, task.entity_key) for task in assigned]),
    }


def tasks_for_split(manifest: dict[str, Any], split: str) -> list[PublicTask]:
    answers = manifest.get("answers_for_scorer_only") or {}
    out = []
    for row in manifest.get("tasks") or []:
        if row["split"] != split:
            continue
        out.append(
            PublicTask(
                task_id=row["task_id"],
                question=row["question"],
                gold_answer=str(answers.get(row["task_id"]) or ""),
                split=row["split"],
                source_group=row["source_group"],
                family=row["family"],
                entity_key=row["entity_key"],
                cluster=str(row.get("cluster") or row["source_group"]),
            )
        )
    return out


def _take(pool: list[PublicTask], taken: set[str], count: int) -> list[PublicTask]:
    chosen = []
    for task in pool:
        if task.entity_key in taken:
            continue
        chosen.append(task)
        if len(chosen) >= count:
            break
    return chosen
