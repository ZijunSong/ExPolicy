"""Freeze OS evaluation tasks that do not overlap known D development sources."""

from __future__ import annotations

import random
import re
from typing import Any, Iterable, Mapping, Sequence

from expolicy.paper_d.content_protocol import PROTOCOL
from expolicy.r9_lab.io_util import sha256_json


def normalize_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_']+", normalize_instruction(text)))


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def family_id(skill_list: Sequence[str]) -> str:
    skills = [str(item) for item in skill_list if str(item).strip()]
    return "skills:" + ",".join(sorted(skills)) if skills else "unclustered"


def _too_close(instruction: str, chosen: Sequence[str], limit: float) -> bool:
    tokens = token_set(instruction)
    if not tokens:
        return True
    return any(jaccard(tokens, token_set(other)) >= limit for other in chosen)


def select_role_tasks(
    records: Sequence[Mapping[str, Any]],
    *,
    n: int,
    excluded_ids: set[str],
    blocked_text: Sequence[str],
    seed: int,
    role: str,
) -> list[dict[str, Any]]:
    """Draw n tasks with a fixed seed. Near-copies and known ids are removed first."""
    pool = []
    for record in records:
        task_id = str(record["task_id"])
        if task_id in excluded_ids or not str(task_id).startswith("os:"):
            continue
        pool.append(record)
    pool.sort(key=lambda record: int(record["sample_index"]))
    random.Random(seed).shuffle(pool)
    chosen_text = list(blocked_text)
    selected = []
    for record in pool:
        instruction = str(record.get("instruction") or "")
        if _too_close(instruction, chosen_text, float(PROTOCOL["near_copy_jaccard"])):
            continue
        selected.append(
            {
                "task_id": str(record["task_id"]),
                "sample_index": str(record["sample_index"]),
                "instruction": instruction,
                "raw_task_hash": sha256_json({"instruction": normalize_instruction(instruction)}),
                "probe_role": role,
                "task_family": family_id(record.get("skill_list") or []),
                "skill_list": [str(item) for item in (record.get("skill_list") or [])],
                "development_or_new": "new",
                "prior_exposure_unknown": True,
            }
        )
        chosen_text.append(instruction)
        if len(selected) == n:
            break
    if len(selected) != n:
        raise RuntimeError(f"only {len(selected)} legal {role} tasks, needed {n}")
    return selected


def select_main_tasks(
    future_records: Sequence[Mapping[str, Any]],
    retention_records: Sequence[Mapping[str, Any]],
    *,
    excluded_ids: set[str],
    blocked_text: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    blocked = list(blocked_text or [])
    future = select_role_tasks(
        future_records,
        n=int(PROTOCOL["n_future"]),
        excluded_ids=excluded_ids,
        blocked_text=blocked,
        seed=int(PROTOCOL["task_seed"]),
        role="future",
    )
    retention = select_role_tasks(
        retention_records,
        n=int(PROTOCOL["n_retention"]),
        excluded_ids=excluded_ids | {row["task_id"] for row in future},
        blocked_text=blocked + [row["instruction"] for row in future],
        seed=int(PROTOCOL["task_seed"]) + 1,
        role="retention",
    )
    tasks = future + retention
    assert_main_tasks_clean(tasks, excluded_ids)
    return tasks


def assert_main_tasks_clean(tasks: Sequence[Mapping[str, Any]], forbidden_ids: set[str]) -> None:
    ids = [str(task["task_id"]) for task in tasks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("main task list has duplicate ids")
    overlap = sorted(set(ids) & forbidden_ids)
    if overlap:
        raise RuntimeError(f"main tasks overlap known sources: {overlap[:8]}")
    roles = {str(task["probe_role"]) for task in tasks}
    if roles != {"future", "retention"}:
        raise RuntimeError(f"unexpected roles {roles}")


def history_subset(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """First frozen tasks in each role. This reuses tasks and is not a new sample."""
    future = [dict(task) for task in tasks if task["probe_role"] == "future"][: int(PROTOCOL["history_future"])]
    retention = [dict(task) for task in tasks if task["probe_role"] == "retention"][: int(PROTOCOL["history_retention"])]
    if len(future) != int(PROTOCOL["history_future"]) or len(retention) != int(PROTOCOL["history_retention"]):
        raise RuntimeError("history subset is short")
    return future + retention


def rank_donors(
    records: Sequence[Mapping[str, Any]],
    *,
    excluded_ids: set[str],
    source_skills: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Rank unused acquisition tasks by skill-set distance. No evaluation scores are used."""
    ranked = []
    for record in records:
        task_id = str(record["task_id"])
        if task_id in excluded_ids:
            continue
        skills = {str(item) for item in (record.get("skill_list") or [])}
        ranked.append((jaccard(skills, source_skills), int(record["sample_index"]), record, skills))
    ranked.sort(key=lambda item: (item[0], item[1]))
    chosen = []
    for distance, _index, record, skills in ranked[:limit]:
        chosen.append(
            {
                "task_id": str(record["task_id"]),
                "sample_index": str(record["sample_index"]),
                "instruction": str(record.get("instruction") or ""),
                "skill_list": sorted(skills),
                "skill_jaccard_to_source": distance,
                "task_family": family_id(record.get("skill_list") or []),
            }
        )
    return chosen
