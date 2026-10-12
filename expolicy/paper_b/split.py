"""Group-disjoint split audit. A reworded copy of the same entity is not a new task."""

from __future__ import annotations

from collections import defaultdict

from expolicy.paper_b.constants import (
    SPLIT_CROSS,
    SPLIT_NEAR,
    SPLIT_RETENTION,
    SPLIT_TRAIN,
    SPLIT_UNRELATED,
    SPLIT_VAL,
    TARGET_FAMILIES,
)
from expolicy.paper_b.models import Task
from expolicy.r9_lab.io_util import sha256_json


def audit_splits(tasks: list[Task], *, full_counts: dict[str, int] | None = None) -> dict:
    errors: list[str] = []
    by_split: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        by_split[task.split].append(task)
    if len({task.task_id for task in tasks}) != len(tasks):
        errors.append("task_id_overlap")
    if len({task.entity_key for task in tasks}) != len(tasks):
        errors.append("entity_overlap")
    if len({task.db_object for task in tasks}) != len(tasks):
        errors.append("db_object_overlap")
    train_groups = _groups(by_split, SPLIT_TRAIN)
    if not _groups(by_split, SPLIT_NEAR) <= train_groups:
        errors.append("near_template_not_in_train")
    if _groups(by_split, SPLIT_CROSS) & train_groups:
        errors.append("cross_template_overlaps_train")
    val_target = {
        task.source_group
        for task in by_split.get(SPLIT_VAL, [])
        if task.family in TARGET_FAMILIES
    }
    if val_target - train_groups:
        errors.append("val_template_not_in_train")
    if _groups(by_split, SPLIT_RETENTION) & train_groups:
        errors.append("retention_template_overlaps_train")
    if _entities(by_split, SPLIT_UNRELATED) & _entities(by_split, SPLIT_TRAIN):
        errors.append("unrelated_overlaps_train")
    counts = {split: len(group) for split, group in sorted(by_split.items())}
    assignment = [
        {
            "task_id": task.task_id,
            "split": task.split,
            "source_group": task.source_group,
            "entity_key": task.entity_key,
            "family": task.family,
        }
        for task in sorted(tasks, key=lambda item: item.task_id)
    ]
    short = False
    if full_counts:
        short = any(counts.get(split, 0) < expected for split, expected in full_counts.items())
    return {
        "ok": not errors,
        "errors": errors,
        "counts": counts,
        "source_groups": {split: sorted(_groups(by_split, split)) for split in sorted(by_split)},
        "assignment_sha256": sha256_json(assignment),
        "precision_warning": short,
        "gold_not_a_split_key": True,
    }


def _groups(by_split: dict[str, list[Task]], split: str) -> set[str]:
    return {task.source_group for task in by_split.get(split, [])}


def _entities(by_split: dict[str, list[Task]], split: str) -> set[str]:
    return {task.entity_key for task in by_split.get(split, [])}
