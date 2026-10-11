"""Outer/inner splits on source groups. A new seed of the same batch is not a new sample."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from expolicy.paper_a.features import template_key


class SplitLeakageError(RuntimeError):
    pass


@dataclass
class GroupSplit:
    batches: list[dict[str, Any]]
    folds: dict[str, int]
    n_outer: int
    deduped: list[dict[str, Any]]
    components: dict[str, str]
    notes: dict[str, Any] = field(default_factory=dict)

    def role(self, batch_id: str, test_fold: int) -> str:
        fold = self.folds[batch_id]
        if fold == test_fold:
            return "test"
        n = self.n_outer
        if n < 3:
            return "train"
        fit_fold = (test_fold + 1) % n
        if fold == fit_fold:
            return "train"
        return "val"

    def nonempty_folds(self) -> list[int]:
        return sorted(set(self.folds.values()))


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, key: str) -> None:
        self.parent.setdefault(key, key)

    def find(self, key: str) -> str:
        self.add(key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        self.parent[self.find(left)] = self.find(right)


def split_by_source_group(
    batches: Sequence[Mapping[str, Any]],
    *,
    n_outer: int = 3,
    seed: int = 20260822,
    forbidden_task_ids: set[str] | None = None,
) -> GroupSplit:
    """Deduplicate by task identity, connect shared templates, then assign group folds.

    Probe tasks are not sample units. Batches that share a task id or a normalized
    template stay in one component, so they cannot land in different outer folds.
    """
    forbidden = forbidden_task_ids or set()
    deduped: list[dict[str, Any]] = []
    seen: dict[tuple[str, tuple[str, ...]], str] = {}
    dropped: list[dict[str, str]] = []
    for batch in batches:
        task_ids = tuple(sorted(str(x) for x in batch.get("task_ids") or []))
        leaked = sorted(set(task_ids) & forbidden)
        if leaked:
            raise SplitLeakageError(f"batch {batch.get('batch_id')} contains held-out ids: {leaked}")
        identity = (str(batch.get("env") or ""), task_ids)
        batch_id = str(batch["batch_id"])
        if identity in seen:
            dropped.append(
                {
                    "batch_id": batch_id,
                    "kept": seen[identity],
                    "reason": "seed_is_not_a_new_sample",
                }
            )
            continue
        seen[identity] = batch_id
        deduped.append(dict(batch))

    uf = _UnionFind()
    template_owner: dict[str, str] = {}
    task_owner: dict[str, str] = {}
    templates_of: dict[str, list[str]] = {}
    for batch in deduped:
        batch_id = str(batch["batch_id"])
        uf.add(batch_id)
        items = list(batch.get("items") or [])
        if not items:
            items = [{"task_id": tid, "env": batch.get("env"), "instruction": tid} for tid in batch.get("task_ids") or []]
        keys = [template_key(item) for item in items]
        templates_of[batch_id] = keys
        for key in keys:
            if key in template_owner:
                uf.union(batch_id, template_owner[key])
            else:
                template_owner[key] = batch_id
        for task_id in batch.get("task_ids") or []:
            task_id = str(task_id)
            if task_id in task_owner:
                uf.union(batch_id, task_owner[task_id])
            else:
                task_owner[task_id] = batch_id

    components: dict[str, list[str]] = defaultdict(list)
    for batch in deduped:
        components[uf.find(str(batch["batch_id"]))].append(str(batch["batch_id"]))
    component_ids = sorted(components, key=lambda root: sorted(components[root])[0])
    rng = random.Random(seed)
    order = list(component_ids)
    rng.shuffle(order)
    n_outer = max(1, min(int(n_outer), len(order) or 1))
    folds: dict[str, int] = {}
    component_of: dict[str, str] = {}
    for index, root in enumerate(order):
        fold = index % n_outer
        for batch_id in components[root]:
            folds[batch_id] = fold
            component_of[batch_id] = "g" + root
    for batch in deduped:
        batch["source_group"] = component_of[str(batch["batch_id"])]

    _assert_no_leakage(deduped, folds, templates_of)
    notes = {
        "dropped_duplicates": dropped,
        "n_components": len(component_ids),
        "n_outer": n_outer,
        "shared_probe_is_not_a_sample": True,
        "reduced_folds": n_outer < 3,
    }
    return GroupSplit(
        batches=deduped,
        folds=folds,
        n_outer=n_outer,
        deduped=deduped,
        components=component_of,
        notes=notes,
    )


def _assert_no_leakage(
    batches: Sequence[Mapping[str, Any]],
    folds: Mapping[str, int],
    templates_of: Mapping[str, Sequence[str]],
) -> None:
    fold_by_template: dict[str, set[int]] = defaultdict(set)
    fold_by_task: dict[str, set[int]] = defaultdict(set)
    for batch in batches:
        batch_id = str(batch["batch_id"])
        fold = folds[batch_id]
        for key in templates_of.get(batch_id) or []:
            fold_by_template[key].add(fold)
        for task_id in batch.get("task_ids") or []:
            fold_by_task[str(task_id)].add(fold)
    leaked = {key: sorted(value) for key, value in fold_by_template.items() if len(value) > 1}
    if leaked:
        raise SplitLeakageError(f"template crosses outer folds: {list(leaked)[:5]}")
    leaked_tasks = {key: sorted(value) for key, value in fold_by_task.items() if len(value) > 1}
    if leaked_tasks:
        raise SplitLeakageError(f"task id crosses outer folds: {list(leaked_tasks)[:5]}")
