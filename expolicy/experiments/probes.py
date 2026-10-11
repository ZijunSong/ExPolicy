"""Capability-matched near-transfer / far / retention probe construction."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from expolicy.core.experience import Experience
from expolicy.environments.search.world import Task, World
from expolicy.experiments.patterns import (
    BUCKET_PRIMARY_CAP,
    TARGET_CAPS,
    gold_doc_key,
    normalized_pattern,
    task_code,
)


NEAR_SPLITS = ("future", "validation", "acquisition", "deployment")
FAR_CAPS_BY_BUCKET = {
    "s1_multi_hop": ("temporal_reasoning", "cross_document_synthesis", "entity_disambiguation"),
    "s2_temporal": ("multi_hop", "cross_document_synthesis", "query_decomposition"),
    "s4_cross_doc": ("multi_hop", "temporal_reasoning", "query_decomposition"),
}


@dataclass
class ProbeSet:
    batch_id: str
    bucket: str
    experience_task_ids: list[str]
    near: list[Task]
    far: list[Task]
    retention: list[Task]
    notes: list[str] = field(default_factory=list)

    def all_tasks(self) -> list[Task]:
        return [*self.near, *self.far, *self.retention]

    def as_ids(self) -> dict[str, list[str]]:
        return {
            "batch_id": self.batch_id,
            "bucket": self.bucket,
            "experience_task_ids": self.experience_task_ids,
            "near": [t.task_id for t in self.near],
            "far": [t.task_id for t in self.far],
            "retention": [t.task_id for t in self.retention],
            "notes": self.notes,
        }


def _same_surface(a: Task, b: Task) -> bool:
    if a.task_id == b.task_id:
        return True
    if a.gold_answer and a.gold_answer == b.gold_answer and gold_doc_key(a) == gold_doc_key(b):
        return True
    if gold_doc_key(a) and gold_doc_key(a) == gold_doc_key(b):
        return True
    return False


def _blocked(task: Task, sources: Sequence[Task]) -> bool:
    return any(_same_surface(task, src) for src in sources)


def select_near_tasks(
    world: World,
    *,
    source_tasks: Sequence[Task],
    n: int,
    rng: random.Random,
) -> tuple[list[Task], list[str]]:
    notes: list[str] = []
    patterns = {normalized_pattern(t.task_id) for t in source_tasks}
    caps = {t.capability for t in source_tasks}
    blocked_ids = {t.task_id for t in source_tasks}

    def pool_ok(t: Task) -> bool:
        return t.task_id not in blocked_ids and not _blocked(t, source_tasks)

    same_pattern = [
        t
        for t in world.tasks.values()
        if normalized_pattern(t.task_id) in patterns and pool_ok(t)
    ]
    same_cap = [
        t
        for t in world.tasks.values()
        if t.capability in caps and normalized_pattern(t.task_id) not in patterns and pool_ok(t)
    ]

    def take(pool: list[Task], k: int, splits: Sequence[str] | None = None) -> list[Task]:
        items = list(pool)
        if splits is not None:
            allowed = set(splits)
            preferred = [t for t in items if t.split in allowed]
            leftover = [t for t in items if t.split not in allowed]
            rng.shuffle(preferred)
            rng.shuffle(leftover)
            items = preferred + leftover
        else:
            rng.shuffle(items)
        out: list[Task] = []
        seen_ans: set[str] = set()
        for t in items:
            if t.gold_answer in seen_ans and len(out) < k // 2:
                continue
            seen_ans.add(t.gold_answer)
            out.append(t)
            if len(out) >= k:
                break
        if len(out) < k:
            extra = [t for t in items if t not in out]
            out.extend(extra[: k - len(out)])
        return out[:k]

    chosen = take(same_pattern, n, NEAR_SPLITS)
    if len(chosen) < n:
        notes.append(f"near_fill_same_cap:{n - len(chosen)}")
        chosen.extend(take([t for t in same_cap if t not in chosen], n - len(chosen), NEAR_SPLITS))
    if len(chosen) < n:
        notes.append(f"near_shortfall:{n - len(chosen)}")
    return chosen[:n], notes


def select_far_tasks(
    world: World,
    *,
    bucket: str,
    blocked: Sequence[Task],
    n: int,
    rng: random.Random,
) -> list[Task]:
    caps = FAR_CAPS_BY_BUCKET[bucket]
    blocked_ids = {t.task_id for t in blocked}
    pool = [
        t
        for t in world.tasks.values()
        if t.capability in caps and t.task_id not in blocked_ids and not _blocked(t, blocked)
    ]
    rng.shuffle(pool)
    preferred = [t for t in pool if t.split in {"future", "validation"}]
    rest = [t for t in pool if t not in preferred]
    ordered = preferred + rest
    # diversify capability
    by_cap: dict[str, list[Task]] = {}
    for t in ordered:
        by_cap.setdefault(t.capability, []).append(t)
    mixed: list[Task] = []
    keys = [c for c in caps if c in by_cap]
    i = 0
    while len(mixed) < n and keys:
        cap = keys[i % len(keys)]
        if by_cap[cap]:
            mixed.append(by_cap[cap].pop(0))
        else:
            keys = [k for k in keys if by_cap[k]]
            if not keys:
                break
            continue
        i += 1
    return mixed[:n]


def select_retention_tasks(
    world: World,
    *,
    blocked_ids: set[str],
    n: int,
    rng: random.Random,
) -> list[Task]:
    pool = [
        t
        for t in world.tasks.values()
        if t.stage == 0 and t.capability == "single_hop" and t.task_id not in blocked_ids
    ]
    preferred = [t for t in pool if t.split in {"retention", "validation"}]
    rest = [t for t in pool if t not in preferred]
    rng.shuffle(preferred)
    rng.shuffle(rest)
    return (preferred + rest)[:n]


def build_probe_set(
    world: World,
    *,
    batch_id: str,
    bucket: str,
    experiences: Sequence[Experience],
    n_near: int = 16,
    n_far: int = 16,
    n_retention: int = 16,
    seed: int = 0,
) -> ProbeSet:
    rng = random.Random(seed)
    source_tasks = [world.tasks[e.task_id] for e in experiences if e.task_id in world.tasks]
    near, notes = select_near_tasks(world, source_tasks=source_tasks, n=n_near, rng=rng)
    blocked = [*source_tasks, *near]
    far = select_far_tasks(world, bucket=bucket, blocked=blocked, n=n_far, rng=rng)
    blocked_ids = {t.task_id for t in [*blocked, *far]}
    retention = select_retention_tasks(world, blocked_ids=blocked_ids, n=n_retention, rng=rng)
    return ProbeSet(
        batch_id=batch_id,
        bucket=bucket,
        experience_task_ids=[e.task_id for e in experiences],
        near=near,
        far=far,
        retention=retention,
        notes=notes,
    )


def split_experiences_into_batches(
    experiences: Sequence[Experience],
    *,
    n_batches: int,
    batch_size: int,
    seed: int,
) -> list[list[Experience]]:
    rng = random.Random(seed)
    by_pattern: dict[str, list[Experience]] = {}
    for exp in experiences:
        by_pattern.setdefault(task_code(exp.task_id), []).append(exp)
    # round-robin patterns so a batch is not one entity family only, but stays in-bucket
    striped: list[Experience] = []
    queues = {k: list(v) for k, v in by_pattern.items()}
    for items in queues.values():
        rng.shuffle(items)
    keys = sorted(queues)
    while any(queues[k] for k in keys):
        for k in keys:
            if queues[k]:
                striped.append(queues[k].pop())
    batches: list[list[Experience]] = []
    used: set[str] = set()
    i = 0
    while len(batches) < n_batches and i < len(striped):
        batch: list[Experience] = []
        while len(batch) < batch_size and i < len(striped):
            exp = striped[i]
            i += 1
            if exp.task_id in used:
                continue
            batch.append(exp)
            used.add(exp.task_id)
        if len(batch) == batch_size:
            batches.append(batch)
        elif batch and len(batches) < n_batches:
            # last incomplete: fill from unused
            rest = [e for e in experiences if e.task_id not in used]
            rng.shuffle(rest)
            for exp in rest:
                batch.append(exp)
                used.add(exp.task_id)
                if len(batch) == batch_size:
                    break
            if len(batch) == batch_size:
                batches.append(batch)
    return batches[:n_batches]


def bucket_pool(experiences: Sequence[Experience], bucket: str) -> list[Experience]:
    caps = set(TARGET_CAPS[bucket])
    primary = BUCKET_PRIMARY_CAP[bucket]
    primary_pool = [e for e in experiences if e.capability == primary]
    extra = [e for e in experiences if e.capability in caps and e.capability != primary]
    # prefer primary capability; S1 uses multi_hop not query_decomp as main target
    return primary_pool + extra
