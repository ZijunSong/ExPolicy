"""Candidate micro-batch construction. No future/shadow performance may be used."""

from __future__ import annotations

import random
from typing import Any, Sequence

from expolicy.r9_lab.constants import ENVS, JOINT_M_CANDIDATES, JOINT_MICRO_BATCH


def generate_candidates(
    pool: Sequence[dict[str, Any]],
    *,
    seed: int,
    round_id: int,
    m: int = JOINT_M_CANDIDATES,
    batch_size: int = JOINT_MICRO_BATCH,
    seen_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Random + skill/diversity coverage. Not an acquisition heuristic over utility."""
    seen = set(seen_ids or [])
    remaining = [x for x in pool if str(x.get("task_id")) not in seen]
    rng = random.Random(seed * 1009 + round_id)
    by_env: dict[str, list[dict[str, Any]]] = {e: [] for e in ENVS}
    for row in remaining:
        env = str(row.get("env") or "")
        by_env.setdefault(env, []).append(row)
    for env in by_env:
        rng.shuffle(by_env[env])
    candidates: list[dict[str, Any]] = []
    for i in range(m):
        batch: list[dict[str, Any]] = []
        # Round-robin envs for balance.
        env_cycle = list(ENVS)
        rng.shuffle(env_cycle)
        guard = 0
        while len(batch) < batch_size and guard < 10_000:
            guard += 1
            env = env_cycle[(len(batch)) % max(1, len(env_cycle))]
            bucket = by_env.get(env) or []
            if not bucket:
                # steal from any env
                stolen = False
                for other in ENVS:
                    if by_env.get(other):
                        batch.append(by_env[other].pop())
                        stolen = True
                        break
                if not stolen:
                    break
                continue
            batch.append(bucket.pop())
        candidates.append(
            {
                "candidate_id": f"r{round_id}_c{i}",
                "round_id": round_id,
                "seed": seed,
                "task_ids": [str(x["task_id"]) for x in batch],
                "items": batch,
                "uses_future": False,
                "uses_shadow": False,
            }
        )
    return candidates
