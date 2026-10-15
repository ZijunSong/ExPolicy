"""Confirmation plan. Six batches share tasks; repeats are not new tasks."""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from expolicy.paper_d.confirm_protocol import PROTOCOL
from expolicy.r9_lab.io_util import sha256_json


def execution_identity(cell: Mapping[str, Any], *, state_hash: str, control_hash: str) -> str:
    return sha256_json(
        {
            "stage": PROTOCOL["stage_id"],
            "batch": cell["batch_id"],
            "condition": cell["condition"],
            "state": state_hash,
            "control": control_hash,
            "task": cell["task_id"],
            "role": cell["probe_role"],
            "repeat": cell["repeat_id"],
            "model": PROTOCOL["served_name"],
            "temperature": PROTOCOL["temperature"],
            "seed": PROTOCOL["inference_seed"],
        }
    )[:24]


def build_plan(tasks: Sequence[Mapping[str, Any]], batch_ids: Sequence[str]) -> list[dict[str, Any]]:
    cells = []
    for repeat_id in range(int(PROTOCOL["repeats"])):
        for task in tasks:
            batch_order = list(batch_ids)
            random.Random(f"{PROTOCOL['order_seed']}:batch:{task['task_id']}:{repeat_id}").shuffle(batch_order)
            for batch_id in batch_order:
                conditions = list(PROTOCOL["conditions"])
                random.Random(f"{PROTOCOL['order_seed']}:condition:{batch_id}:{task['task_id']}:{repeat_id}").shuffle(conditions)
                for position, condition in enumerate(conditions):
                    cells.append(
                        {
                            "block_id": "confirm",
                            "state_id": "S0",
                            "batch_id": batch_id,
                            "condition": condition,
                            "task_id": task["task_id"],
                            "sample_index": task["sample_index"],
                            "probe_role": task["probe_role"],
                            "raw_task_hash": task["raw_task_hash"],
                            "task_family": task.get("task_family") or "unclustered",
                            "development_or_new": "confirm",
                            "repeat_id": repeat_id,
                            "randomized_position": position,
                            "instruction": task["instruction"],
                        }
                    )
    return cells


def expected_count(n_tasks: int, n_batches: int = 6) -> int:
    return n_batches * len(PROTOCOL["conditions"]) * n_tasks * int(PROTOCOL["repeats"])
