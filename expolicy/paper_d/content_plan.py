"""Fixed evaluation blocks. Counts are part of the protocol, not a result."""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from expolicy.paper_d.content_protocol import PROTOCOL
from expolicy.r9_lab.io_util import sha256_json


CONDITIONS = tuple(PROTOCOL["conditions"])


def build_block(
    tasks: Sequence[Mapping[str, Any]],
    *,
    block_id: str,
    state_id: str,
    batches: Sequence[str] | None = None,
    development_or_new: str = "new",
) -> list[dict[str, Any]]:
    cells = []
    batch_ids = list(batches or PROTOCOL["batches"])
    for batch_id in batch_ids:
        for task in tasks:
            for repeat_id in range(int(PROTOCOL["repeats"])):
                order = list(CONDITIONS)
                rng = random.Random(
                    f"{PROTOCOL['order_seed']}:{block_id}:{state_id}:{batch_id}:{task['task_id']}:{repeat_id}"
                )
                rng.shuffle(order)
                for position, condition in enumerate(order):
                    cells.append(
                        {
                            "block_id": block_id,
                            "state_id": state_id,
                            "batch_id": batch_id,
                            "condition": condition,
                            "task_id": task["task_id"],
                            "sample_index": task["sample_index"],
                            "instruction": task["instruction"],
                            "raw_task_hash": task["raw_task_hash"],
                            "task_family": task.get("task_family") or "unclustered",
                            "probe_role": task["probe_role"],
                            "development_or_new": development_or_new,
                            "repeat_id": repeat_id,
                            "randomized_position": position,
                        }
                    )
    return cells


def execution_identity(cell: Mapping[str, Any], *, state_hash: str, control_hash: str) -> str:
    return sha256_json(
        {
            "stage": PROTOCOL["stage_id"],
            "block": cell["block_id"],
            "env": "os",
            "state": state_hash,
            "batch": cell["batch_id"],
            "condition": cell["condition"],
            "control": control_hash,
            "task": cell["task_id"],
            "sample": cell["sample_index"],
            "role": cell["probe_role"],
            "repeat": cell["repeat_id"],
            "model": PROTOCOL["served_name"],
            "temperature": PROTOCOL["temperature"],
            "top_p": PROTOCOL["top_p"],
            "max_tokens": PROTOCOL["max_tokens"],
            "seed": PROTOCOL["inference_seed"],
        }
    )[:24]


def expected_counts() -> dict[str, int]:
    conditions = len(CONDITIONS)
    batches = len(PROTOCOL["batches"])
    repeats = int(PROTOCOL["repeats"])
    main_tasks = int(PROTOCOL["n_future"]) + int(PROTOCOL["n_retention"])
    history_tasks = int(PROTOCOL["history_future"]) + int(PROTOCOL["history_retention"])
    return {
        "main": batches * conditions * main_tasks * repeats,
        "history": batches * conditions * history_tasks * repeats,
        "case": batches * conditions * 2 * repeats,
    }
