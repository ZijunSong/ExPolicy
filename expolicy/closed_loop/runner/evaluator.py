"""Shadow / retention / future_test evaluation under the current agent state."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from expolicy.closed_loop.agent_state import ClosedLoopRuntime
from expolicy.closed_loop.constants import CAPS, DATASETS
from expolicy.closed_loop.io_utils import append_jsonl, write_json
from expolicy.environments.public_search.evaluator import score_public
from expolicy.environments.search.world import Task


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def evaluate_tasks(
    runtime: ClosedLoopRuntime,
    tasks: Sequence[Task],
    *,
    out: Path | None = None,
    phase: str = "eval",
    heartbeat=None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    tokens = 0
    tools = 0
    for i, task in enumerate(tasks):
        if heartbeat:
            heartbeat(phase=phase, i=i, n=len(tasks), task_id=task.task_id, cap=task.capability)
        exp = runtime.run_task(task)
        row = score_public(exp, task)
        row["dispatch"] = runtime.last_dispatch
        row["memory_injected"] = runtime.last_memory_injected
        row["persistent_state_hash"] = runtime.persistent_state_hash()
        row["phase"] = phase
        rows.append(row)
        tokens += int(exp.cost.tokens or 0)
        tools += int(exp.cost.tool_calls or 0)
        if out is not None:
            append_jsonl(out / f"{phase}_rows.jsonl", row)
    by_cap: dict[str, list[float]] = defaultdict(list)
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cap[str(row.get("capability") or "")].append(float(row.get("em") or 0.0))
        ds = str(row.get("dataset") or "")
        if ds:
            by_ds[ds].append(float(row.get("em") or 0.0))
    em_by_cap = {c: mean(by_cap.get(c) or [0.0]) for c in CAPS}
    per_ds = {ds: mean(vs) for ds, vs in by_ds.items()}
    ds_vals = [per_ds[ds] for ds in DATASETS if ds in per_ds]
    macro = mean(ds_vals) if ds_vals else mean([em_by_cap[c] for c in CAPS])
    summary = {
        "phase": phase,
        "n": len(rows),
        "macro": macro,
        "em_by_capability": em_by_cap,
        "em_by_dataset": per_ds,
        "tokens": tokens,
        "tools": tools,
        "wall_seconds": time.time() - t0,
        "dispatch_counts": _count([r.get("dispatch") for r in rows]),
    }
    if out is not None:
        write_json(out / f"{phase}_summary.json", summary)
    return summary


def _count(items: Sequence[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        key = str(x or "")
        out[key] = out.get(key, 0) + 1
    return out
