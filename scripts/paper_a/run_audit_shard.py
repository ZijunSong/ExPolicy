#!/usr/bin/env python3
"""Score one disjoint slice of the no-op audit.

Each worker has its own vLLM port and its own environment container. Requests
on a port stay one-at-a-time, so this is data parallel rather than a packed
batch. Packed batches can change scores; this does not relabel a serial run
as a concurrent replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from expolicy.paper_a.audit import (  # noqa: E402
    BRANCH_SENTINEL_PREFIX,
    NOOP_BRANCHES,
    AuditTask,
    _check_task_sets,
    _component_diffs,
    _default_prepare,
    _prompt_leak,
    component_fingerprint,
)
from expolicy.paper_a.config import load_config  # noqa: E402
from expolicy.paper_a.execute import load_acquired_experiences, probe_tasks, run_probes  # noqa: E402
from expolicy.r9_lab.benchmark.official_env import OfficialLabEnv  # noqa: E402
from expolicy.r9_lab.benchmark.split import load_split  # noqa: E402
from expolicy.r9_lab.constants import LAB_SYSTEM_PROMPT  # noqa: E402
from expolicy.r9_lab.io_util import atomic_write_json, read_json  # noqa: E402
from expolicy.r9_lab.llm_client import VLLMClient  # noqa: E402
from expolicy.r9_lab.state.clone import clone_state  # noqa: E402


def _safe(task_id: str) -> str:
    return task_id.replace("/", "_").replace(":", "_")


def _outcome_path(root: Path, env: str, batch_id: str, mode: str, repeat_id: int, branch: str, task_id: str) -> Path:
    return root / env / batch_id / mode / f"r{repeat_id}" / branch / f"{_safe(task_id)}.json"


def _tasks_for_shard(tasks: list[AuditTask], shard: int, shards: int) -> list[AuditTask]:
    ordered = sorted(tasks, key=lambda task: task.probe_task_id)
    return [task for index, task in enumerate(ordered) if index % shards == shard]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "paper_a_v1" / "parallel")
    args = parser.parse_args(argv)
    cfg = load_config()
    split = load_split(cfg.path("split.path"))
    tasks = [
        AuditTask(row["probe_task_id"], row["eval_split"], row["instruction"], args.env)
        for row in probe_tasks(split, args.env)
    ]
    problems = _check_task_sets(tasks)
    if problems:
        raise SystemExit("; ".join(problems))
    mine = _tasks_for_shard(tasks, args.shard, args.shards)
    frozen = read_json(cfg.path("frozen_batches.path"))
    batches = [batch for batch in frozen["batches"] if batch["env"] == args.env][: int(cfg.raw["measurement"]["audit_batches_per_env"])]
    client = VLLMClient.from_model_id("Q2", port=args.port, timeout=600.0)
    lab = OfficialLabEnv(args.env)
    modes = ["serial"]
    if cfg.raw["measurement"]["require_formal_concurrency_replay"]:
        modes.append("formal_concurrency")
    repeats = int(cfg.raw["measurement"]["noise_repeats"])
    try:
        for batch in batches:
            experiences = load_acquired_experiences(cfg.path("legacy.r9_matrix"), batch["batch_id"])
            if not experiences:
                raise SystemExit(f"missing acquisition for {batch['batch_id']}")
            s0_dir = cfg.path("legacy.r9_matrix") / "_s0" / args.env / "state"
            work = args.output_root / "states" / args.env / batch["batch_id"] / f"shard{args.shard}"
            branches = {}
            for name in NOOP_BRANCHES:
                # Clone the saved directory. Do not rewrite the shared S0.
                state = clone_state(s0_dir, work / name / "state")
                _default_prepare(name, state, experiences)
                branches[name] = state
            queries = [task.instruction for task in mine]
            fingerprints = {
                name: component_fingerprint(
                    state,
                    queries,
                    system_prompt=LAB_SYSTEM_PROMPT,
                    tool_config={"env": args.env, "max_round_locked": True},
                    model_identity={"port": args.port},
                    env_snapshot_id=args.env,
                )
                for name, state in branches.items()
            }
            atomic_write_json(
                args.output_root / "components" / args.env / batch["batch_id"] / f"shard{args.shard}.json",
                {"diffs_vs_base": _component_diffs(fingerprints), "shard": args.shard},
            )
            leaks = {name: _prompt_leak(state, mine, name) for name, state in branches.items()}
            if any(leaks.values()) and BRANCH_SENTINEL_PREFIX:
                atomic_write_json(
                    args.output_root / "components" / args.env / batch["batch_id"] / f"leaks{args.shard}.json",
                    leaks,
                )
            probes = [
                {
                    "probe_task_id": task.probe_task_id,
                    "eval_split": task.eval_split,
                    "instruction": task.instruction,
                    "sample_index": task.probe_task_id.split(":", 1)[-1],
                }
                for task in mine
            ]
            for mode in modes:
                for repeat_id in range(repeats):
                    for branch, state in branches.items():
                        pending = [
                            probe
                            for probe in probes
                            if not _outcome_path(
                                args.output_root / "outcomes",
                                args.env,
                                batch["batch_id"],
                                mode,
                                repeat_id,
                                branch,
                                probe["probe_task_id"],
                            ).exists()
                        ]
                        if not pending:
                            continue
                        print(
                            json.dumps(
                                {
                                    "shard": args.shard,
                                    "env": args.env,
                                    "batch": batch["batch_id"],
                                    "mode": mode,
                                    "repeat": repeat_id,
                                    "branch": branch,
                                    "n": len(pending),
                                }
                            ),
                            flush=True,
                        )
                        rows = run_probes(lab, client, state, pending)
                        for row in rows:
                            path = _outcome_path(
                                args.output_root / "outcomes",
                                args.env,
                                batch["batch_id"],
                                mode,
                                repeat_id,
                                branch,
                                str(row["probe_task_id"]),
                            )
                            row["mode"] = mode
                            row["repeat_id"] = repeat_id
                            row["branch"] = branch
                            row["batch_id"] = batch["batch_id"]
                            row["port"] = args.port
                            atomic_write_json(path, row)
    finally:
        lab.close()
    print(json.dumps({"shard_done": args.shard, "env": args.env}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
