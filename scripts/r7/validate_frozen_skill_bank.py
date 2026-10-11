#!/usr/bin/env python3
"""Freeze skill-bank hashes and verify SearchEnv tool-sequence differences."""

from __future__ import annotations

import argparse
from pathlib import Path

from expolicy.closed_loop.constants import CAPS, DEFAULT_RETRIEVER_URL, R7_ROOT
from expolicy.closed_loop.frozen_skill_bank import assert_frozen, run_skill, write_manifest
from expolicy.closed_loop.io_utils import write_json
from expolicy.closed_loop.split_data import load_split, split_tasks
from expolicy.environments.public_search.environment import PublicSearchEnvironment, PublicWorld
from expolicy.environments.public_search.retriever import RetrieverClient


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--retriever-url", default=DEFAULT_RETRIEVER_URL)
    p.add_argument("--output-dir", default=str(R7_ROOT / "audit"))
    p.add_argument("--n-per-cap", type=int, default=2)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(out / "FROZEN_SKILL_BANK_MANIFEST.json")
    assert_frozen(out / "FROZEN_SKILL_BANK_MANIFEST.json")

    health = RetrieverClient(args.retriever_url).health()
    split = load_split()
    pool = split_tasks(split, "calibration")
    world = PublicWorld()
    for t in pool:
        world.add_task(t)
    env = PublicSearchEnvironment(world, RetrieverClient(args.retriever_url), top_k=5)
    rows = []
    for cap in CAPS:
        tasks = [t for t in pool if t.capability == cap][: args.n_per_cap]
        for task in tasks:
            exp = run_skill(cap, env, task, model=None)
            tools = [a.get("tool") for a in exp.actions]
            ok = tools.count("search") >= 2 and tools.count("open") >= 1 and tools[-1] == "answer"
            rows.append(
                {
                    "capability": cap,
                    "task_id": task.task_id,
                    "tools": tools,
                    "n_search": tools.count("search"),
                    "n_open": tools.count("open"),
                    "behavior_ok": ok,
                    "dispatch": "frozen_skill",
                }
            )
            if not ok:
                raise SystemExit(f"skill {cap} did not change tool flow on {task.task_id}: {tools}")
    write_json(out / "FROZEN_SKILL_VALIDATION.json", {"health": health, "manifest": manifest, "rows": rows})
    print("frozen skill bank OK", "n", len(rows), "retriever", health)


if __name__ == "__main__":
    main()
