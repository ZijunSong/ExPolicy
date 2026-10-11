#!/usr/bin/env python3
"""Bamboogle zero-shot transfer after main 10-seed acquisition is frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expolicy.environments.public_search.retriever import RetrieverClient
from expolicy.environments.public_search.skills import run_public_skill_on_task
from expolicy.experiments.common import mean, write_json
from expolicy.experiments.r5_3 import (
    MIN_SUPPORT,
    R53_ROOT,
    count_only_store,
    eval_with_skills,
    load_base_rows,
    load_json,
    load_split,
    split_tasks,
    world_from_tasks,
)
from expolicy.environments.public_search.environment import PublicSearchEnvironment
from expolicy.experiments.r5_3 import _load_jsonl_exp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=str(R53_ROOT))
    p.add_argument("--output-dir", default=str(R53_ROOT))
    p.add_argument("--retriever-url", default="http://127.0.0.1:8000")
    p.add_argument("--methods", nargs="+", default=["frontier", "ace_skill_prioritized", "passive"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = p.parse_args()
    root = Path(args.input_dir)
    out = Path(args.output_dir)
    split = load_split(root)
    tasks = split_tasks(split, "bamboogle") if split.get("bamboogle") else []
    if not tasks:
        # records stored under key bamboogle
        from expolicy.environments.public_search.datasets import to_task
        from expolicy.environments.public_search.capability_map import apply_mapping

        recs = apply_mapping(split.get("bamboogle") or [])
        tasks = [to_task(r, capability=r.get("capability") or "bridge_decomposition", split="bamboogle") for r in recs]
    world = world_from_tasks(tasks)
    env = PublicSearchEnvironment(world, RetrieverClient(args.retriever_url))
    base_rows = load_base_rows(root)
    # if bamboogle wasn't in base_eval, we can still run no_skill via missing-row error — require rows
    missing = [t.task_id for t in tasks if t.task_id not in base_rows]
    rows_out = []
    for method in args.methods:
        ems = []
        for seed in args.seeds:
            run = root / method / f"seed_{seed}"
            if not (run / "DONE").exists():
                continue
            acquired = _load_jsonl_exp(run / "trajectories.jsonl")
            store, _ = count_only_store(acquired, min_support=MIN_SUPPORT, skill_dir=run / "skill_versions_bamboogle")
            payload = eval_with_skills(tasks, store=store, base_rows=base_rows, env=env, model=None, dispatch="oracle")
            ems.append(payload["macro"] if payload.get("macro") is not None else mean(float(r.get("em") or 0) for r in payload["rows"]))
            rows_out.append(
                {
                    "method": method,
                    "seed": seed,
                    "bamboogle_em": mean(float(r.get("em") or 0) for r in payload["rows"]),
                    "bamboogle_f1": mean(float(r.get("f1") or 0) for r in payload["rows"]),
                    "n": len(payload["rows"]),
                    "n_skill": payload["n_skill_invocations"],
                    "n_active_skills": payload["n_active_skills"],
                }
            )
        if ems:
            print(f"[bamboogle] {method} mean EM={sum(ems)/len(ems):.4f} n_seeds={len(ems)}", flush=True)
    # no_skill row
    noskill = eval_with_skills(tasks, store=None, base_rows=base_rows, env=env, model=None)
    rows_out.insert(
        0,
        {
            "method": "no_skill",
            "seed": -1,
            "bamboogle_em": mean(float(r.get("em") or 0) for r in noskill["rows"]),
            "bamboogle_f1": mean(float(r.get("f1") or 0) for r in noskill["rows"]),
            "n": len(noskill["rows"]),
            "n_skill": 0,
            "n_active_skills": 0,
        },
    )
    from expolicy.experiments.r5_1 import write_csv

    write_csv(out / "R5_3_BAMBOOGLE_TRANSFER.csv", rows_out)
    write_json(out / "bamboogle_transfer.json", rows_out)
    print(f"[bamboogle] wrote {out / 'R5_3_BAMBOOGLE_TRANSFER.csv'} missing_base={len(missing)}", flush=True)
    if missing:
        raise SystemExit(f"bamboogle missing base rows: {len(missing)}. Re-run base_eval including bamboogle.")


if __name__ == "__main__":
    main()
