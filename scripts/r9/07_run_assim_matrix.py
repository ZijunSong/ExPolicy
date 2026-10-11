#!/usr/bin/env python3
"""36x4 independent clones. Only shadow+retention. Per-cell DONE.json under matrix/{batch_id}/{action}/."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from expolicy.r9_lab.audit.leakage import assert_future_not_in_update
from expolicy.r9_lab.benchmark.official_env import OfficialLabEnv, OfficialLabInfraError, summarize_episodes
from expolicy.r9_lab.benchmark.split import load_split, records_in_bucket
from expolicy.r9_lab.constants import ACTIONS, ENVS, OUTPUT_ROOT, VLLM_PORT, model_by_id
from expolicy.r9_lab.io_util import read_json
from expolicy.r9_lab.llm_client import VLLMClient, probe_vllm
from expolicy.r9_lab.manifest import load_manifest
from expolicy.r9_lab.matrix_runtime import as_experiences, record_to_item, run_items
from expolicy.r9_lab.resume import run_cell_once
from expolicy.r9_lab.state.persistent_state import PersistentState
from expolicy.r9_lab.watchdog_util import append_recorded_pid, write_status_live


def _s0_dir(out: Path, env: str) -> Path:
    return out / "matrix" / "_s0" / env / "state"


def ensure_s0_baseline(
    *,
    out: Path,
    env: str,
    split: dict,
    lab: OfficialLabEnv,
    client,
    gpu: str | None,
    resume: bool,
) -> dict:
    cell = out / "matrix" / "_s0" / env

    def _job() -> dict:
        s0 = PersistentState.create(_s0_dir(out, env))
        shadow_items = [record_to_item(r) for r in records_in_bucket(split, env, "shadow_reward_probe")]
        ret_items = [record_to_item(r) for r in records_in_bucket(split, env, "retention_probe")]
        write_status_live(cell / "STATUS_LIVE.json", gpu=gpu, stage="s0_shadow", batch_id=env)
        shadow_res = run_items(lab, client, shadow_items, s0, status_path=cell / "STATUS_LIVE.json", gpu=gpu, stage="s0_shadow", batch_id=env)
        write_status_live(cell / "STATUS_LIVE.json", gpu=gpu, stage="s0_retention", batch_id=env)
        ret_res = run_items(lab, client, ret_items, s0, status_path=cell / "STATUS_LIVE.json", gpu=gpu, stage="s0_retention", batch_id=env)
        shadow = summarize_episodes(shadow_res)
        retention = summarize_episodes(ret_res)
        return {
            "env": env,
            "shadow": shadow["success"],
            "retention": retention["success"],
            "shadow_n": shadow["n"],
            "retention_n": retention["n"],
            "tokens": int(shadow["tokens"]) + int(retention["tokens"]),
            "used_future_test": False,
        }

    result = run_cell_once(cell, _job, resume=resume)
    if result.get("skipped"):
        raw = read_json(cell / "DONE.json")
        return raw.get("payload") or raw
    return result.get("payload") or {}


def ensure_acquire(
    *,
    out: Path,
    batch: dict,
    split: dict,
    lab: OfficialLabEnv,
    client,
    gpu: str | None,
    resume: bool,
) -> list[dict]:
    cell = out / "matrix" / batch["batch_id"] / "_acquire"

    def _job() -> dict:
        write_status_live(
            cell / "STATUS_LIVE.json",
            gpu=gpu,
            stage="matrix_acquire",
            batch_id=batch["batch_id"],
        )
        results = run_items(
            lab,
            client,
            batch["items"],
            None,
            status_path=cell / "STATUS_LIVE.json",
            gpu=gpu,
            stage="matrix_acquire",
            batch_id=batch["batch_id"],
        )
        experiences = as_experiences(results, env=batch["env"], bucket="acquisition_pool")
        assert_future_not_in_update([str(x["task_id"]) for x in experiences], split)
        stats = summarize_episodes(results)
        return {
            "batch_id": batch["batch_id"],
            "env": batch["env"],
            "experiences": experiences,
            "batch_success_rate": stats["success"],
            "mean_steps": stats["mean_steps"],
            "invalid_action_rate": stats["invalid_action_rate"],
            "tokens": stats["tokens"],
            "wall_time": stats["wall_time"],
            "used_future_test": False,
        }

    result = run_cell_once(cell, _job, resume=resume)
    if result.get("skipped"):
        raw = read_json(cell / "DONE.json")
        payload = raw.get("payload") or raw
    else:
        payload = result.get("payload") or {}
    return payload.get("experiences") or []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--batch-id", action="append", dest="batch_ids")
    p.add_argument("--action", action="append", dest="actions")
    p.add_argument("--env", action="append", dest="envs")
    p.add_argument("--gpu", default=None)
    p.add_argument("--port", type=int, default=VLLM_PORT)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--allow-synthetic", action="store_true")
    args = p.parse_args(argv)
    out = args.output_root
    append_recorded_pid(
        out / "jobs",
        os.getpid(),
        f"07_run_assim_matrix shard={args.shard}/{args.shards} gpu={args.gpu} port={args.port}",
    )
    frozen = read_json(out / "FROZEN_BATCHES.json") if (out / "FROZEN_BATCHES.json").exists() else {"batches": []}
    actions = args.actions or list(ACTIONS)
    batches = frozen.get("batches") or []
    if args.envs:
        batches = [b for b in batches if b["env"] in set(args.envs)]
    if args.batch_ids:
        batches = [b for b in batches if b["batch_id"] in set(args.batch_ids)]
    shards = max(1, int(args.shards))
    shard = int(args.shard) % shards
    batches = [b for i, b in enumerate(batches) if i % shards == shard]

    if args.dry_run:
        submitted = 0
        skipped = 0
        for batch in batches:
            for action in actions:
                cell = out / "matrix" / batch["batch_id"] / action

                def _run(b=batch, a=action) -> dict:
                    return {"batch_id": b["batch_id"], "action": a, "dry_run": True, "used_future_test": False}

                result = run_cell_once(cell, _run, resume=args.resume)
                if result.get("skipped"):
                    skipped += 1
                else:
                    submitted += 1
        print(json.dumps({"submitted": submitted, "skipped": skipped, "dry_run": True}))
        return 0

    man = load_manifest(out / "model_manifest.json")
    if not man.get("selected_model_id"):
        raise RuntimeError("07_run_assim_matrix requires frozen selected actor")
    spec = model_by_id(str(man["selected_model_id"]))
    split = load_split(out / "split.json")
    if split.get("synthetic") and not args.allow_synthetic:
        raise RuntimeError("Refusing synthetic split.json unless --allow-synthetic")
    probe_vllm(int(args.port))
    client = VLLMClient.from_model_id(spec.model_id, port=int(args.port), timeout=600.0)

    submitted = 0
    skipped = 0
    labs: dict[str, OfficialLabEnv] = {}
    try:
        needed_envs = sorted({b["env"] for b in batches}) or list(ENVS)
        for env in needed_envs:
            try:
                labs[env] = OfficialLabEnv(env)
            except OfficialLabInfraError as exc:
                print(json.dumps({"env": env, "infra_error": str(exc)}))
                raise
        baselines = {}
        for env in needed_envs:
            baselines[env] = ensure_s0_baseline(
                out=out,
                env=env,
                split=split,
                lab=labs[env],
                client=client,
                gpu=args.gpu,
                resume=args.resume,
            )
        for batch in batches:
            env = batch["env"]
            lab = labs[env]
            acquire_exps = None
            shadow_items = [record_to_item(r) for r in records_in_bucket(split, env, "shadow_reward_probe")]
            ret_items = [record_to_item(r) for r in records_in_bucket(split, env, "retention_probe")]
            val_items = [record_to_item(r) for r in records_in_bucket(split, env, "assim_validation")]
            for action in actions:
                cell = out / "matrix" / batch["batch_id"] / action

                def _run(b=batch, a=action) -> dict:
                    nonlocal acquire_exps
                    write_status_live(
                        cell / "STATUS_LIVE.json",
                        gpu=args.gpu,
                        stage="matrix_cell",
                        batch_id=b["batch_id"],
                        action=a,
                    )
                    if acquire_exps is None:
                        acquire_exps = ensure_acquire(
                            out=out,
                            batch=b,
                            split=split,
                            lab=lab,
                            client=client,
                            gpu=args.gpu,
                            resume=args.resume,
                        )
                    # Replay cached experiences; do not re-roll acquire.
                    from expolicy.r9_lab.matrix_runtime import apply_action, eval_success, make_validate_fn
                    from expolicy.r9_lab.evaluation.utility import compute_utility
                    from expolicy.r9_lab.state.clone import clone_state
                    from expolicy.r9_lab.matrix_runtime import failure_type_stats, state_size

                    s0 = PersistentState.load(_s0_dir(out, env))
                    branch = clone_state(s0, cell / "state")
                    validate_fn = make_validate_fn(
                        lab,
                        client,
                        val_items[:8],
                        cell / "_val",
                        status_path=cell / "STATUS_LIVE.json",
                        gpu=args.gpu,
                        batch_id=f"{b['batch_id']}:{a}",
                    )
                    assim = apply_action(a, branch, acquire_exps, client=client, validate_fn=validate_fn)
                    assert_future_not_in_update(
                        [str(x.get("task_id") or "") for x in acquire_exps],
                        split,
                    )
                    shadow = eval_success(
                        lab,
                        client,
                        shadow_items,
                        branch,
                        status_path=cell / "STATUS_LIVE.json",
                        gpu=args.gpu,
                        stage="matrix_shadow",
                        batch_id=f"{b['batch_id']}:{a}",
                    )
                    retention = eval_success(
                        lab,
                        client,
                        ret_items,
                        branch,
                        status_path=cell / "STATUS_LIVE.json",
                        gpu=args.gpu,
                        stage="matrix_retention",
                        batch_id=f"{b['batch_id']}:{a}",
                    )
                    baseline = baselines[env]
                    util = compute_utility(
                        shadow_before=float(baseline["shadow"]),
                        shadow_after=float(shadow["success"]),
                        retention_before=float(baseline["retention"]),
                        retention_after=float(retention["success"]),
                    )
                    sizes = state_size(branch)
                    payload = {
                        "env": env,
                        "action": a,
                        "batch_id": b["batch_id"],
                        "difficulty": b.get("difficulty_mean") or 0,
                        "skill_count": b.get("skill_count_mean") or 0,
                        "assim": assim,
                        "batch_success_rate": sum(1 for x in acquire_exps if x.get("success")) / max(1, len(acquire_exps)),
                        "mean_steps": sum(float(x.get("steps") or 0) for x in acquire_exps) / max(1, len(acquire_exps)),
                        "invalid_action_rate": sum(float(x.get("invalid_actions") or 0) for x in acquire_exps)
                        / max(1, sum(float(x.get("steps") or 0) for x in acquire_exps)),
                        "tokens": int(shadow.get("tokens") or 0) + int(retention.get("tokens") or 0),
                        "wall_time": float(shadow.get("wall_time") or 0) + float(retention.get("wall_time") or 0),
                        "shadow_after": shadow["success"],
                        "retention_after": retention["success"],
                        "shadow_n": shadow["n"],
                        "retention_n": retention["n"],
                        "base_success_rate": baseline["shadow"],
                        "failure_type_stats": {
                            "n": len(acquire_exps),
                            "n_fail": sum(1 for x in acquire_exps if not x.get("success")),
                            "n_infra": sum(1 for x in acquire_exps if x.get("infra_failure")),
                            "n_parser_invalid": sum(1 for x in acquire_exps if not x.get("parser_valid", True)),
                        },
                        "utility": util,
                        "U": util["U"],
                        "used_future_test": False,
                        **sizes,
                    }
                    return payload

                result = run_cell_once(cell, _run, resume=args.resume)
                if result.get("skipped"):
                    skipped += 1
                else:
                    submitted += 1
    finally:
        for lab in labs.values():
            lab.close()
    print(json.dumps({"submitted": submitted, "skipped": skipped, "shard": shard, "shards": shards, "n_batches": len(batches)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
