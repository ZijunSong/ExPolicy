#!/usr/bin/env python3
"""A-v1.1 pilot runner. Default is a dry run. --execute is required before any model or environment call."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from expolicy.paper_a.v11 import (  # noqa: E402
    PROTOCOL_ID,
    PilotRefusal,
    assert_v11_launch,
    cache_usable,
    choose_pilot_batches,
    choose_smoke_tasks,
    classify_divergence,
    code_hash,
    legacy_reinterpretation,
    load_v11,
    paired_discard_utility,
    protocol_hash,
    selection_signal,
    split_probe_tasks,
)
from expolicy.r9_lab.io_util import read_json, sha256_file, write_json  # noqa: E402

BRANCHES = ("BASE", "DISCARD", "REJECTED_EDIT")
ACTIONS = ("DISCARD", "EPISODIC_REPLAY", "SEMANTIC_MEMORY")
TIE = ("DISCARD", "EPISODIC_REPLAY", "SEMANTIC_MEMORY")


def run_dir(cfg: dict[str, Any]) -> Path:
    return REPO / cfg["paths"]["output_root"] / cfg["paths"]["run_id"]


def _lock_count(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read().strip()
        handle.seek(0)
        return int(raw or "0"), handle


def charge(path: Path, n: int, limit: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        used = int(handle.read().strip() or "0")
        if used + n > limit:
            raise PilotRefusal(f"episode cap {limit} would be exceeded ({used}+{n})")
        used += n
        handle.seek(0)
        handle.truncate()
        handle.write(str(used))
        return used


def used_episodes(path: Path) -> int:
    if not path.exists():
        return 0
    return int(path.read_text(encoding="utf-8").strip() or "0")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify_assets(cfg: dict[str, Any]) -> None:
    from expolicy.r9_lab.io_util import sha256_json

    split = REPO / cfg["split"]["path"]
    frozen_path = REPO / cfg["frozen_batches"]["path"]
    manifest = REPO / cfg["model"]["manifest_path"]
    if sha256_file(split) != cfg["split"]["expect_file_sha256"]:
        raise PilotRefusal("split file hash mismatch")
    frozen = read_json(frozen_path)
    frozen_hash = sha256_json({key: value for key, value in frozen.items() if key != "sha256"})
    if frozen.get("sha256") != cfg["frozen_batches"]["expect_sha256"] or frozen_hash != frozen.get("sha256"):
        raise PilotRefusal("frozen batch hash mismatch")
    if int(frozen.get("n_batches") or 0) != int(cfg["frozen_batches"]["expect_n_batches"]):
        raise PilotRefusal("frozen batch count mismatch")
    body = read_json(manifest)
    if body.get("content_hash") != cfg["model"]["expect_manifest_content_hash"]:
        raise PilotRefusal("model manifest hash mismatch")


def service_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for server in cfg["servers"]:
        url = f"http://127.0.0.1:{server['port']}/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            model_id = payload["data"][0]["id"] if payload.get("data") else "unverified"
            status = "up" if model_id == cfg["model"]["served_name"] else "unexpected_model"
        except Exception as exc:  # noqa: BLE001
            model_id = "unverified"
            status = f"down:{type(exc).__name__}"
        rows.append(
            {
                "gpu": server["gpu"],
                "port": server["port"],
                "served_model": model_id,
                "status": status,
                "checkpoint": cfg["model"]["local_path"],
                "container_image": "unverified",
                "data_version": "unverified",
            }
        )
    return {"servers": rows, "sparql": "http://127.0.0.1:3001/sparql"}


def build_plan(cfg: dict[str, Any]) -> dict[str, Any]:
    split = read_json(REPO / cfg["split"]["path"])
    frozen = read_json(REPO / cfg["frozen_batches"]["path"])
    plan: dict[str, Any] = {"environments": {}}
    for env in cfg["environments"]:
        batches = choose_pilot_batches(
            frozen,
            env,
            n_discovery=int(cfg["pilot"]["n_discovery"]),
            n_reserved=int(cfg["pilot"]["n_reserved"]),
            seed=int(cfg["pilot"]["seed"]),
        )
        smoke = choose_smoke_tasks(
            split,
            env,
            n_shadow=int(cfg["pilot"]["smoke"]["n_shadow"]),
            n_retention=int(cfg["pilot"]["smoke"]["n_retention"]),
            seed=int(cfg["pilot"]["seed"]),
            exclude={"db:367", "db:414"},
        )
        probes = {}
        for bucket, key in (("shadow_reward_probe", "selection_shadow"), ("retention_probe", "selection_retention")):
            rows = split["envs"][env][bucket]
            probes[bucket] = split_probe_tasks(rows, int(cfg["pilot"]["probes"][env][key]), seed=int(cfg["pilot"]["seed"]))
        observed = []
        for batch in batches["discovery"] + batches["reserved"]:
            done = REPO / cfg["paths"]["acquire_root"] / batch["batch_id"] / "DISCARD" / "DONE.json"
            observed.append({"batch_id": batch["batch_id"], "seen_in_r9": done.exists()})
        plan["environments"][env] = {
            "discovery": [b["batch_id"] for b in batches["discovery"]],
            "reserved": [b["batch_id"] for b in batches["reserved"]],
            "shortfall": batches["shortfall"],
            "smoke_counts": {bucket: len(rows) for bucket, rows in smoke.items()},
            "probe_counts": {
                side: {bucket: len(probes[bucket][side]) for bucket in probes}
                for side in ("selection", "confirmation")
            },
            "r9_overlap": observed,
            "smoke": smoke,
            "probes": probes,
            "batches": batches,
        }
    return plan


def cmd_preflight(cfg: dict[str, Any]) -> int:
    verify_assets(cfg)
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    digest = protocol_hash(cfg)
    code = code_hash(
        [
            REPO / "expolicy/paper_a/audit.py",
            REPO / "expolicy/paper_a/stats.py",
            REPO / "expolicy/paper_a/v11.py",
            REPO / "expolicy/paper_a/execute.py",
            REPO / "expolicy/r9_lab/assimilators/semantic.py",
            REPO / "expolicy/r9_lab/runner.py",
            REPO / "scripts/paper_a/run_a_v11.py",
        ]
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    plan = build_plan(cfg)
    write_json(out / "PROTOCOL_LOCK.json", {"protocol_id": PROTOCOL_ID, "protocol_hash": digest, "code_hash": code, "head": head, "config": cfg})
    write_json(out / "PILOT_SPLIT_MANIFEST.json", {env: {k: v for k, v in body.items() if k not in {"smoke", "probes", "batches"}} for env, body in plan["environments"].items()})
    write_json(out / "SERVICE_AND_ENV_MANIFEST.json", service_manifest(cfg))
    (out / "LEGACY_REINTERPRETATION.md").write_text(legacy_reinterpretation(REPO / cfg["paths"]["legacy_audit"]), encoding="utf-8")
    (out / "CODE_CHANGE_SUMMARY.md").write_text(
        "\n".join(
            [
                "# A-v1.1 代码改动",
                "",
                f"基线提交 `{cfg['source_commit']}`。当前 HEAD `{head}`。",
                "",
                "已改的测量缺陷：后一块全零不能把前面的不确定块抬成通过；分项保留每个 no-op 分支和题型；全零或 n=1 的正态区间不再被当成总体噪声为零；分数不同只记 variation_observed，不再直接写成已定位的模型噪声。",
                "",
                "仍只是风险、这一轮没有假装已经证明的部分：旧审计轨迹没有逐步哈希，db:367、db:414 和 OS 翻转的首个分叉仍是 VARIATION_UNLOCATED。OS 不进入本轮。",
                "",
                "语义生成失败记为 OPERATOR_INVALID，不再静默退回启发式。注入长度只在本协议版本加 2048 token 上限。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(out / "PLAN.json", plan)
    print(f"preflight wrote {out}")
    return 0


def _probe(env: str, bucket: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "probe_task_id": row.get("task_id") or row.get("probe_task_id"),
        "eval_split": bucket,
        "instruction": row.get("instruction") or "",
        "sample_index": row.get("sample_index") or str(row.get("task_id", "")).split(":", 1)[-1],
        "env": env,
    }


def smoke_jobs(cfg: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    repeats = int(cfg["pilot"]["smoke"]["repeats"])
    for env, body in plan["environments"].items():
        tasks = []
        for bucket, rows in body["smoke"].items():
            tasks.extend((_probe(env, bucket, row), index) for index, row in enumerate(rows))
        for offset, (probe, index) in enumerate(tasks):
            rotation = offset % len(BRANCHES)
            order = list(BRANCHES[rotation:]) + list(BRANCHES[:rotation])
            for repeat in range(repeats):
                for branch in order:
                    jobs.append({"kind": "smoke", "env": env, "probe": probe, "branch": branch, "repeat": repeat, "order_index": index})
    return jobs


def _cache_key(cfg: dict[str, Any], job: dict[str, Any]) -> str:
    lock = read_json(run_dir(cfg) / "PROTOCOL_LOCK.json")
    return protocol_hash(
        {
            "protocol": lock["protocol_hash"],
            "code": lock["code_hash"],
            "model": cfg["model"]["local_path"],
            "job": {k: job[k] for k in job if k != "experiences"},
            "operator": cfg["pilot"]["operator_version"],
            "decoding": {
                "temperature": cfg["model"]["temperature"],
                "max_tokens": cfg["model"]["max_tokens"],
                "seed": cfg["model"]["inference_seed"],
            },
        }
    )


def _prepare_states(cfg: dict[str, Any], env: str, root: Path, experiences: list[dict[str, Any]]):
    from expolicy.r9_lab.assimilators.discard import apply_discard
    from expolicy.r9_lab.assimilators.skillopt import apply_skillopt
    from expolicy.r9_lab.state.clone import clone_state
    from expolicy.r9_lab.state.persistent_state import PersistentState

    s0 = PersistentState.load(REPO / cfg["paths"]["s0"] / env / "state")
    states = {}
    base = clone_state(s0, root / env / "BASE")
    states["BASE"] = base
    discarded = clone_state(s0, root / env / "DISCARD")
    apply_discard(discarded, experiences)
    states["DISCARD"] = discarded
    rejected = clone_state(s0, root / env / "REJECTED_EDIT")
    apply_skillopt(rejected, experiences, validate_fn=lambda _skill: 0.0)
    states["REJECTED_EDIT"] = rejected
    return states


def _run_one(lab, client, state, probe, cfg) -> dict[str, Any]:
    from expolicy.paper_a.execute import run_probes

    rows = run_probes(
        lab,
        client,
        state,
        [probe],
        seed=int(cfg["model"]["inference_seed"]),
        max_tokens=int(cfg["model"]["max_tokens"]),
        temperature=float(cfg["model"]["temperature"]),
    )
    return rows[0]


def cmd_worker(cfg: dict[str, Any], work_path: Path, port: int) -> int:
    from expolicy.r9_lab.assimilators.episodic import apply_episodic
    from expolicy.r9_lab.assimilators.semantic import commit_semantic_entries, generate_semantic_v11
    from expolicy.r9_lab.benchmark.official_env import OfficialLabEnv
    from expolicy.r9_lab.llm_client import VLLMClient
    from expolicy.r9_lab.runner import set_injection_token_cap
    from expolicy.r9_lab.state.clone import clone_state
    from expolicy.r9_lab.state.persistent_state import PersistentState

    jobs = read_json(work_path)
    if not jobs:
        return 0
    set_injection_token_cap(int(cfg["pilot"]["injection_token_cap"]))
    out = run_dir(cfg)
    ledger = out / "episodes.used"
    client = VLLMClient.from_model_id(cfg["model"]["model_id"], port=port, timeout=600.0)
    labs: dict[str, Any] = {}
    smoke_states: dict[str, Any] = {}
    try:
        for job in jobs:
            env = job["env"]
            if env not in labs:
                labs[env] = OfficialLabEnv(env)
            destination = out / job["relpath"]
            key = job["cache_key"]
            if destination.exists() and cache_usable(read_json(destination), key):
                continue
            if job["kind"] in {"smoke", "readback", "cell"}:
                charge(ledger, 1 if job["kind"] != "cell" else len(job["probes"]), int(cfg["pilot"]["episode_cap"]))
            if job["kind"] == "smoke":
                if env not in smoke_states:
                    smoke_states[env] = _prepare_states(cfg, env, out / "smoke_states", job["experiences"])
                row = _run_one(labs[env], client, smoke_states[env][job["branch"]], job["probe"], cfg)
                row.update({"cache_key": key, "complete": row.get("infra_status") == "ok", "branch": job["branch"], "repeat": job["repeat"], "env": env})
                write_json(destination, row)
            elif job["kind"] == "readback":
                state = _materialize(cfg, env, job, client, out / "readback_states" / job["state_name"])
                row = _run_one(labs[env], client, state, job["probe"], cfg)
                row.update({"cache_key": key, "complete": row.get("infra_status") == "ok", "action": job["action"], "repeat": job["repeat"], "env": env})
                write_json(destination, row)
            elif job["kind"] == "cell":
                state = _materialize(cfg, env, job, client, out / "cells" / job["state_name"])
                from expolicy.paper_a.execute import run_probes

                rows = run_probes(
                    labs[env],
                    client,
                    state,
                    job["probes"],
                    seed=int(cfg["model"]["inference_seed"]),
                    max_tokens=int(cfg["model"]["max_tokens"]),
                    temperature=float(cfg["model"]["temperature"]),
                )
                write_json(
                    destination,
                    {"cache_key": key, "complete": all(r.get("infra_status") == "ok" for r in rows), "items": rows, "env": env, "action": job["action"], "batch_id": job["batch_id"]},
                )
            append_jsonl(out / "COST_LEDGER.jsonl", {"kind": job["kind"], "env": env, "port": port, "relpath": job["relpath"]})
            print(f"worker_done port={port} {job['relpath']}", flush=True)
    finally:
        for lab in labs.values():
            lab.close()
    return 0


def _materialize(cfg, env, job, client, dest: Path):
    from expolicy.r9_lab.assimilators.discard import apply_discard
    from expolicy.r9_lab.assimilators.episodic import apply_episodic
    from expolicy.r9_lab.assimilators.semantic import commit_semantic_entries, generate_semantic_v11
    from expolicy.r9_lab.state.clone import clone_state
    from expolicy.r9_lab.state.persistent_state import PersistentState

    s0 = PersistentState.load(REPO / cfg["paths"]["s0"] / env / "state")
    state = clone_state(s0, dest)
    action = job["action"]
    experiences = job["experiences"]
    if action == "DISCARD":
        apply_discard(state, experiences)
    elif action == "EPISODIC_REPLAY":
        applied = apply_episodic(state, experiences)
        append_jsonl(run_dir(cfg) / "EPISODIC_WRITE.jsonl", {"batch_id": job.get("batch_id"), "env": env, **applied})
    elif action == "SEMANTIC_MEMORY":
        record = run_dir(cfg) / "semantic_raw" / str(job.get("batch_id") or job.get("state_name"))
        entries, meta = generate_semantic_v11(experiences, client, record_dir=record)
        append_jsonl(run_dir(cfg) / "SEMANTIC_GENERATION_RECORDS.jsonl", {"batch_id": job.get("batch_id"), "env": env, **meta})
        if meta.get("generation_status") != "ok":
            write_json(dest.parent / "OPERATOR_INVALID.json", meta)
            raise PilotRefusal(f"semantic generation invalid for {job.get('batch_id')}")
        commit_semantic_entries(state, experiences, entries)
    return state


def _experiences(cfg: dict[str, Any], batch_id: str) -> list[dict[str, Any]]:
    from expolicy.paper_a.execute import load_acquired_experiences

    rows = load_acquired_experiences(REPO / cfg["paths"]["acquire_root"], batch_id)
    if not rows:
        raise PilotRefusal(f"missing acquired experiences for {batch_id}")
    return rows


def _shard(jobs: list[dict[str, Any]], n: int) -> list[list[dict[str, Any]]]:
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    # Keep every branch of one task on the same worker.
    groups: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        probe = job.get("probe") or {}
        key = f"{job.get('env')}:{probe.get('probe_task_id') or job.get('batch_id')}:{job.get('action') or ''}"
        groups.setdefault(key, []).append(job)
    for index, group in enumerate(groups.values()):
        buckets[index % n].extend(group)
    return buckets


def _launch(cfg: dict[str, Any], jobs: list[dict[str, Any]]) -> None:
    manifest = read_json(run_dir(cfg) / "SERVICE_AND_ENV_MANIFEST.json")
    live = [row for row in manifest["servers"] if row["status"] == "up"]
    if len(live) < 1:
        raise PilotRefusal("no verified model server is up")
    shards = _shard(jobs, len(live))
    script = str(REPO / "scripts/paper_a/run_a_v11.py")
    config = str(REPO / "configs/paper_a_v11.yaml")
    procs = []
    for index, shard in enumerate(shards):
        if not shard:
            continue
        path = run_dir(cfg) / f"work-{index}.json"
        write_json(path, shard)
        cmd = [sys.executable, script, "--config", config, "worker", "--work", str(path), "--port", str(live[index]["port"])]
        log = open(run_dir(cfg) / f"worker-{index}.log", "a", encoding="utf-8")
        procs.append(subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
    codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in codes):
        raise PilotRefusal(f"worker exit codes {codes}")


def _assign_smoke(cfg, plan) -> list[dict[str, Any]]:
    jobs = smoke_jobs(cfg, plan)
    for job in jobs:
        batch_id = plan["environments"][job["env"]]["discovery"][0]
        job["experiences"] = _experiences(cfg, batch_id)
        safe = str(job["probe"]["probe_task_id"]).replace(":", "_")
        job["relpath"] = f"smoke/{job['env']}/{safe}/{job['branch']}/r{job['repeat']}.json"
        job["cache_key"] = _cache_key(cfg, {k: v for k, v in job.items() if k != "experiences"})
    return jobs


def _load_smoke(cfg) -> list[dict[str, Any]]:
    rows = []
    root = run_dir(cfg) / "smoke"
    for path in sorted(root.rglob("r*.json")):
        row = read_json(path)
        parts = path.relative_to(root).parts
        row["env"] = parts[0]
        row["task_id"] = parts[1].replace("_", ":", 1)
        rows.append(row)
    return rows


def _eligibility(cfg: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    rows = _load_smoke(cfg)
    by_env: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_env.setdefault(row["env"], []).append(row)
    environments = []
    notes = []
    for env, env_rows in by_env.items():
        status = "EXPLORATORY_ELIGIBLE"
        grouped: dict[tuple, dict[int, dict[str, Any]]] = {}
        for row in env_rows:
            grouped.setdefault((row["task_id"], row["branch"]), {})[int(row["repeat"])] = row
        for (task_id, branch), repeats in grouped.items():
            if branch == "BASE" or len(repeats) < 2:
                continue
            pair = [repeats[0], repeats[1]]
            if any(item.get("branch_name_in_prompt") for item in pair):
                status = "BLOCKED_STRUCTURAL"
            traces = [{"reset_hash": item.get("reset_hash"), "steps": item.get("steps") or [], "y": item.get("y")} for item in pair]
            if pair[0].get("y") != pair[1].get("y"):
                kind = classify_divergence(traces)
                if kind == "VARIATION_UNLOCATED":
                    status = "BLOCKED_UNLOCATED"
                append_jsonl(
                    run_dir(cfg) / "FIRST_DIVERGENCE_CASES.jsonl",
                    {"env": env, "task_id": task_id, "branch": branch, "class": kind},
                )
        if status == "EXPLORATORY_ELIGIBLE":
            environments.append(env)
        notes.append({"env": env, "status": status, "n": len({row["task_id"] for row in env_rows})})
    eligibility = {
        "protocol_hash": read_json(run_dir(cfg) / "PROTOCOL_LOCK.json")["protocol_hash"],
        "environments": environments,
        "requested_environments": environments,
        "episodes_used": used_episodes(run_dir(cfg) / "episodes.used"),
        "notes": notes,
        "structural_error": None,
    }
    write_json(run_dir(cfg) / "ELIGIBILITY.json", eligibility)
    lines = ["# 测量冒烟", ""]
    for note in notes:
        lines.append(f"- {note['env']}: {note['status']}，任务数 {note['n']}。16 题不能分辨 3 个百分点，这本身不是精度不足。")
    (run_dir(cfg) / "MEASUREMENT_SMOKE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (run_dir(cfg) / "NOOP_TASK_RESULTS.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({k: row[k] for k in row if k != "steps"}, ensure_ascii=False) + "\n")
    return eligibility


def cmd_audit_smoke(cfg: dict[str, Any], execute: bool) -> int:
    plan = read_json(run_dir(cfg) / "PLAN.json")
    jobs = _assign_smoke(cfg, plan)
    print(f"audit-smoke jobs={len(jobs)} execute={execute}")
    if not execute:
        return 0
    assert_v11_launch(
        {"protocol_hash": read_json(run_dir(cfg) / "PROTOCOL_LOCK.json")["protocol_hash"], "environments": list(cfg["environments"]), "requested_environments": list(cfg["environments"]), "episodes_used": used_episodes(run_dir(cfg) / "episodes.used")},
        expected_hash=read_json(run_dir(cfg) / "PROTOCOL_LOCK.json")["protocol_hash"],
        episodes=len(jobs),
        budget=int(cfg["pilot"]["episode_cap"]),
    )
    _launch(cfg, jobs)
    _eligibility(cfg, plan)
    return 0


def _readback_jobs(cfg, plan, eligibility) -> list[dict[str, Any]]:
    jobs = []
    for env in eligibility["environments"]:
        body = plan["environments"][env]
        batch_ids = body["discovery"][:2]
        probes = []
        for bucket in ("shadow_reward_probe", "retention_probe"):
            probes.extend(_probe(env, bucket, row) for row in body["smoke"][bucket][:2])
        for batch_id in batch_ids:
            experiences = _experiences(cfg, batch_id)
            for action in ("EPISODIC_REPLAY", "SEMANTIC_MEMORY"):
                for repeat, probe in enumerate(probes):
                    if repeat >= 2 and action == "EPISODIC_REPLAY":
                        pass
                    job = {
                        "kind": "readback",
                        "env": env,
                        "action": action,
                        "probe": probe,
                        "repeat": 0,
                        "batch_id": batch_id,
                        "experiences": experiences,
                        "state_name": f"{env}-{batch_id}-{action}",
                    }
                    # Four probes x two actions x two batches would be 16 per env. Two envs = 32.
                    safe = str(probe["probe_task_id"]).replace(":", "_")
                    job["relpath"] = f"readback/{env}/{batch_id}/{action}/{safe}.json"
                    job["cache_key"] = _cache_key(cfg, {k: v for k, v in job.items() if k != "experiences"})
                    jobs.append(job)
    return jobs[:32]


def cmd_operator(cfg, execute: bool) -> int:
    eligibility = read_json(run_dir(cfg) / "ELIGIBILITY.json") if (run_dir(cfg) / "ELIGIBILITY.json").exists() else None
    plan = read_json(run_dir(cfg) / "PLAN.json")
    if not eligibility or not eligibility.get("environments"):
        _not_run(cfg, "OPERATOR_SMOKE.md", "测量没有环境进入探索，操作符冒烟不运行。")
        return 0
    jobs = _readback_jobs(cfg, plan, eligibility)
    print(f"operator-smoke jobs={len(jobs)} execute={execute}")
    if not execute:
        return 0
    assert_v11_launch(eligibility, expected_hash=eligibility["protocol_hash"], episodes=len(jobs), budget=int(cfg["pilot"]["episode_cap"]))
    try:
        _launch(cfg, jobs)
    except PilotRefusal as exc:
        (run_dir(cfg) / "OPERATOR_SMOKE.md").write_text(f"# 操作符冒烟\n\nOPERATOR_NOT_READY: {exc}\n", encoding="utf-8")
        return 0
    (run_dir(cfg) / "OPERATOR_SMOKE.md").write_text("# 操作符冒烟\n\n读回已完成。语义生成记录见 SEMANTIC_GENERATION_RECORDS.jsonl。\n", encoding="utf-8")
    return 0


def _cell_jobs(cfg, plan, envs, batches_key: str, side: str) -> list[dict[str, Any]]:
    jobs = []
    for env in envs:
        body = plan["environments"][env]
        probes = []
        for bucket in ("shadow_reward_probe", "retention_probe"):
            if side == "all":
                probes.extend(_probe(env, bucket, row) for row in body["probes"][bucket]["selection"])
                probes.extend(_probe(env, bucket, row) for row in body["probes"][bucket]["confirmation"])
            else:
                probes.extend(_probe(env, bucket, row) for row in body["probes"][bucket][side])
        for batch in body["batches"][batches_key]:
            experiences = _experiences(cfg, batch["batch_id"])
            for action in ACTIONS:
                job = {
                    "kind": "cell",
                    "env": env,
                    "action": action,
                    "batch_id": batch["batch_id"],
                    "probes": probes,
                    "experiences": experiences,
                    "state_name": f"{side}-{batch['batch_id']}-{action}",
                    "relpath": f"matrix/{side}/{env}/{batch['batch_id']}/{action}.json",
                }
                job["cache_key"] = _cache_key(
                    cfg,
                    {
                        "env": env,
                        "action": action,
                        "batch_id": batch["batch_id"],
                        "side": side,
                        "probe_ids": [probe["probe_task_id"] for probe in probes],
                    },
                )
                jobs.append(job)
    return jobs


def cmd_matrix(cfg, phase: str, execute: bool) -> int:
    eligibility_path = run_dir(cfg) / "ELIGIBILITY.json"
    eligibility = read_json(eligibility_path) if eligibility_path.exists() else None
    plan = read_json(run_dir(cfg) / "PLAN.json")
    if phase == "confirmation":
        lock = run_dir(cfg) / "SELECTION_LOCK.json"
        if not lock.exists() or read_json(lock).get("signal") != "SIGNAL":
            _not_run(cfg, "CONFIRMATION_RESULTS.csv", "选择阶段没有可核验的信号，确认阶段不运行。")
            return 0
    if not eligibility or not eligibility.get("environments"):
        _not_run(cfg, "SELECTION_RESULTS.csv", "没有可探索的环境。")
        return 0
    if phase == "selection":
        jobs = _cell_jobs(cfg, plan, eligibility["environments"], "discovery", "selection")
    else:
        jobs = _cell_jobs(cfg, plan, eligibility["environments"], "discovery", "confirmation")
        jobs += _cell_jobs(cfg, plan, eligibility["environments"], "reserved", "all")
    episodes = sum(len(job["probes"]) for job in jobs)
    print(f"matrix phase={phase} cells={len(jobs)} episodes={episodes} execute={execute}")
    if not execute:
        return 0
    assert_v11_launch(
        {**eligibility, "episodes_used": used_episodes(run_dir(cfg) / "episodes.used"), "requested_environments": eligibility["environments"]},
        expected_hash=eligibility["protocol_hash"],
        episodes=episodes,
        budget=int(cfg["pilot"]["episode_cap"]),
    )
    _launch(cfg, jobs)
    return 0


def _utility_rows(cfg, side: str) -> list[dict[str, Any]]:
    rows = []
    root = run_dir(cfg) / "matrix" / side
    if not root.exists():
        return rows
    cells: dict[tuple, dict[str, dict[str, float]]] = {}
    for path in root.rglob("*.json"):
        if path.name == "OPERATOR_INVALID.json":
            continue
        payload = read_json(path)
        if "items" not in payload:
            continue
        key = (payload["env"], payload["batch_id"])
        bucket = cells.setdefault(key, {})
        bucket[payload["action"]] = {item["probe_task_id"]: item.get("y") for item in payload["items"]}
        for item in payload["items"]:
            rows.append({"env": payload["env"], "batch_id": payload["batch_id"], "action": payload["action"], "task_id": item["probe_task_id"], "split": item["eval_split"], "y": item.get("y")})
    enriched = []
    for row in rows:
        discard = cells[(row["env"], row["batch_id"])].get("DISCARD", {})
        base = discard.get(row["task_id"])
        other = row["y"]
        row["delta"] = None if base is None or other is None else float(other) - float(base)
        enriched.append(row)
    return enriched


def _batch_utility(rows: list[dict[str, Any]], action: str) -> float | None:
    owned = [row for row in rows if row["action"] == action and row["delta"] is not None]
    if not owned:
        return None
    def mean(split: str) -> float:
        vals = [row["delta"] for row in owned if row["split"] == split]
        return sum(vals) / len(vals) if vals else 0.0
    shadow = mean("shadow_reward_probe")
    retention = mean("retention_probe")
    return shadow - 0.5 * max(0.0, -retention)


def cmd_analyze(cfg, phase: str) -> int:
    rows = _utility_rows(cfg, phase if phase != "confirmation" else "confirmation")
    if phase == "confirmation":
        rows += _utility_rows(cfg, "all")
    if not rows:
        _not_run(cfg, f"{phase.upper()}_RESULTS.csv", "该阶段没有结果。")
        return 0
    path = run_dir(cfg) / ("SELECTION_RESULTS.csv" if phase == "selection" else "CONFIRMATION_RESULTS.csv")
    header = ["env", "batch_id", "action", "task_id", "split", "y", "delta"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(key)) for key in header) + "\n")
    if phase != "selection":
        return 0
    winners = []
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["env"], row["batch_id"]), []).append(row)
    for (env, batch_id), batch_rows in sorted(grouped.items()):
        scores = {action: _batch_utility(batch_rows, action) for action in ACTIONS}
        usable = {action: score for action, score in scores.items() if score is not None}
        ordered = sorted(usable, key=lambda action: (-usable[action], TIE.index(action)))
        winner = ordered[0] if ordered else "DISCARD"
        flips = [row for row in batch_rows if row["action"] == winner and row["delta"] not in (None, 0.0)]
        winners.append({"env": env, "source_group": batch_id, "winner": winner, "n_task_flips": len(flips), "tie": len(set(usable.values())) <= 1, "scores": scores})
    signal = selection_signal(winners)
    write_json(run_dir(cfg) / "SELECTION_LOCK.json", {"signal": signal, "winners": winners, "frozen": True})
    print(f"selection signal={signal}")
    return 0


def cmd_decide(cfg) -> int:
    out = run_dir(cfg)
    eligibility = read_json(out / "ELIGIBILITY.json") if (out / "ELIGIBILITY.json").exists() else {}
    operator = (out / "OPERATOR_SMOKE.md").read_text(encoding="utf-8") if (out / "OPERATOR_SMOKE.md").exists() else "NOT_RUN"
    lock = read_json(out / "SELECTION_LOCK.json") if (out / "SELECTION_LOCK.json").exists() else {}
    confirmation_path = out / "CONFIRMATION_RESULTS.csv"
    confirmation = confirmation_path.exists() and not confirmation_path.read_text(encoding="utf-8").startswith("NOT_RUN")
    invalid_semantic = False
    semantic_path = out / "SEMANTIC_GENERATION_RECORDS.jsonl"
    if semantic_path.exists():
        for line in semantic_path.read_text(encoding="utf-8").splitlines():
            if line and json.loads(line).get("generation_status") == "OPERATOR_INVALID":
                invalid_semantic = True
    if "OPERATOR_NOT_READY" in operator or invalid_semantic:
        label = "A_OPERATOR_NOT_READY"
    elif not eligibility.get("environments") and eligibility.get("notes"):
        label = "A_MEASUREMENT_BLOCKED"
    elif lock.get("signal") == "SIGNAL" and not confirmation:
        label = "A_INCONCLUSIVE_BUDGET_EXHAUSTED"
    elif lock.get("signal") == "SIGNAL" and confirmation:
        winners = {row["winner"] for row in lock.get("winners") or []}
        carriers = winners & {"EPISODIC_REPLAY", "SEMANTIC_MEMORY"}
        label = "A_TWO_CARRIER_SIGNAL" if len(carriers) >= 2 else "A_UPDATE_ONLY_SIGNAL"
    elif lock.get("signal") == "WEAK_SIGNAL":
        label = "A_STOP_CURRENT_SETUP"
    elif not lock:
        label = "A_MEASUREMENT_BLOCKED" if eligibility else "A_INCONCLUSIVE_BUDGET_EXHAUSTED"
    else:
        label = "A_STOP_CURRENT_SETUP"
    episodes = used_episodes(out / "episodes.used")
    text = "\n".join(
        [
            "# A-v1.1 决定",
            "",
            f"结论：`{label}`。",
            "",
            f"实际尝试回合：{episodes}。硬上限 {cfg['pilot']['episode_cap']}。",
            "",
            "这一轮不训练路由器，不重跑 36×3，不把 SkillOpt 变成正式动作。",
            "",
        ]
    )
    (out / "DECISION.md").write_text(text, encoding="utf-8")
    print(label)
    return 0


def _not_run(cfg, name: str, reason: str) -> None:
    path = run_dir(cfg) / name
    if name.endswith(".md"):
        path.write_text(f"NOT_RUN\n\n{reason}\n", encoding="utf-8")
    elif name.endswith(".csv"):
        path.write_text(f"NOT_RUN,{reason}\n", encoding="utf-8")
    else:
        path.write_text(reason + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--phase", default="selection")
    parser.add_argument("--work", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("command")
    args = parser.parse_args()
    cfg = load_v11(REPO / args.config)
    os.chdir(REPO)
    if args.command == "worker":
        return cmd_worker(cfg, Path(args.work), args.port)
    if args.command == "preflight":
        return cmd_preflight(cfg)
    if args.command == "audit-smoke":
        return cmd_audit_smoke(cfg, args.execute)
    if args.command == "operator-smoke":
        return cmd_operator(cfg, args.execute)
    if args.command == "matrix":
        return cmd_matrix(cfg, args.phase, args.execute)
    if args.command == "analyze":
        return cmd_analyze(cfg, args.phase)
    if args.command == "decide":
        return cmd_decide(cfg)
    if args.command == "all":
        cmd_preflight(cfg)
        try:
            if args.execute:
                cmd_audit_smoke(cfg, True)
                cmd_operator(cfg, True)
                cmd_matrix(cfg, "selection", True)
                cmd_analyze(cfg, "selection")
                cmd_matrix(cfg, "confirmation", True)
                cmd_analyze(cfg, "confirmation")
        except PilotRefusal as exc:
            append_jsonl(run_dir(cfg) / "FAILURES.jsonl", {"reason": str(exc)})
        return cmd_decide(cfg)
    raise PilotRefusal(f"unknown command {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PilotRefusal as exc:
        print(f"REFUSED: {exc}")
        raise SystemExit(2)
