#!/usr/bin/env python3
"""Run the no-op audit as one-request-at-a-time workers on several GPUs, then merge."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from expolicy.paper_a.audit import AuditTask, audit_noop  # noqa: E402
from expolicy.paper_a.config import load_config  # noqa: E402
from expolicy.paper_a.execute import asset_cache_key, load_acquired_experiences, probe_tasks  # noqa: E402
from expolicy.r9_lab.benchmark.split import load_split  # noqa: E402
from expolicy.r9_lab.constants import LAB_SYSTEM_PROMPT  # noqa: E402
from expolicy.r9_lab.io_util import read_json, write_json  # noqa: E402
from expolicy.r9_lab.state.persistent_state import PersistentState  # noqa: E402

WORKERS = (
    {"gpu": "0", "port": 18102},
    {"gpu": "3", "port": 18113},
    {"gpu": "4", "port": 18114},
    {"gpu": "5", "port": 18115},
    {"gpu": "6", "port": 18116},
    {"gpu": "7", "port": 18117},
)


def _wait_port(port: int, timeout: float = 900.0) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(5)
    raise RuntimeError(f"port {port} not ready: {last}")


def _merge_batch(cfg, env: str, batch: dict, parallel: Path) -> dict:
    split = load_split(cfg.path("split.path"))
    tasks = [AuditTask(row["probe_task_id"], row["eval_split"], row["instruction"], env) for row in probe_tasks(split, env)]
    cache: dict[tuple, dict] = {}
    root = parallel / "outcomes" / env / batch["batch_id"]
    if root.exists():
        for path in root.rglob("*.json"):
            row = read_json(path)
            cache[(row["mode"], int(row["repeat_id"]), row["branch"], row["probe_task_id"])] = row

    def evaluate(state, eval_tasks, *, mode, repeat_id, concurrency, branch):  # noqa: ARG001
        rows = []
        for task in eval_tasks:
            found = cache.get((mode, int(repeat_id), branch, task.probe_task_id))
            if found is None:
                rows.append({"probe_task_id": task.probe_task_id, "y": None, "infra_status": "missing"})
            else:
                rows.append(found)
        return rows

    experiences = load_acquired_experiences(cfg.path("legacy.r9_matrix"), batch["batch_id"]) or []
    s0 = PersistentState.load(cfg.path("legacy.r9_matrix") / "_s0" / env / "state")
    report = audit_noop(
        s0,
        tasks,
        evaluate,
        experiences=experiences,
        work_dir=parallel / "merge_states" / env / batch["batch_id"],
        model_identity={"data_parallel": True, "requests_per_server": 1},
        system_prompt=LAB_SYSTEM_PROMPT,
        tool_config={"env": env, "max_round_locked": True},
        env_snapshot_id=env,
        repeats=int(cfg.raw["measurement"]["noise_repeats"]),
        equivalence=float(cfg.raw["measurement"]["equivalence_success"]),
        ci_z=float(cfg.raw["measurement"]["ci_z"]),
        formal_concurrency=int(cfg.raw["measurement"]["formal_concurrency"]),
        require_formal_concurrency_replay=bool(cfg.raw["measurement"]["require_formal_concurrency_replay"]),
    )
    diffs = []
    component_dir = parallel / "components" / env / batch["batch_id"]
    if component_dir.exists():
        for path in component_dir.glob("shard*.json"):
            body = read_json(path)
            if body.get("diffs_vs_base"):
                diffs.append(body)
    payload = report.to_dict()
    if diffs:
        payload["status"] = "FAIL_SYSTEMATIC"
        payload["reasons"] = list(payload.get("reasons") or []) + ["a shard saw no-op branches diverge"]
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", action="append", dest="envs")
    args = parser.parse_args(argv)
    cfg = load_config()
    envs = args.envs or [env for env in cfg.environments if env != "kg"]
    parallel = ROOT / "outputs" / "paper_a_v1" / "parallel"
    parallel.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    shard_script = ROOT / "scripts" / "paper_a" / "run_audit_shard.py"
    for port in [worker["port"] for worker in WORKERS]:
        print(json.dumps({"wait_port": port}), flush=True)
        _wait_port(port)
    reports = []
    for env in envs:
        procs = []
        for index, worker in enumerate(WORKERS):
            log = parallel / "logs" / f"{env}-shard{index}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8")
            procs.append(
                subprocess.Popen(
                    [
                        py,
                        str(shard_script),
                        "--env",
                        env,
                        "--shard",
                        str(index),
                        "--shards",
                        str(len(WORKERS)),
                        "--port",
                        str(worker["port"]),
                    ],
                    cwd=str(ROOT),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        codes = [proc.wait() for proc in procs]
        if any(code != 0 for code in codes):
            print(json.dumps({"env": env, "worker_codes": codes}))
            return 1
        frozen = read_json(cfg.path("frozen_batches.path"))
        batches = [batch for batch in frozen["batches"] if batch["env"] == env][: int(cfg.raw["measurement"]["audit_batches_per_env"])]
        env_reports = []
        for batch in batches:
            payload = _merge_batch(cfg, env, batch, parallel)
            write_json(ROOT / "outputs" / "paper_a_v1" / "audit" / env / batch["batch_id"] / "report.json", payload)
            env_reports.append(payload)
            print(json.dumps({"merged": batch["batch_id"], "status": payload["status"]}), flush=True)
        reports.append({"env": env, "batches": env_reports})
    _key, hashes = asset_cache_key(cfg)
    statuses = [batch["status"] for env_report in reports for batch in env_report["batches"]]
    if statuses and all(status == "PASS" for status in statuses):
        status = "PASS"
    elif any(status == "FAIL_SYSTEMATIC" for status in statuses):
        status = "FAIL_SYSTEMATIC"
    else:
        status = "MEASUREMENT_INCONCLUSIVE"
    write_json(
        ROOT / "outputs" / "paper_a_v1" / "audit_partial.json",
        {"status": status, "complete": False, "envs": reports, "model_hash": hashes.get("model_hash"), "workers": list(WORKERS)},
    )
    print(json.dumps({"parallel_audit": status, "envs": envs}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
