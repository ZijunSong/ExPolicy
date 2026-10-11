#!/usr/bin/env python3
"""R5.4 condition runner.

Phases: prep | base_eval | condition | watchdog
`--condition NAME --seeds 0,1,2 --budget 48 --resume` runs one condition sequentially.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from expolicy.environments.public_search.retriever import RetrieverClient
from expolicy.experiments.common import ROOT, write_json
from expolicy.experiments.r5_3 import heartbeat
from expolicy.experiments.r5_4 import (
    CONDITIONS,
    DEFAULT_MODEL,
    R54_ROOT,
    RETRIEVER_URL,
    fit_clusters,
    load_base_rows,
    load_predictions,
    load_split_r54,
    records_to_tasks,
    run_one_seed,
    train_r54_router,
    world_from_tasks,
)
from expolicy.experiments.r_common import apply_gpu
from expolicy.logging.event_logger import append_jsonl
from expolicy.environments.public_search.evaluator import score_public

STALL_SEC = 15 * 60
MAX_RESTARTS = 5
POLL_SEC = 30


def _venv_python() -> str:
    cand = ROOT / ".venv" / "bin" / "python"
    return str(cand) if cand.exists() else sys.executable


def parse_seeds(text: str | None, values: list[int] | None) -> list[int]:
    if values:
        return [int(x) for x in values]
    if text:
        return [int(x.strip()) for x in str(text).split(",") if x.strip()]
    return [0, 1, 2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="condition", help="prep|base_eval|condition|watchdog")
    p.add_argument("--condition", default=None)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--budget", type=int, default=48)
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES") or None)
    p.add_argument("--gpus", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--root", default=str(R54_ROOT))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--retriever-url", dest="retriever_url", default=os.environ.get("EXPOLICY_RETRIEVER_URL", RETRIEVER_URL))
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--n-shards", dest="n_shards", type=int, default=1)
    return p.parse_args(argv)


def wait_retriever(url: str, timeout: int = 600) -> dict:
    client = RetrieverClient(url, timeout=10.0)
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            last = client.health()
            if last.get("ok"):
                print(f"[retriever] ok {last}", flush=True)
                return last
        except Exception as exc:
            last = {"error": str(exc)}
        time.sleep(5)
    raise SystemExit(f"retriever not healthy at {url}: {last}")


def phase_prep(args: argparse.Namespace) -> None:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    print("[prep] train router on R5.3 calib+acq only", flush=True)
    train_r54_router(root)
    print("[prep] fit frozen E5 KMeans on R5.4 acquisition questions", flush=True)
    meta = fit_clusters(root)
    print(json.dumps(meta.get("meta") or meta, indent=2), flush=True)
    preds = load_predictions(root)
    print(f"[prep] predictions={len(preds)}", flush=True)


def phase_base_eval(args: argparse.Namespace) -> None:
    apply_gpu(args.gpu)
    root = Path(args.root)
    out = root / "base_eval"
    out.mkdir(parents=True, exist_ok=True)
    wait_retriever(args.retriever_url)
    split = load_split_r54(root)
    preds = load_predictions(root)
    tasks = records_to_tasks(split.get("future_probe") or [], split="future_probe", predictions=preds)
    n_shards = max(1, int(args.n_shards))
    shard = int(args.shard or 0)
    selected = []
    for t in tasks:
        h = int(hashlib.md5(t.task_id.encode()).hexdigest(), 16)
        if h % n_shards == shard:
            selected.append(t)
    shard_path = out / f"shard_{shard}.jsonl"
    done_ids = set()
    if args.resume and shard_path.exists():
        for line in shard_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["task_id"])
    remaining = [t for t in selected if t.task_id not in done_ids]
    print(f"[base_eval] shard={shard}/{n_shards} n={len(selected)} remaining={len(remaining)}", flush=True)
    if remaining:
        from expolicy.experiments.r5_4 import build_public_agent

        world = world_from_tasks(tasks)
        agent, state, backend, env = build_public_agent(
            world, model_path=args.model, output_dir=out / f"agent_shard_{shard}", retriever_url=args.retriever_url
        )
        for i, task in enumerate(remaining):
            heartbeat(out, phase="base_eval", shard=shard, i=i, n=len(remaining), task_id=task.task_id)
            heartbeat(out / f"shard_status_{shard}", phase="base_eval", shard=shard, i=i, n=len(remaining), task_id=task.task_id)
            exp = agent.run_task(task, state=state)
            exp.run_id = "base"
            row = score_public(exp, task)
            row["split"] = task.split
            row["dataset"] = (task.metadata or {}).get("dataset")
            row["predicted_capability"] = (task.metadata or {}).get("predicted_capability")
            row["true_capability"] = (task.metadata or {}).get("true_capability")
            row["tokens"] = exp.cost.tokens
            row["tool_calls"] = exp.cost.tool_calls
            row["predicted"] = exp.predicted_answer
            row["gold"] = task.gold_answer
            append_jsonl(shard_path, row)
            append_jsonl(out / f"experiences_shard_{shard}.jsonl", json.loads(exp.model_dump_json()))
    (out / f"DONE_shard_{shard}").write_text("ok\n", encoding="utf-8")
    print(f"[base_eval] shard {shard} done", flush=True)


def merge_base_eval(root: Path) -> None:
    out = root / "base_eval"
    rows = []
    for path in sorted(out.glob("shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    merged = out / "rows.jsonl"
    with merged.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    write_json(out / "summary.json", {"n": len(rows)})
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    print(f"[base_eval] merged {len(rows)} rows", flush=True)


def phase_condition(args: argparse.Namespace) -> None:
    apply_gpu(args.gpu)
    root = Path(args.root)
    wait_retriever(args.retriever_url)
    if not load_base_rows(root):
        raise SystemExit("base_eval missing; run --phase base_eval first")
    condition = args.condition
    if not condition or condition not in CONDITIONS:
        raise SystemExit(f"need --condition in {list(CONDITIONS)}")
    seeds = parse_seeds(args.seeds, None)
    cache: dict = {}
    for seed in seeds:
        print(f"[r5.4] condition={condition} seed={seed} gpu={args.gpu}", flush=True)
        rec = run_one_seed(
            condition=condition,
            seed=int(seed),
            output_dir=root,
            model=args.model,
            budget=int(args.budget),
            gpu=args.gpu,
            resume=bool(args.resume),
            retriever_url=args.retriever_url,
            cache=cache,
        )
        print(json.dumps({k: rec.get(k) for k in ("status", "condition", "seed", "future_macro", "delta_future") if k in rec}, default=str), flush=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _status_age(path: Path) -> float | None:
    live = path / "STATUS_LIVE.json"
    if not live.exists():
        return None
    return time.time() - live.stat().st_mtime


def launch_sharded_base_eval(args: argparse.Namespace) -> None:
    gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip()]
    root = Path(args.root)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    python = _venv_python()
    for i, gpu in enumerate(gpus):
        if (root / "base_eval" / f"DONE_shard_{i}").exists() and args.resume:
            print(f"[skip-done] base_eval shard {i}", flush=True)
            continue
        log_path = log_dir / f"base_eval_gpu{gpu}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        env["EXPOLICY_RETRIEVER_URL"] = args.retriever_url
        cmd = [
            python,
            str(ROOT / "scripts" / "run_r5_4_condition.py"),
            "--phase",
            "base_eval",
            "--resume",
            "--root",
            str(root),
            "--model",
            args.model,
            "--retriever-url",
            args.retriever_url,
            "--gpu",
            gpu,
            "--shard",
            str(i),
            "--n-shards",
            str(len(gpus)),
        ]
        handle = log_path.open("a", encoding="utf-8")
        handle.write(f"\n===== base_eval gpu={gpu} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        handle.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
        jobs.append({"gpu": gpu, "shard": i, "proc": proc, "pid": proc.pid, "handle": handle, "log": str(log_path), "restarts": 0, "launched_at": time.time()})
        print(f"[launch] base_eval shard={i} gpu={gpu} pid={proc.pid}", flush=True)
    while jobs:
        time.sleep(POLL_SEC)
        still = []
        for job in jobs:
            proc = job["proc"]
            code = proc.poll()
            age = _status_age(root / "base_eval" / f"shard_status_{job['shard']}") or _status_age(root / "base_eval")
            done = (root / "base_eval" / f"DONE_shard_{job['shard']}").exists()
            stalled = False
            if code is None:
                if age is None and time.time() - job["launched_at"] > 900:
                    stalled = True
                if age is not None and age > STALL_SEC:
                    stalled = True
            if stalled:
                print(f"[stall] base_eval shard={job['shard']} — kill and resume", flush=True)
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                job["restarts"] += 1
                if job["restarts"] > MAX_RESTARTS:
                    raise SystemExit(f"base_eval shard {job['shard']} stalled too many times")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = job["gpu"]
                env["PYTHONUNBUFFERED"] = "1"
                env["EXPOLICY_RETRIEVER_URL"] = args.retriever_url
                cmd = [
                    python,
                    str(ROOT / "scripts" / "run_r5_4_condition.py"),
                    "--phase",
                    "base_eval",
                    "--resume",
                    "--root",
                    str(root),
                    "--model",
                    args.model,
                    "--retriever-url",
                    args.retriever_url,
                    "--gpu",
                    job["gpu"],
                    "--shard",
                    str(job["shard"]),
                    "--n-shards",
                    str(len(gpus)),
                ]
                handle = Path(job["log"]).open("a", encoding="utf-8")
                proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
                job.update({"proc": proc, "pid": proc.pid, "handle": handle, "launched_at": time.time()})
                still.append(job)
                continue
            if code is None:
                still.append(job)
                continue
            if code == 0 or done:
                print(f"[done] base_eval shard={job['shard']}", flush=True)
            else:
                print(f"[exit] base_eval shard={job['shard']} code={code} — resume", flush=True)
                job["restarts"] += 1
                if job["restarts"] > MAX_RESTARTS:
                    raise SystemExit(f"base_eval shard {job['shard']} failed")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = job["gpu"]
                env["PYTHONUNBUFFERED"] = "1"
                env["EXPOLICY_RETRIEVER_URL"] = args.retriever_url
                cmd = [
                    python,
                    str(ROOT / "scripts" / "run_r5_4_condition.py"),
                    "--phase",
                    "base_eval",
                    "--resume",
                    "--root",
                    str(root),
                    "--model",
                    args.model,
                    "--retriever-url",
                    args.retriever_url,
                    "--gpu",
                    job["gpu"],
                    "--shard",
                    str(job["shard"]),
                    "--n-shards",
                    str(len(gpus)),
                ]
                handle = Path(job["log"]).open("a", encoding="utf-8")
                proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
                job.update({"proc": proc, "pid": proc.pid, "handle": handle, "launched_at": time.time()})
                still.append(job)
        jobs = still
    merge_base_eval(root)


def main() -> None:
    args = parse_args()
    Path(args.root).mkdir(parents=True, exist_ok=True)
    (Path(args.root) / "logs").mkdir(parents=True, exist_ok=True)
    if args.phase == "prep":
        phase_prep(args)
    elif args.phase == "base_eval":
        if args.gpus:
            launch_sharded_base_eval(args)
        else:
            phase_base_eval(args)
    elif args.phase == "condition":
        phase_condition(args)
    else:
        raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    main()
