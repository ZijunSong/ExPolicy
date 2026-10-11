#!/usr/bin/env python3
"""R6.1 Exact AVA runner.

Phases: prep | shadow_base | mask_eval | condition | merge_masks
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from expolicy.environments.public_search.retriever import RetrieverClient
from expolicy.evaluation.shadow_probe import ALL_MASKS
from expolicy.experiments.common import ROOT, write_json
from expolicy.experiments.r5_3 import heartbeat
from expolicy.experiments.r5_4 import DEFAULT_MODEL, R54_ROOT, RETRIEVER_URL
from expolicy.experiments.r6_ava import (
    BATCH_SIZE,
    BUDGET_DEFAULT,
    CONDITIONS,
    NUM_CANDIDATES,
    R6_ROOT,
    copy_future_base_eval,
    run_one_seed,
)
from expolicy.experiments.r_common import apply_gpu

STALL_SEC = 15 * 60
POLL_SEC = 30
MAX_RESTARTS = 6


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
    p.add_argument("--phase", default="condition", help="prep|shadow_base|mask_eval|merge_masks|condition")
    p.add_argument("--condition", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", nargs="*", default=None)
    p.add_argument("--seeds-csv", dest="seeds_csv", default=None)
    p.add_argument("--budget", type=int, default=BUDGET_DEFAULT)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-candidates", dest="num_candidates", type=int, default=NUM_CANDIDATES)
    p.add_argument("--split", default=str(R6_ROOT / "split_r6_ava.json"))
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES") or None)
    p.add_argument("--gpus", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--root", default=str(R6_ROOT))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--retriever-url", dest="retriever_url", default=os.environ.get("EXPOLICY_RETRIEVER_URL", RETRIEVER_URL))
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--n-shards", dest="n_shards", type=int, default=1)
    p.add_argument("--split-name", dest="split_name", default=None)
    p.add_argument("--mask", default=None)
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
    copy_future_base_eval(root)
    split = Path(args.split)
    if not split.exists():
        subprocess.check_call(
            [
                _venv_python(),
                str(ROOT / "scripts" / "build_r6_ava_split.py"),
                "--source-split",
                str(R54_ROOT / "split_r5_4.json"),
                "--out",
                str(split),
            ],
            cwd=str(ROOT),
        )
    print("[prep] split + future base_eval ready", flush=True)


def phase_shadow_base(args: argparse.Namespace) -> None:
    """LLM base eval of the 180 shadow-probe tasks (sharded)."""
    apply_gpu(args.gpu)
    root = Path(args.root)
    out = root / "base_eval_shadow"
    out.mkdir(parents=True, exist_ok=True)
    wait_retriever(args.retriever_url)
    import hashlib
    import json

    from expolicy.experiments.r5_3 import build_public_agent, world_from_tasks
    from expolicy.experiments.r5_4 import load_predictions, records_to_tasks
    from expolicy.experiments.r6_ava import load_split_r6
    from expolicy.environments.public_search.evaluator import score_public
    from expolicy.logging.event_logger import append_jsonl

    split = load_split_r6(root)
    preds = load_predictions(R54_ROOT)
    tasks = records_to_tasks(split.get("shadow_probe") or [], split="shadow_probe", predictions=preds)
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
    print(f"[shadow_base] shard={shard}/{n_shards} n={len(selected)} remaining={len(remaining)}", flush=True)
    if remaining:
        world = world_from_tasks(tasks)
        agent, state, backend, env = build_public_agent(
            world, model_path=args.model, output_dir=out / f"agent_shard_{shard}", retriever_url=args.retriever_url
        )
        for i, task in enumerate(remaining):
            heartbeat(out, phase="shadow_base", shard=shard, i=i, n=len(remaining), task_id=task.task_id)
            heartbeat(out / f"shard_status_{shard}", phase="shadow_base", shard=shard, i=i, n=len(remaining), task_id=task.task_id)
            exp = agent.run_task(task, state=state)
            exp.run_id = "base"
            row = score_public(exp, task)
            row["split"] = "shadow_probe"
            row["dataset"] = (task.metadata or {}).get("dataset")
            row["predicted_capability"] = (task.metadata or {}).get("predicted_capability")
            row["true_capability"] = (task.metadata or {}).get("true_capability")
            row["tokens"] = exp.cost.tokens
            row["tool_calls"] = exp.cost.tool_calls
            row["predicted"] = exp.predicted_answer
            row["gold"] = task.gold_answer
            append_jsonl(shard_path, row)
    (out / f"DONE_shard_{shard}").write_text("ok\n", encoding="utf-8")
    print(f"[shadow_base] shard {shard} done", flush=True)


def merge_shadow_base(root: Path) -> None:
    import json

    out = root / "base_eval_shadow"
    rows = []
    for path in sorted(out.glob("shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    with (out / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    write_json(out / "summary.json", {"n": len(rows)})
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    print(f"[shadow_base] merged {len(rows)} rows", flush=True)


def launch_sharded(args: argparse.Namespace, phase: str, extra: list[str] | None = None) -> None:
    gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip()]
    root = Path(args.root)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    python = _venv_python()
    jobs = []
    extra = extra or []
    for i, gpu in enumerate(gpus):
        log_path = log_dir / f"{phase}_gpu{gpu}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        env["EXPOLICY_RETRIEVER_URL"] = args.retriever_url
        cmd = [
            python,
            str(ROOT / "scripts" / "run_r6_ava_stage1.py"),
            "--phase",
            phase,
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
            *extra,
        ]
        handle = log_path.open("a", encoding="utf-8")
        handle.write(f"\n===== {phase} gpu={gpu} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        handle.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
        jobs.append({"gpu": gpu, "shard": i, "proc": proc, "pid": proc.pid, "log": str(log_path), "restarts": 0, "launched_at": time.time(), "handle": handle, "cmd": cmd, "env": env})
        print(f"[launch] {phase} shard={i} gpu={gpu} pid={proc.pid}", flush=True)
    while jobs:
        time.sleep(POLL_SEC)
        still = []
        for job in jobs:
            proc = job["proc"]
            code = proc.poll()
            log_age = time.time() - Path(job["log"]).stat().st_mtime
            stalled = code is None and log_age > STALL_SEC and time.time() - job["launched_at"] > STALL_SEC
            if stalled:
                print(f"[stall] {phase} shard={job['shard']} log_age={log_age:.0f}s — kill/resume", flush=True)
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                code = proc.poll()
            if code is None and not stalled:
                still.append(job)
                continue
            if code == 0:
                print(f"[done] {phase} shard={job['shard']}", flush=True)
                continue
            job["restarts"] += 1
            if job["restarts"] > MAX_RESTARTS:
                raise SystemExit(f"{phase} shard {job['shard']} failed")
            print(f"[exit] {phase} shard={job['shard']} code={code} resume", flush=True)
            handle = Path(job["log"]).open("a", encoding="utf-8")
            proc = subprocess.Popen(job["cmd"], cwd=str(ROOT), env=job["env"], stdout=handle, stderr=subprocess.STDOUT)
            job.update({"proc": proc, "pid": proc.pid, "handle": handle, "launched_at": time.time()})
            still.append(job)
        jobs = still


def phase_mask_eval_one(args: argparse.Namespace) -> None:
    cmd = [
        _venv_python(),
        str(ROOT / "scripts" / "eval_r6_shadow_utility.py"),
        "--root",
        str(args.root),
        "--split-name",
        args.split_name or "shadow_probe",
        "--shard",
        str(args.shard or 0),
        "--n-shards",
        str(args.n_shards or 1),
        "--resume",
        "--model",
        args.model,
        "--retriever-url",
        args.retriever_url,
    ]
    if args.mask:
        cmd.extend(["--mask", args.mask])
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        raise SystemExit(rc)


def merge_all_masks(root: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_r6_shadow_utility import merge_mask, none_from_base  # noqa: E402
    from expolicy.experiments.r5_4 import load_predictions, records_to_tasks
    from expolicy.experiments.r6_ava import load_split_r6

    split = load_split_r6(root)
    preds = load_predictions(R54_ROOT)
    for split_name in ("shadow_probe", "future_probe"):
        tasks = records_to_tasks(split.get(split_name) or [], split=split_name, predictions=preds)
        none_from_base(root, split_name, tasks)
        for mask in ALL_MASKS:
            if mask == "none":
                continue
            dest = root / "mask_eval" / split_name / f"{mask.replace('+', '__')}.json"
            work = root / "mask_eval" / split_name / "work" / mask.replace("+", "__")
            if dest.exists():
                continue
            if any(work.glob("shard_*.jsonl")):
                merge_mask(root, split_name, mask)
    (root / "MASK_EVAL.DONE").write_text("ok\n", encoding="utf-8")


def phase_condition(args: argparse.Namespace) -> None:
    apply_gpu(args.gpu)
    root = Path(args.root)
    wait_retriever(args.retriever_url)
    condition = args.condition
    if not condition or condition not in CONDITIONS:
        raise SystemExit(f"need --condition in {list(CONDITIONS)}")
    seeds = parse_seeds(args.seeds_csv, args.seeds)
    if args.seed is not None:
        seeds = [int(args.seed)]
    cache: dict = {}
    for seed in seeds:
        print(f"[r6] condition={condition} seed={seed} gpu={args.gpu}", flush=True)
        rec = run_one_seed(
            condition=condition,
            seed=int(seed),
            output_dir=root,
            model=args.model,
            budget=int(args.budget),
            gpu=args.gpu,
            resume=bool(args.resume),
            retriever_url=args.retriever_url,
            batch_size=int(args.batch_size),
            num_candidates=int(args.num_candidates),
            cache=cache,
        )
        print({k: rec.get(k) for k in ("status", "condition", "seed", "future_macro", "delta_future") if k in rec}, flush=True)


def main() -> None:
    args = parse_args()
    Path(args.root).mkdir(parents=True, exist_ok=True)
    (Path(args.root) / "logs").mkdir(parents=True, exist_ok=True)
    if args.phase == "prep":
        phase_prep(args)
    elif args.phase == "shadow_base":
        if args.gpus:
            launch_sharded(args, "shadow_base")
            merge_shadow_base(Path(args.root))
        else:
            phase_shadow_base(args)
            if args.n_shards <= 1:
                merge_shadow_base(Path(args.root))
            else:
                done = all((Path(args.root) / "base_eval_shadow" / f"DONE_shard_{i}").exists() for i in range(args.n_shards))
                if done:
                    merge_shadow_base(Path(args.root))
    elif args.phase == "mask_eval":
        if args.gpus:
            for split_name in ("shadow_probe", "future_probe"):
                launch_sharded(args, "mask_eval", extra=["--split-name", split_name])
            merge_all_masks(Path(args.root))
        else:
            phase_mask_eval_one(args)
    elif args.phase == "merge_masks":
        merge_all_masks(Path(args.root))
    elif args.phase == "condition":
        phase_condition(args)
    else:
        raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    main()
