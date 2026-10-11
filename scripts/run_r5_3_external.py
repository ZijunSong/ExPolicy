#!/usr/bin/env python3
"""R5.3 external generalization runner.

Phases: smoke | splits | capability | base_eval | calibration | skill_validation
        | acquire | bamboogle | analyze | all

--gpus 4,5,6,7 launches the 4-way watchdog. Single --gpu runs seeds sequentially.
Always --resume after a kill. Never writes into Helion R5/R5.1/R5.2 output dirs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from expolicy.environments.public_search.evaluator import score_public
from expolicy.environments.public_search.retriever import RetrieverClient
from expolicy.experiments.common import ROOT, write_json
from expolicy.experiments.r5_3 import (
    DEFAULT_MODEL,
    R53_ROOT,
    build_public_agent,
    heartbeat,
    load_split,
    run_one_seed,
    split_tasks,
    train_routers,
    world_from_tasks,
    write_frontier_port_audit,
)
from expolicy.experiments.r_common import apply_gpu
from expolicy.logging.event_logger import append_jsonl

STALL_SEC = 15 * 60
MAX_RESTARTS = 3
POLL_SEC = 30
SEED_GPU_MAP = {
    "4": (0, 4, 8),
    "5": (1, 5, 9),
    "6": (2, 6),
    "7": (3, 7),
}


def _venv_python() -> str:
    cand = ROOT / ".venv" / "bin" / "python"
    return str(cand) if cand.exists() else sys.executable


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _status_age(out: Path) -> float | None:
    path = out / "STATUS_LIVE.json"
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime


def _run_dir(root: Path, method: str, seed: int) -> Path:
    return root / method / f"seed_{seed}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="acquire", help="smoke|splits|capability|base_eval|calibration|skill_validation|acquire|bamboogle|analyze|all")
    p.add_argument("--methods", nargs="+", default=["frontier", "ace_skill_prioritized", "passive"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--budget", type=int, default=48)
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES") or None)
    p.add_argument("--gpus", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", default=str(R53_ROOT))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--retriever-url", dest="retriever_url", default=os.environ.get("EXPOLICY_RETRIEVER_URL", "http://127.0.0.1:8000"))
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki", "musique"])
    p.add_argument("--min-support", dest="min_support", type=int, default=8)
    p.add_argument("--dispatch", default="oracle")
    p.add_argument("--stage", default="pilot")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--n-shards", dest="n_shards", type=int, default=1)
    p.add_argument("--skip-prep", dest="skip_prep", action="store_true")
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


def phase_smoke(args: argparse.Namespace) -> None:
    apply_gpu(args.gpu)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    health = wait_retriever(args.retriever_url, timeout=120)
    client = RetrieverClient(args.retriever_url)
    hits = client.search("Barack Obama birthplace", topk=5)
    if not hits:
        raise SystemExit("smoke: search returned 0 hits")
    opened = client.open_id(str(hits[0]["document_id"]))
    if opened is None:
        # open via cached search doc
        opened = hits[0]
    from expolicy.environments.public_search.datasets import to_task
    from expolicy.environments.search.world import Task

    split_path = root / "splits" / "split.json"
    tasks = []
    if split_path.exists():
        split = load_split(root)
        tasks = split_tasks(split, "future_probe")[:10]
    if not tasks:
        recs = [
            {
                "task_id": f"smoke:{i}",
                "dataset": "hotpotqa",
                "question": q,
                "gold_answer": "smoke",
                "aliases": [],
                "hops": 2,
                "supporting_titles": [],
            }
            for i, q in enumerate(
                [
                    "What is the capital of France?",
                    "Who wrote Pride and Prejudice?",
                    "Which city is the Eiffel Tower in?",
                ]
            )
        ]
        tasks = [to_task(r, capability="bridge_decomposition", split="smoke") for r in recs]
    world = world_from_tasks(tasks)
    agent, state, backend, env = build_public_agent(
        world, model_path=args.model, output_dir=root / "smoke" / "agent", retriever_url=args.retriever_url
    )
    rows = []
    for task in tasks[:10]:
        heartbeat(root / "smoke", phase="smoke", task_id=task.task_id)
        exp = agent.run_task(task, state=state)
        row = score_public(exp, task)
        rows.append(row)
        print(f"[smoke] {task.task_id} pred={exp.predicted_answer!r} tools={exp.cost.tool_calls}", flush=True)
    write_json(
        root / "smoke" / "result.json",
        {"n": len(rows), "retriever": health, "n_with_search": sum(1 for r in rows if r.get("n_search", 0) > 0), "rows": rows},
    )
    (root / "smoke" / "DONE").write_text("ok\n", encoding="utf-8")
    print("[smoke] ok", flush=True)


def phase_splits(args: argparse.Namespace) -> None:
    cmd = [
        _venv_python(),
        str(ROOT / "scripts" / "build_r5_3_public_splits.py"),
        "--datasets",
        *args.datasets,
        "bamboogle",
        "--output-dir",
        args.output_dir,
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))


def phase_capability(args: argparse.Namespace) -> None:
    cmd = [
        _venv_python(),
        str(ROOT / "scripts" / "build_public_capability_map.py"),
        "--input-dir",
        args.output_dir,
        "--output-dir",
        args.output_dir,
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))
    write_frontier_port_audit(Path(args.output_dir))


def _all_eval_tasks(root: Path):
    split = load_split(root)
    tasks = []
    for key in ("calibration", "skill_validation", "future_probe", "bamboogle"):
        tasks.extend(split_tasks(split, key) if key != "bamboogle" else [])
    if split.get("bamboogle"):
        from expolicy.environments.public_search.datasets import to_task

        for rec in split["bamboogle"]:
            tasks.append(
                to_task(
                    rec,
                    capability=rec.get("capability") or "bridge_decomposition",
                    split="bamboogle",
                )
            )
    return tasks


def phase_base_eval(args: argparse.Namespace) -> None:
    apply_gpu(args.gpu)
    root = Path(args.output_dir)
    out = root / "base_eval"
    out.mkdir(parents=True, exist_ok=True)
    wait_retriever(args.retriever_url)
    tasks = _all_eval_tasks(root)
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


def phase_calibration(args: argparse.Namespace) -> None:
    from expolicy.core.experience import Experience
    from expolicy.diagnosis.capability_profile import CapabilityDiagnoser
    from expolicy.experiments.r5_3 import _load_jsonl_exp, macro_from_rows

    root = Path(args.output_dir)
    out = root / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    split = load_split(root)
    cal_ids = {t.task_id for t in split_tasks(split, "calibration")}
    base = root / "base_eval"
    experiences = []
    for path in sorted(base.glob("experiences_shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("task_id") in cal_ids:
                experiences.append(Experience.model_validate(payload))
    if not experiences:
        # rebuild from rows? insufficient for diagnoser tool stats; require experiences
        raise SystemExit("calibration experiences missing; run base_eval first")
    rows = []
    merged = base / "rows.jsonl"
    if merged.exists():
        for line in merged.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("task_id") in cal_ids:
                    rows.append(rec)
    diagnoser = CapabilityDiagnoser()
    profile = diagnoser.diagnose(experiences)
    # pass@4 proxy: we do not re-roll; report failures only
    fail = [e for e in experiences if not e.success]
    by_cap = {}
    for cap in sorted({e.capability for e in experiences if e.capability}):
        items = [e for e in experiences if e.capability == cap]
        fails = [e for e in items if not e.success][:16]
        by_cap[cap] = {
            "n": len(items),
            "accuracy": sum(1 for e in items if e.success) / max(1, len(items)),
            "n_fail_capped": len(fails),
            "avg_tool_calls": sum(e.cost.tool_calls for e in items) / max(1, len(items)),
            "search_miss_proxy": sum(1 for e in items if not any(c.get("tool") == "open" for c in e.tool_calls)) / max(1, len(items)),
            "band": next((x.band.value for x in profile.frontier_capabilities if x.name == cap), "unknown"),
        }
    metrics = {
        "n": len(experiences),
        "macro": macro_from_rows(rows) if rows else {},
        "per_capability": by_cap,
        "tokens": int(sum(e.cost.tokens for e in experiences)),
        "tool_calls": int(sum(e.cost.tool_calls for e in experiences)),
        "rollouts": len(experiences),
        "capability_scores": profile.capability_scores,
        "uncertainty": profile.uncertainty,
    }
    write_json(out / "metrics.json", metrics)
    write_json(out / "profile.json", json.loads(profile.model_dump_json()) if hasattr(profile, "model_dump_json") else profile.__dict__)
    with (out / "experiences.jsonl").open("w", encoding="utf-8") as handle:
        for exp in experiences:
            handle.write(exp.model_dump_json() + "\n")
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    print("[calibration] wrote profile", flush=True)


def phase_router(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    split = load_split(root)
    text, embed, metrics = train_routers(split, seed=0)
    write_json(root / "router_metrics.json", metrics)
    md = [
        "# ROUTER_AUDIT",
        "",
        "Trained only on calibration + acquisition-side labels. Future labels used only for accuracy.",
        "",
        json.dumps(metrics, indent=2),
        "",
        "Do not tune prompts or add future patterns to chase 1.0 accuracy.",
    ]
    (root / "ROUTER_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("[router] wrote ROUTER_AUDIT.md", flush=True)


def phase_skill_validation(args: argparse.Namespace) -> None:
    cmd = [
        _venv_python(),
        str(ROOT / "scripts" / "validate_r5_3_skills.py"),
        "--datasets",
        *args.datasets,
        "--output-dir",
        args.output_dir,
        "--model",
        args.model,
        "--retriever-url",
        args.retriever_url,
        "--resume",
    ]
    if args.gpu:
        cmd.extend(["--gpu", str(args.gpu)])
    subprocess.check_call(cmd, cwd=str(ROOT))


def run_worker(args: argparse.Namespace) -> None:
    apply_gpu(args.gpu)
    root = Path(args.output_dir)
    wait_retriever(args.retriever_url)
    write_frontier_port_audit(root)
    cache: dict = {}
    for method in args.methods:
        for seed in args.seeds:
            print(f"[r5.3] method={method} seed={seed} gpu={args.gpu}", flush=True)
            rec = run_one_seed(
                method=method,
                seed=int(seed),
                output_dir=root,
                model=args.model,
                budget=args.budget,
                gpu=args.gpu,
                resume=args.resume,
                retriever_url=args.retriever_url,
                min_support=args.min_support,
                dispatch=args.dispatch,
                cache=cache,
            )
            print(json.dumps({k: rec.get(k) for k in ("status", "method", "seed", "future_macro", "delta_future") if k in rec}, default=str), flush=True)


def _launch(job: dict, python: str) -> subprocess.Popen:
    gpu = str(job["gpu"])
    log_path = Path(job["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    env["EXPOLICY_RETRIEVER_URL"] = job["retriever_url"]
    cmd = [
        python,
        str(ROOT / "scripts" / "run_r5_3_external.py"),
        "--phase",
        job["phase"],
        "--methods",
        *(job.get("methods") or ["frontier"]),
        "--seeds",
        *[str(s) for s in job.get("seeds") or [0]],
        "--budget",
        str(job.get("budget") or 48),
        "--gpu",
        gpu,
        "--resume",
        "--output-dir",
        job["output_dir"],
        "--model",
        job["model"],
        "--retriever-url",
        job["retriever_url"],
        "--min-support",
        str(job.get("min_support") or 8),
        "--dispatch",
        job.get("dispatch") or "oracle",
    ]
    if job.get("shard") is not None:
        cmd.extend(["--shard", str(job["shard"]), "--n-shards", str(job["n_shards"])])
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} restart={job['restarts']} =====\n")
    handle.flush()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    job["pid"] = proc.pid
    job["handle"] = handle
    job["proc"] = proc
    job["launched_at"] = time.time()
    print(f"[launch] {job['name']} gpu={gpu} pid={proc.pid} log={log_path}", flush=True)
    return proc


def _kill(job: dict) -> None:
    proc: subprocess.Popen | None = job.get("proc")
    pid = job.get("pid")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    elif pid and _alive(pid):
        os.kill(pid, 15)
        time.sleep(2)
        if _alive(pid):
            os.kill(pid, 9)
    handle = job.get("handle")
    if handle:
        try:
            handle.close()
        except Exception:
            pass


def _worker_done(job: dict) -> bool:
    root = Path(job["output_dir"])
    phase = job["phase"]
    if phase == "base_eval":
        return (root / "base_eval" / f"DONE_shard_{job['shard']}").exists()
    if phase == "skill_validation":
        return (root / "skill_validation" / "DONE").exists()
    for method in job.get("methods") or []:
        for seed in job.get("seeds") or []:
            if not (_run_dir(root, method, seed) / "DONE").exists():
                return False
    return True


def _worker_status_age(job: dict) -> float | None:
    root = Path(job["output_dir"])
    ages = []
    phase = job["phase"]
    if phase == "base_eval":
        return _status_age(root / "base_eval" / f"shard_status_{job['shard']}") or _status_age(root / "base_eval")
    if phase == "skill_validation":
        return _status_age(root / "skill_validation")
    for method in job.get("methods") or []:
        for seed in job.get("seeds") or []:
            out = _run_dir(root, method, seed)
            if (out / "DONE").exists():
                continue
            age = _status_age(out)
            if age is not None:
                ages.append(age)
    if not ages:
        return None
    return min(ages)


def watchdog(jobs: list[dict], gpus: list[str], *, status_path: Path) -> None:
    python = _venv_python()
    pending = list(jobs)
    running: list[dict] = []
    finished: list[dict] = []
    failed: list[dict] = []
    free = list(gpus)

    def start_ready() -> None:
        while pending and free:
            job = pending[0]
            if _worker_done(job):
                print(f"[skip-done] {job['name']}", flush=True)
                finished.append(pending.pop(0))
                continue
            job = pending.pop(0)
            job["gpu"] = free.pop(0)
            _launch(job, python)
            running.append(job)

    start_ready()
    write_json(status_path, {"running": [j["name"] for j in running], "pending": [j["name"] for j in pending]})
    while running or pending:
        time.sleep(POLL_SEC)
        still = []
        for job in running:
            proc: subprocess.Popen | None = job.get("proc")
            code = proc.poll() if proc is not None else None
            age = _worker_status_age(job)
            if age is not None:
                job["saw_status"] = True
            stalled = False
            if code is None:
                if age is None and not job.get("saw_status") and time.time() - job["launched_at"] > 900:
                    stalled = True
                if age is not None and age > STALL_SEC:
                    stalled = True
            if stalled:
                print(f"[stall] {job['name']} pid={job.get('pid')} age={age} — kill and resume", flush=True)
                _kill(job)
                if job.get("gpu") is not None:
                    free.append(job["gpu"])
                job["gpu"] = None
                job["restarts"] += 1
                if job["restarts"] <= MAX_RESTARTS:
                    pending.insert(0, job)
                else:
                    failed.append(job)
                    print(f"[fail] {job['name']} exceeded restarts", flush=True)
                continue
            if code is None:
                still.append(job)
                continue
            handle = job.get("handle")
            if handle:
                try:
                    handle.close()
                except Exception:
                    pass
            if job.get("gpu") is not None:
                free.append(job["gpu"])
            job["gpu"] = None
            if code == 0 and _worker_done(job):
                print(f"[done] {job['name']} exit=0", flush=True)
                finished.append(job)
            else:
                print(f"[exit] {job['name']} code={code} — resume", flush=True)
                job["restarts"] += 1
                if job["restarts"] <= MAX_RESTARTS and not _worker_done(job):
                    pending.insert(0, job)
                elif _worker_done(job):
                    finished.append(job)
                else:
                    failed.append(job)
        running = still
        start_ready()
        write_json(
            status_path,
            {
                "ts": time.time(),
                "running": [{"name": j["name"], "gpu": j["gpu"], "pid": j.get("pid")} for j in running],
                "pending": [j["name"] for j in pending],
                "finished": [j["name"] for j in finished],
                "failed": [j["name"] for j in failed],
            },
        )
    if failed:
        raise SystemExit(f"failed jobs: {[j['name'] for j in failed]}")


def assign_seeds_to_gpus(seeds: list[int], gpus: list[str]) -> dict[str, list[int]]:
    assigned = {g: [] for g in gpus}
    if set(gpus) <= set(SEED_GPU_MAP) and set(seeds) <= set(range(10)):
        for g in gpus:
            assigned[g] = [s for s in SEED_GPU_MAP[g] if s in seeds]
        leftover = [s for s in seeds if not any(s in assigned[g] for g in gpus)]
        for i, s in enumerate(leftover):
            assigned[gpus[i % len(gpus)]].append(s)
        return assigned
    for i, s in enumerate(seeds):
        assigned[gpus[i % len(gpus)]].append(s)
    return assigned


def launch_sharded(args: argparse.Namespace, phase: str, n: int | None = None) -> None:
    gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip()]
    n = n or len(gpus)
    root = Path(args.output_dir)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, gpu in enumerate(gpus[:n]):
        jobs.append(
            {
                "name": f"{phase}_shard{i}_gpu{gpu}",
                "phase": phase,
                "methods": list(args.methods),
                "seeds": list(args.seeds),
                "gpu": gpu,
                "restarts": 0,
                "budget": args.budget,
                "output_dir": str(root),
                "model": args.model,
                "retriever_url": args.retriever_url,
                "min_support": args.min_support,
                "dispatch": args.dispatch,
                "shard": i,
                "n_shards": n,
                "log": str(log_dir / f"{phase}_gpu{gpu}.log"),
            }
        )
    watchdog(jobs, [j["gpu"] for j in jobs], status_path=root / f"SUITE_{phase}.json")
    if phase == "base_eval":
        merge_base_eval(root)


def launch_acquire(args: argparse.Namespace) -> None:
    gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip()]
    root = Path(args.output_dir)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    mapping = assign_seeds_to_gpus(list(args.seeds), gpus)
    jobs = []
    for gpu, seeds in mapping.items():
        if not seeds:
            continue
        jobs.append(
            {
                "name": f"gpu{gpu}_{'_'.join(args.methods)}_{'-'.join(str(s) for s in seeds)}",
                "phase": "acquire",
                "methods": list(args.methods),
                "seeds": seeds,
                "gpu": gpu,
                "restarts": 0,
                "budget": args.budget,
                "output_dir": str(root),
                "model": args.model,
                "retriever_url": args.retriever_url,
                "min_support": args.min_support,
                "dispatch": args.dispatch,
                "log": str(log_dir / f"r53_gpu{gpu}.log"),
            }
        )
    watchdog(jobs, [j["gpu"] for j in jobs], status_path=root / "SUITE_STATUS.json")
    print("[suite] acquire complete", flush=True)


def write_retriever_audit(root: Path, url: str) -> None:
    try:
        health = RetrieverClient(url).health()
    except Exception as exc:
        health = {"ok": False, "error": str(exc)}
    md = [
        "# RETRIEVER_AUDIT",
        "",
        "Backend: Wikipedia 2018 corpus + E5-base-v2 dense retrieval (Search-R1 compatible).",
        "No mock retrieval, no gold supporting facts injected as search hits.",
        "",
        f"url: `{url}`",
        "",
        "```json",
        json.dumps(health, indent=2),
        "```",
    ]
    (root / "RETRIEVER_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_retriever_audit(root, args.retriever_url)
    print("==== STEP 1 smoke ====", flush=True)
    smoke_args = argparse.Namespace(**vars(args))
    smoke_args.gpu = (args.gpus or "4").split(",")[0]
    phase_smoke(smoke_args)
    print("==== STEP 2 splits ====", flush=True)
    phase_splits(args)
    print("==== STEP 3 capability map ====", flush=True)
    phase_capability(args)
    print("==== STEP 4 base_eval + calibration ====", flush=True)
    if args.gpus:
        launch_sharded(args, "base_eval")
    else:
        args.shard = 0
        args.n_shards = 1
        phase_base_eval(args)
        merge_base_eval(root)
    phase_calibration(args)
    phase_router(args)
    print("==== STEP 5 skill validation ====", flush=True)
    val_args = argparse.Namespace(**vars(args))
    val_args.gpu = (args.gpus or args.gpu or "4").split(",")[0]
    phase_skill_validation(val_args)
    print("==== STEP 6 acquisition ====", flush=True)
    if args.gpus:
        launch_acquire(args)
    else:
        args.phase = "acquire"
        run_worker(args)
    print("==== STEP 6b analyze pilot ====", flush=True)
    subprocess.check_call(
        [_venv_python(), str(ROOT / "scripts" / "analyze_r5_3_external.py"), "--input-dir", str(root), "--stage", args.stage],
        cwd=str(ROOT),
    )


def main() -> None:
    args = parse_args()
    # fix accidental walrus leftover if parser imported broken module - r5_3 import is clean
    if args.phase == "all":
        run_all(args)
        return
    if args.phase == "smoke":
        phase_smoke(args)
        return
    if args.phase == "splits":
        phase_splits(args)
        return
    if args.phase == "capability":
        phase_capability(args)
        return
    if args.phase == "base_eval":
        if args.gpus and args.shard is None:
            launch_sharded(args, "base_eval")
        else:
            phase_base_eval(args)
        return
    if args.phase == "calibration":
        phase_calibration(args)
        return
    if args.phase == "router":
        phase_router(args)
        return
    if args.phase == "skill_validation":
        phase_skill_validation(args)
        return
    if args.phase == "acquire":
        if args.gpus:
            launch_acquire(args)
        else:
            run_worker(args)
        return
    if args.phase == "bamboogle":
        cmd = [
            _venv_python(),
            str(ROOT / "scripts" / "eval_r5_3_bamboogle.py"),
            "--input-dir",
            args.output_dir,
            "--output-dir",
            args.output_dir,
            "--retriever-url",
            args.retriever_url,
            "--methods",
            *args.methods,
            "--seeds",
            *[str(s) for s in args.seeds],
        ]
        subprocess.check_call(cmd, cwd=str(ROOT))
        return
    if args.phase == "analyze":
        subprocess.check_call(
            [_venv_python(), str(ROOT / "scripts" / "analyze_r5_3_external.py"), "--input-dir", args.output_dir, "--stage", args.stage],
            cwd=str(ROOT),
        )
        return
    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    main()
