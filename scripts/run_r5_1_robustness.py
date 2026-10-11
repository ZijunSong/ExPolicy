#!/usr/bin/env python3
"""R5.1 robustness driver: seed extension, temporal coverage, causality, dispatch.

Reuses R5 acquisition / SkillAssimilator / probe eval. Never writes into
outputs/r5_acquisition_skill/. Always --resume after a kill.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from expolicy.experiments.common import ROOT, load_world, mean, write_json
from expolicy.experiments.r5_1 import (
    EVAL_AT,
    R51_ROOT,
    R5_ROOT,
    TARGET_CAP_LIST,
    TARGET_CAPS,
    assert_not_r5_output,
    auc_over_budget,
    cap_counts,
    ensure_seed_splits,
    entropy_from_counts,
    eval_with_dispatch,
    first_activation,
    fit_store_for_mode,
    frontier_pool_audit,
    future_by_cap,
    heartbeat,
    load_acquired,
    load_base_metrics,
    load_base_rows,
    load_json,
    load_split,
    replay_threshold_store,
    sha256_file,
    split_overlap_report,
    tasks_by_ids,
    train_router_for_split,
    verify_r5_frozen,
    write_csv,
)
from expolicy.experiments.r_common import apply_gpu
from expolicy.logging.event_logger import append_jsonl

STALL_SEC = 12 * 60
MAX_RESTARTS = 3
POLL_SEC = 30
DEFAULT_MODEL = "/data/ppnm/models/Qwen2.5-3B-Instruct"


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
    if method == "base":
        return root / "base" / f"seed_{seed}"
    return root / method / f"seed_{seed}"


def _launch(job: dict, python: str) -> subprocess.Popen:
    gpu = str(job["gpu"])
    log_path = Path(job["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        python,
        str(ROOT / "scripts" / "run_r5_acquisition_skill.py"),
        "--method",
        job["method"],
        "--seed",
        str(job["seed"]),
        "--gpu",
        gpu,
        "--model",
        job["model"],
        "--output_dir",
        job["output_dir"],
        "--budget",
        str(job["budget"]),
        "--min_support",
        str(job.get("min_support", 8)),
        "--resume",
    ]
    if job.get("splits"):
        cmd.extend(["--splits", job["splits"]])
    if job.get("base_dir"):
        cmd.extend(["--base_dir", job["base_dir"]])
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


def _cuda_visible(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
        env = dict(x.split(b"=", 1) for x in raw.split(b"\0") if b"=" in x)
        val = env.get(b"CUDA_VISIBLE_DEVICES", b"").decode()
        return val or None
    except Exception:
        return None


def _live_run(out: Path) -> dict | None:
    status = out / "STATUS_LIVE.json"
    if (out / "DONE").exists() or not status.exists():
        return None
    try:
        payload = json.loads(status.read_text(encoding="utf-8"))
    except Exception:
        return None
    pid = payload.get("pid")
    age = time.time() - status.stat().st_mtime
    if pid and _alive(int(pid)) and age < STALL_SEC:
        return {"pid": int(pid), "age": age, "phase": payload.get("phase")}
    return None


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
            out = Path(job["run_dir"])
            if (out / "DONE").exists():
                print(f"[skip-done] {job['name']}", flush=True)
                finished.append(pending.pop(0))
                continue
            live = _live_run(out)
            if live:
                job = pending.pop(0)
                job["pid"] = live["pid"]
                job["proc"] = None
                job["adopted"] = True
                job["launched_at"] = time.time()
                occupied = _cuda_visible(live["pid"]) or str(job.get("gpu") or "")
                job["gpu"] = occupied or "adopted"
                if occupied and occupied in free:
                    free.remove(occupied)
                print(f"[adopt] {job['name']} pid={job['pid']} gpu={job['gpu']} phase={live.get('phase')}", flush=True)
                running.append(job)
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
            if proc is not None:
                code = proc.poll()
            else:
                code = None if job.get("pid") and _alive(int(job["pid"])) else 0
            out = Path(job["run_dir"])
            age = _status_age(out)
            stalled = False
            if code is None:
                if age is None and time.time() - job["launched_at"] > 180:
                    stalled = True
                if age is not None and age > STALL_SEC:
                    stalled = True
            if stalled:
                print(f"[stall] {job['name']} pid={job.get('pid')} age={age} — kill and resume", flush=True)
                _kill(job)
                if job.get("gpu") and job["gpu"] != "adopted":
                    free.append(job["gpu"])
                job["gpu"] = None
                job["restarts"] += 1
                job["adopted"] = False
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
            if job.get("gpu") and job["gpu"] != "adopted":
                free.append(job["gpu"])
            job["gpu"] = None
            if (out / "DONE").exists():
                print(f"[done] {job['name']} exit={code}", flush=True)
                finished.append(job)
            else:
                print(f"[exit] {job['name']} code={code} — resume", flush=True)
                job["restarts"] += 1
                job["adopted"] = False
                if job["restarts"] <= MAX_RESTARTS:
                    pending.insert(0, job)
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


def gpu_binding_smoke(gpu: str | None) -> dict[str, Any]:
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    code = (
        "import os, torch; "
        "print(json.dumps({"
        "'cuda': torch.cuda.is_available(), "
        "'n': torch.cuda.device_count() if torch.cuda.is_available() else 0, "
        "'visible': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
        "}))"
    )
    # json import inside - fix
    code = (
        "import os, json, torch\n"
        "payload = {'cuda': bool(torch.cuda.is_available()), "
        "'n': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0, "
        "'visible': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\n"
        "print(json.dumps(payload))"
    )
    proc = subprocess.run([_venv_python(), "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"GPU smoke failed: {proc.stderr[-500:]}")
    line = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")][-1]
    payload = json.loads(line)
    if gpu and payload.get("visible") not in {str(gpu), gpu}:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES not bound: {payload}")
    if gpu and payload.get("n") not in {0, 1}:
        # n==0 can happen if CUDA is momentarily busy; n>1 means binding failed
        if payload.get("n") > 1:
            raise RuntimeError(f"expected 1 visible GPU, got {payload}")
    return payload


def make_acq_job(
    *,
    method: str,
    seed: int,
    output_dir: Path,
    model: str,
    budget: int,
    log_dir: Path,
    splits: Path | None = None,
    base_dir: Path | None = None,
    min_support: int = 8,
) -> dict:
    run_dir = _run_dir(output_dir, method, seed)
    return {
        "name": f"{method}_s{seed}",
        "method": method,
        "seed": seed,
        "restarts": 0,
        "gpu": None,
        "output_dir": str(output_dir),
        "model": model,
        "budget": budget,
        "min_support": min_support,
        "splits": str(splits) if splits else None,
        "base_dir": str(base_dir) if base_dir else None,
        "log": str(log_dir / f"r51_{output_dir.name}_{method}_s{seed}.log"),
        "run_dir": str(run_dir),
    }


def smoke_integrity(out: Path, r5_root: Path, seeds: list[int], gpu: str | None) -> dict[str, Any]:
    assert_not_r5_output(out)
    frozen = verify_r5_frozen(r5_root)
    gpu_info = gpu_binding_smoke(gpu) if gpu else {"skipped": True}
    overlap = {}
    for seed in seeds:
        try:
            split = load_split(out, seed, r5_root=r5_root)
        except FileNotFoundError:
            continue
        ov = split_overlap_report(split)
        overlap[str(seed)] = ov
        if ov["future_acq"] or ov["future_ret"] or ov["acq_ret"]:
            raise RuntimeError(f"overlap seed={seed}: {ov}")
        if seed in (0, 1, 2):
            r5_hash = sha256_file(r5_root / "splits" / f"seed_{seed}.json")
            if r5_hash != frozen["split_hashes"][seed]:
                raise RuntimeError("R5 split hash mutated during smoke")
    payload = {
        "r5_not_overwritten": True,
        "r5_audit": frozen["audit"],
        "split_hashes": frozen["split_hashes"],
        "overlap": overlap,
        "gpu": gpu_info,
        "resume_contract": "run_r5_acquisition_skill.py --resume; DONE short-circuits; failed run does not write sibling DONE",
    }
    write_json(out / "SMOKE.json", payload)
    print(json.dumps({"smoke": "PASS", **{k: payload[k] for k in ("r5_audit", "gpu")}}, indent=2), flush=True)
    return payload


def track_seed_extension(args) -> None:
    out = Path(args.output)
    assert_not_r5_output(out)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)
    methods = list(args.methods)
    ensure_seed_splits(out, seeds, r5_root=Path(args.r5_root))
    if not args.skip_smoke:
        smoke_integrity(out, Path(args.r5_root), seeds, args.gpu)
    jobs = []
    for seed in seeds:
        split_path = out / "splits" / f"seed_{seed}.json"
        base_dir = out / "base" / f"seed_{seed}"
        for method in methods:
            jobs.append(
                make_acq_job(
                    method=method,
                    seed=seed,
                    output_dir=out,
                    model=args.model,
                    budget=args.budget,
                    log_dir=log_dir,
                    splits=split_path,
                    base_dir=base_dir,
                )
            )
    if getattr(args, "gpus", None):
        gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip()]
    else:
        gpus = [str(args.gpu or "4")]
    watchdog(jobs, gpus, status_path=out / "SUITE_STATUS.json")


def track_temporal(args) -> None:
    out = Path(args.output)
    r5 = Path(args.r5_root)
    assert_not_r5_output(out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)
    if not args.skip_smoke:
        smoke_integrity(out, r5, seeds, args.gpu)
    world = load_world()
    audits = []
    for seed in seeds:
        split = load_split(out, seed, r5_root=r5)
        audits.append(frontier_pool_audit(world, split, seed=seed))
    write_json(out / "AUDIT_TEMPORAL_POOL.json", {"seeds": audits})
    lines = ["# AUDIT_TEMPORAL_POOL", "", "Frozen acquisition pool frontier-score diagnostic. No future tasks copied.", ""]
    for row in audits:
        lines += [
            f"## seed {row['seed']}",
            "",
            f"- pool n={row['n_pool']} by_cap={row['pool_by_cap']}",
            f"- temporal in pool={row['n_temporal_in_pool']} best_rank={row['temporal_best_rank']} mean_percentile={row['temporal_mean_percentile']}",
            f"- temporal in top 48/64/96 = {row['temporal_in_top_48']}/{row['temporal_in_top_64']}/{row['temporal_in_top_96']}",
            f"- would reach 8 temporal at 48/64/96: {row['would_reach_8_temporal_at_budget_48']}/{row['would_reach_8_temporal_at_budget_64']}/{row['would_reach_8_temporal_at_budget_96']}",
            "",
        ]
    (out / "AUDIT_TEMPORAL_POOL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # reuse R5 frontier/passive; only launch frontier_capnorm
    methods = [m for m in args.methods if m == "frontier_capnorm"] or ["frontier_capnorm"]
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in seeds:
        split_path = r5 / "splits" / f"seed_{seed}.json"
        base_dir = r5 / "base" / f"seed_{seed}"
        for method in methods:
            jobs.append(
                make_acq_job(
                    method=method,
                    seed=seed,
                    output_dir=out,
                    model=args.model,
                    budget=args.budget,
                    log_dir=log_dir,
                    splits=split_path,
                    base_dir=base_dir,
                )
            )
    if jobs:
        watchdog(jobs, [str(args.gpu or "5")], status_path=out / "SUITE_STATUS.json")
    _write_temporal_tables(out, r5, seeds, list(args.methods))


def _write_temporal_tables(out: Path, r5: Path, seeds: list[int], methods: list[str]) -> None:
    world = load_world()
    acq_rows = []
    act_rows = []
    cap_rows = []
    for method in methods:
        for seed in seeds:
            run_dir = out / method / f"seed_{seed}" if method == "frontier_capnorm" else r5 / method / f"seed_{seed}"
            if not (run_dir / "final_metrics.json").exists():
                continue
            metrics = load_json(run_dir / "final_metrics.json")
            acquired = load_acquired(run_dir)
            hist = dict(metrics.get("caps_chosen") or Counter_from_acq(acquired))
            t_count = int(hist.get("temporal_reasoning") or 0)
            first = metrics.get("first_skill_activation") or first_activation(run_dir)
            t_first = None
            running = 0
            for i, exp in enumerate(acquired, 1):
                if exp.capability == "temporal_reasoning":
                    running += 1
                    if running >= 8 and t_first is None:
                        t_first = i
            acq_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "temporal_acquired": t_count,
                    "temporal_first_reach_8": t_first,
                    "s2_activated": "temporal_reasoning" in (first or {}),
                    "s2_activation_budget": (first or {}).get("temporal_reasoning"),
                    "caps_chosen": json.dumps(hist),
                    "acquisition_entropy": entropy_from_counts({k: int(v) for k, v in hist.items()}),
                    "n_target_caps_covered": sum(1 for c in TARGET_CAP_LIST if int(hist.get(c) or 0) > 0),
                    "future_em": metrics.get("future_em"),
                    "delta_future": metrics.get("delta_future"),
                }
            )
            act_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "first_skill_activation": json.dumps(first),
                    "n_active_skills": metrics.get("n_active_skills"),
                    "s2_gain": ((metrics.get("future_per_cap") or {}).get("temporal_reasoning") or {}).get("em"),
                }
            )
            split = load_split(out, seed, r5_root=r5)
            base = load_base_metrics(r5, seed, fallback_root=out)
            for cap in TARGET_CAP_LIST:
                em = float(((metrics.get("future_per_cap") or {}).get(cap) or {}).get("em") or 0.0)
                bem = float(((base.get("future_per_cap") or {}).get(cap) or {}).get("em") or 0.0)
                cap_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "capability": cap,
                        "acquired": int(hist.get(cap) or 0),
                        "future_em": em,
                        "base_em": bem,
                        "gain": em - bem,
                        "activated": cap in (first or {}),
                    }
                )
    write_csv(out / "TEMPORAL_ACQUISITION_COUNTS.csv", acq_rows)
    write_csv(out / "TEMPORAL_ACTIVATION.csv", act_rows)
    write_csv(out / "TEMPORAL_CAPABILITY_OUTCOME.csv", cap_rows)
    md = ["# TEMPORAL_SUMMARY", "", "Coverage-aware frontier vs original frontier / passive.", ""]
    for row in acq_rows:
        md.append(
            f"- `{row['method']}` seed={row['seed']}: temporal_acq={row['temporal_acquired']} "
            f"reach8={row['temporal_first_reach_8']} S2_on={row['s2_activated']} "
            f"ΔFuture={row['delta_future']} entropy={row['acquisition_entropy']:.3f}"
        )
    (out / "TEMPORAL_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _ = world


def Counter_from_acq(acquired) -> dict:
    from collections import Counter

    return dict(Counter(e.capability or "unknown" for e in acquired))


def track_content_causality(args) -> None:
    out = Path(args.output)
    r5 = Path(args.r5_root)
    assert_not_r5_output(out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)
    methods = list(args.methods)
    modes = list(args.fit_modes)
    if not args.skip_smoke:
        smoke_integrity(out, r5, seeds, args.gpu)
        _run_content_one(out, r5, methods[0], seeds[0], modes, smoke=True)
        print("[smoke] content_causality seed done", flush=True)
    per_seed = []
    per_cap = []
    world = load_world()
    for method in methods:
        for seed in seeds:
            rows = _run_content_one(out, r5, method, seed, modes, smoke=False, world=world)
            per_seed.extend(rows["per_seed"])
            per_cap.extend(rows["per_cap"])
    write_csv(out / "CONTENT_CAUSALITY_PER_SEED.csv", per_seed)
    write_csv(out / "CONTENT_CAUSALITY_PER_CAP.csv", per_cap)
    md = ["# CONTENT_CAUSALITY_SUMMARY", "", "Activation mask frozen from original R5. Only fit() content changes.", ""]
    grouped = {}
    for row in per_seed:
        grouped.setdefault((row["method"], row["fit_mode"]), []).append(row)
    for (method, mode), items in grouped.items():
        md.append(
            f"- `{method}` / `{mode}`: mean future EM={mean(float(r['future_em']) for r in items):.4f} "
            f"mean ΔFuture={mean(float(r['delta_future']) for r in items):.4f}"
        )
    (out / "CONTENT_CAUSALITY_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "DONE").write_text("ok\n", encoding="utf-8")


def _run_content_one(out, r5, method, seed, modes, *, smoke=False, world=None):
    world = world or load_world()
    run_r5 = r5 / method / f"seed_{seed}"
    if not (run_r5 / "DONE").exists():
        raise RuntimeError(f"need original R5 trajectory {run_r5}")
    split = load_split(out, seed, r5_root=r5)
    acquired = load_acquired(run_r5)
    mask = first_activation(run_r5)
    probe_ids = list(split["future_probe_ids"]) + list(split["retention_probe"])
    future_map = future_by_cap(world, split)
    gold_docs = [d for tasks in future_map.values() for t in tasks for d in t.gold_doc_ids]
    blocked = set(probe_ids)
    base = load_base_metrics(r5, seed)
    base_rows = load_base_rows(r5, seed)
    retention = tasks_by_ids(world, split["retention_probe"])
    per_seed = []
    per_cap = []
    use_modes = modes[:1] if smoke else modes
    for mode in use_modes:
        job_out = out / method / f"seed_{seed}" / mode
        job_out.mkdir(parents=True, exist_ok=True)
        heartbeat(job_out, phase="fit", method=method, seed=seed, mode=mode)
        if (job_out / "DONE").exists():
            payload = load_json(job_out / "metrics.json")
        else:
            store, events = fit_store_for_mode(
                acquired,
                mode=mode,
                min_support=8,
                activation_mask=mask,
                probe_ids=probe_ids,
                gold_docs=gold_docs,
                skill_dir=job_out / "skills",
                seed=seed,
                method=method,
                r5_root=r5,
                blocked=blocked,
                budget=len(acquired),
            )
            write_json(job_out / "fit_events.json", events)
            ev = eval_with_dispatch(
                world,
                future_map,
                retention,
                store=store,
                base_rows=base_rows,
                dispatch="oracle",
                out=job_out,
            )
            payload = {
                "method": method,
                "seed": seed,
                "fit_mode": mode,
                "future_em": ev["future_em"],
                "retention_em": ev["retention_em"],
                "delta_future": ev["future_em"] - float(base["future"]["em"]),
                "delta_retention": ev["retention_em"] - float(base["retention"]["em"]),
                "n_active_skills": ev["n_active_skills"],
                "activation_mask": mask,
                "future_per_cap": ev["future_per_cap"],
            }
            write_json(job_out / "metrics.json", payload)
            (job_out / "DONE").write_text("ok\n", encoding="utf-8")
        per_seed.append(payload)
        for cap in TARGET_CAP_LIST:
            em = float(((payload.get("future_per_cap") or {}).get(cap) or {}).get("em") or 0.0)
            bem = float(((base.get("future_per_cap") or {}).get(cap) or {}).get("em") or 0.0)
            per_cap.append(
                {
                    "method": method,
                    "seed": seed,
                    "fit_mode": mode,
                    "capability": cap,
                    "future_em": em,
                    "gain": em - bem,
                    "activated": cap in mask,
                }
            )
    return {"per_seed": per_seed, "per_cap": per_cap}


def track_activation_threshold(args) -> None:
    out = Path(args.output)
    r5 = Path(args.r5_root)
    assert_not_r5_output(out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)
    methods = list(args.methods)
    thresholds = list(args.thresholds)
    if not args.skip_smoke:
        smoke_integrity(out, r5, seeds, args.gpu)
        _run_threshold_one(out, r5, methods[0], seeds[0], thresholds[:1])
        print("[smoke] activation_threshold seed done", flush=True)
    world = load_world()
    per_seed = []
    curves = []
    acts = []
    for method in methods:
        for seed in seeds:
            for k in thresholds:
                rec = _run_threshold_one(out, r5, method, seed, [k], world=world)
                per_seed.extend(rec["per_seed"])
                curves.extend(rec["curves"])
                acts.extend(rec["acts"])
    write_csv(out / "THRESHOLD_PER_SEED.csv", per_seed)
    write_csv(out / "THRESHOLD_LEARNING_CURVE.csv", curves)
    write_csv(out / "THRESHOLD_ACTIVATION.csv", acts)
    md = ["# THRESHOLD_SUMMARY", "", "Same acquisition trajectories; only activation threshold K changes.", ""]
    for method in methods:
        for k in thresholds:
            items = [r for r in per_seed if r["method"] == method and int(r["k"]) == int(k)]
            if not items:
                continue
            md.append(
                f"- `{method}` K={k}: mean ΔFuture={mean(float(r['delta_future']) for r in items):.4f} "
                f"mean time-to-first-skill={mean(float(r['time_to_first_skill'] or 99) for r in items):.1f}"
            )
    (out / "THRESHOLD_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "DONE").write_text("ok\n", encoding="utf-8")


def _run_threshold_one(out, r5, method, seed, thresholds, world=None):
    world = world or load_world()
    run_r5 = r5 / method / f"seed_{seed}"
    split = load_split(out, seed, r5_root=r5)
    acquired = load_acquired(run_r5)
    probe_ids = list(split["future_probe_ids"]) + list(split["retention_probe"])
    future_map = future_by_cap(world, split)
    gold_docs = [d for tasks in future_map.values() for t in tasks for d in t.gold_doc_ids]
    base = load_base_metrics(r5, seed)
    base_rows = load_base_rows(r5, seed)
    retention = tasks_by_ids(world, split["retention_probe"])
    per_seed, curves, acts = [], [], []
    for k in thresholds:
        job_out = out / method / f"seed_{seed}" / f"k{k}"
        job_out.mkdir(parents=True, exist_ok=True)
        heartbeat(job_out, phase="threshold", method=method, seed=seed, k=k)
        curve_pts = []
        first = {}
        last = None
        if (job_out / "DONE").exists():
            payload = load_json(job_out / "metrics.json")
            last = payload
            for line in (job_out / "checkpoint_metrics.jsonl").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    curves.append(row)
                    curve_pts.append((int(row["budget"]), float(row["delta_future"])))
            first = payload.get("first_skill_activation") or {}
        else:
            for budget in EVAL_AT:
                if budget == 0:
                    ev = {
                        "future_em": float(base["future"]["em"]),
                        "retention_em": float(base["retention"]["em"]),
                        "future_per_cap": base.get("future_per_cap") or {},
                        "n_active_skills": 0,
                    }
                    first = {}
                else:
                    store, events, first = replay_threshold_store(
                        acquired,
                        min_support=int(k),
                        budget=budget,
                        probe_ids=probe_ids,
                        gold_docs=gold_docs,
                        skill_dir=job_out / f"skills_b{budget}",
                    )
                    ev = eval_with_dispatch(
                        world,
                        future_map,
                        retention,
                        store=store,
                        base_rows=base_rows,
                        dispatch="oracle",
                        out=job_out,
                    )
                    for evn in events:
                        acts.append({"method": method, "seed": seed, "k": k, **evn})
                delta = ev["future_em"] - float(base["future"]["em"])
                row = {
                    "method": method,
                    "seed": seed,
                    "k": k,
                    "budget": budget,
                    "future_em": ev["future_em"],
                    "delta_future": delta,
                    "n_active_skills": ev.get("n_active_skills") or 0,
                }
                append_jsonl(job_out / "checkpoint_metrics.jsonl", row)
                curves.append(row)
                curve_pts.append((budget, delta))
                last = ev
            times = sorted(first.values()) if first else []
            payload = {
                "method": method,
                "seed": seed,
                "k": k,
                "future_em": last["future_em"],
                "delta_future": last["future_em"] - float(base["future"]["em"]),
                "retention_em": last["retention_em"],
                "time_to_first_skill": times[0] if times else None,
                "time_to_second_skill": times[1] if len(times) > 1 else None,
                "auc_delta_future": auc_over_budget(curve_pts),
                "first_skill_activation": first,
                "n_active_skills": last.get("n_active_skills"),
            }
            write_json(job_out / "metrics.json", payload)
            (job_out / "DONE").write_text("ok\n", encoding="utf-8")
        per_seed.append(payload if last is payload or "k" in (last or {}) else payload)
        # always append the saved payload
        if payload not in per_seed:
            pass
    # dedup last append logic: per_seed already has payload
    return {"per_seed": per_seed, "curves": curves, "acts": acts}


def track_dispatch(args) -> None:
    out = Path(args.output)
    r5 = Path(args.r5_root)
    assert_not_r5_output(out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)
    methods = list(args.methods)
    modes = list(args.dispatch_modes)
    if not args.skip_smoke:
        smoke_integrity(out, r5, seeds, args.gpu)
        _run_dispatch_seed(out, r5, methods[0], seeds[0], modes[:2])
        print("[smoke] deployable_dispatch seed done", flush=True)
    world = load_world()
    router_metrics = {}
    confusion_rows = []
    down_rows = []
    cap_rows = []
    for seed in seeds:
        split = load_split(out, seed, r5_root=r5)
        router, metrics = train_router_for_split(world, split, seed=seed)
        router_metrics[str(seed)] = metrics
        labels = metrics.get("labels") or []
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                confusion_rows.append(
                    {"seed": seed, "true": a, "pred": b, "count": (metrics.get("confusion") or [[0]])[i][j]}
                )
        for method in methods:
            rec = _run_dispatch_seed(out, r5, method, seed, modes, world=world, router=router)
            down_rows.extend(rec["down"])
            cap_rows.extend(rec["cap"])
    write_json(out / "DISPATCH_ROUTER_METRICS.json", router_metrics)
    write_csv(out / "DISPATCH_CONFUSION.csv", confusion_rows)
    write_csv(out / "DISPATCH_DOWNSTREAM.csv", down_rows)
    write_csv(out / "DISPATCH_PER_CAP.csv", cap_rows)
    md = ["# DISPATCH_SUMMARY", "", "Text-router uses question text only from the acquisition pool.", ""]
    for seed, met in router_metrics.items():
        md.append(
            f"- seed {seed}: acc={met.get('accuracy'):.3f} macro-F1={met.get('macro_f1'):.3f} "
            f"leakage_pass={met.get('leakage_pass')} recall={met.get('per_class_recall')}"
        )
    for method in methods:
        oracle = [r for r in down_rows if r["method"] == method and r["dispatch"] == "oracle"]
        router_d = [r for r in down_rows if r["method"] == method and r["dispatch"] == "text_router"]
        if oracle and router_d:
            md.append(
                f"- `{method}` oracle EM={mean(float(r['future_em']) for r in oracle):.4f} "
                f"text-router EM={mean(float(r['future_em']) for r in router_d):.4f} "
                f"gap={mean(float(r['future_em']) for r in oracle) - mean(float(r['future_em']) for r in router_d):.4f}"
            )
    (out / "DISPATCH_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "DONE").write_text("ok\n", encoding="utf-8")


def _run_dispatch_seed(out, r5, method, seed, modes, world=None, router=None):
    world = world or load_world()
    split = load_split(out, seed, r5_root=r5)
    if router is None:
        router, _ = train_router_for_split(world, split, seed=seed)
    run_r5 = r5 / method / f"seed_{seed}"
    acquired = load_acquired(run_r5)
    mask = first_activation(run_r5)
    probe_ids = list(split["future_probe_ids"]) + list(split["retention_probe"])
    future_map = future_by_cap(world, split)
    gold_docs = [d for tasks in future_map.values() for t in tasks for d in t.gold_doc_ids]
    blocked = set(probe_ids)
    base = load_base_metrics(r5, seed)
    base_rows = load_base_rows(r5, seed)
    retention = tasks_by_ids(world, split["retention_probe"])
    job_root = out / method / f"seed_{seed}"
    job_root.mkdir(parents=True, exist_ok=True)
    store, _events = fit_store_for_mode(
        acquired,
        mode="real",
        min_support=8,
        activation_mask=mask,
        probe_ids=probe_ids,
        gold_docs=gold_docs,
        skill_dir=job_root / "skills_real",
        seed=seed,
        method=method,
        r5_root=r5,
        blocked=blocked,
        budget=len(acquired),
    )
    alias = {
        "oracle": "oracle",
        "oracle_dispatch": "oracle",
        "text_router": "text_router",
        "text_router_dispatch": "text_router",
        "no_skill": "no_skill",
        "all_skills_fixed_order": "all_skills_fixed_order",
    }
    down, cap_rows = [], []
    for raw in modes:
        mode = alias.get(raw, raw)
        job_out = job_root / mode
        job_out.mkdir(parents=True, exist_ok=True)
        heartbeat(job_out, phase="dispatch", method=method, seed=seed, mode=mode)
        if (job_out / "DONE").exists():
            payload = load_json(job_out / "metrics.json")
        else:
            ev = eval_with_dispatch(
                world,
                future_map,
                retention,
                store=None if mode == "no_skill" else store,
                base_rows=base_rows,
                dispatch=mode,
                router=router,
                out=job_out,
            )
            payload = {
                "method": method,
                "seed": seed,
                "dispatch": mode,
                "future_em": ev["future_em"],
                "retention_em": ev["retention_em"],
                "delta_future": ev["future_em"] - float(base["future"]["em"]),
                "delta_retention": ev["retention_em"] - float(base["retention"]["em"]),
                "s0_fp_rate": ev["s0_fp_rate"],
                "n_skill_invocations": ev["n_skill_invocations"],
                "future_per_cap": ev["future_per_cap"],
            }
            write_json(job_out / "metrics.json", payload)
            (job_out / "DONE").write_text("ok\n", encoding="utf-8")
        down.append(payload)
        for cap in TARGET_CAP_LIST:
            em = float(((payload.get("future_per_cap") or {}).get(cap) or {}).get("em") or 0.0)
            bem = float(((base.get("future_per_cap") or {}).get(cap) or {}).get("em") or 0.0)
            cap_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "dispatch": mode,
                    "capability": cap,
                    "future_em": em,
                    "gain": em - bem,
                }
            )
    return {"down": down, "cap": cap_rows}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="R5.1 robustness & causal validation driver")
    p.add_argument(
        "--track",
        required=True,
        choices=(
            "seed_extension",
            "temporal_coverage",
            "activation_threshold",
            "content_causality",
            "deployable_dispatch",
            "gpu6",
        ),
        help="Which R5.1 track to run",
    )
    p.add_argument("--methods", nargs="+", default=["frontier", "passive"])
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--thresholds", type=int, nargs="+", default=[4, 8, 12, 16], help="activation K values")
    p.add_argument(
        "--fit-modes",
        dest="fit_modes",
        nargs="+",
        default=["real", "count_only", "within_cap_shuffle", "cross_cap_content"],
        help="C2 fit modes",
    )
    p.add_argument(
        "--dispatch",
        dest="dispatch_modes",
        nargs="+",
        default=["oracle", "text_router", "no_skill", "all_skills_fixed_order"],
        help="Track D dispatch conditions",
    )
    p.add_argument("--freeze-activation-mask-from-r5", action="store_true", default=True)
    p.add_argument("--budget", type=int, default=48)
    p.add_argument("--output", type=str, default=None, help="Track output directory")
    p.add_argument("--r5-root", default=str(R5_ROOT))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--gpu", default=None, help="Physical GPU id, also exported as CUDA_VISIBLE_DEVICES")
    p.add_argument("--gpus", default=None, help="Comma-separated GPU list for the seed_extension watchdog")
    p.add_argument("--skip_smoke", action="store_true")
    p.add_argument("--skip_analyze", action="store_true")
    return p


def default_output(track: str) -> Path:
    mapping = {
        "seed_extension": R51_ROOT / "seed_extension",
        "temporal_coverage": R51_ROOT / "temporal",
        "activation_threshold": R51_ROOT / "activation_threshold",
        "content_causality": R51_ROOT / "content_causality",
        "deployable_dispatch": R51_ROOT / "deployable_dispatch",
        "gpu6": R51_ROOT / "content_causality",
    }
    return mapping[track]


def default_seeds(track: str) -> list[int]:
    if track == "seed_extension":
        return [3, 4, 5, 6, 7, 8, 9]
    return [0, 1, 2]


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    apply_gpu(args.gpu)
    if args.output is None:
        args.output = str(default_output(args.track))
    if args.seeds is None:
        args.seeds = default_seeds(args.track)
    assert_not_r5_output(Path(args.output))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(f"[r5.1] track={args.track} gpu={args.gpu or os.environ.get('CUDA_VISIBLE_DEVICES')} out={args.output}", flush=True)
    if args.track == "seed_extension":
        track_seed_extension(args)
        if not args.skip_analyze:
            subprocess.check_call(
                [_venv_python(), str(ROOT / "scripts" / "analyze_r5_1_robustness.py"), "--only", "seed_extension", "--root", str(R51_ROOT), "--r5-root", args.r5_root],
                cwd=str(ROOT),
            )
    elif args.track == "temporal_coverage":
        track_temporal(args)
        if not args.skip_analyze:
            subprocess.check_call(
                [_venv_python(), str(ROOT / "scripts" / "analyze_r5_1_robustness.py"), "--only", "temporal", "--root", str(R51_ROOT), "--r5-root", args.r5_root],
                cwd=str(ROOT),
            )
    elif args.track == "content_causality":
        track_content_causality(args)
    elif args.track == "activation_threshold":
        track_activation_threshold(args)
    elif args.track == "deployable_dispatch":
        track_dispatch(args)
    elif args.track == "gpu6":
        args.output = str(R51_ROOT / "content_causality")
        track_content_causality(args)
        args.output = str(R51_ROOT / "activation_threshold")
        Path(args.output).mkdir(parents=True, exist_ok=True)
        track_activation_threshold(args)
    print("[r5.1] track complete", flush=True)


if __name__ == "__main__":
    main()
