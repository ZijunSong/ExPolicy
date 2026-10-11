#!/usr/bin/env python3
"""R7 orchestrator: CPU gates then 4-GPU jobs on GPU 4-7 with stall watchdog.

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/r7/orchestrate_r7.py \
    --stage all-prepolicy --scheduler auto --resume --watchdog \
    --out outputs/r7_policy_generalization
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from expolicy.r7.constants import (
    ALL_PROBE_MODELS,
    DEFAULT_GPUS,
    MAX_RESTARTS,
    MODEL_ROOT,
    POLL_SEC,
    PRIMARY_MODELS,
    R7_DATA,
    R7_ROOT,
    RETRIEVER_PORT,
    RETRIEVER_URL,
    STALL_SEC,
)
from expolicy.experiments.common import ROOT, write_json

PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable


def log(msg: str, out: Path | None = None) -> None:
    line = f"[r7 {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if out is not None:
        logdir = out / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        (logdir / "orchestrate.log").open("a", encoding="utf-8").write(line + "\n")


def run(cmd: list[str], name: str, out: Path, extra_env: dict | None = None) -> None:
    log(" ".join(cmd), out)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["EXPOLICY_RETRIEVER_URL"] = RETRIEVER_URL
    env["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    if extra_env:
        env.update(extra_env)
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    if rc != 0:
        raise SystemExit(f"{name} failed rc={rc}")


def retriever_ok() -> bool:
    try:
        import httpx

        r = httpx.get(f"{RETRIEVER_URL}/health", timeout=5.0)
        return r.status_code == 200 and bool(r.json().get("ok"))
    except Exception:
        return False


def launch_retriever(out: Path) -> None:
    if retriever_ok():
        log("retriever already healthy", out)
        return
    meta = json.loads((ROOT / "data" / "wiki18" / "BACKEND_PATHS.json").read_text())
    logdir = out / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        str(ROOT / "expolicy" / "environments" / "public_search" / "retriever_server.py"),
        "--index_path",
        meta["index"],
        "--corpus_path",
        meta["corpus"],
        "--retriever_model",
        meta["e5_model"],
        "--retriever_name",
        "e5",
        "--topk",
        "5",
        "--port",
        str(RETRIEVER_PORT),
        "--device",
        "cpu",
    ]
    log("launch retriever: " + " ".join(cmd), out)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    handle = (logdir / "retriever.log").open("a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    (out / "retriever.pid").write_text(str(proc.pid), encoding="utf-8")
    t0 = time.time()
    while time.time() - t0 < 3600:
        if retriever_ok():
            log(f"retriever up pid={proc.pid}", out)
            return
        if proc.poll() is not None:
            raise SystemExit(f"retriever exited {proc.returncode}; see logs/retriever.log")
        time.sleep(5)
    raise SystemExit("retriever did not become healthy")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def pid_tree(pid: int) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-P", str(pid)], text=True)
        kids = [int(x) for x in out.split() if x.strip()]
    except subprocess.CalledProcessError:
        kids = []
    all_pids = list(kids)
    for k in kids:
        all_pids.extend(pid_tree(k))
    return all_pids


def kill_tree(pid: int) -> None:
    pids = [pid] + pid_tree(pid)
    for p in pids:
        if alive(p):
            try:
                os.kill(p, signal.SIGTERM)
            except OSError:
                pass
    time.sleep(5)
    for p in pids:
        if alive(p):
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass


def job_done(job: dict) -> bool:
    d = Path(job["out"])
    if job.get("phase") == "compat":
        smoke = d / f"{job['model']}.smoke.json"
        return smoke.exists()
    return (d / "DONE").exists()


def status_age(job: dict) -> float | None:
    root = Path(job["out"])
    if not root.exists():
        return None
    files = list(root.rglob("STATUS_LIVE.json"))
    if not files:
        return None
    newest = max(p.stat().st_mtime for p in files)
    return time.time() - newest


def launch_job(job: dict, gpu: str, out: Path) -> subprocess.Popen:
    Path(job["out"]).mkdir(parents=True, exist_ok=True)
    log_path = out / "logs" / f"gpu{gpu}_{job['id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} {job['id']} =====\n")
    handle.flush()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["EXPOLICY_RETRIEVER_URL"] = RETRIEVER_URL
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    cmd = [
        PY,
        str(ROOT / "scripts" / "r7" / "run_r7_worker.py"),
        "--phase",
        job["phase"],
        "--model",
        job["model"],
        "--gpu",
        gpu,
        "--out",
        job["out"],
        "--suite",
        job.get("suite") or str(R7_DATA),
    ]
    for k in ("condition", "method", "batch_id", "seed", "per_dataset", "budget", "optimized_skills"):
        if job.get(k) is not None and job.get(k) != "":
            flag = "--" + k.replace("_", "-")
            cmd.extend([flag, str(job[k])])
    log(f"GPU{gpu} {' '.join(cmd)}", out)
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)


def schedule(jobs: list[dict], gpus: list[str], out: Path, *, watchdog: bool) -> None:
    pending = [j for j in jobs if not job_done(j)]
    log(f"schedule {len(jobs)} jobs, {len(pending)} pending, gpus={gpus}", out)
    running: dict[str, dict] = {}  # gpu -> {job, proc, restarts}
    queue = list(pending)
    restarts: dict[str, int] = {}
    while queue or running:
        # launch
        for gpu in gpus:
            if gpu in running:
                continue
            if not queue:
                continue
            job = queue.pop(0)
            proc = launch_job(job, gpu, out)
            running[gpu] = {"job": job, "proc": proc, "started": time.time()}
            write_json(out / "logs" / f"gpu{gpu}.pid", {"pid": proc.pid, "job": job["id"]})
        # poll
        time.sleep(POLL_SEC if watchdog else 10)
        for gpu in list(running):
            slot = running[gpu]
            job = slot["job"]
            proc: subprocess.Popen = slot["proc"]
            rc = proc.poll()
            if job_done(job):
                log(f"DONE {job['id']} gpu={gpu}", out)
                if rc is None:
                    kill_tree(proc.pid)
                running.pop(gpu, None)
                continue
            stalled = False
            age = status_age(job)
            if watchdog and age is not None and age > STALL_SEC:
                stalled = True
            if watchdog and rc is None and age is None and time.time() - slot["started"] > STALL_SEC:
                stalled = True
            if rc is not None and not job_done(job):
                stalled = True
                log(f"exit rc={rc} without DONE {job['id']}", out)
            if stalled:
                n = restarts.get(job["id"], 0)
                log(f"STALL {job['id']} gpu={gpu} restarts={n} age={age}", out)
                kill_tree(proc.pid)
                running.pop(gpu, None)
                if n >= MAX_RESTARTS:
                    write_json(Path(job["out"]) / "FAILED.json", {"reason": "max_restarts", "restarts": n})
                    log(f"GIVE UP {job['id']}", out)
                    continue
                restarts[job["id"]] = n + 1
                queue.insert(0, job)
        # GPU util snapshot
        try:
            snap = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
                text=True,
            )
            (out / "logs" / "nvidia.snap").write_text(snap, encoding="utf-8")
        except Exception:
            pass
    leftover = [j["id"] for j in jobs if not job_done(j)]
    if leftover:
        raise SystemExit(f"jobs failed: {leftover}")


def phase_audit(out: Path) -> None:
    audit = out / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    run([PY, str(ROOT / "scripts" / "r7" / "audit_novelty_collision.py"), "--out", str(audit)], "novelty", out)
    run(
        [
            PY,
            str(ROOT / "scripts" / "r7" / "audit_repo_and_skill_ownership.py"),
            "--out",
            str(audit),
        ],
        "ownership",
        out,
    )
    run(
        [
            PY,
            str(ROOT / "scripts" / "r7" / "audit_r5_r6_cases.py"),
            "--out",
            str(audit / "cases"),
        ],
        "cases",
        out,
    )
    run([PY, "-m", "pytest", "-q", "tests/r7"], "pytest", out)


def phase_splits(out: Path) -> None:
    if (R7_DATA / "split.json").exists() and (out / "audit" / "SPLIT_AUDIT.md").exists():
        log("skip splits (frozen)", out)
        return
    run(
        [
            PY,
            str(ROOT / "scripts" / "r7" / "build_r7_search_suite.py"),
            "--datasets",
            "nq",
            "triviaqa",
            "popqa",
            "hotpotqa",
            "2wiki",
            "musique",
            "bamboogle",
            "--retriever",
            "e5",
            "--wiki-index",
            "existing",
            "--split-seed",
            "20260815",
            "--calibration-per-dataset",
            "64",
            "--acquisition-per-dataset",
            "300",
            "--skill-validation-per-dataset",
            "80",
            "--future-per-dataset",
            "150",
            "--bamboogle-use-full-if-insufficient",
            "--out",
            str(R7_DATA),
        ],
        "build_splits",
        out,
    )
    run(
        [
            PY,
            str(ROOT / "scripts" / "r7" / "validate_r7_splits.py"),
            "--root",
            str(R7_DATA),
            "--require-task-id-disjoint",
            "--require-question-disjoint",
            "--require-answer-overlap-report",
            "--write-sha256",
            "--out",
            str(out / "audit" / "split"),
        ],
        "validate_splits",
        out,
    )


def compatible_models(out: Path) -> list[str]:
    d = out / "audit" / "model_compatibility"
    names = []
    for name in ALL_PROBE_MODELS:
        cfg = d / f"{name}.config.json"
        smoke = d / f"{name}.smoke.json"
        if not cfg.exists():
            continue
        info = json.loads(cfg.read_text())
        if not info.get("enter_policy_table") and not info.get("is_causal_lm"):
            continue
        if smoke.exists():
            rec = json.loads(smoke.read_text())
            if rec.get("ok"):
                names.append(name)
            elif rec.get("need_tp"):
                log(f"{name} needs TP; defer, do not quantize", out)
        else:
            names.append(name)
    return names or list(PRIMARY_MODELS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all-prepolicy", help="audit|splits|compat|interventional|content|matrix|acquisition|all-prepolicy|all")
    p.add_argument("--config", default="configs/r7_policy_generalization.yaml")
    p.add_argument("--scheduler", default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--watchdog", action="store_true")
    p.add_argument("--out", default=str(R7_ROOT))
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES") or DEFAULT_GPUS)
    p.add_argument("--models", nargs="+", default=None)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip()]
    log(f"stage={args.stage} gpus={gpus}", out)

    if args.stage in {"audit", "all-prepolicy", "all"}:
        phase_audit(out)
        (out / "PHASE_AUDIT.DONE").write_text("ok\n")
    if args.stage in {"splits", "all-prepolicy", "all"}:
        phase_splits(out)
        (out / "PHASE_SPLITS.DONE").write_text("ok\n")

    need_gpu = args.stage in {"compat", "interventional", "content", "matrix", "acquisition", "all-prepolicy", "all"}
    if need_gpu:
        launch_retriever(out)

    if args.stage in {"compat", "all-prepolicy", "all"}:
        jobs = []
        compat_out = out / "audit" / "model_compatibility"
        compat_out.mkdir(parents=True, exist_ok=True)
        # static first
        run(
            [
                PY,
                str(ROOT / "scripts" / "r7" / "audit_model_compatibility.py"),
                "--offline-only",
                "--out",
                str(compat_out),
            ],
            "compat_static",
            out,
        )
        for name in ALL_PROBE_MODELS:
            cfgp = compat_out / f"{name}.config.json"
            if cfgp.exists() and not json.loads(cfgp.read_text()).get("is_causal_lm"):
                if name == "harness-1":
                    log("harness-1 causal check from static config", out)
            jobs.append(
                {
                    "id": f"compat:{name}",
                    "phase": "compat",
                    "model": name,
                    "out": str(compat_out),
                }
            )
        # filter non-causal
        keep = []
        for j in jobs:
            cfgp = compat_out / f"{j['model']}.config.json"
            if cfgp.exists():
                info = json.loads(cfgp.read_text())
                if not info.get("is_causal_lm") and not info.get("enter_policy_table"):
                    log(f"skip non-causal {j['model']}", out)
                    continue
            keep.append(j)
        schedule(keep, gpus, out, watchdog=args.watchdog)
        # rewrite combined md
        run(
            [
                PY,
                str(ROOT / "scripts" / "r7" / "audit_model_compatibility.py"),
                "--offline-only",
                "--out",
                str(compat_out),
            ],
            "compat_md",
            out,
        )
        (out / "PHASE_COMPAT.DONE").write_text("ok\n")

    models = args.models or compatible_models(out)
    primary = [m for m in PRIMARY_MODELS if m in models] or models[:4]
    log(f"models in play: {models}; primary={primary}", out)

    if args.stage in {"interventional", "all-prepolicy", "all"}:
        jobs = []
        for name in primary[:3] or models[:3]:
            jobs.append(
                {
                    "id": f"ival:{name}",
                    "phase": "interventional_validate",
                    "model": name,
                    "out": str(out / "assimilation" / "interventional_validate" / name),
                    "per_dataset": 40,
                }
            )
        schedule(jobs, gpus, out, watchdog=args.watchdog)
        (out / "PHASE_INTERVENTIONAL.DONE").write_text("ok\n")

    if args.stage in {"content", "all-prepolicy", "all"}:
        # content sensitivity on the anchor model first (12 batches), plus 1 batch on 2 other models
        jobs = []
        anchor = "Qwen2.5-3B-Instruct" if "Qwen2.5-3B-Instruct" in models else models[0]
        for bid in range(12):
            jobs.append(
                {
                    "id": f"cbatch:{anchor}:{bid}",
                    "phase": "content_batch",
                    "model": anchor,
                    "batch_id": bid,
                    "out": str(out / "assimilation" / "content_batches" / anchor / f"batch_{bid}"),
                    "per_dataset": 80,
                }
            )
        schedule(jobs, gpus, out, watchdog=args.watchdog)
        run([PY, str(ROOT / "scripts" / "r7" / "analyze_r7.py"), "--stage", "content", "--out", str(out)], "analyze_content", out)
        (out / "PHASE_CONTENT.DONE").write_text("ok\n")

    content_ok = False
    decision_pre = out / "assimilation" / "ASSIMILATOR_DECISION.json"
    if decision_pre.exists():
        content_ok = bool(json.loads(decision_pre.read_text()).get("content_sensitive_assimilator_valid"))

    if args.stage in {"matrix", "all-prepolicy", "all"}:
        conditions = ["base", "old_replacement_skill", "interventional_skill"]
        if content_ok:
            conditions.append("content_optimized_skill")
        jobs = []
        opt_anchor = out / "assimilation" / "content_batches" / (primary[0] if primary else "Qwen2.5-3B-Instruct")
        # pick best batch skills for content_optimized
        best_skills = ""
        best_gain = -1e9
        croot = out / "assimilation" / "content_batches"
        if croot.exists():
            for bdir in croot.glob("* /batch_*".replace(" ", "")):
                pass
            for bdir in croot.rglob("batch_*"):
                sp = bdir / "summary.json"
                op = bdir / "optimized.json"
                if sp.exists() and op.exists():
                    g = float(json.loads(sp.read_text()).get("future_gain") or 0)
                    if g > best_gain:
                        best_gain = g
                        best_skills = str(op)
        for name in primary:
            for cond in conditions:
                for seed in (0, 1, 2):
                    jobs.append(
                        {
                            "id": f"matrix:{name}:{cond}:s{seed}",
                            "phase": "matrix",
                            "model": name,
                            "condition": cond,
                            "seed": seed,
                            "out": str(out / "model_skill_matrix" / name / cond / f"seed_{seed}"),
                            "per_dataset": 80,
                            "optimized_skills": best_skills if cond == "content_optimized_skill" else "",
                        }
                    )
        schedule(jobs, gpus, out, watchdog=args.watchdog)
        run([PY, str(ROOT / "scripts" / "r7" / "analyze_r7.py"), "--stage", "matrix", "--out", str(out)], "analyze_matrix", out)
        (out / "PHASE_MATRIX.DONE").write_text("ok\n")

    if args.stage in {"acquisition", "all"}:
        # only if assimilator valid
        if not content_ok and decision_pre.exists():
            log("skip acquisition: content-sensitive assimilator failed gate", out)
        else:
            methods = ["passive", "ace_skill", "legacy_frontier", "coverage", "exact_utility"]
            acq_models = primary[:4]
            jobs = []
            for name in acq_models:
                for method in methods:
                    for seed in (0, 1, 2):
                        jobs.append(
                            {
                                "id": f"acq:{name}:{method}:s{seed}",
                                "phase": "acquisition",
                                "model": name,
                                "method": method,
                                "seed": seed,
                                "out": str(out / "acquisition" / name / method / f"seed_{seed}"),
                                "budget": 48,
                                "per_dataset": 80,
                            }
                        )
            schedule(jobs, gpus, out, watchdog=args.watchdog)
        run([PY, str(ROOT / "scripts" / "r7" / "analyze_r7.py"), "--stage", "final", "--out", str(out)], "analyze_final", out)
        (out / "PHASE_ACQUISITION.DONE").write_text("ok\n")

    if args.stage in {"all-prepolicy", "all"}:
        run([PY, str(ROOT / "scripts" / "r7" / "analyze_r7.py"), "--stage", "final", "--out", str(out)], "analyze_final", out)
    log("orchestrator stage complete", out)


if __name__ == "__main__":
    main()
