#!/usr/bin/env python3
"""Watch R7 GPU 4-7 jobs: recover stalls, then chain matrix → parametric → analyze.

Does not delete completed artifacts. Restarts only the stalled (model, condition, seed).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
OUT = ROOT / "outputs" / "r7_policy_generalization"
LOG = OUT / "logs"
PY = str(ROOT / ".venv" / "bin" / "python")
GPUS = "4,5,6,7"
STALL = 45 * 60
POLL = 60
RETRIEVER_URL = "http://127.0.0.1:18080"


def log(msg: str) -> None:
    line = f"[r7watch {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "watchdog_continue.log").open("a", encoding="utf-8").write(line + "\n")


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


def pgrep_af(pattern: str) -> list[str]:
    try:
        out = subprocess.check_output(["pgrep", "-af", pattern], text=True)
    except subprocess.CalledProcessError:
        return []
    return [
        ln
        for ln in out.splitlines()
        if pattern in ln and "pgrep" not in ln and "watchdog_continue" not in ln
    ]


def retriever_ok() -> bool:
    try:
        import httpx

        r = httpx.get(f"{RETRIEVER_URL}/health", timeout=5.0)
        return r.status_code == 200 and bool(r.json().get("ok"))
    except Exception:
        return False


def launch_retriever() -> None:
    if retriever_ok():
        return
    meta = json.loads((ROOT / "data" / "wiki18" / "BACKEND_PATHS.json").read_text())
    logdir = LOG
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
        "18080",
        "--device",
        "cpu",
    ]
    handle = (logdir / "retriever.log").open("a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    (OUT / "retriever.pid").write_text(str(proc.pid), encoding="utf-8")
    t0 = time.time()
    while time.time() - t0 < 600:
        if retriever_ok():
            log(f"retriever up pid={proc.pid}")
            return
        if proc.poll() is not None:
            raise SystemExit("retriever exited")
        time.sleep(5)
    raise SystemExit("retriever did not become healthy")


def launch_detached(script: str, log_name: str) -> int:
    log_path = LOG / log_name
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n===== watchdog launch {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    handle.flush()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = GPUS
    env["EXPOLICY_RETRIEVER_URL"] = RETRIEVER_URL
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    proc = subprocess.Popen(
        [PY, str(ROOT / "scripts" / "r7" / script)],
        cwd=str(ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log(f"launch {script} pid={proc.pid}")
    return proc.pid


def queue_running(name: str) -> bool:
    return bool(pgrep_af(name))


def worker_status() -> list[dict]:
    rows = []
    try:
        out = subprocess.check_output(["pgrep", "-af", "run_r7_worker.py"], text=True)
    except subprocess.CalledProcessError:
        return rows
    for line in out.splitlines():
        if "run_r7_worker.py" not in line:
            continue
        parts = line.split()
        rec = {"pid": int(parts[0]), "line": line}
        for key in ("--phase", "--model", "--gpu", "--condition", "--seed"):
            if key in parts:
                rec[key[2:]] = parts[parts.index(key) + 1]
        rows.append(rec)
    return rows


def newest_live(path: Path) -> float | None:
    if not path.exists():
        return None
    lives = list(path.rglob("STATUS_LIVE.json"))
    if not lives:
        return None
    return time.time() - max(p.stat().st_mtime for p in lives)


MATRIX_MODELS = [
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen3-4B-Instruct-2507",
    "Phi-4-reasoning",
    "gpt-oss-20b",
    "Qwen3-4B-Thinking-2507",
    "Qwen3-1.7B",
    "gemma-4-E4B-it",
    "harness-1",
    "Qwen3-30B-A3B-Thinking-2507",
]
MATRIX_CONDS = ["base", "old_replacement_skill", "interventional_skill"]
PARAM_MODELS = [
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen3-4B-Instruct-2507",
    "Qwen3-4B-Thinking-2507",
    "Phi-4-reasoning",
    "gpt-oss-20b",
    "Qwen3-1.7B",
    "gemma-4-E4B-it",
    "harness-1",
    "Qwen3-30B-A3B-Thinking-2507",
]


def _seed_done_or_failed(model: str, cond: str) -> bool:
    d = OUT / "model_skill_matrix" / model / cond / "seed_0"
    return ((d / "DONE").exists() and (d / "summary.json").exists()) or (d / "FAILED.json").exists()


def matrix_complete() -> bool:
    if (OUT / "PHASE_MATRIX.DONE").exists():
        return True
    return all(_seed_done_or_failed(m, c) for m in MATRIX_MODELS for c in MATRIX_CONDS)


def parametric_complete() -> bool:
    if (OUT / "PHASE_PARAMETRIC.DONE").exists():
        return True
    ready = []
    for model in PARAM_MODELS:
        shared = OUT / "model_skill_matrix" / model / "_shared_base"
        base_done = (OUT / "model_skill_matrix" / model / "base" / "seed_0" / "DONE").exists()
        if shared.joinpath("rows.jsonl").exists() or base_done:
            ready.append(model)
    if not ready:
        return False
    ok = True
    for model in ready:
        d = OUT / "assimilation" / "parametric" / model
        if not (((d / "DONE").exists() and (d / "summary.json").exists()) or (d / "FAILED.json").exists()):
            ok = False
    return ok


def snap_gpu() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "-i", GPUS, "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
            text=True,
        )
    except Exception as exc:
        return str(exc)


def progress_line() -> str:
    bits = []
    root = OUT / "model_skill_matrix"
    for live in sorted(root.glob("*/*/seed_0/base/STATUS_LIVE.json")):
        try:
            rec = json.loads(live.read_text())
        except Exception:
            continue
        age = time.time() - live.stat().st_mtime
        bits.append(f"{live.parts[-5]}/{live.parts[-4]} i={rec.get('i')}/{rec.get('n')} age={age:.0f}s")
    workers = worker_status()
    return f"workers={len(workers)} " + " | ".join(bits[:6])


def recover_stalled_workers() -> None:
    for rec in worker_status():
        pid = rec["pid"]
        model = rec.get("model")
        phase = rec.get("phase")
        cond = rec.get("condition") or "base"
        if phase == "matrix" and model:
            path = OUT / "model_skill_matrix" / model / cond / "seed_0"
        elif phase == "parametric" and model:
            path = OUT / "assimilation" / "parametric" / model
        else:
            continue
        age = newest_live(path)
        if age is not None and age > STALL and alive(pid):
            log(f"KILL STALL pid={pid} phase={phase} model={model} cond={cond} age={age:.0f}s")
            kill_tree(pid)


def run_analyze(stage: str) -> None:
    log(f"analyze {stage}")
    rc = subprocess.call(
        [PY, str(ROOT / "scripts" / "r7" / "analyze_r7.py"), "--stage", stage, "--out", str(OUT)],
        cwd=str(ROOT),
    )
    log(f"analyze {stage} rc={rc}")


def write_pathbench_audit() -> None:
    dest = OUT / "audit" / "PATHBENCH_AUDIT.md"
    dest.write_text(
        "# PATH-Bench readiness\n\n"
        "Paper: PATH-Bench (arXiv:2608.01149), path-dependent lifelong agents "
        "(BigCodeBench / WildToolBench tracks, SEU baseline).\n\n"
        "**Official code/data release was not found** as of 2026-08-16. "
        "GitHub repositories named PathBench refer to unrelated pathology / robotics projects.\n\n"
        "Protocol: do not mock PATH-Bench. `pathbench_ready = false`.\n"
        "R7-Lifelong stays blocked until the official release is pinned.\n",
        encoding="utf-8",
    )


def ensure_queue(script: str, log_name: str, pid_name: str) -> None:
    if queue_running(script):
        return
    pid = launch_detached(script, log_name)
    (LOG / pid_name).write_text(str(pid), encoding="utf-8")


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    log("watchdog_continue start")
    launch_retriever()

    if not (OUT / "PHASE_MATRIX.DONE").exists():
        ensure_queue("queue_matrix.py", "matrix_queue.out", "matrix_queue.pid")
        while not matrix_complete():
            if not retriever_ok():
                log("retriever down; relaunch")
                launch_retriever()
            if not queue_running("queue_matrix.py"):
                log("matrix queue died; restart")
                ensure_queue("queue_matrix.py", "matrix_queue.out", "matrix_queue.pid")
            matrix_gpus = {str(rec.get("gpu")) for rec in worker_status() if rec.get("phase") == "matrix"}
            if len(matrix_gpus) < 4 and not (OUT / "PHASE_PARAMETRIC.DONE").exists():
                if not queue_running("queue_parametric.py"):
                    log(f"overlap parametric on idle GPUs; matrix using {sorted(matrix_gpus)}")
                    ensure_queue("queue_parametric.py", "parametric_queue.out", "parametric_queue.pid")
            recover_stalled_workers()
            (LOG / "nvidia.snap").write_text(snap_gpu(), encoding="utf-8")
            log("matrix " + progress_line())
            time.sleep(POLL)
        run_analyze("matrix")
        (OUT / "PHASE_MATRIX.DONE").write_text("ok\n", encoding="utf-8")
        log("PHASE_MATRIX.DONE")

    if not (OUT / "PHASE_PARAMETRIC.DONE").exists():
        ensure_queue("queue_parametric.py", "parametric_queue.out", "parametric_queue.pid")
        while not parametric_complete():
            if not retriever_ok():
                log("retriever down; relaunch")
                launch_retriever()
            if not queue_running("queue_parametric.py"):
                log("parametric queue died; restart")
                ensure_queue("queue_parametric.py", "parametric_queue.out", "parametric_queue.pid")
            recover_stalled_workers()
            (LOG / "nvidia.snap").write_text(snap_gpu(), encoding="utf-8")
            log("parametric " + progress_line())
            time.sleep(POLL)
        run_analyze("final")
        (OUT / "PHASE_PARAMETRIC.DONE").write_text("ok\n", encoding="utf-8")
        log("PHASE_PARAMETRIC.DONE")

    write_pathbench_audit()
    run_analyze("final")
    log("watchdog_continue complete")


if __name__ == "__main__":
    main()
