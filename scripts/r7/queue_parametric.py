#!/usr/bin/env python3
"""Queue Backend-3 parametric vs skill carrier jobs onto GPUs 4-7."""

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
GPUS = [g.strip() for g in os.environ.get("R7_PARAM_GPUS", "4,5,6,7").split(",") if g.strip()] or ["4", "5", "6", "7"]
STALL = 45 * 60
MAX_RESTARTS = 8
MODELS = [
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
LARGE_MODELS = {"Qwen3-30B-A3B-Thinking-2507"}
LARGE_FREE_MB = 70000
RESTARTS: dict[str, int] = {}


def log(msg: str) -> None:
    line = f"[pqueue {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "parametric_queue.log").open("a", encoding="utf-8").write(line + "\n")


def job_out(model: str) -> Path:
    return OUT / "assimilation" / "parametric" / model


def is_done(model: str) -> bool:
    d = job_out(model)
    return (d / "DONE").exists() and (d / "summary.json").exists()


def is_failed(model: str) -> bool:
    return (job_out(model) / "FAILED.json").exists()


def matrix_ready(model: str) -> bool:
    shared = OUT / "model_skill_matrix" / model / "_shared_base"
    return (shared / "rows.jsonl").exists() or (
        OUT / "model_skill_matrix" / model / "base" / "seed_0" / "DONE"
    ).exists()


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


def live_age(model: str) -> float | None:
    d = job_out(model)
    if not d.exists():
        return None
    lives = list(d.rglob("STATUS_LIVE.json"))
    if not lives:
        return None
    return time.time() - max(p.stat().st_mtime for p in lives)


def _worker_lines() -> list[str]:
    try:
        out = subprocess.check_output(["pgrep", "-af", "run_r7_worker.py"], text=True)
    except subprocess.CalledProcessError:
        return []
    return [ln for ln in out.splitlines() if "run_r7_worker.py" in ln]


def occupied_gpus() -> set[str]:
    busy = set()
    for line in _worker_lines():
        parts = line.split()
        if "--gpu" in parts:
            busy.add(parts[parts.index("--gpu") + 1])
    return busy


def matrix_busy_gpus() -> set[str]:
    busy = set()
    for line in _worker_lines():
        if "--phase" not in line:
            continue
        parts = line.split()
        phase = parts[parts.index("--phase") + 1] if "--phase" in parts else ""
        if phase == "matrix" and "--gpu" in parts:
            busy.add(parts[parts.index("--gpu") + 1])
    return busy


def running_jobs() -> dict[str, str]:
    """gpu -> 'pid:model' for live parametric workers."""
    mapping: dict[str, str] = {}
    for line in _worker_lines():
        if "queue_parametric" in line:
            continue
        parts = line.split()
        phase = parts[parts.index("--phase") + 1] if "--phase" in parts else ""
        if phase != "parametric":
            continue
        pid = parts[0]
        gpu = parts[parts.index("--gpu") + 1] if "--gpu" in parts else None
        model = parts[parts.index("--model") + 1] if "--model" in parts else None
        if gpu and model:
            mapping[gpu] = f"{pid}:{model}"
    return mapping


def launch(model: str, gpu: str) -> None:
    d = job_out(model)
    d.mkdir(parents=True, exist_ok=True)
    log_path = LOG / f"parametric_gpu{gpu}_{model}.log"
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} =====\n")
    handle.flush()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["EXPOLICY_RETRIEVER_URL"] = "http://127.0.0.1:18080"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    cmd = [
        PY,
        str(ROOT / "scripts" / "r7" / "run_r7_worker.py"),
        "--phase",
        "parametric",
        "--model",
        model,
        "--gpu",
        gpu,
        "--per-dataset",
        "80",
        "--out",
        str(d),
    ]
    subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log(f"launch parametric {model} on GPU {gpu}")


def gpu_free_mb() -> dict[str, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return {}
    rec: dict[str, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            rec[parts[0]] = int(float(parts[1]))
    return rec


def next_model(busy_models: set[str], gpu: str | None = None) -> str | None:
    free = gpu_free_mb() if gpu is not None else {}
    for model in MODELS:
        if is_done(model) or is_failed(model) or model in busy_models:
            continue
        if RESTARTS.get(model, 0) >= MAX_RESTARTS:
            d = job_out(model)
            d.mkdir(parents=True, exist_ok=True)
            (d / "FAILED.json").write_text(
                json.dumps({"reason": "max_restarts", "restarts": RESTARTS[model]}),
                encoding="utf-8",
            )
            log(f"GIVE UP parametric {model}")
            continue
        if not matrix_ready(model):
            continue
        if model in LARGE_MODELS and gpu is not None and free.get(gpu, 0) < LARGE_FREE_MB:
            continue
        return model
    return None


def all_done() -> bool:
    return all(is_done(m) or is_failed(m) or not matrix_ready(m) for m in MODELS) and all(
        is_done(m) or is_failed(m) for m in MODELS if matrix_ready(m)
    )


def main() -> None:
    log("parametric queue start")
    while True:
        busy = running_jobs()
        busy_models = {v.split(":", 1)[1] for v in busy.values() if ":" in v}
        if all_done():
            log("all parametric jobs DONE or FAILED")
            return
        for gpu, v in list(busy.items()):
            pid_s, _, model = v.partition(":")
            try:
                pid = int(pid_s)
            except ValueError:
                continue
            age = live_age(model)
            hung = age is not None and age > STALL
            dead = not alive(pid)
            if hung or (dead and not is_done(model)):
                RESTARTS[model] = RESTARTS.get(model, 0) + 1
                log(f"STALL parametric {model} gpu={gpu} pid={pid} age={age} restarts={RESTARTS[model]}")
                if alive(pid):
                    kill_tree(pid)
                busy.pop(gpu, None)
                busy_models.discard(model)
        taken = occupied_gpus() | matrix_busy_gpus() | set(busy)
        for gpu in GPUS:
            if gpu in taken:
                continue
            model = next_model(busy_models, gpu=gpu)
            if model is None:
                continue
            launch(model, gpu)
            busy_models.add(model)
            busy[gpu] = f"new:{model}"
            taken.add(gpu)
        time.sleep(45)


if __name__ == "__main__":
    main()
