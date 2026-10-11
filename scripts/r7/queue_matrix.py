#!/usr/bin/env python3
"""Queue Phase-5 static headroom matrix onto GPUs 4-7. Nested heartbeat stall check."""

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
GPUS = ["4", "5", "6", "7"]
STALL = 45 * 60
MAX_RESTARTS = 8
LARGE_FREE_MB = 70000
PRIMARY = [
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen3-4B-Instruct-2507",
    "Phi-4-reasoning",
    "gpt-oss-20b",
    "Qwen3-4B-Thinking-2507",
]
FILL = [
    "Qwen3-1.7B",
    "gemma-4-E4B-it",
    "harness-1",
    "Qwen3-30B-A3B-Thinking-2507",
]
ALL_MODELS = PRIMARY + FILL
CONDS = ["base", "old_replacement_skill", "interventional_skill"]
LARGE_MODELS = {"Qwen3-30B-A3B-Thinking-2507"}
RESTARTS: dict[str, int] = {}


def log(msg: str) -> None:
    line = f"[mqueue {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "matrix_queue.log").open("a", encoding="utf-8").write(line + "\n")


def job_out(model: str, cond: str, seed: int) -> Path:
    return OUT / "model_skill_matrix" / model / cond / f"seed_{seed}"


def shared_base(model: str) -> Path:
    return OUT / "model_skill_matrix" / model / "_shared_base"


def is_done(model: str, cond: str, seed: int) -> bool:
    d = job_out(model, cond, seed)
    return (d / "DONE").exists() and (d / "summary.json").exists()


def is_failed(model: str, cond: str, seed: int) -> bool:
    return (job_out(model, cond, seed) / "FAILED.json").exists()


def job_key(model: str, cond: str, seed: int) -> str:
    return f"{model}:{cond}:{seed}"


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


def gpu_free_mb(gpu: str) -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().split()[0])
    except Exception:
        return 0


def live_age(model: str, cond: str, seed: int) -> float | None:
    d = job_out(model, cond, seed)
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


def running_jobs() -> dict[str, str]:
    """gpu -> 'pid:model:cond:seed' for live matrix workers."""
    mapping = {}
    for line in _worker_lines():
        parts = line.split()
        phase = parts[parts.index("--phase") + 1] if "--phase" in parts else ""
        if phase != "matrix":
            continue
        pid = parts[0]
        gpu = parts[parts.index("--gpu") + 1] if "--gpu" in parts else None
        model = parts[parts.index("--model") + 1] if "--model" in parts else None
        cond = parts[parts.index("--condition") + 1] if "--condition" in parts else None
        seed = parts[parts.index("--seed") + 1] if "--seed" in parts else "0"
        if gpu and model and cond is not None:
            mapping[gpu] = f"{pid}:{model}:{cond}:{seed}"
    return mapping


def copy_seed0_to(model: str, cond: str, seed: int) -> None:
    src = job_out(model, cond, 0)
    dst = job_out(model, cond, seed)
    if is_done(model, cond, seed):
        return
    if not is_done(model, cond, 0):
        return
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "DONE"):
        (dst / name).write_bytes((src / name).read_bytes())
    note = {
        "copied_from": str(src),
        "reason": "temperature=0 greedy decoding; seed_1/2 are deterministic copies of seed_0",
        "seed": seed,
    }
    (dst / "SEED_COPY.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    log(f"copied {model}/{cond}/seed_0 -> seed_{seed}")


def launch(model: str, cond: str, seed: int, gpu: str) -> None:
    d = job_out(model, cond, seed)
    d.mkdir(parents=True, exist_ok=True)
    log_path = LOG / f"matrix_gpu{gpu}_{model}_{cond}_s{seed}.log"
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
        "matrix",
        "--model",
        model,
        "--gpu",
        gpu,
        "--seed",
        str(seed),
        "--condition",
        cond,
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
    log(f"launch {model}/{cond}/seed_{seed} on GPU {gpu}")


def ready(model: str, cond: str, gpu: str | None = None) -> bool:
    if is_failed(model, cond, 0):
        return False
    if model in LARGE_MODELS and gpu is not None and gpu_free_mb(gpu) < LARGE_FREE_MB:
        return False
    if cond == "base":
        return True
    return (shared_base(model) / "rows.jsonl").exists() or is_done(model, "base", 0)


def next_job(busy_keys: set[str], gpu: str | None = None) -> tuple[str, str, int] | None:
    # Prefer finishing a model's skill conds after its base exists.
    ordered = []
    for model in ALL_MODELS:
        has_base = (shared_base(model) / "rows.jsonl").exists() or is_done(model, "base", 0)
        if not has_base:
            ordered.append((model, "base", 0))
        for cond in CONDS:
            if cond == "base" and has_base and is_done(model, "base", 0):
                continue
            ordered.append((model, cond, 0))
    seen = set()
    for model, cond, seed in ordered:
        key = f"{model}:{cond}:{seed}"
        if key in seen:
            continue
        seen.add(key)
        if is_done(model, cond, seed) or is_failed(model, cond, seed):
            continue
        if key in busy_keys:
            continue
        if RESTARTS.get(key, 0) >= MAX_RESTARTS:
            d = job_out(model, cond, seed)
            d.mkdir(parents=True, exist_ok=True)
            (d / "FAILED.json").write_text(
                json.dumps({"reason": "max_restarts", "restarts": RESTARTS[key]}),
                encoding="utf-8",
            )
            log(f"GIVE UP {key}")
            continue
        if not ready(model, cond, gpu):
            continue
        return model, cond, seed
    return None


def all_seed0_done() -> bool:
    return all(is_done(m, c, 0) or is_failed(m, c, 0) for m in ALL_MODELS for c in CONDS)


def main() -> None:
    log("matrix queue start")
    while True:
        busy = running_jobs()
        busy_keys = set()
        for v in busy.values():
            # pid:model:cond:seed
            bits = v.split(":", 3)
            if len(bits) >= 4:
                busy_keys.add(f"{bits[1]}:{bits[2]}:{bits[3]}")
        if all_seed0_done():
            for m in ALL_MODELS:
                for c in CONDS:
                    if is_done(m, c, 0):
                        copy_seed0_to(m, c, 1)
                        copy_seed0_to(m, c, 2)
            log("all matrix seed_0 jobs DONE or FAILED; seeds 1/2 copied where possible")
            return
        # stall: kill hung-but-alive workers; queue will relaunch
        for gpu, v in list(busy.items()):
            bits = v.split(":", 3)
            if len(bits) < 4:
                continue
            pid_s, model, cond, seed = bits
            try:
                pid = int(pid_s)
                seed_i = int(seed)
            except ValueError:
                continue
            age = live_age(model, cond, seed_i)
            hung = age is not None and age > STALL
            dead = not alive(pid)
            if hung or (dead and not is_done(model, cond, seed_i)):
                key = job_key(model, cond, seed_i)
                RESTARTS[key] = RESTARTS.get(key, 0) + 1
                log(f"STALL {key} gpu={gpu} pid={pid} age={age} restarts={RESTARTS[key]} dead={dead}")
                if alive(pid):
                    kill_tree(pid)
                busy.pop(gpu, None)
                busy_keys.discard(key)
        taken = occupied_gpus() | set(busy)
        for gpu in GPUS:
            if gpu in taken:
                continue
            job = next_job(busy_keys, gpu)
            if job is None:
                continue
            model, cond, seed = job
            launch(model, cond, seed, gpu)
            busy_keys.add(f"{model}:{cond}:{seed}")
            busy[gpu] = f"new:{model}:{cond}:{seed}"
            taken.add(gpu)
        time.sleep(45)


if __name__ == "__main__":
    main()
