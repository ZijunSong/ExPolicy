#!/usr/bin/env python3
"""Queue remaining content-sensitivity batches onto free GPUs 4,6,7. GPU5 reserved for thinking ival until DONE."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
OUT = ROOT / "outputs" / "r7_policy_generalization"
LOG = OUT / "logs"
PY = str(ROOT / ".venv" / "bin" / "python")
ANCHOR = "Qwen2.5-3B-Instruct"
GPUS_CONTENT = ["4", "6", "7"]
GPU_IVAL = "5"
N_BATCH = 12
STALL = 20 * 60
IVAL_DONE = OUT / "assimilation" / "interventional_validate" / "Qwen3-4B-Thinking-2507" / "DONE"


def log(msg: str) -> None:
    line = f"[cqueue {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "content_queue.log").open("a", encoding="utf-8").write(line + "\n")


def batch_dir(bid: int) -> Path:
    return OUT / "assimilation" / "content_batches" / ANCHOR / f"batch_{bid}"


def is_done(bid: int) -> bool:
    return (batch_dir(bid) / "DONE").exists()


def worker_pids() -> dict[str, str]:
    """gpu -> cmdline pid if content_batch running."""
    try:
        out = subprocess.check_output(["pgrep", "-af", "run_r7_worker.py"], text=True)
    except subprocess.CalledProcessError:
        return {}
    mapping = {}
    for line in out.splitlines():
        if "content_batch" not in line:
            continue
        pid = line.split()[0]
        gpu = None
        bid = None
        parts = line.split()
        if "--gpu" in parts:
            gpu = parts[parts.index("--gpu") + 1]
        if "--batch-id" in parts:
            bid = parts[parts.index("--batch-id") + 1]
        if gpu:
            mapping[gpu] = f"{pid}:b{bid}"
    return mapping


def live_age(bid: int) -> float | None:
    lives = list(batch_dir(bid).rglob("STATUS_LIVE.json"))
    if not lives:
        return None
    newest = max(p.stat().st_mtime for p in lives)
    return time.time() - newest


def ival_running() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-af", "run_r7_worker.py"], text=True)
    except subprocess.CalledProcessError:
        return False
    return any("interventional_validate" in line and "Qwen3-4B-Thinking-2507" in line for line in out.splitlines())


def free_gpus() -> list[str]:
    gpus = list(GPUS_CONTENT)
    if IVAL_DONE.exists() or not ival_running():
        gpus.append(GPU_IVAL)
    return gpus


def launch(bid: int, gpu: str) -> None:
    d = batch_dir(bid)
    d.mkdir(parents=True, exist_ok=True)
    src = OUT / "assimilation" / "interventional_validate" / ANCHOR / "base"
    vb = d / "val_base"
    if src.exists() and not (vb / "rows.jsonl").exists():
        vb.mkdir(parents=True, exist_ok=True)
        for name in ("rows.jsonl", "summary.json", "DONE"):
            sp = src / name
            if sp.exists():
                (vb / name).write_bytes(sp.read_bytes())
        if not (vb / "DONE").exists():
            (vb / "DONE").write_text("ok\n")
    log_path = LOG / f"cbatch_gpu{gpu}_b{bid}.log"
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n===== queue launch {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
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
        "content_batch",
        "--model",
        ANCHOR,
        "--gpu",
        gpu,
        "--batch-id",
        str(bid),
        "--per-dataset",
        "80",
        "--out",
        str(d),
    ]
    subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    log(f"launch batch {bid} on GPU {gpu}")


def main() -> None:
    log("content queue start")
    while True:
        if all(is_done(b) for b in range(N_BATCH)):
            log("all 12 content batches DONE")
            return
        busy = worker_pids()
        pending = [b for b in range(N_BATCH) if not is_done(b) and not any(f":b{b}" in v for v in busy.values())]
        for gpu in free_gpus():
            if gpu in busy:
                continue
            if not pending:
                break
            bid = pending.pop(0)
            launch(bid, gpu)
            busy[gpu] = f"new:b{bid}"
        # stall report only (do not auto-kill unless live is very stale AND pid missing)
        for b in range(N_BATCH):
            if is_done(b):
                continue
            age = live_age(b)
            running = any(f":b{b}" in v for v in worker_pids().values())
            if age is not None and age > STALL and not running:
                log(f"stale batch {b} age={age:.0f}s not running — will relaunch on free GPU")
        time.sleep(60)


if __name__ == "__main__":
    main()
