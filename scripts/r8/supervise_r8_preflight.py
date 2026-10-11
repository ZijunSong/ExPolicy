#!/usr/bin/env python3
"""Launch and babysit R8-A GPU workers on GPUs 4/5/6/7. Never kills foreign PIDs."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from expolicy.closed_loop.constants import ROOT
from expolicy.closed_loop.io_utils import read_json, write_json

PY = str(ROOT / ".venv" / "bin" / "python")
AUDIT = ROOT / "scripts" / "r8" / "audit_r7_structural.py"
OUT = ROOT / "outputs" / "r8_preflight"
LOG_DIR = OUT / "logs"
PIDFILE = LOG_DIR / "supervise.pid"
STATEFILE = LOG_DIR / "supervise_state.json"
GPUS = ("4", "5", "6", "7")
STALL_SEC = 20 * 60
POLL_SEC = 45
MIN_FREE_MB = 20000
WORKER_MARK = "audit_r7_structural.py"


def log(msg: str) -> None:
    line = time.strftime("%F %T") + " " + msg
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "supervise.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def gpu_table() -> list[dict]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        idx, used, total, util = [x.strip() for x in line.split(",")]
        rows.append({"index": idx, "used": int(used), "total": int(total), "free": int(total) - int(used), "util": int(util)})
    return rows


def our_workers() -> dict[str, dict]:
    try:
        text = subprocess.check_output(["ps", "-ef"], text=True)
    except Exception:
        return {}
    found: dict[str, dict] = {}
    for line in text.splitlines():
        if WORKER_MARK not in line or "--mode" not in line or "grep" in line:
            continue
        if "supervise_r8_preflight.py" in line:
            continue
        parts = line.split()
        if len(parts) < 8 or not parts[1].isdigit():
            continue
        pid = int(parts[1])
        cmd = " ".join(parts[7:])
        gpu = None
        toks = cmd.split()
        for i, tok in enumerate(toks):
            if tok == "--gpu" and i + 1 < len(toks):
                gpu = toks[i + 1]
        if gpu is None:
            continue
        found[gpu] = {"pid": pid, "cmd": cmd, "gpu": gpu}
    return found


def retriever_ok() -> bool:
    try:
        import httpx

        r = httpx.get("http://127.0.0.1:18080/health", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def newest_live_age() -> float:
    lives = list(OUT.glob("jobs/**/STATUS_LIVE.json"))
    if not lives:
        return 1e9
    return time.time() - max(p.stat().st_mtime for p in lives)


def launch(gpu: str, smoke: bool = False) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"worker_gpu{gpu}.out"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["EXPOLICY_RETRIEVER_URL"] = env.get("EXPOLICY_RETRIEVER_URL") or "http://127.0.0.1:18080"
    cmd = [PY, str(AUDIT), "--mode", "worker", "--gpu", str(gpu)]
    if smoke:
        cmd.append("--smoke")
    handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env, start_new_session=True)
    log(f"LAUNCH worker gpu={gpu} pid={proc.pid} log={log_path}")
    return proc.pid


def kill_our(pid: int, why: str) -> None:
    log(f"KILL our worker pid={pid} reason={why}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(8)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def already_supervised() -> bool:
    if not PIDFILE.exists():
        return False
    try:
        old = int(PIDFILE.read_text().strip())
    except Exception:
        return False
    if old != os.getpid() and os.path.exists(f"/proc/{old}"):
        cmdline = Path(f"/proc/{old}/cmdline").read_text(errors="ignore")
        if "supervise_r8_preflight.py" in cmdline:
            return True
    return False


def incomplete_jobs() -> list[dict]:
    path = OUT / "JOBS.json"
    if not path.exists():
        return []
    missing = []
    for job in read_json(path):
        dest = OUT / "jobs" / job["kind"] / job["job_id"] / "DONE"
        if not dest.exists():
            missing.append(job)
    return missing


def jobs_complete() -> bool:
    return not incomplete_jobs()


def main() -> None:
    smoke = "--smoke" in sys.argv
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if already_supervised() and "--force" not in sys.argv:
        print("supervise_r8_preflight already running", PIDFILE.read_text().strip())
        return
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    log(f"supervisor start pid={os.getpid()} smoke={smoke}")
    subprocess.check_call([PY, str(AUDIT), "--mode", "emit"] + (["--smoke"] if smoke else []), cwd=str(ROOT))
    state = {"summarized": False}
    write_json(STATEFILE, state)
    while True:
        try:
            workers = our_workers()
            health = retriever_ok()
            gpus = gpu_table()
            free_s = " ".join(f"{r['index']}:{r['free']}MB/{r['util']}%" for r in gpus)
            age = newest_live_age()
            log(f"workers={len(workers)} retriever={'OK' if health else 'DOWN'} live_age={age:.0f}s free=[{free_s}]")
            if not health:
                log("WARNING retriever down — not killing foreign processes")

            for gpu, w in list(workers.items()):
                live_path = LOG_DIR / f"worker_gpu{gpu}.live.json"
                live = {}
                if live_path.exists():
                    try:
                        live = read_json(live_path)
                    except Exception:
                        live = {}
                live_pid = live.get("pid")
                phase = str(live.get("phase") or "")
                if live_pid and int(live_pid) != int(w["pid"]):
                    continue
                if phase in {"done", "idle", "waiting"}:
                    continue
                w_age = age
                if live_path.exists():
                    w_age = time.time() - live_path.stat().st_mtime
                if w_age > STALL_SEC and not jobs_complete():
                    log(f"STALL gpu={gpu} age={w_age:.0f}s")
                    kill_our(w["pid"], "stall")
                    launch(gpu, smoke=smoke)

            gpu_info = {r["index"]: r for r in gpus}
            n_left = len(incomplete_jobs())
            for gpu in GPUS:
                if gpu in workers:
                    continue
                if jobs_complete() or len(workers) >= max(1, n_left):
                    break
                row = gpu_info.get(gpu)
                if row and row["free"] < MIN_FREE_MB:
                    log(f"skip gpu={gpu} free={row['free']}MB < {MIN_FREE_MB}")
                    continue
                pid = launch(gpu, smoke=smoke)
                workers[gpu] = {"pid": pid, "cmd": "launched", "gpu": gpu}

            if jobs_complete() and not state.get("summarized"):
                log("ALL JOBS DONE — summarize")
                subprocess.check_call([PY, str(AUDIT), "--mode", "summarize"], cwd=str(ROOT))
                state["summarized"] = True
                write_json(STATEFILE, state)
                log("supervisor complete")
                break
        except Exception as exc:
            log(f"supervisor error: {exc!r}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
