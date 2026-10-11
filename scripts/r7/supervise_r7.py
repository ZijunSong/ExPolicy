#!/usr/bin/env python3
"""Detached R7 supervisor: keep M0/baselines alive, resume stalls, use spare GPUs.

Safe to run under nohup/setsid. Never kills non-R7 processes.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from expolicy.closed_loop.constants import R7_ROOT, ROOT
from expolicy.closed_loop.io_utils import read_json, write_json

PY = str(ROOT / ".venv" / "bin" / "python")
CONDITION_SCRIPT = ROOT / "scripts" / "r7" / "run_r7_condition.py"
ANALYZE = ROOT / "scripts" / "r7" / "analyze_r7.py"
LOG_DIR = R7_ROOT / "logs"
PIDFILE = LOG_DIR / "supervise.pid"
STATEFILE = LOG_DIR / "supervise_state.json"

SEED_GPUS = {0: "5", 1: "6", 2: "7"}
FULL_CONDITIONS = [
    "expolicy",
    "passive_noop",
    "passive_episodic",
    "passive_count_skill",
    "random_joint",
    "no_feedback",
    "reward_shuffle",
    "assimilation_fixed",
]
STALL_SEC = 15 * 60
POLL_SEC = 60
MIN_FREE_MB = 40000  # enough for Qwen2.5-3B + headroom


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


def our_workers() -> dict[tuple[str, int], dict]:
    try:
        text = subprocess.check_output(["ps", "-ef"], text=True)
    except Exception:
        return {}
    found = {}
    for line in text.splitlines():
        if "run_r7_condition.py" not in line or "grep" in line:
            continue
        parts = line.split()
        if len(parts) < 8 or not parts[1].isdigit():
            continue
        pid = int(parts[1])
        cmd = " ".join(parts[7:])
        cond = seed = gpu = None
        toks = cmd.split()
        for i, tok in enumerate(toks):
            if tok == "--condition" and i + 1 < len(toks):
                cond = toks[i + 1]
            if tok == "--seed" and i + 1 < len(toks):
                seed = int(toks[i + 1])
            if tok == "--gpu" and i + 1 < len(toks):
                gpu = toks[i + 1]
        if cond is None or seed is None:
            continue
        found[(cond, seed)] = {"pid": pid, "gpu": gpu, "cmd": cmd}
    return found


def is_done(cond: str, seed: int) -> bool:
    return (R7_ROOT / cond / f"seed{seed}" / "DONE").exists()


def live(cond: str, seed: int) -> dict | None:
    path = R7_ROOT / cond / f"seed{seed}" / "STATUS_LIVE.json"
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def m0_posterior_ok(seed: int = 0) -> bool:
    sd = R7_ROOT / "expolicy" / f"seed{seed}"
    rounds = sorted(sd.glob("round_*/policy_update.json"))
    if not rounds:
        return False
    rec = read_json(rounds[0])
    acq = float((rec.get("acquisition") or {}).get("parameter_delta_norm") or 0)
    assim = float((rec.get("assimilation") or {}).get("parameter_delta_norm") or 0)
    return acq > 0 and assim > 0


def launch(cond: str, seed: int, gpu: str) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{cond}_seed{seed}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["EXPOLICY_RETRIEVER_URL"] = env.get("EXPOLICY_RETRIEVER_URL") or "http://127.0.0.1:18080"
    cmd = [
        PY,
        str(CONDITION_SCRIPT),
        "--condition",
        cond,
        "--seed",
        str(seed),
        "--gpu",
        str(gpu),
        "--resume",
    ]
    handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=handle,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
    )
    log(f"LAUNCH {cond} seed{seed} gpu={gpu} pid={proc.pid} log={log_path}")
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


def retriever_ok() -> bool:
    try:
        import httpx

        r = httpx.get("http://127.0.0.1:18080/health", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def pick_spare_gpu(busy: set[str]) -> str | None:
    rows = {r["index"]: r for r in gpu_table()}
    # Prefer GPU 4 (retriever card, ~120GB typically free; 3B actor is small).
    for idx in ("4", "5", "6", "7"):
        if idx in busy:
            continue
        row = rows.get(idx)
        if row and row["free"] >= MIN_FREE_MB:
            return idx
    # GPU 0-3 only when truly idle (others left).
    for idx in ("0", "1", "2", "3"):
        if idx in busy:
            continue
        row = rows.get(idx)
        if row and row["used"] < 2000:
            return idx
    return None


def next_condition(seed: int) -> str | None:
    for cond in FULL_CONDITIONS:
        if not is_done(cond, seed):
            return cond
    return None


def maybe_analyze(state: dict) -> None:
    if state.get("analyzed"):
        return
    if all(is_done(c, s) for c in FULL_CONDITIONS for s in SEED_GPUS) and is_done("all_skills", 0):
        log("ALL CONDITIONS DONE — running analyze_r7.py")
        subprocess.call([PY, str(ANALYZE)], cwd=str(ROOT))
        state["analyzed"] = True
        write_json(STATEFILE, state)


def already_supervised() -> bool:
    if not PIDFILE.exists():
        return False
    try:
        old = int(PIDFILE.read_text().strip())
    except Exception:
        return False
    if old != os.getpid() and os.path.exists(f"/proc/{old}"):
        cmdline = Path(f"/proc/{old}/cmdline").read_text(errors="ignore")
        if "supervise_r7.py" in cmdline:
            return True
    return False


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if already_supervised():
        print("supervise_r7 already running", PIDFILE.read_text().strip())
        return
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    state = read_json(STATEFILE) if STATEFILE.exists() else {"analyzed": False, "baselines_armed": False}
    log(f"supervisor start pid={os.getpid()}")
    while True:
        try:
            workers = our_workers()
            health = retriever_ok()
            gpus = gpu_table()
            free_s = " ".join(f"{r['index']}:{r['free']}MB/{r['util']}%" for r in gpus)
            log(f"workers={len(workers)} retriever={'OK' if health else 'DOWN'} free=[{free_s}]")

            # stall recovery for our jobs only
            now = time.time()
            for (cond, seed), w in list(workers.items()):
                st = live(cond, seed)
                mtime = 0.0
                path = R7_ROOT / cond / f"seed{seed}" / "STATUS_LIVE.json"
                if path.exists():
                    mtime = path.stat().st_mtime
                age = now - mtime if mtime else 1e9
                if age > STALL_SEC and not is_done(cond, seed):
                    log(f"STALL {cond} seed{seed} age={age:.0f}s")
                    kill_our(w["pid"], "stall")
                    gpu = w.get("gpu") or SEED_GPUS.get(seed, "5")
                    launch(cond, seed, gpu)

            # ensure M0 running
            for seed, gpu in SEED_GPUS.items():
                if is_done("expolicy", seed):
                    continue
                if ("expolicy", seed) not in workers:
                    launch("expolicy", seed, gpu)

            # arm baselines after first M0 seed proves posterior update
            if not state.get("baselines_armed") and any(is_done("expolicy", s) for s in SEED_GPUS):
                if m0_posterior_ok():
                    state["baselines_armed"] = True
                    write_json(STATEFILE, state)
                    log("BASELINES ARMED: M0 posterior delta > 0")
                else:
                    log("M0 DONE but posterior gate failed — not arming baselines")

            # per-seed sequential remainder on mapped GPU
            if state.get("baselines_armed"):
                for seed, gpu in SEED_GPUS.items():
                    if not is_done("expolicy", seed):
                        continue
                    nxt = next_condition(seed)
                    if nxt is None or nxt == "expolicy":
                        continue
                    if (nxt, seed) in workers:
                        continue
                    # only one R7 actor per mapped GPU
                    if any(w.get("gpu") == gpu for w in workers.values()):
                        continue
                    launch(nxt, seed, gpu)

            # all_skills upper bound: independent, can use spare GPU
            if not is_done("all_skills", 0) and ("all_skills", 0) not in workers:
                busy = {w.get("gpu") for w in workers.values() if w.get("gpu")}
                spare = pick_spare_gpu(busy)
                if spare is not None:
                    launch("all_skills", 0, spare)
                elif state.get("baselines_armed"):
                    # fall back onto seed0 GPU when it is idle
                    if not any(w.get("gpu") == SEED_GPUS[0] for w in workers.values()):
                        launch("all_skills", 0, SEED_GPUS[0])

            maybe_analyze(state)
            if state.get("analyzed"):
                log("supervisor complete")
                break
        except Exception as exc:
            log(f"supervisor error: {exc!r}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
