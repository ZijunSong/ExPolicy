#!/usr/bin/env python3
"""Babysit R8-B jobs on GPUs 4-7. Never kills foreign PIDs."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
OUT = ROOT / "outputs" / "r8_backend_validation"
LOG = OUT / "logs"
PY = str(ROOT / ".venv" / "bin" / "python")
GPUS = ("4", "5", "6", "7")
STALL_SEC = 30 * 60
POLL_SEC = 45
MARK = "run_backend_validation.py"


def log(msg: str) -> None:
    line = time.strftime("%F %T") + " " + msg
    print(line, flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG / "supervise_backend.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def our_jobs() -> dict[str, dict]:
    try:
        text = subprocess.check_output(["ps", "-ef"], text=True)
    except Exception:
        return {}
    found = {}
    for line in text.splitlines():
        if MARK not in line or "supervise_r8_backend" in line:
            continue
        parts = line.split()
        if len(parts) < 8 or not parts[1].isdigit():
            continue
        pid = int(parts[1])
        cmd = " ".join(parts[7:])
        job = ""
        toks = cmd.split()
        for i, tok in enumerate(toks):
            if tok == "--job" and i + 1 < len(toks):
                job = toks[i + 1]
        if job:
            found[job] = {"pid": pid, "cmd": cmd}
    return found


def pending_jobs(smoke: bool, seeds: list[int]) -> list[str]:
    jobs = [] if smoke else ["foundation"]
    prefix = "_smoke" if smoke else ""
    for seed in seeds:
        jobs.append(f"none_s{seed}{prefix}")
        jobs.append(f"memrl_s{seed}{prefix}")
        jobs.append(f"skillopt_s{seed}{prefix}")
    out = []
    for name in jobs:
        dest = OUT / "jobs" / ( "foundation_no_learning" if name == "foundation" else name )
        if (dest / "DONE").exists() or (dest / "FAIL").exists():
            continue
        out.append(name)
    return out


def launch(job: str, python_bin: str, extra: list[str]) -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    logf = LOG / f"job_{job}.out"
    cmd = [
        python_bin,
        str(ROOT / "scripts" / "r8" / "run_backend_validation.py"),
        "--job",
        job,
        "--python",
        python_bin,
        *extra,
    ]
    with logf.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT))
    log(f"LAUNCH {job} pid={proc.pid}")
    return proc.pid


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--python", default=PY)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max_parallel", type=int, default=3)
    p.add_argument("--seeds", default="0,1,2")
    args = p.parse_args()
    extra = ["--smoke"] if args.smoke else []
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.smoke:
        seeds = seeds[:1]
    launched: dict[str, dict] = {}
    while True:
        todo = pending_jobs(args.smoke, seeds)
        live = our_jobs()
        if not todo and not live and not launched:
            log("all backend jobs complete")
            subprocess.check_call(
                [
                    args.python,
                    str(ROOT / "scripts" / "r8" / "analyze_backend_validation.py"),
                    "--input",
                    str(OUT),
                    "--out",
                    str(OUT / "BACKEND_GATE"),
                ]
            )
            return
        # stall recovery: only our PIDs
        now = time.time()
        for job, meta in list(launched.items()):
            logp = LOG / f"job_{job}.out"
            age = now - logp.stat().st_mtime if logp.exists() else 0
            pid = meta["pid"]
            if pid_alive(pid) and age > STALL_SEC:
                log(f"STALL kill own pid {pid} job={job}")
                os.kill(pid, 9)
                dest = OUT / "jobs" / ("foundation_no_learning" if job == "foundation" else job)
                lock = dest / "FAIL"
                lock.write_text('{"reason":"stall"}\\n', encoding="utf-8")
                launched.pop(job, None)
            elif not pid_alive(pid):
                launched.pop(job, None)
        running = len(our_jobs())
        for job in todo:
            if running >= args.max_parallel:
                break
            if job in our_jobs() or job in launched:
                continue
            pid = launch(job, args.python, extra)
            launched[job] = {"pid": pid, "t": time.time()}
            running += 1
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
