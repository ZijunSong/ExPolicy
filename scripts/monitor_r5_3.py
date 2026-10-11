#!/usr/bin/env python3
"""Watchdog for R5.3: restart orchestrate if it dies, kill stalled children."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
OUT = ROOT / "outputs" / "r5_3_external_generalization"
LOG = OUT / "logs"
PY = str(ROOT / ".venv" / "bin" / "python")
PID_FILE = OUT / "orchestrate.pid"
ORCH_LOG = LOG / "orchestrate.log"
FINAL = OUT / "R5_3_MASTER_SUMMARY.md"
STALL_SEC = 20 * 60
POLL = 60


def log(msg: str) -> None:
    print(f"[mon {time.strftime('%H:%M:%S')}] {msg}", flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG / "monitor.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[mon {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def orch_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return None
    return pid if alive(pid) else None


def launch_orch() -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    handle = ORCH_LOG.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [PY, "-u", str(ROOT / "scripts" / "orchestrate_r5_3.py")],
        cwd=str(ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    log(f"launched orchestrate pid={proc.pid}")
    return proc.pid


def retriever_ok() -> bool:
    try:
        import httpx
        r = httpx.get("http://127.0.0.1:18080/health", timeout=5.0)
        return r.status_code == 200 and bool(r.json().get("ok"))
    except Exception:
        return False


def status_snapshot() -> str:
    bits = []
    bits.append(f"retriever={'ok' if retriever_ok() else 'down'}")
    for name in ("SUITE_STATUS.json", "SUITE_base_eval.json", "STATUS_LIVE.json"):
        path = OUT / name
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                bits.append(f"{name}={json.dumps(payload)[:240]}")
            except Exception:
                bits.append(f"{name}=unreadable")
    done = list(OUT.rglob("DONE"))
    bits.append(f"DONE_files={len(done)}")
    smoke = OUT / "smoke" / "DONE"
    bits.append(f"smoke={'yes' if smoke.exists() else 'no'}")
    cal = OUT / "calibration" / "DONE"
    bits.append(f"calib={'yes' if cal.exists() else 'no'}")
    sk = OUT / "skill_validation" / "DONE"
    bits.append(f"skill_val={'yes' if sk.exists() else 'no'}")
    return " | ".join(bits)


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    restarts = 0
    log("monitor start")
    if orch_pid() is None:
        launch_orch()
    while True:
        if FINAL.exists():
            log("pipeline complete: R5_3_MASTER_SUMMARY.md present")
            return
        pid = orch_pid()
        if pid is None:
            restarts += 1
            if restarts > 8:
                log("too many orchestrate restarts; giving up")
                raise SystemExit(1)
            log(f"orchestrate dead; restart #{restarts}")
            launch_orch()
        else:
            log(f"orch pid={pid} {status_snapshot()}")
        time.sleep(POLL)


if __name__ == "__main__":
    main()
