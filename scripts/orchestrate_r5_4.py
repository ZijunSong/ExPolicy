#!/usr/bin/env python3
"""R5.4 Stage-1 orchestrator: retriever health, base_eval, 4-GPU conditions, watchdog."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
PY = str(ROOT / ".venv" / "bin" / "python")
OUT = ROOT / "outputs" / "r5_4_rasa"
LOG = OUT / "logs"
PORT = 18080


def log(msg: str) -> None:
    line = f"[orch {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "orchestrate.log").open("a", encoding="utf-8").write(line + "\n")


def retriever_ok() -> bool:
    try:
        import httpx

        r = httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5.0)
        return r.status_code == 200 and bool(r.json().get("ok"))
    except Exception:
        return False


def run(cmd: list[str], name: str) -> None:
    log(" ".join(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["EXPOLICY_RETRIEVER_URL"] = f"http://127.0.0.1:{PORT}"
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    if rc != 0:
        raise SystemExit(f"{name} failed rc={rc}")


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while not retriever_ok():
        if time.time() - t0 > 3600:
            raise SystemExit("retriever not healthy")
        log("waiting for retriever :18080")
        time.sleep(10)
    log("retriever healthy")

    if not (OUT / "base_eval" / "DONE").exists():
        log("==== base_eval 4-way ====")
        run(
            [
                PY,
                str(ROOT / "scripts" / "run_r5_4_condition.py"),
                "--phase",
                "base_eval",
                "--gpus",
                "4,5,6,7",
                "--resume",
                "--root",
                str(OUT),
            ],
            "base_eval",
        )
    else:
        log("skip base_eval (DONE)")

    log("==== Stage-1 GPU wrappers + watchdog ====")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["EXPOLICY_RETRIEVER_URL"] = f"http://127.0.0.1:{PORT}"
    rc = subprocess.call([PY, str(ROOT / "scripts" / "watchdog_r5_4.py")], cwd=str(ROOT), env=env)
    if rc != 0:
        raise SystemExit(f"watchdog failed rc={rc}")
    if not (OUT / "STAGE1_JOBS.DONE").exists():
        raise SystemExit("Stage-1 wrappers did not finish cleanly")
    log("==== analyze Stage-1 ====")
    run(
        [
            PY,
            str(ROOT / "scripts" / "analyze_r5_4.py"),
            "--root",
            str(OUT),
            "--stage",
            "1",
            "--bootstrap",
            "10000",
            "--output",
            str(OUT / "R5_4_MASTER_SUMMARY.md"),
        ],
        "analyze",
    )
    log("Stage-1 complete")


if __name__ == "__main__":
    main()
