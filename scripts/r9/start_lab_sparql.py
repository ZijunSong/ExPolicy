#!/usr/bin/env python3
"""Start official Freebase Virtuoso SPARQL on :3001 after virtuoso_db.zip is complete.

Does not patch LifelongAgentBench. Kill only this helper's child virtuoso if needed.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/ppnm/third_party/freebase_virtuoso")
ZIP_PATH = ROOT / "virtuoso_db_onedrive.zip"
DB_DIR = ROOT / "virtuoso_db"
PY = ROOT / "virtuoso.py"
LOG = Path("/data/ppnm/expolicy/outputs/r9_lab/logs/sparql_virtuoso.log")


def zip_complete() -> bool:
    if not ZIP_PATH.exists():
        return False
    # aria2 writes .aria2 control file until finished
    if Path(str(ZIP_PATH) + ".aria2").exists():
        return False
    return ZIP_PATH.stat().st_size > 40 * 1024**3


def extract() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    if (DB_DIR / "virtuoso.db").exists() and (DB_DIR / "virtuoso.db").stat().st_size > 20 * 1024**3:
        print("virtuoso.db already present", flush=True)
        return
    print(f"extracting {ZIP_PATH} -> {DB_DIR}", flush=True)
    subprocess.check_call(["unzip", "-o", str(ZIP_PATH), "-d", str(DB_DIR)])


def start() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(PY), "start", "3001", "-d", str(DB_DIR)]
    print("RUNNING", cmd, flush=True)
    subprocess.check_call(cmd)


def wait_ready(timeout: float = 600.0) -> None:
    import urllib.request

    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://127.0.0.1:3001/sparql", timeout=5)
            print("SPARQL ready", flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            time.sleep(5)
    raise SystemExit(f"SPARQL not ready: {last}")


def main() -> int:
    if not zip_complete():
        print("zip not complete yet", flush=True)
        return 2
    extract()
    start()
    wait_ready()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
