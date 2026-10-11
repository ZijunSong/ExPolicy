#!/usr/bin/env python3
"""R9 preflight: GPU snapshot, docker, java, disk. Never kills other users."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

from expolicy.r9_lab.benchmark.lab_data import ensure_lab_assets
from expolicy.r9_lab.constants import ALLOWED_GPUS, OUTPUT_ROOT, ROOT
from expolicy.r9_lab.io_util import atomic_write_json, is_done, mark_done
from expolicy.r9_lab.resume import run_cell_once


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=60)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {cmd}: {exc}\n"


def snapshot_text() -> str:
    parts = [
        time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        f"hostname={os.uname().nodename}",
        "",
        "=== nvidia-smi ===",
        _run(["nvidia-smi"]),
        "=== df -h ===",
        _run(["df", "-h"]),
        "=== free -h ===",
        _run(["free", "-h"]),
        "=== docker --version ===",
        _run(["docker", "--version"]),
        "=== java ===",
        _run(["java", "-version"]),
        f"allowed_gpus={ALLOWED_GPUS}",
        "never_pkill_python=true",
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = p.parse_args(argv)
    audit = args.output_root / "audit"
    audit.mkdir(parents=True, exist_ok=True)

    def _job() -> dict:
        snap = audit / "SYSTEM_SNAPSHOT.txt"
        if not snap.exists():
            snap.write_text(snapshot_text(), encoding="utf-8")
        # GPU occupancy — record only, never kill.
        gpu = _run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"])
        (audit / "GPU_SNAPSHOT.txt").write_text(gpu, encoding="utf-8")
        lab_assets = None
        try:
            lab_assets = ensure_lab_assets(timeout=25.0)
            (audit / "LAB_DATA_MATERIALIZE.json").write_text(
                __import__("json").dumps(lab_assets, indent=2, default=str) + "\n", encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            (audit / "LAB_DATA_MATERIALIZE.json").write_text(
                __import__("json").dumps({"ok": False, "error": str(exc)}, indent=2) + "\n", encoding="utf-8"
            )
            lab_assets = {"ok": False, "error": str(exc)}
        return {"snapshot": str(snap), "killed_anyone": False, "lab_assets": lab_assets}

    result = run_cell_once(args.output_root / "preflight", _job, resume=args.resume)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
