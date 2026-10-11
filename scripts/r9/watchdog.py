#!/usr/bin/env python3
"""Watchdog: read STATUS_LIVE.json, stall=20min, kill ONLY recorded PIDs of this round.

Never pkill python / killall python.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from expolicy.r9_lab.constants import OUTPUT_ROOT, STALL_SEC
from expolicy.r9_lab.watchdog_util import (
    collect_status_files,
    is_stalled,
    kill_recorded_pid,
    load_status,
    recorded_pids,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--once", action="store_true")
    p.add_argument("--kill-stalled", action="store_true")
    p.add_argument("--interval", type=int, default=60)
    args = p.parse_args(argv)
    while True:
        files = collect_status_files(args.root)
        pids = recorded_pids(files)
        report = []
        for path in files:
            row = load_status(path) or {}
            stalled = is_stalled(row, stall_sec=STALL_SEC)
            item = {"path": str(path), "pid": row.get("pid"), "stalled": stalled, "stage": row.get("stage")}
            if stalled and args.kill_stalled and row.get("pid") in pids:
                item["kill"] = kill_recorded_pid(int(row["pid"]))
            report.append(item)
        print(json.dumps({"recorded_pids": pids, "n_status": len(files), "items": report}, default=str))
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
