#!/usr/bin/env python3
"""Watch R7 workers. STALL if 15 min without rollout/eval/checkpoint. Resume our workers only."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from expolicy.closed_loop.constants import R7_ROOT, ROOT

STALL_SEC = 15 * 60


def live_files(root: Path) -> list[Path]:
    return list(root.rglob("STATUS_LIVE.json")) + list(root.rglob("state_after.json")) + list(root.rglob("experience_batch.jsonl"))


def newest_mtime(root: Path) -> float:
    files = live_files(root)
    if not files:
        return 0.0
    return max(p.stat().st_mtime for p in files)


def our_workers() -> list[dict]:
    try:
        out = subprocess.check_output(["ps", "-ef"], text=True)
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        if "run_r7_condition.py" in line or "_seed_queue.py" in line or "smoke_r7_closed_loop.py" in line:
            if "grep" in line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                rows.append({"pid": int(parts[1]), "cmd": " ".join(parts[7:])})
    return rows


def retriever_ok(url: str) -> bool:
    try:
        import httpx

        r = httpx.get(url.rstrip("/") + "/health", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(R7_ROOT))
    p.add_argument("--retriever-url", default="http://127.0.0.1:18080")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--once", action="store_true")
    p.add_argument("--auto-resume", action="store_true")
    args = p.parse_args()
    root = Path(args.root)
    log = root / "logs" / "watchdog.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    def wlog(msg: str) -> None:
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    while True:
        mtime = newest_mtime(root)
        age = time.time() - mtime if mtime else 1e9
        workers = our_workers()
        health = retriever_ok(args.retriever_url)
        done = list(root.glob("*/seed*/DONE"))
        wlog(f"workers={len(workers)} retriever={'OK' if health else 'DOWN'} age={age:.0f}s dones={len(done)}")
        if not health:
            wlog("STALL? retriever down — will not kill retriever; record only")
        if workers and age > STALL_SEC:
            wlog("STALL detected: 15min without checkpoint/rollout")
            if args.auto_resume:
                for w in workers:
                    wlog(f"kill our worker pid={w['pid']}")
                    try:
                        os.kill(w["pid"], 15)
                    except Exception as exc:
                        wlog(f"kill failed {exc}")
                time.sleep(5)
                # relaunch suite resume for unfinished seeds
                cmd = [
                    str(ROOT / ".venv" / "bin" / "python"),
                    str(ROOT / "scripts" / "r7" / "run_r7_suite.py"),
                    "--resume",
                ]
                wlog("relaunch " + " ".join(cmd))
                subprocess.Popen(cmd, cwd=str(ROOT))
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
