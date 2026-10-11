#!/usr/bin/env python3
"""Assimilator isolation: MemRL and SkillOpt persistent stores must not share paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=Path, default=ROOT / "outputs" / "r8_backend_validation" / "jobs")
    args = p.parse_args()
    if not args.jobs.exists():
        print("SKIP no jobs yet")
        return
    mem_dirs = []
    skill_dirs = []
    for job in args.jobs.iterdir():
        if not job.is_dir():
            continue
        if job.name.startswith("memrl"):
            mem_dirs.append(job.resolve())
        if job.name.startswith("skillopt"):
            skill_dirs.append(job.resolve())
    # Distinct job directories.
    if len(mem_dirs) != len(set(mem_dirs)):
        raise SystemExit("duplicate memrl job dirs")
    overlap = set(mem_dirs) & set(skill_dirs)
    if overlap:
        raise SystemExit(f"shared job dir between backends: {overlap}")
    print(json.dumps({"memrl_jobs": len(mem_dirs), "skillopt_jobs": len(skill_dirs), "ok": True}))


if __name__ == "__main__":
    main()
