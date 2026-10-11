#!/usr/bin/env python3
"""Joint oracle (only if Gate H PASS)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expolicy.r9_lab.constants import OUTPUT_ROOT
from expolicy.r9_lab.io_util import read_json
from expolicy.r9_lab.resume import run_cell_once


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    out = args.output_root
    gate = read_json(out / "R9_GATE_H.json") if (out / "R9_GATE_H.json").exists() else {}
    if not gate.get("pass"):
        print(json.dumps({"skipped": True, "reason": "H_NOT_PASS", "case": gate.get("case")}))
        return 0

    def _job() -> dict:
        if args.dry_run:
            return {"status": "DRY_RUN", "used_future_for_selection": False}
        raise RuntimeError("09_run_joint_oracle requires GPU/LAB after H PASS")

    print(json.dumps(run_cell_once(out / "joint_oracle", _job, resume=args.resume), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
