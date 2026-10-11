#!/usr/bin/env python3
"""Analyze Gate J (only if H PASS and joint oracle ran)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expolicy.r9_lab.constants import OUTPUT_ROOT
from expolicy.r9_lab.evaluation.gate_j import analyze_gate_j
from expolicy.r9_lab.io_util import atomic_write_json, read_json
from expolicy.r9_lab.resume import run_cell_once


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = p.parse_args(argv)
    out = args.output_root
    gate = read_json(out / "R9_GATE_H.json") if (out / "R9_GATE_H.json").exists() else {}
    if not gate.get("pass"):
        print(json.dumps({"skipped": True, "reason": "H_NOT_PASS"}))
        return 0

    def _job() -> dict:
        raw = out / "R9_JOINT_ORACLE.csv"
        seed_rows = []
        if (out / "joint_oracle_seeds.json").exists():
            seed_rows = read_json(out / "joint_oracle_seeds.json")
        report = analyze_gate_j(seed_rows)
        atomic_write_json(out / "R9_GATE_J.json", report)
        return report

    print(json.dumps(run_cell_once(out / "gate_j", _job, resume=args.resume), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
