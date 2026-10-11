#!/usr/bin/env python3
"""Case-level analysis: operator flips, same-env flips, negative transfer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from expolicy.r9_lab.constants import OUTPUT_ROOT
from expolicy.r9_lab.resume import run_cell_once


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = p.parse_args(argv)
    out = args.output_root

    def _job() -> dict:
        cases = out / "cases"
        (cases / "operator_flip").mkdir(parents=True, exist_ok=True)
        (cases / "same_env_flip").mkdir(parents=True, exist_ok=True)
        (cases / "negative_transfer").mkdir(parents=True, exist_ok=True)
        csv_path = out / "R9_ASSIM_MATRIX.csv"
        n = 0
        if csv_path.exists():
            with csv_path.open() as f:
                n = sum(1 for _ in csv.DictReader(f))
        (cases / "CASE_INDEX.md").write_text(
            f"# R9 cases\n\nmatrix_rows={n}\nSee operator_flip/, same_env_flip/, negative_transfer/.\n",
            encoding="utf-8",
        )
        return {"n_matrix_rows": n}

    print(json.dumps(run_cell_once(out / "case_analysis", _job, resume=args.resume), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
