#!/usr/bin/env python3
"""Static skill/backend headroom matrix (instruction entrypoint)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from expolicy.experiments.common import ROOT
from expolicy.r7.constants import DEFAULT_GPUS, R7_ROOT

PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models-config", default="configs/r7_models.yaml")
    p.add_argument("--suite", default="data/r7_search_suite")
    p.add_argument("--conditions", nargs="+", default=["base", "old_replacement_skill", "interventional_skill", "content_optimized_skill"])
    p.add_argument("--per-dataset", type=int, default=80)
    p.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    p.add_argument("--scheduler", default="auto")
    p.add_argument("--out", default=str(R7_ROOT / "model_skill_matrix"))
    args = p.parse_args()
    cmd = [
        PY,
        str(ROOT / "scripts" / "r7" / "orchestrate_r7.py"),
        "--stage",
        "matrix",
        "--scheduler",
        args.scheduler,
        "--resume",
        "--watchdog",
        "--out",
        str(R7_ROOT),
        "--gpus",
        os.environ.get("CUDA_VISIBLE_DEVICES") or DEFAULT_GPUS,
    ]
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
