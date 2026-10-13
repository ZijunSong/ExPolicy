#!/usr/bin/env python3
"""B-v2.1 runtime repair and stage-A calibration. Does not open the confirmation set."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 B-v2.1 执行器验收")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper_b_v21.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs" / "paper_b_v2" / "pilot" / "B_V2_DATA_MANIFEST.json")
    parser.add_argument("--answers", type=Path, default=ROOT / "outputs" / "paper_b_v2" / "pilot" / "B_V2_SCORER_ANSWERS.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "paper_b_v21" / "stage_a")
    parser.add_argument("--repair-calibration", action="store_true")
    parser.add_argument("--stage", choices=("a", "b"), default="a")
    args = parser.parse_args()
    if args.stage == "b":
        from expolicy.paper_b.v21_collect import run_stage_b

        report = run_stage_b(args.config, args.manifest, args.answers, args.out)
        print(json.dumps({"gate": report["gate"], "coverage": str(args.out / "B_V21_PAIR_COVERAGE.md")}, ensure_ascii=False))
        return 0 if report["gate"] == "pass" else 3
    if args.repair_calibration:
        from expolicy.paper_b.v21_stage import rerun_calibration

        decision = rerun_calibration(args.config, args.manifest, args.answers, args.out)
    else:
        from expolicy.paper_b.v21_stage import run_stage_a

        decision = run_stage_a(args.config, args.manifest, args.answers, args.out, git=_git())
    print(json.dumps({"status": decision["status"], "decision": str(args.out / "B_V21_DECISION.md")}, ensure_ascii=False))
    return 2 if "BLOCKED" in decision["status"] else 0


def _git() -> dict[str, str]:
    def show(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            return ""

    return {
        "commit": show("rev-parse", "HEAD"),
        "branch": show("branch", "--show-current"),
        "status": show("status", "-sb"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
