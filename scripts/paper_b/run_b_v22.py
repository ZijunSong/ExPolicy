#!/usr/bin/env python3
"""B-v2.2 exploratory audit and one limited mechanism run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from expolicy.paper_b.v22_pilot import run_exploratory

    parser = argparse.ArgumentParser(description="运行 B-v2.2 机制小试")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper_b_v22.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs" / "paper_b_v2" / "pilot" / "B_V2_DATA_MANIFEST.json")
    parser.add_argument("--answers", type=Path, default=ROOT / "outputs" / "paper_b_v2" / "pilot" / "B_V2_SCORER_ANSWERS.json")
    parser.add_argument("--pool", type=Path, default=ROOT / "outputs" / "paper_b_v21" / "stage_b" / "B_V21_RAW_EPISODES.jsonl")
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "paper_b_v21" / "stage_a" / "B_V21_CALIBRATION_RESULTS.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "paper_b_v22")
    args = parser.parse_args()
    decision = run_exploratory(args.config, args.manifest, args.answers, args.pool, args.calibration, args.out)
    print(json.dumps({"status": decision["status"], "decision": str(args.out / "B_V22_DECISION.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
