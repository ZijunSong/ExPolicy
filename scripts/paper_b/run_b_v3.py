#!/usr/bin/env python3
"""B-v3 action-space check. Does not rerun the B-v2.2 menu."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from expolicy.paper_b.v3_probe import run_action_space

    parser = argparse.ArgumentParser(description="运行 B-v3 动作空间检查")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper_b_v22.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs" / "paper_b_v2" / "pilot" / "B_V2_DATA_MANIFEST.json")
    parser.add_argument("--answers", type=Path, default=ROOT / "outputs" / "paper_b_v2" / "pilot" / "B_V2_SCORER_ANSWERS.json")
    parser.add_argument("--states", type=Path, default=ROOT / "outputs" / "paper_b_v22" / "fork_states.json")
    parser.add_argument("--forks", type=Path, default=ROOT / "outputs" / "paper_b_v22" / "fork_episodes.jsonl")
    parser.add_argument("--pool", type=Path, default=ROOT / "outputs" / "paper_b_v21" / "stage_b" / "B_V21_RAW_EPISODES.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "paper_b_v3")
    args = parser.parse_args()
    decision = run_action_space(args.config, args.manifest, args.answers, args.states, args.forks, args.pool, args.out)
    print(json.dumps({"status": decision["status"], "decision": str(args.out / "B_V3_ACTION_SPACE_DECISION.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
