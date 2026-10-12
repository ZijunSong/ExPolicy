#!/usr/bin/env python3
"""Run the B-v1 local harness. This does not call a model server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from expolicy.paper_b.config import load_config
    from expolicy.paper_b.pipeline import run_protocol
    from expolicy.paper_b.world import load_tasks_jsonl

    parser = argparse.ArgumentParser(description="运行 B-v1 本地搜索执行器上的规则归纳")
    parser.add_argument("--profile", choices=("feasibility", "pilot"), default="feasibility")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tasks", type=Path, default=None, help="外部 JSONL 任务。不传则使用协议里的合成划分")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    out = args.out or (ROOT / "outputs" / "paper_b" / args.profile)
    tasks = load_tasks_jsonl(args.tasks) if args.tasks else None
    decision = run_protocol(config, out, profile=args.profile, tasks=tasks)
    print(
        json.dumps(
            {
                "status": decision["status"],
                "paper_claim_allowed": decision["paper_claim_allowed"],
                "decision": str(out / "B_DECISION.md"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
