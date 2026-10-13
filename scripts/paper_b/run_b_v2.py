#!/usr/bin/env python3
"""B-v2 entry point. The B-v1 command remains a synthetic regression only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from expolicy.paper_b.pilot_v2 import run_development_pilot
    from expolicy.paper_b.pipeline_v2 import build_real_manifest, load_v2_config, run_smoke

    parser = argparse.ArgumentParser(description="运行 B-v2 真实搜索协议")
    parser.add_argument("stage", choices=("smoke", "pilot"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper_b_v2.yaml")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    raw = load_v2_config(args.config)
    if args.out is None:
        args.out = ROOT / "outputs" / "paper_b_v2" / args.stage
    if args.stage == "pilot":
        manifest = build_real_manifest(raw)
        decision = run_development_pilot(raw, manifest, args.out)
    else:
        decision = run_smoke(raw, args.out)
    print(json.dumps({"status": decision["status"], "decision": str(args.out / "B_V2_DECISION.md")}, ensure_ascii=False))
    return 2 if str(decision["status"]).endswith("BLOCKED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
