"""D-v1.1 real entry. The default is a dry run and does not invent LAB scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expolicy.paper_d.constants import REPO_ROOT
from expolicy.paper_d.real_config import load_real_config
from expolicy.paper_d.real_execution import (
    RealExecutionRefused,
    execute_smoke,
    preflight,
    write_real_decision,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="D-v1.1 real diagnostic. Default is dry-run.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "paper_d_real_v11.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "paper_d_real_v11" / "preflight")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "smoke",
            "operator",
            "matrix",
            "content",
            "analyze",
            "freeze-card",
            "prospective-train",
            "prospective-test",
            "decide",
        ),
    )
    args = parser.parse_args()
    config = load_real_config(args.config)
    report = preflight(config)
    smoke = {"status": "NOT_RUN", "episodes": 0}
    if args.command == "smoke":
        try:
            smoke = execute_smoke(config, execute=args.execute)
        except RealExecutionRefused as exc:
            smoke = {"status": "REFUSED", "episodes": 0, "reason": str(exc)}
    elif args.command in {"operator", "matrix", "content", "prospective-train", "prospective-test"} and args.execute:
        smoke = {"status": "REFUSED", "episodes": 0, "reason": "execute stays behind a passing smoke gate"}
    elif args.command in {"analyze", "freeze-card", "prospective-test"} and not args.execute:
        smoke = {"status": "NOT_RUN", "episodes": 0}
    write_real_decision(args.output, preflight_report=report, smoke=smoke)
    (args.output / "D_PREFLIGHT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"command={args.command} preflight={report['status']} smoke={smoke['status']} episodes={smoke['episodes']}")


if __name__ == "__main__":
    main()
