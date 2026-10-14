"""Run the D-v1 harness. This does not start a LifelongAgentBench job."""

from __future__ import annotations

import argparse
from pathlib import Path

from expolicy.paper_d.constants import REPO_ROOT
from expolicy.paper_d.pilot import run_self_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic D-v1 diagnostic self-check")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "paper_d" / "harness")
    args = parser.parse_args()
    result = run_self_check(args.output)
    decision = result["decision"]
    print(
        f"paper_claim={decision['paper_claim']} harness={decision['harness_status']} "
        f"verified_artifacts={decision['inventory_verified']}"
    )


if __name__ == "__main__":
    main()
