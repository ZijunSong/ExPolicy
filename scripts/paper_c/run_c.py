"""Run the C-v1 harness. This does not start a public-environment experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from expolicy.paper_c.pilot import run_self_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic C-v1 protocol self-check")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_self_check(args.output)
    decision = result["decision"]
    print(
        f"paper_claim={decision['paper_claim']} c0={decision['c0_status']} "
        f"c1={decision['c1_status']} c2={decision['c2_status']} pilot={decision['pilot_status']}"
    )


if __name__ == "__main__":
    main()
