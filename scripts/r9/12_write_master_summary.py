#!/usr/bin/env python3
"""Write R9_MASTER_SUMMARY.md and R9_DECISION.json (allowed states only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expolicy.r9_lab.constants import ALLOWED_DECISIONS, OUTPUT_ROOT
from expolicy.r9_lab.io_util import atomic_write_json, read_json
from expolicy.r9_lab.resume import run_cell_once


def decide(out: Path) -> str:
    man = read_json(out / "model_manifest.json") if (out / "model_manifest.json").exists() else {}
    if man.get("status") == "MODEL_LADDER_NO_FULL_PASS" or man.get("selected_model_id") is None and man:
        if man.get("status") in ALLOWED_DECISIONS or man.get("status") == "MODEL_LADDER_NO_FULL_PASS":
            return "MODEL_LADDER_NO_FULL_PASS"
    gate_h = read_json(out / "R9_GATE_H.json") if (out / "R9_GATE_H.json").exists() else None
    gate_j = read_json(out / "R9_GATE_J.json") if (out / "R9_GATE_J.json").exists() else None
    if gate_j and gate_h and gate_h.get("pass"):
        d = str(gate_j.get("decision") or "JOINT_IDENTIFIABILITY_FAIL")
        return d if d in ALLOWED_DECISIONS else "JOINT_IDENTIFIABILITY_FAIL"
    if gate_h:
        d = str(gate_h.get("decision") or "SUBSTRATE_FIXED_DOMINATES")
        return d if d in ALLOWED_DECISIONS else "SUBSTRATE_FIXED_DOMINATES"
    found = out / "R9_FOUNDATION.csv"
    if found.exists():
        return "FOUNDATION_FAIL"
    if man.get("selected_model_id"):
        return "FOUNDATION_FAIL"
    return "MODEL_LADDER_NO_FULL_PASS"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = p.parse_args(argv)
    out = args.output_root

    def _job() -> dict:
        decision = decide(out)
        if decision not in ALLOWED_DECISIONS:
            raise RuntimeError(f"illegal decision {decision}")
        atomic_write_json(out / "R9_DECISION.json", {"decision": decision, "allowed": list(ALLOWED_DECISIONS)})
        sections = [
            "# R9_MASTER_SUMMARY",
            "",
            "1. Novelty status",
            "2. Benchmark / Qwen Model Ladder / commits",
            "3. Model-screen result and frozen actor",
            "4. Formal foundation result",
            "5. Operator sanity",
            "6. 36×4 counterfactual matrix",
            "7. Global fixed vs hindsight oracle",
            "8. Per-env fixed vs hindsight oracle",
            "9. Within-env operator diversity",
            "10. Feature separability",
            "11. Gate H",
            "12. Joint oracle experiment（若运行）",
            "13. Gate J（若运行）",
            "14. Cost",
            "15. Case analysis",
            "16. Final decision",
            f"**{decision}**",
            "17. What is NOT supported",
            "18. Exact next action",
            "",
        ]
        (out / "R9_MASTER_SUMMARY.md").write_text("\n".join(sections), encoding="utf-8")
        return {"decision": decision}

    print(json.dumps(run_cell_once(out / "master_summary", _job, resume=args.resume), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
