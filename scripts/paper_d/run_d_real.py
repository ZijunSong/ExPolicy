"""D-v1.1 real entry. The default is a dry run and does not invent LAB scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from expolicy.paper_d.constants import REPO_ROOT
from expolicy.paper_d.persistent_ledger import PersistentLedger
from expolicy.paper_d.real_config import load_real_config
from expolicy.paper_d.real_dispatch import prepare_s0, run_jobs, vertical_jobs, write_json
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
            smoke = execute_smoke(
                config,
                execute=args.execute,
                runner=(lambda current: _run_vertical(current, args.output)) if args.execute else None,
            )
        except RealExecutionRefused as exc:
            smoke = {"status": "REFUSED", "episodes": 0, "reason": str(exc)}
    elif args.command in {"operator", "matrix", "content", "prospective-train", "prospective-test"} and args.execute:
        smoke = {"status": "REFUSED", "episodes": 0, "reason": "execute stays behind a passing smoke gate"}
    elif args.command in {"analyze", "freeze-card", "prospective-test"} and not args.execute:
        smoke = {"status": "NOT_RUN", "episodes": 0}
    write_real_decision(args.output, preflight_report=report, smoke=smoke)
    (args.output / "D_PREFLIGHT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"command={args.command} preflight={report['status']} smoke={smoke['status']} episodes={smoke['episodes']}")
    if smoke["status"] in {"REFUSED", "BLOCKED"}:
        sys.exit(1)


def _run_vertical(config: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    state_hash = prepare_s0(output)
    jobs = vertical_jobs(config, output, state_hash=state_hash)
    write_json(output / "D_EXECUTION_MANIFEST.json", {"jobs": [{k: v for k, v in job.items() if k != "instruction_hint"} for job in jobs]})
    ledger = PersistentLedger(output / "D_COST_LEDGER.sqlite", cap=int(config["pilot_episode_cap_all_inclusive"]))
    artifacts = run_jobs(jobs, ledger=ledger, run_root=output)
    ledger.close()
    summary = output / "D_VERTICAL_INTEGRATION.md"
    lines = ["# 纵向真实执行", "", "这些记录来自 OfficialLabEnv 和已核验的 Qwen 服务。答错仍算执行完成。", ""]
    for row in artifacts:
        job = row.get("job") or {}
        lines.append(
            f"- {job.get('env')} {job.get('branch')} task={job.get('task_id')} "
            f"success={row.get('success')} calls={row.get('n_model_calls')} "
            f"unchanged={row.get('state_unchanged')} cache={row.get('cache_hit')}"
        )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "episodes": len(artifacts),
        "artifact_paths": [str((row.get("job") or {}).get("result_path") or "") for row in artifacts],
        "successes": [bool(row.get("success")) for row in artifacts],
    }


if __name__ == "__main__":
    main()
