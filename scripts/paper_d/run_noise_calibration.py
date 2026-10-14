"""Offline smoke review and the capped OS calibration pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expolicy.paper_d.constants import REPO_ROOT
from expolicy.paper_d.calibration_run import planned_cells, probe_tasks, run_calibration
from expolicy.paper_d.noise_export import audit_history_state, bias_rows, load_smoke_rows, triad_rows, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="D noise calibration. Analysis does not start episodes.")
    parser.add_argument(
        "command",
        choices=("export-smoke-noise", "audit-consumption", "plan-noise-calibration", "run-noise-calibration"),
    )
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "outputs" / "paper_d_real_v11" / "20261014-smoke-r2")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "paper_d_real_v11" / "20261014-noise-review")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--port", type=int, default=18102)
    parser.add_argument("--served-name", default="qwen3-14b-r9")
    parser.add_argument("--prior-ledger", type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "export-smoke-noise":
        rows = load_smoke_rows(args.input)
        columns = list(rows[0]) if rows else ["env"]
        write_csv(args.output / "D_SMOKE_NOISE_ROWS.csv", rows, columns)
        triads = triad_rows(rows)
        write_csv(args.output / "D_SMOKE_TRIADS.csv", triads, list(triads[0]) if triads else ["env"])
        bias = bias_rows(triads)
        write_csv(args.output / "D_NOOP_BIAS_VARIANCE.csv", bias, list(bias[0]) if bias else ["env"])
        (args.output / "D_DATA_DENOMINATORS.md").write_text(_denominators(rows, triads), encoding="utf-8")
        (args.output / "D_NOISE_REVIEW.md").write_text(_review(bias, triads), encoding="utf-8")
        print(f"rows={len(rows)} triads={len(triads)} disagree={sum(1 for row in triads if not row['branches_agree'])}")
        return
    if args.command == "audit-consumption":
        reports = []
        for env in ("db", "os", "kg"):
            path = args.input / "states" / env / "S_history_source"
            reports.append({"env": env, **audit_history_state(path, ["monitor user", "select count"])})
        (args.output / "D_STATE_CONSUMPTION_AUDIT.md").write_text(_consumption(reports), encoding="utf-8")
        (args.output / "D_STATE_CONSUMPTION_AUDIT.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("consumption", {item["env"]: item["status"] for item in reports})
        return
    probes = probe_tasks(args.input)
    cells = planned_cells(probes)
    (args.output / "D_CALIBRATION_PLANNED_CELLS.jsonl").write_text(
        "".join(json.dumps(cell, ensure_ascii=False) + "\n" for cell in cells),
        encoding="utf-8",
    )
    print(f"planned={len(cells)} probes={len(probes)}")
    if args.command == "plan-noise-calibration" or not args.execute:
        return
    summary = run_calibration(
        smoke_root=args.input,
        output=args.output,
        port=args.port,
        served_name=args.served_name,
        prior_ledger=args.prior_ledger or (args.input / "D_COST_LEDGER.sqlite"),
    )
    print(summary)


def _denominators(rows: list[dict], triads: list[dict]) -> str:
    tasks = {(row["env"], row["task_id"]) for row in rows}
    return "\n".join(
        [
            "# 数据分母",
            "",
            f"- 有效 smoke 行：{len(rows)}。这是分支级 episode，不是独立题数。",
            f"- 三分支格子：{len(triads)}。",
            f"- 环境×任务 ID：{len(tasks)}。",
            "- 早期在调用模型前失败的 576 条不进入这些行。",
            "- `structural_status=ok` 只表示持久状态前后 hash 未变。",
            "",
        ]
    )


def _review(bias: list[dict], triads: list[dict]) -> str:
    disagree = [row for row in triads if not row["branches_agree"]]
    lines = [
        "# 噪声复核",
        "",
        "分歧格子的初始请求相同。分数差是同一输入下的运行差异，不是经验写入。",
        "",
        f"不一致格子 {len(disagree)} / {len(triads)}。",
        "",
    ]
    for row in bias:
        lines.append(
            f"- {row['env']} {row['state_id']} {row['probe_role']}: "
            f"DISCARD-BASE={row['mean_discard_minus_base']:.4f}, "
            f"不一致率={row['disagree_rate']:.4f}, n={row['n_triads']}"
        )
    lines.append("")
    return "\n".join(lines)


def _consumption(reports: list[dict]) -> str:
    lines = ["# 状态消费审计", ""]
    for item in reports:
        lines.append(
            f"- {item['env']}: episodic={item['n_episodic']} success={item['n_success']} "
            f"committed={item['n_committed']} intersection={item['intersection']} status=`{item['status']}`"
        )
    lines.extend(["", "committed 交集为 0 时，默认检索不会把历史送进 actor。旧 smoke 不能证明两个有效记忆状态都被试过。", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
