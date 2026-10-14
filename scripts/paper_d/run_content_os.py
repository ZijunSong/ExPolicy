"""OS content attribution. Dry-run unless --execute is set. No router and no second model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from expolicy.paper_d.constants import REPO_ROOT
from expolicy.paper_d.content_analysis import (
    align_plan,
    batch_equal_task_deltas,
    benefiting_tasks,
    bootstrap_interval,
    decide,
    holm,
    leave_one_task_out,
    role_delta,
    single_task_dependent,
    utility_from_deltas,
)
from expolicy.paper_d.content_execute import (
    CALIBRATION_LEDGER,
    acquire_donors,
    build_condition_states,
    controls_invalid,
    freeze,
    run_cells,
    structure_problem,
)
from expolicy.paper_d.content_plan import expected_counts
from expolicy.paper_d.content_protocol import PROTOCOL
from expolicy.paper_d.persistent_ledger import PersistentLedger
from expolicy.r9_lab.io_util import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="D OS content attribution. Does not train a router or start another model.")
    parser.add_argument("command", choices=("preflight", "freeze", "execute", "analyze", "decide"))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "paper_d_content_os" / "20261014-content-os")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prior-ledger", type=Path, default=CALIBRATION_LEDGER)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "preflight":
        print(json.dumps({"sha_expected": "2f21a4a41e59056037be35c8fca091b9ecaa89c1", "counts": expected_counts(), **PROTOCOL}, ensure_ascii=False, indent=2))
        return
    if args.command == "freeze" or not (args.output / "D_CONTENT_TASK_MANIFEST.json").is_file():
        print(freeze(args.output))
        if args.command == "freeze":
            return
    if args.command == "execute":
        if not args.execute:
            print("dry-run: plan is frozen; pass --execute to call the official environment")
            return
        _execute(args.output, args.prior_ledger)
        _analyze(args.output)
        return
    if args.command in {"analyze", "decide"}:
        _analyze(args.output)


def _execute(output: Path, prior_ledger: Path) -> None:
    ledger = PersistentLedger(output / "D_CONTENT_COST_LEDGER.sqlite", cap=int(PROTOCOL["cumulative_cap"]))
    if prior_ledger.is_file() and ledger.spent() == 0:
        from expolicy.paper_d.content_execute import _import_ledger

        _import_ledger(ledger, prior_ledger)
    scope_path = output / "D_BUDGET_SCOPE.json"
    existing_scope = json.loads(scope_path.read_text(encoding="utf-8")) if scope_path.is_file() else {}
    baseline = int(existing_scope.get("imported_spent") or ledger.spent())
    write_json(scope_path, {"imported_spent": baseline, "phase_cap": PROTOCOL["phase_episode_cap"], "cumulative_cap": PROTOCOL["cumulative_cap"], "current_spent": ledger.spent()})
    bundles_path = output / "D_CONTENT_SOURCE_BUNDLES.json"
    bundles = json.loads(bundles_path.read_text(encoding="utf-8"))
    if not bundles.get("donors"):
        acquire_donors(output, ledger, baseline)
    prepared = build_condition_states(output)
    audits = json.loads((output / "D_CONTENT_CONTROL_AUDIT.json").read_text(encoding="utf-8"))
    reason = controls_invalid(audits)
    if reason:
        _decide_early(output, "D_CONTENT_CONTROL_INVALID", reason)
        return
    plan = [json.loads(line) for line in (output / "D_CONTENT_EXECUTION_PLAN.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads((output / "D_CONTENT_TASK_MANIFEST.json").read_text(encoding="utf-8"))
    instructions = {
        task["task_id"]: task["instruction"]
        for task in manifest["main"] + manifest["history_subset"] + manifest["cases"]
    }
    for cell in plan:
        cell["instruction"] = instructions[cell["task_id"]]
    structure_ids = [task["task_id"] for task in manifest["main"] if task["probe_role"] == "future"][:2]
    structure_ids += [task["task_id"] for task in manifest["main"] if task["probe_role"] == "retention"][:2]
    main = [cell for cell in plan if cell["block_id"] == "main"]
    first = [cell for cell in main if cell["task_id"] in structure_ids]
    rest = [cell for cell in main if cell["task_id"] not in structure_ids]
    tokens = [0]
    retries = [0]
    rows, stop = run_cells(first, output=output, ledger=ledger, prepared=prepared, baseline_spent=baseline, tokens_used=tokens, retries=retries)
    problem = structure_problem(rows)
    if problem:
        _write_rows(output, rows)
        _decide_early(output, "D_CONTENT_EXECUTION_BLOCKED", problem)
        return
    more, stop = run_cells(rest, output=output, ledger=ledger, prepared=prepared, baseline_spent=baseline, tokens_used=tokens, retries=retries)
    rows.extend(more)
    _write_rows(output, rows)
    if stop:
        _decide_early(output, "D_CONTENT_INCONCLUSIVE", f"主块因{stop}提前停止")
        return
    _maybe_secondary(output, plan, rows, ledger, prepared, baseline, tokens, retries)


def _maybe_secondary(output, plan, rows, ledger, prepared, baseline, tokens, retries) -> None:
    done = [row for row in rows if row.get("infra_status") == "ok" and row.get("tokens")]
    mean_tokens = (sum(int(row["tokens"]) for row in done) / len(done)) if done else 10000
    for block_id in ("history", "case"):
        cells = [cell for cell in plan if cell["block_id"] == block_id]
        forecast = mean_tokens * len(cells)
        if tokens[0] + forecast > int(PROTOCOL["phase_token_cap"]) or ledger.spent() - baseline + len(cells) > int(PROTOCOL["phase_episode_cap"]):
            (output / f"D_CONTENT_{block_id.upper()}_NOT_RUN.txt").write_text(
                "资源预报超过本轮 token 或 episode 上限，按事先规则不跑这个次要块。\n",
                encoding="utf-8",
            )
            continue
        extra, stop = run_cells(cells, output=output, ledger=ledger, prepared=prepared, baseline_spent=baseline, tokens_used=tokens, retries=retries)
        rows.extend(extra)
        _write_rows(output, rows)
        if stop:
            break


def _write_rows(output: Path, rows: list[dict]) -> None:
    (output / "D_CONTENT_TASK_RESULTS.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _decide_early(output: Path, label: str, reason: str) -> None:
    (output / "D_CONTENT_DECISION.md").write_text(f"# 决定\n\n`{label}`\n\n{reason}\n", encoding="utf-8")
    print(label, reason)


def _analyze(output: Path) -> None:
    existing = output / "D_CONTENT_DECISION.md"
    if existing.is_file() and any(label in existing.read_text(encoding="utf-8") for label in ("D_CONTENT_CONTROL_INVALID", "D_CONTENT_EXECUTION_BLOCKED")):
        print("decision already written")
        return
    manifest = json.loads((output / "D_CONTENT_TASK_MANIFEST.json").read_text(encoding="utf-8"))
    plan = [json.loads(line) for line in (output / "D_CONTENT_EXECUTION_PLAN.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    results_path = output / "D_CONTENT_TASK_RESULTS.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()] if results_path.is_file() else []
    main_plan = [cell for cell in plan if cell["block_id"] == "main"]
    aligned = align_plan(main_plan, rows)
    report = _effects(aligned["outcomes"], manifest["main"], "main", "S0")
    report["execution_blocked"] = False
    report["control_invalid"] = False
    label = decide(report)
    _write_tables(output, report, aligned)
    (output / "D_CONTENT_DECISION.md").write_text(_decision_md(label, report, aligned), encoding="utf-8")
    print(label)


def _effects(outcomes, tasks, block_id: str, state_id: str) -> dict:
    contrasts = {}
    utilities = {}
    per_task = {}
    for left, right, name in (
        ("REAL", "NO_UPDATE", "REAL-NO_UPDATE"),
        ("REAL", "EMPTY", "REAL-EMPTY"),
        ("REAL", "MATCHED_WRONG", "REAL-MATCHED_WRONG"),
    ):
        future_batches = [
            role_delta(outcomes, tasks, block_id=block_id, state_id=state_id, batch_id=batch, role="future", left=left, right=right)
            for batch in PROTOCOL["batches"]
        ]
        retention_batches = [
            role_delta(outcomes, tasks, block_id=block_id, state_id=state_id, batch_id=batch, role="retention", left=left, right=right)
            for batch in PROTOCOL["batches"]
        ]
        future = batch_equal_task_deltas(future_batches)
        retention = batch_equal_task_deltas(retention_batches)
        interval = bootstrap_interval([item["delta"] for item in future["tasks"]]) if future["estimable"] else {"estimable": False}
        utility_point = utility_from_deltas(future.get("mean"), retention.get("mean"))
        utility_interval = _bootstrap_utility(future, retention) if future["estimable"] and retention["estimable"] else {"estimable": False, "point": utility_point}
        contrasts[name] = future
        per_task[name] = future.get("tasks") or []
        utilities[name] = {"point": utility_point, **{key: utility_interval.get(key) for key in ("low", "high", "p", "estimable")}}
        contrasts[name + ":retention"] = retention
        contrasts[name + ":future_interval"] = interval
    future_view = {
        "estimable": all(contrasts[name]["estimable"] for name in PROTOCOL["primary_contrasts"]),
    }
    pvalues = {}
    for name in PROTOCOL["primary_contrasts"]:
        interval = contrasts[name + ":future_interval"]
        future_view[name] = {"point": contrasts[name]["mean"], "low": interval.get("low"), "high": interval.get("high"), "p": interval.get("p")}
        if interval.get("estimable"):
            pvalues[name] = float(interval["p"])
    future_view["holm"] = holm(pvalues) if pvalues else {}
    benefits = benefiting_tasks({name: per_task[name] for name in PROTOCOL["primary_contrasts"]}) if future_view["estimable"] else {}
    return {
        "future": future_view,
        "utility": {name: utilities[name] for name in PROTOCOL["primary_contrasts"]},
        "retention": {name: contrasts[name + ":retention"] for name in PROTOCOL["primary_contrasts"]},
        "benefits": benefits,
        "single_task_dependent": single_task_dependent(per_task["REAL-NO_UPDATE"]) if per_task["REAL-NO_UPDATE"] else False,
        "leave_one_out": leave_one_task_out(per_task["REAL-NO_UPDATE"]),
        "per_task": per_task,
    }


def _bootstrap_utility(future: dict, retention: dict) -> dict:
    rng = random.Random(20261014)
    fvalues = [float(item["delta"]) for item in future["tasks"]]
    rvalues = [float(item["delta"]) for item in retention["tasks"]]
    stats = []
    for _ in range(1000):
        fmean = sum(fvalues[rng.randrange(len(fvalues))] for _ in fvalues) / len(fvalues)
        rmean = sum(rvalues[rng.randrange(len(rvalues))] for _ in rvalues) / len(rvalues)
        stats.append(fmean + 0.5 * min(0.0, rmean))
    stats.sort()
    return {"estimable": True, "point": utility_from_deltas(future["mean"], retention["mean"]), "low": stats[25], "high": stats[974]}


def _write_tables(output: Path, report: dict, aligned: dict) -> None:
    _csv(output / "D_CONTENT_PRIMARY_EFFECTS.csv", [
        {"contrast": name, **report["future"][name], "holm_p": report["future"]["holm"].get(name)}
        for name in PROTOCOL["primary_contrasts"]
    ])
    _csv(output / "D_CONTENT_RETENTION_UTILITY.csv", [
        {"contrast": name, "retention_mean": (report["retention"][name] or {}).get("mean"), **report["utility"][name]}
        for name in PROTOCOL["primary_contrasts"]
    ])
    _csv(output / "D_CONTENT_LEAVE_ONE_OUT.csv", report["leave_one_out"])
    family_rows = []
    for name, tasks in report["per_task"].items():
        buckets: dict[str, list[float]] = {}
        for item in tasks:
            buckets.setdefault(str(item.get("family")), []).append(float(item["delta"]))
        for family, values in sorted(buckets.items()):
            family_rows.append({"contrast": name, "family": family, "n": len(values), "mean": sum(values) / len(values)})
    _csv(output / "D_CONTENT_TASK_FAMILY_EFFECTS.csv", family_rows)
    write_json(output / "D_CONTENT_ALIGNMENT.json", {"planned": aligned["planned"], "paired": aligned["paired"], "missing": len(aligned["missing"])})


def _csv(path: Path, rows: list[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _decision_md(label: str, report: dict, aligned: dict) -> str:
    lines = [
        "# 决定",
        "",
        f"`{label}`",
        "",
        "这是新任务上的内容归因，不是 D-v1.1 的正式 GO，也不训练 Router。",
        "",
        f"主计划配对 {aligned['paired']} / {aligned['planned']}，缺失 {len(aligned['missing'])}。缺失没有补 0。",
        "",
    ]
    for name in PROTOCOL["primary_contrasts"]:
        item = report["future"][name]
        lines.append(
            f"- {name} future 差 {item['point']}，区间 [{item['low']}, {item['high']}]，Holm p={report['future']['holm'].get(name)}"
        )
    benefits = report["benefits"]
    lines.append(
        f"- 同时胜过基线和两种对照的新 future 任务 {benefits.get('n_benefited')} 个，受损 {benefits.get('n_harmed')} 个，覆盖族 {benefits.get('n_families')}。"
    )
    lines.append(f"- 单题驱动：{report['single_task_dependent']}。")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
