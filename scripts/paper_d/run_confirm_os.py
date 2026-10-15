"""Last OS content confirmation. Dry-run unless --execute. No router and no second model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expolicy.paper_d.confirm_analysis import (
    batch_effects,
    cross_bootstrap,
    decide,
    leave_one_batch_out,
    leave_one_task_out,
    positive_batch_count,
    robustness_ok,
)
from expolicy.paper_d.confirm_controls import load_tokenizer_counter
from expolicy.paper_d.confirm_execute import (
    PRIOR_LEDGER,
    S0_SOURCE,
    audit_controls,
    experience_from,
    import_ledger,
    run_with_retry,
    select_written,
)
from expolicy.paper_d.confirm_plan import build_plan, expected_count
from expolicy.paper_d.confirm_pools import freeze_pools, load_excluded
from expolicy.paper_d.confirm_protocol import PROTOCOL
from expolicy.paper_d.constants import REPO_ROOT
from expolicy.paper_d.content_controls import clone_state, write_empty, write_no_update, write_real
from expolicy.paper_d.persistent_ledger import PersistentLedger
from expolicy.r9_lab.assimilators.episodic import apply_episodic
from expolicy.r9_lab.io_util import write_json
from expolicy.r9_lab.state.persistent_state import PersistentState


def main() -> None:
    parser = argparse.ArgumentParser(description="D OS confirmation. Does not train a router or start another model.")
    parser.add_argument(
        "command",
        choices=("preflight", "freeze-pools", "acquire-sources", "match-and-audit-controls", "freeze-evaluation", "execute", "analyze", "decide"),
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "paper_d_content_os" / "20261015-confirm")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prior-ledger", type=Path, default=REPO_ROOT / PRIOR_LEDGER)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "preflight":
        _reanalysis(args.output)
        print(json.dumps({"stage": PROTOCOL["stage_id"], "prior_ledger": str(args.prior_ledger), "phase_cap": PROTOCOL["phase_episode_cap"]}, indent=2))
        return
    if args.command == "freeze-pools" or not (args.output / "D_CONFIRM_SOURCE_PLAN.json").is_file():
        _freeze(args.output)
        if args.command == "freeze-pools":
            return
    if args.command == "acquire-sources":
        if not args.execute:
            print("dry-run")
            return
        _acquire(args.output, args.prior_ledger)
        return
    if args.command == "match-and-audit-controls":
        _match(args.output)
        return
    if args.command == "freeze-evaluation":
        _freeze_eval(args.output)
        return
    if args.command == "execute":
        if not args.execute:
            print("dry-run")
            return
        _execute(args.output, args.prior_ledger)
        return
    if args.command in {"analyze", "decide"}:
        _analyze(args.output)


def _freeze(output: Path) -> None:
    _reanalysis(output)
    split = json.loads((REPO_ROOT / "outputs/r9_lab/split.json").read_text(encoding="utf-8"))["envs"]["os"]
    excluded, blocked = load_excluded(REPO_ROOT)
    pools = freeze_pools(split, excluded, blocked)
    write_json(output / "D_CONFIRM_SOURCE_PLAN.json", pools)
    n_tasks = len(pools["evaluation"])
    write_json(
        output / "D_CONFIRM_PROTOCOL_LOCK.json",
        {**PROTOCOL, "variant": pools["variant"], "n_evaluation_tasks": n_tasks, "n_donors": len(pools["donor_order"]), "planned_eval": expected_count(n_tasks)},
    )
    (output / "D_CONFIRM_EXPOSURE_AUDIT.md").write_text(
        "\n".join(
            [
                "# 确认任务暴露",
                "",
                f"排除已追踪的 D 任务 {len(excluded)} 个。`prior_exposure_unknown` 仍在：不能声称整个 R9/A 没用过这些题。",
                f"确认题 {pools['n_future']} future + {pools['n_retention']} retention，变体 `{pools['variant']}`。",
                f"源任务 {sum(len(batch['sources']) for batch in pools['batches'])}，donor 顺序 {len(pools['donor_order'])}。",
                "skill 组合只作描述，不作为独立任务族。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print({"variant": pools["variant"], "tasks": n_tasks, "donors": len(pools["donor_order"]), "planned": expected_count(n_tasks)})


def _ledger(output: Path, prior: Path) -> tuple[PersistentLedger, int]:
    ledger = PersistentLedger(output / "D_CONFIRM_COST_LEDGER.sqlite", cap=int(PROTOCOL["cumulative_cap"]))
    scope = output / "D_BUDGET_SCOPE.json"
    if ledger.spent() == 0 and prior.is_file():
        import_ledger(ledger, prior)
    existing = json.loads(scope.read_text(encoding="utf-8")) if scope.is_file() else {}
    baseline = int(existing.get("imported_spent") or ledger.spent())
    write_json(scope, {"imported_spent": baseline, "current_spent": ledger.spent(), "phase_cap": PROTOCOL["phase_episode_cap"]})
    return ledger, baseline


def _s0(output: Path) -> PersistentState:
    path = output / "states" / "bases" / "S0"
    if (path / "STATE.json").is_file():
        return PersistentState.load(path)
    return clone_state(PersistentState.load(S0_SOURCE), str(path))


def _acquire(output: Path, prior: Path) -> None:
    pools = json.loads((output / "D_CONFIRM_SOURCE_PLAN.json").read_text(encoding="utf-8"))
    ledger, baseline = _ledger(output, prior)
    state = _s0(output)
    state.save()
    retries = [0]
    source_rows = []
    if not (output / "D_CONFIRM_SOURCE_RESULTS.jsonl").is_file():
        for batch in pools["batches"]:
            for task in batch["sources"]:
                if ledger.spent() - baseline >= int(PROTOCOL["phase_episode_cap"]):
                    break
                cell = {**task, "batch_id": batch["batch_id"], "condition": "SOURCE", "repeat_id": 0, "block_id": "source"}
                row = run_with_retry(cell, output=output, ledger=ledger, state_path=state.root_path(), state_hash=state.content_hash(), control_hash="source", retries=retries, job_kind="confirm_source")
                print(f"source {batch['batch_id']} {task['task_id']} outcome={None if row is None else row.get('outcome')}", flush=True)
                if row:
                    source_rows.append(row)
        _jsonl(output / "D_CONFIRM_SOURCE_RESULTS.jsonl", source_rows)
    else:
        source_rows = [json.loads(line) for line in (output / "D_CONFIRM_SOURCE_RESULTS.jsonl").read_text().splitlines() if line.strip()]
    by_batch: dict[str, list] = {batch["batch_id"]: [] for batch in pools["batches"]}
    lookup = {(batch["batch_id"], task["task_id"]): task for batch in pools["batches"] for task in batch["sources"]}
    for row in source_rows:
        task = lookup[(row["batch_id"], row["task_id"])]
        by_batch[row["batch_id"]].append(experience_from(task, row))
    batches = []
    for batch in pools["batches"]:
        summary = select_written(by_batch[batch["batch_id"]])
        batches.append({"batch_id": batch["batch_id"], **summary})
    donor_rows = []
    if not (output / "D_CONFIRM_DONOR_LIBRARY.jsonl").is_file():
        for task in pools["donor_order"]:
            if ledger.spent() - baseline >= int(PROTOCOL["phase_episode_cap"]):
                break
            cell = {**task, "batch_id": "donor", "condition": "DONOR", "repeat_id": 0, "block_id": "donor"}
            row = run_with_retry(cell, output=output, ledger=ledger, state_path=state.root_path(), state_hash=state.content_hash(), control_hash="donor", retries=retries, job_kind="confirm_donor")
            print(f"donor {task['task_id']} outcome={None if row is None else row.get('outcome')}", flush=True)
            if row and row.get("success") and row.get("infra_status") == "ok":
                donor_rows.append({**experience_from(task, row), "batch_id": "donor"})
        _jsonl(output / "D_CONFIRM_DONOR_LIBRARY.jsonl", donor_rows)
    write_json(output / "D_CONFIRM_SOURCE_RESULTS_SUMMARY.json", {"batches": batches})


def _match(output: Path) -> None:
    pools = json.loads((output / "D_CONFIRM_SOURCE_PLAN.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "D_CONFIRM_SOURCE_RESULTS_SUMMARY.json").read_text(encoding="utf-8"))
    donors = [json.loads(line) for line in (output / "D_CONFIRM_DONOR_LIBRARY.jsonl").read_text().splitlines() if line.strip()]
    audit = audit_controls(pools, summary, donors, load_tokenizer_counter())
    write_json(output / "D_CONFIRM_CONTROL_AUDIT.json", audit)
    rows = []
    for batch in audit["batches"]:
        for ratio in batch["choice"].get("ratios") or []:
            rows.append({"batch_id": batch["batch_id"], **ratio})
    _csv(output / "D_CONFIRM_LENGTH_MATCH.csv", rows)
    if audit["blocked"]:
        (output / "D_CONFIRM_DECISION.md").write_text(f"# 决定\n\n`D_CONFIRM_BLOCKED`\n\n控制构造未通过：{audit['blocked']}\n", encoding="utf-8")
    print({"blocked": audit["blocked"], "batches": [(item["batch_id"], item["n_written"], item["choice"]["status"]) for item in audit["batches"]]})


def _freeze_eval(output: Path) -> None:
    audit = json.loads((output / "D_CONFIRM_CONTROL_AUDIT.json").read_text(encoding="utf-8"))
    if audit["blocked"]:
        print("controls blocked; evaluation plan not released")
        return
    pools = json.loads((output / "D_CONFIRM_SOURCE_PLAN.json").read_text(encoding="utf-8"))
    plan = build_plan(pools["evaluation"], [batch["batch_id"] for batch in pools["batches"]])
    _jsonl(output / "D_CONFIRM_EVALUATION_PLAN.jsonl", [{key: value for key, value in cell.items() if key != "instruction"} for cell in plan])
    print({"plan": len(plan)})


def _execute(output: Path, prior: Path) -> None:
    if not (output / "D_CONFIRM_EVALUATION_PLAN.jsonl").is_file():
        print("no evaluation plan")
        return
    audit = json.loads((output / "D_CONFIRM_CONTROL_AUDIT.json").read_text(encoding="utf-8"))
    if audit["blocked"]:
        print("refusing to evaluate blocked controls")
        return
    pools = json.loads((output / "D_CONFIRM_SOURCE_PLAN.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "D_CONFIRM_SOURCE_RESULTS_SUMMARY.json").read_text(encoding="utf-8"))
    prepared = _states(output, summary, audit)
    instructions = {task["task_id"]: task["instruction"] for task in pools["evaluation"]}
    plan = [json.loads(line) for line in (output / "D_CONFIRM_EVALUATION_PLAN.jsonl").read_text().splitlines() if line.strip()]
    for cell in plan:
        cell["instruction"] = instructions[cell["task_id"]]
    structure_ids = [task["task_id"] for task in pools["evaluation"] if task["probe_role"] == "future"][:2]
    first = [cell for cell in plan if cell["task_id"] in structure_ids]
    rest = [cell for cell in plan if cell["task_id"] not in structure_ids]
    ledger, baseline = _ledger(output, prior)
    retries = [0]
    tokens = [0]
    rows = _run_plan(first, output, ledger, prepared, baseline, retries, tokens)
    if any(not row.get("user_contains_official_instruction") or row.get("infra_status") != "ok" for row in rows):
        _jsonl(output / "D_CONFIRM_TASK_RESULTS.jsonl", rows)
        (output / "D_CONFIRM_DECISION.md").write_text("# 决定\n\n`D_CONFIRM_BLOCKED`\n\n结构检查未通过。\n", encoding="utf-8")
        return
    rows.extend(_run_plan(rest, output, ledger, prepared, baseline, retries, tokens))
    _jsonl(output / "D_CONFIRM_TASK_RESULTS.jsonl", rows)
    _analyze(output)


def _run_plan(cells, output, ledger, prepared, baseline, retries, tokens):
    rows = []
    for index, cell in enumerate(cells, start=1):
        if ledger.spent() - baseline >= int(PROTOCOL["phase_episode_cap"]) or tokens[0] >= int(PROTOCOL["phase_token_cap"]):
            print("cap reached", flush=True)
            break
        spec = prepared[(cell["batch_id"], cell["condition"])]
        row = run_with_retry(
            cell, output=output, ledger=ledger, state_path=Path(spec["path"]),
            state_hash=spec["state_hash"], control_hash=spec["control_hash"], retries=retries, job_kind="confirm_eval",
        )
        if row:
            tokens[0] += 0 if row.get("cache_hit") else int(row.get("tokens") or 0)
            rows.append(row)
            print(f"confirm {index}/{len(cells)} {cell['batch_id']} {cell['task_id']} {cell['condition']} outcome={row.get('outcome')}", flush=True)
    return rows


def _states(output: Path, summary: dict, audit: dict) -> dict[tuple[str, str], dict[str, Any]]:
    base = PersistentState.load(output / "states" / "bases" / "S0")
    prepared = {}
    manifest = []
    audit_by_batch = {item["batch_id"]: item for item in audit["batches"]}
    for batch in summary["batches"]:
        batch_id = batch["batch_id"]
        written = batch["written"]
        donors = (audit_by_batch[batch_id]["choice"].get("donors") or []) if audit_by_batch[batch_id]["choice"]["status"] == "matched" else []
        real = write_real(base, written, str(output / "states" / batch_id / "REAL")) if written else write_no_update(base, str(output / "states" / batch_id / "REAL"))
        empty = write_empty(PersistentState.load(real["path"]), real.get("written_ids") or [], str(output / "states" / batch_id / "EMPTY")) if written else write_no_update(base, str(output / "states" / batch_id / "EMPTY"))
        no_update = write_no_update(base, str(output / "states" / batch_id / "NO_UPDATE"))
        wrong_path = output / "states" / batch_id / "MATCHED_WRONG"
        if donors:
            wrong_state = clone_state(base, str(wrong_path))
            apply_episodic(wrong_state, donors)
            wrong = {"path": str(wrong_path), "state_hash": PersistentState.load(wrong_path).content_hash()}
        else:
            wrong = write_no_update(base, str(wrong_path))
        specs = {"REAL": real, "EMPTY": empty, "NO_UPDATE": no_update, "MATCHED_WRONG": wrong}
        for condition, spec in specs.items():
            spec["control_hash"] = condition + ":" + spec["state_hash"]
            prepared[(batch_id, condition)] = spec
            manifest.append({"batch_id": batch_id, "condition": condition, "state_hash": spec["state_hash"], "path": spec["path"]})
    write_json(output / "D_CONFIRM_STATE_MANIFEST.json", manifest)
    return prepared


def _analyze(output: Path) -> None:
    existing = output / "D_CONFIRM_DECISION.md"
    if existing.is_file() and "D_CONFIRM_BLOCKED" in existing.read_text(encoding="utf-8"):
        print("decision already written")
        return
    pools = json.loads((output / "D_CONFIRM_SOURCE_PLAN.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "D_CONFIRM_CONTROL_AUDIT.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output / "D_CONFIRM_TASK_RESULTS.jsonl").read_text().splitlines() if line.strip()]
    future_ids = [task["task_id"] for task in pools["evaluation"] if task["probe_role"] == "future"]
    retention_ids = [task["task_id"] for task in pools["evaluation"] if task["probe_role"] == "retention"]
    summary = json.loads((output / "D_CONFIRM_SOURCE_RESULTS_SUMMARY.json").read_text(encoding="utf-8"))
    batches = _tables(rows, pools, summary)
    effects = [batch_effects(batch, future_ids, retention_ids) for batch in batches]
    bootstrap = cross_bootstrap(batches, future_ids, retention_ids)
    loo = leave_one_batch_out(batches, future_ids, retention_ids)
    task_loo = leave_one_task_out(batches, future_ids, retention_ids)
    report = {
        "controls_valid": bool(audit.get("controls_valid")),
        "control_blocked": bool(audit.get("blocked")),
        "execution_blocked": False,
        "bootstrap": bootstrap,
        "robustness_ok": robustness_ok(effects, loo) if bootstrap.get("estimable") else False,
        "single_batch_or_task": positive_batch_count(effects) <= 1,
    }
    label = decide(report)
    _csv(output / "D_CONFIRM_BATCH_EFFECTS.csv", [{"batch_id": row["batch_id"], **{name: row["gaps"][name] for name in PROTOCOL["primary_estimates"]}} for row in effects])
    _csv(output / "D_CONFIRM_OVERALL_EFFECTS.csv", [{"contrast": name, **(bootstrap.get("intervals") or {}).get(name, {})} for name in PROTOCOL["primary_estimates"]])
    _csv(output / "D_CONFIRM_SIMULTANEOUS_INTERVALS.csv", [{"contrast": name, "low": (bootstrap.get("intervals") or {}).get(name, {}).get("simultaneous_low"), "high": (bootstrap.get("intervals") or {}).get(name, {}).get("simultaneous_high")} for name in PROTOCOL["primary_estimates"]])
    _csv(output / "D_CONFIRM_LEAVE_ONE_BATCH_OUT.csv", loo)
    _csv(output / "D_CONFIRM_LEAVE_ONE_TASK_OUT.csv", task_loo)
    (output / "D_CONFIRM_DECISION.md").write_text(f"# 决定\n\n`{label}`\n\nformal_claim 仍是 false。这不是条件经验策略的 GO。\n", encoding="utf-8")
    print(label)


def _tables(rows: list[dict], pools: dict, summary: dict) -> list[dict]:
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        if row.get("outcome") is None or row.get("infra_status") != "ok":
            continue
        grouped.setdefault((row["batch_id"], row["condition"], row["probe_role"], row["task_id"]), []).append(float(row["outcome"]))
    written = {item["batch_id"]: item["n_written"] for item in summary["batches"]}
    repeats = int(PROTOCOL["repeats"])
    batches = []
    for batch in pools["batches"]:
        future: dict[str, dict] = {}
        retention: dict[str, dict] = {}
        for task in pools["evaluation"]:
            target = future if task["probe_role"] == "future" else retention
            target[task["task_id"]] = {}
            for condition in PROTOCOL["conditions"]:
                values = grouped.get((batch["batch_id"], condition, task["probe_role"], task["task_id"]))
                target[task["task_id"]][condition] = sum(values) / len(values) if values and len(values) == repeats else None
        batches.append({"batch_id": batch["batch_id"], "n_written": written.get(batch["batch_id"], 0), "future": future, "retention": retention})
    return batches


def _reanalysis(output: Path) -> None:
    from expolicy.paper_d.confirm_utility import content_utility_gap, update_utility

    path = REPO_ROOT / "outputs/paper_d_content_os/20261014-content-os/D_CONTENT_TASK_RESULTS.jsonl"
    lines = [
        "# 旧 content-os-v1 效用重算",
        "",
        "这是只读复算，不覆盖原始 CSV，也不改写旧决定。旧 future 差不受这个非线性定义影响。",
        "旧 CLI 对 REAL−对照的 retention 差直接取惩罚。正确做法是每个 batch 先算相对 NO_UPDATE 的 u，再对两个 u 的差做等权平均。",
        "",
        "反例：REAL 的未来差 0、retention 差 +0.1，u=0；EMPTY 的未来差 0、retention 差 −0.1，u=−0.05。内容效用差是 +0.05。直接惩罚 REAL−EMPTY 的 retention 差会得到 0。",
        "反例：两个 batch 的 retention 差是 +0.2 和 −0.2，未来差都是 0。逐 batch 惩罚后平均 u 是 −0.05，不是先把 retention 平均成 0 再得到 u=0。",
        "",
    ]
    if path.is_file():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        grouped: dict[tuple, list[float]] = {}
        for row in rows:
            if row.get("block_id") != "main" or row.get("outcome") is None:
                continue
            grouped.setdefault((row["batch_id"], row["condition"], row["probe_role"], row["task_id"]), []).append(float(row["outcome"]))
        def mean(batch, condition, role, tasks):
            values = []
            for task in tasks:
                vals = grouped.get((batch, condition, role, task))
                if not vals:
                    return None
                values.append(sum(vals) / len(vals))
            return sum(values) / len(values)
        tasks = {}
        for (batch, _condition, role, task), _values in grouped.items():
            tasks.setdefault((batch, role), set()).add(task)
        utilities = {name: [] for name in ("REAL", "EMPTY", "MATCHED_WRONG")}
        for batch in ("os-B1", "os-B2"):
            lines.append(f"## {batch}")
            for condition in utilities:
                future = mean(batch, condition, "future", tasks[(batch, "future")]) - mean(batch, "NO_UPDATE", "future", tasks[(batch, "future")])
                retention = mean(batch, condition, "retention", tasks[(batch, "retention")]) - mean(batch, "NO_UPDATE", "retention", tasks[(batch, "retention")])
                utilities[condition].append(update_utility(future, retention))
                lines.append(f"- {condition}: ΔF={future:.4f}, ΔR={retention:.4f}, u={utilities[condition][-1]:.4f}")
        lines.append("")
        lines.append(f"两批等权 u(REAL)−u(EMPTY)={content_utility_gap(utilities['REAL'], utilities['EMPTY']):.4f}")
        lines.append(f"两批等权 u(REAL)−u(WRONG)={content_utility_gap(utilities['REAL'], utilities['MATCHED_WRONG']):.4f}")
        lines.append(f"两批等权 u(REAL)={sum(utilities['REAL'])/2:.4f}")
        lines.append("")
    (output / "D_V1_CONTENT_UTILITY_REANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    import csv
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
