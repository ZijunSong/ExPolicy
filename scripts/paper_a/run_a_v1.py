#!/usr/bin/env python3
"""Run A-v1 stages. Measurement, matrix, and sequence stay idle unless --execute is set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from expolicy.paper_a.artifacts import append_ledger, write_failures, write_manifest, write_summary, write_table, write_text
from expolicy.paper_a.config import load_config
from expolicy.paper_a.constants import PILOT_VARIANTS
from expolicy.paper_a.crossfit import crossfit_action_selection
from expolicy.paper_a.decision import build_decision, render_audit_md, render_decision_md
from expolicy.paper_a.groups import split_by_source_group
from expolicy.paper_a.inventory import inventory_legacy_matrix
from expolicy.paper_a.matrix import to_router_records
from expolicy.paper_a.router import fit_ridge_router
from expolicy.paper_a.schema import read_rows_csv
from expolicy.r9_lab.benchmark.split import all_bucket_ids, load_split
from expolicy.r9_lab.io_util import read_json, write_json


def _out(cfg) -> Path:
    path = cfg.output_root()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_decision(cfg, out: Path) -> None:
    audit = _read_if(out / "audit.json")
    space = _read_if(out / "crossfit.json")
    router = _read_if(out / "router.json")
    sequence = _read_if(out / "sequence.json")
    decision = build_decision(
        audit=audit,
        space=space,
        router=router,
        sequence=sequence,
        min_gain=float(cfg.raw["decision"]["min_gain_vs_env_only"]),
        retention_tolerance=float(cfg.raw["decision"]["retention_tolerance"]),
    )
    write_text(out / "A_DECISION.md", render_decision_md(decision, audit=audit, space=space, router=router, sequence=sequence))
    write_summary(out / "decision_summary.json", decision)


def _read_if(path: Path):
    if not path.exists():
        return None
    return read_json(path)


def cmd_inventory(cfg) -> int:
    out = _out(cfg)
    legacy = inventory_legacy_matrix(cfg.path("legacy.r9_matrix"))
    write_json(out / "LEGACY_INDEX.json", {"splice_into_a_v1": False, "legacy": legacy})
    plan = {
        "protocol_id": cfg.protocol_id,
        "stages": ["inventory", "audit", "matrix", "analyze", "sequence", "decide"],
        "execute_required_for": ["audit", "matrix", "sequence"],
        "actions": list(cfg.actions),
        "excluded_actions": list(cfg.raw["excluded_actions"]),
        "legacy_cells": legacy["n_cells"],
        "legacy_spliced": False,
        "budget_probe_episodes_before_approval": cfg.raw["budget"]["probe_episodes_before_remaining_approval"],
        "note": "Inventory does not score probes. Historical R9 aggregates stay in the legacy index.",
    }
    write_json(out / "PLAN.json", plan)
    _write_decision(cfg, out)
    print(json.dumps({"inventory": True, "legacy_cells": legacy["n_cells"], "decision": str(out / "A_DECISION.md")}))
    return 0


def cmd_analyze(cfg, *, attribution: bool) -> int:
    out = _out(cfg)
    matrix_path = out / "A_UTILITY_MATRIX.csv"
    feature_path = out / "features.json"
    if not matrix_path.exists() or not feature_path.exists():
        print("analyze needs A_UTILITY_MATRIX.csv and features.json from a completed matrix run")
        return 2
    rows = read_rows_csv(matrix_path)
    if rows and "probe_task_id" not in rows[0]:
        print("matrix rows have no probe_task_id; refusing to treat an aggregate table as A-v1")
        return 2
    features = read_json(feature_path)
    frozen = read_json(cfg.path("frozen_batches.path"))
    split_payload = load_split(cfg.path("split.path"))
    forbidden = all_bucket_ids(split_payload, "future_test")
    grouped = split_by_source_group(
        frozen["batches"],
        n_outer=int(cfg.raw["router"]["outer_folds"]),
        seed=int(cfg.raw["split"]["expect_seed"]),
        forbidden_task_ids=forbidden,
    )
    group_of = {batch["batch_id"]: batch["source_group"] for batch in grouped.batches}
    for row in rows:
        if row["batch_id"] in group_of:
            row["source_group"] = group_of[row["batch_id"]]
    space = crossfit_action_selection(
        rows,
        equivalence=float(cfg.raw["measurement"]["equivalence_success"]),
        delta_floor=float(cfg.raw["space"]["delta_floor"]),
        resolution_tasks=float(cfg.raw["space"]["resolution_tasks"]),
        min_positive_batches=int(cfg.raw["space"]["min_positive_batches"]),
        min_non_discard_actions=int(cfg.raw["space"]["min_non_discard_actions"]),
        min_envs_with_crossover=int(cfg.raw["space"]["min_envs_with_crossover"]),
        n_boot=int(cfg.raw["router"]["bootstrap_replicates"]),
        seed=int(cfg.raw["split"]["expect_seed"]),
    )
    records = to_router_records(rows, features)
    for record in records:
        if record["batch_id"] in group_of:
            record["source_group"] = group_of[record["batch_id"]]
    variants = list(cfg.raw["router"]["pilot_variants"])
    if attribution:
        variants.extend(cfg.raw["router"]["attribution_variants"])
    else:
        variants = [variant for variant in variants if variant in PILOT_VARIANTS or variant in cfg.raw["router"]["pilot_variants"]]
    router = fit_ridge_router(
        records,
        grouped,
        variants=variants,
        alpha=float(cfg.raw["router"]["ridge_alpha"]),
        n_boot=int(cfg.raw["router"]["bootstrap_replicates"]),
        bootstrap_quantile=float(cfg.raw["router"]["bootstrap_quantile"]),
        threshold_grid=list(cfg.raw["router"]["threshold_grid"]),
        default_threshold=float(cfg.raw["router"]["default_threshold"]),
        rules=list(cfg.raw["simple_rule"]),
        seed=int(cfg.raw["split"]["expect_seed"]),
        min_tree_samples=int(cfg.raw["router"]["min_tree_samples"]),
        primary=tuple(cfg.raw["router"]["primary_comparison"]),
    )
    space["feature_policy_gain"] = router.get("feature_policy_gain")
    write_json(out / "crossfit.json", {key: value for key, value in space.items() if key != "rows"})
    write_table(out / "A_CROSSFIT_RESULTS.csv", space["rows"])
    router_rows = []
    for fold in router["folds"]:
        for payload in fold["variants"].values():
            for outcome in payload["outcomes"]:
                router_rows.append({"fold": fold["test_fold"], **outcome})
    from expolicy.paper_a.artifacts import _jsonable

    write_json(out / "router.json", _jsonable(router))
    write_table(out / "A_ROUTER_RESULTS.csv", _router_as_matrix_rows(cfg, router_rows))
    _write_decision(cfg, out)
    print(json.dumps({"analyze": True, "space": space["status"], "prediction": router["primary"]["diff"]}))
    return 0


def _router_as_matrix_rows(cfg, outcomes: list[dict]) -> list[dict]:
    rows = []
    for outcome in outcomes:
        rows.append(
            {
                "protocol_id": cfg.protocol_id,
                "model_hash": cfg.raw["model"]["expect_manifest_content_hash"],
                "env": outcome.get("env"),
                "source_group": outcome.get("source_group"),
                "batch_id": outcome.get("batch_id"),
                "action": outcome.get("chosen_action"),
                "probe_task_id": f"variant:{outcome.get('variant')}:fold:{outcome.get('fold')}",
                "eval_split": "outer_test",
                "utility": outcome.get("utility"),
                "delta_shadow": outcome.get("delta_shadow"),
                "delta_retention": outcome.get("delta_retention"),
                "infra_status": "ok" if outcome.get("utility") is not None else "missing",
                "legacy": False,
            }
        )
    return rows


def _require_execute(stage: str, execute: bool) -> int | None:
    if execute:
        return None
    print(f"{stage} scores the model and the environments. Re-run with --execute when compute is available.")
    return 2


def _open_client(cfg):
    from expolicy.r9_lab.constants import model_by_id, vllm_port_for_model
    from expolicy.r9_lab.llm_client import VLLMClient, probe_vllm

    spec = model_by_id(str(cfg.raw["model"]["model_id"]))
    port = vllm_port_for_model(spec.model_id)
    probe_vllm(port)
    return VLLMClient.from_model_id(spec.model_id, port=port, timeout=600.0)


def _write_budget(cfg, out: Path, measured: dict) -> None:
    from expolicy.paper_a.execute import plan_episode_counts

    plan = plan_episode_counts(cfg)
    per = float(measured["elapsed_s_per_episode"])
    projection = {
        "status": "APPROVAL_REQUIRED",
        "measured": {key: value for key, value in measured.items() if key != "items"},
        "plan": plan,
        "projected_audit_seconds": per * plan["audit_episodes"],
        "projected_matrix_seconds": per * plan["matrix_episodes"],
        "projected_sequence_seconds": per * plan["sequence_episodes"],
        "note": "20-episode probe only. Re-run the same stage with --approve-remaining to score the rest. No GO/NO_GO is claimed from this probe.",
    }
    write_json(out / "BUDGET.json", projection)
    append_ledger(out / "A_COST_LEDGER.jsonl", [projection["measured"]])


def _env_report(env: str, env_reports: list[dict]) -> dict:
    if not env_reports:
        return {"env": env, "batches": [], "status": "MEASUREMENT_INCONCLUSIVE", "reasons": ["no batches"], "penalty": {}}
    statuses = [item.get("status") for item in env_reports]
    status = "PASS" if all(item == "PASS" for item in statuses) else next(item for item in statuses if item != "PASS")
    return {
        "env": env,
        "batches": env_reports,
        "status": status,
        "reasons": [reason for item in env_reports for reason in item.get("reasons") or []],
        "penalty": env_reports[0].get("penalty") or {},
    }


def _write_audit_snapshot(out: Path, reports: list[dict], hashes: dict) -> None:
    statuses = [report.get("status") for report in reports]
    if statuses and all(item == "PASS" for item in statuses):
        status = "PASS"
    elif any(item == "FAIL_SYSTEMATIC" for item in statuses):
        status = "FAIL_SYSTEMATIC"
    elif statuses:
        status = "MEASUREMENT_INCONCLUSIVE"
    else:
        status = "NOT_RUN"
    write_json(
        out / "audit_partial.json",
        {
            "status": status,
            "complete": False,
            "reasons": [reason for report in reports for reason in report.get("reasons") or []],
            "envs": reports,
            "model_hash": hashes.get("model_hash"),
        },
    )


def cmd_audit(cfg, *, execute: bool, approve_remaining: bool) -> int:
    blocked = _require_execute("audit", execute)
    if blocked is not None:
        return blocked
    from expolicy.paper_a.execute import (
        asset_cache_key,
        load_acquired_experiences,
        measure_probe_cost,
        probe_tasks,
        run_measurement_audit,
    )
    from expolicy.r9_lab.benchmark.official_env import OfficialLabEnv, OfficialLabInfraError
    from expolicy.r9_lab.benchmark.split import load_split
    from expolicy.r9_lab.state.persistent_state import PersistentState

    out = _out(cfg)
    _key, hashes = asset_cache_key(cfg)
    client = _open_client(cfg)
    frozen = read_json(cfg.path("frozen_batches.path"))
    per_env = int(cfg.raw["measurement"]["audit_batches_per_env"])
    if not approve_remaining:
        env = cfg.environments[0]
        lab = OfficialLabEnv(env)
        try:
            s0 = PersistentState.load(cfg.path("legacy.r9_matrix") / "_s0" / env / "state")
            measured = measure_probe_cost(
                lab,
                client,
                s0,
                probe_tasks(load_split(cfg.path("split.path")), env),
                n=int(cfg.raw["budget"]["probe_episodes_before_remaining_approval"]),
                work_dir=out / "budget_probe",
            )
        finally:
            lab.close()
        _write_budget(cfg, out, measured)
        print(json.dumps({"budget": "APPROVAL_REQUIRED", "elapsed_s": measured["elapsed_s"], "tokens": measured["tokens"]}))
        return 0
    reports = []
    for env in cfg.environments:
        chosen = [batch for batch in frozen["batches"] if batch["env"] == env][:per_env]
        if len(chosen) < per_env:
            print(f"audit needs {per_env} batches for {env}")
            return 2
        try:
            lab = OfficialLabEnv(env)
        except OfficialLabInfraError as exc:
            reports.append(
                {
                    "env": env,
                    "batches": [],
                    "status": "MEASUREMENT_INCONCLUSIVE",
                    "reasons": [f"environment unavailable: {exc}"],
                    "penalty": {},
                }
            )
            _write_audit_snapshot(out, reports, hashes)
            print(json.dumps({"env": env, "skipped": str(exc)}))
            continue
        try:
            s0 = PersistentState.load(cfg.path("legacy.r9_matrix") / "_s0" / env / "state")
            env_reports = []
            for batch in chosen:
                report_path = out / "audit" / env / batch["batch_id"] / "report.json"
                if report_path.exists():
                    env_reports.append(read_json(report_path))
                    continue
                experiences = load_acquired_experiences(cfg.path("legacy.r9_matrix"), batch["batch_id"])
                if not experiences:
                    print(f"missing acquisition for {batch['batch_id']}")
                    return 2
                print(json.dumps({"audit_batch": batch["batch_id"], "env": env}), flush=True)
                report = run_measurement_audit(
                    cfg,
                    env=env,
                    experiences=experiences,
                    lab=lab,
                    client=client,
                    work_dir=out / "audit" / env / batch["batch_id"],
                    model_identity=hashes,
                    s0=s0,
                )
                write_json(report_path, report)
                env_reports.append(report)
                reports_so_far = reports + [
                    _env_report(env, env_reports)
                ]
                _write_audit_snapshot(out, reports_so_far, hashes)
        finally:
            lab.close()
        reports.append(_env_report(env, env_reports))
    status = "PASS" if reports and all(report["status"] == "PASS" for report in reports) else (
        "FAIL_SYSTEMATIC" if any(report["status"] == "FAIL_SYSTEMATIC" for report in reports) else "MEASUREMENT_INCONCLUSIVE"
    )
    merged = {"status": status, "reasons": [reason for report in reports for reason in report["reasons"]], "envs": reports, "penalty": reports[0]["penalty"]}
    write_json(out / "audit.json", merged)
    write_text(out / "A_MEASUREMENT_AUDIT.md", render_audit_md(merged))
    write_manifest(out / "manifest.json", {"protocol_id": cfg.protocol_id, "stage": "audit", **hashes})
    _write_decision(cfg, out)
    print(json.dumps({"audit": status}))
    return 0


def _a_v1_measurement_gate(audit_root: Path) -> str | None:
    reports = sorted(audit_root.rglob("report.json")) if audit_root.exists() else []
    if not reports:
        return "A-v1 matrix refused: measurement audit reports are missing"
    blocked = []
    for path in reports:
        status = str(read_json(path).get("status"))
        if status != "PASS":
            blocked.append(f"{path.parent.name}:{status}")
    if blocked:
        return "A-v1 matrix refused because measurement is not passed: " + ", ".join(blocked)
    return None


def cmd_matrix(cfg, *, execute: bool, approve_remaining: bool) -> int:
    blocked = _require_execute("matrix", execute)
    if blocked is not None:
        return blocked
    gate = _a_v1_measurement_gate(cfg.output_root() / "audit")
    if gate:
        print(gate)
        return 2
    if not approve_remaining:
        print("matrix was not started. Run `audit --execute` first; it measures 20 episodes and writes BUDGET.json. Then re-run matrix with --execute --approve-remaining.")
        return 2
    from expolicy.paper_a.execute import asset_cache_key, build_live_matrix, load_capability
    from expolicy.r9_lab.benchmark.official_env import OfficialLabEnv
    from expolicy.r9_lab.state.persistent_state import PersistentState

    out = _out(cfg)
    key, hashes = asset_cache_key(cfg)
    client = _open_client(cfg)
    frozen = read_json(cfg.path("frozen_batches.path"))
    labs = {}
    try:
        for env in cfg.environments:
            labs[env] = OfficialLabEnv(env)
        s0 = {env: PersistentState.load(cfg.path("legacy.r9_matrix") / "_s0" / env / "state") for env in cfg.environments}
        built = build_live_matrix(
            cfg,
            frozen["batches"],
            labs=labs,
            client=client,
            s0_by_env=s0,
            capability_by_env=load_capability(cfg),
            cache_dir=out / "cache",
            config_hash=key,
            model_hash=hashes["model_hash"],
        )
    finally:
        for lab in labs.values():
            lab.close()
    write_table(out / "A_UTILITY_MATRIX.csv", built["rows"])
    write_json(out / "features.json", built["features"])
    write_failures(out / "failures.jsonl", built["failures"])
    append_ledger(out / "A_COST_LEDGER.jsonl", built["costs"])
    write_json(out / "LEGACY_INDEX.json", {"splice_into_a_v1": False, "ignored_cells": built["legacy"]})
    write_manifest(out / "manifest.json", {"protocol_id": cfg.protocol_id, "stage": "matrix", "config_hash": key, **hashes})
    write_summary(out / "matrix_summary.json", {"n_rows": len(built["rows"]), "n_failures": len(built["failures"]), "config_hash": key})
    print(json.dumps({"matrix_rows": len(built["rows"]), "failures": len(built["failures"])}))
    return 0


def cmd_sequence(cfg, *, execute: bool, ignore_gate: bool, approve_remaining: bool) -> int:
    blocked = _require_execute("sequence", execute)
    if blocked is not None:
        return blocked
    out = _out(cfg)
    decision = _read_if(out / "decision_summary.json") or {}
    if not ignore_gate and decision.get("space_status") != "A-SPACE-GO":
        print("sequence stays idle until the space gate is A-SPACE-GO. Pass --ignore-gate only for a debugging run.")
        return 2
    if not approve_remaining:
        print("sequence was not started. Re-run with --execute --approve-remaining after the budget probe.")
        return 2
    from expolicy.paper_a.execute import asset_cache_key, load_capability, run_sequences
    from expolicy.paper_a.groups import split_by_source_group
    from expolicy.paper_a.matrix import to_router_records
    from expolicy.paper_a.router import fit_ridge_router
    from expolicy.paper_a.schema import read_rows_csv
    from expolicy.paper_a.sequence import EnvOnlySelector, FixedSelector, selector_from_router
    from expolicy.r9_lab.benchmark.official_env import OfficialLabEnv
    from expolicy.r9_lab.benchmark.split import all_bucket_ids, load_split
    from expolicy.r9_lab.constants import A0_DISCARD
    from expolicy.r9_lab.state.persistent_state import PersistentState

    if not (out / "A_UTILITY_MATRIX.csv").exists() or not (out / "features.json").exists():
        print("sequence needs the utility matrix and pre-decision features")
        return 2
    _key, hashes = asset_cache_key(cfg)
    rows = read_rows_csv(out / "A_UTILITY_MATRIX.csv")
    features = read_json(out / "features.json")
    frozen = read_json(cfg.path("frozen_batches.path"))
    split_payload = load_split(cfg.path("split.path"))
    grouped = split_by_source_group(
        frozen["batches"],
        n_outer=int(cfg.raw["router"]["outer_folds"]),
        seed=int(cfg.raw["split"]["expect_seed"]),
        forbidden_task_ids=all_bucket_ids(split_payload, "future_test"),
    )
    records = to_router_records(rows, features)
    group_of = {batch["batch_id"]: batch["source_group"] for batch in grouped.batches}
    for record in records:
        if record["batch_id"] in group_of:
            record["source_group"] = group_of[record["batch_id"]]
    router = fit_ridge_router(
        records,
        grouped,
        variants=["M", "B3"],
        alpha=float(cfg.raw["router"]["ridge_alpha"]),
        n_boot=int(cfg.raw["router"]["bootstrap_replicates"]),
        bootstrap_quantile=float(cfg.raw["router"]["bootstrap_quantile"]),
        threshold_grid=list(cfg.raw["router"]["threshold_grid"]),
        default_threshold=float(cfg.raw["router"]["default_threshold"]),
        rules=list(cfg.raw["simple_rule"]),
        seed=int(cfg.raw["split"]["expect_seed"]),
        min_tree_samples=int(cfg.raw["router"]["min_tree_samples"]),
    )
    fold = router["folds"][0]["variants"]["M"]
    if not fold.get("_models"):
        print("the frozen router has no train fold; sequence was not started")
        return 2
    env_only = fold["env_only"]
    means = {}
    for record in records:
        for action, value in (record.get("U") or {}).items():
            if value is not None:
                means.setdefault(action, []).append(float(value))
    best_global = max(means, key=lambda action: sum(means[action]) / len(means[action])) if means else A0_DISCARD
    selectors = {
        "M": selector_from_router(router, fold=0, variant="M"),
        "env-only": EnvOnlySelector(env_only),
        "best-global": FixedSelector(best_global),
        "DISCARD": FixedSelector(A0_DISCARD),
    }
    client = _open_client(cfg)
    labs = {}
    try:
        for env in cfg.environments:
            labs[env] = OfficialLabEnv(env)
        s0 = {env: PersistentState.load(cfg.path("legacy.r9_matrix") / "_s0" / env / "state") for env in cfg.environments}
        summary = run_sequences(
            cfg,
            labs=labs,
            client=client,
            s0_by_env=s0,
            selectors=selectors,
            capability_by_env=load_capability(cfg),
            work_dir=out / "sequence",
        )
    finally:
        for lab in labs.values():
            lab.close()
    write_json(out / "sequence.json", summary)
    write_manifest(out / "manifest.json", {"protocol_id": cfg.protocol_id, "stage": "sequence", **hashes, "trained_on_states": ["S0"]})
    _write_decision(cfg, out)
    print(json.dumps({"sequence": True, "ood_state": True}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A-v1 protocol runner")
    parser.add_argument("--config", type=Path, default=None)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("inventory")
    audit = sub.add_parser("audit")
    audit.add_argument("--execute", action="store_true")
    audit.add_argument("--approve-remaining", action="store_true")
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--execute", action="store_true")
    matrix.add_argument("--approve-remaining", action="store_true")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--attribution", action="store_true")
    sequence = sub.add_parser("sequence")
    sequence.add_argument("--execute", action="store_true")
    sequence.add_argument("--approve-remaining", action="store_true")
    sequence.add_argument("--ignore-gate", action="store_true")
    sub.add_parser("decide")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, verify_files=True)
    if args.stage == "inventory":
        return cmd_inventory(cfg)
    if args.stage == "audit":
        return cmd_audit(cfg, execute=args.execute, approve_remaining=args.approve_remaining)
    if args.stage == "matrix":
        return cmd_matrix(cfg, execute=args.execute, approve_remaining=args.approve_remaining)
    if args.stage == "analyze":
        return cmd_analyze(cfg, attribution=args.attribution)
    if args.stage == "sequence":
        return cmd_sequence(cfg, execute=args.execute, ignore_gate=args.ignore_gate, approve_remaining=args.approve_remaining)
    if args.stage == "decide":
        _write_decision(cfg, _out(cfg))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
