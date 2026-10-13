"""Harness run of the C protocol. Public environments stay unrun."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from expolicy.paper_c.artifacts import write_products
from expolicy.paper_c.assimilator import SyntheticContentAssimilator
from expolicy.paper_c.audit import audit_assimilator, build_gain_audit
from expolicy.paper_c.config import load_config
from expolicy.paper_c.constants import (
    ACCOUNT_COMPUTE,
    BASELINE_POLICIES,
    ENVS,
    PILOT_POLICIES,
    STATUS_OK,
    M,
)
from expolicy.paper_c.curves import (
    attach_offline_scores,
    aulc_by_sequence,
    evaluate_frozen_curves,
    final_by_sequence,
)
from expolicy.paper_c.decision import build_decision
from expolicy.paper_c.features import (
    AcquisitionHistory,
    env_feature_names,
    model_feature_names,
    pre_acquisition_features,
    visibility_report,
)
from expolicy.paper_c.gates import (
    front_door_open,
    judge_c0,
    judge_c1,
    judge_c2,
    judge_pilot,
    judge_transfer,
)
from expolicy.paper_c.ledger import BudgetLedger
from expolicy.paper_c.manifest import build_manifest, write_manifest
from expolicy.paper_c.policies import fit_best_static, freeze_baseline, make_policy
from expolicy.paper_c.predictor import (
    correlation,
    fit_gain_predictor,
    grouped_holdout,
    unseen_family_advantage,
)
from expolicy.paper_c.sequence import run_learning_sequence
from expolicy.paper_c.stats import recovery_point
from expolicy.paper_c.world import FAMILIES, SyntheticWorld, make_snapshot, mismatch_of
from expolicy.r9_lab.state.persistent_state import PersistentState


def _history(spec: Mapping[str, Sequence[float]]) -> AcquisitionHistory:
    history = AcquisitionHistory()
    for family, outcomes in spec.items():
        for outcome in outcomes:
            history.observe(family, float(outcome))
    return history


def _snapshot_histories() -> dict[str, AcquisitionHistory]:
    flat = {family: (0.5, 0.5) for family in FAMILIES}
    return {
        "S0": _history(flat),
        "Sa": _history({"alpha": (0.0, 0.0), "beta": (1.0, 1.0), "gamma": (1.0, 1.0), "delta": (0.0, 1.0)}),
        "Sb": _history({"alpha": (1.0, 1.0), "beta": (0.0, 0.0), "gamma": (1.0, 1.0), "delta": (0.0, 1.0)}),
        "Sboth": _history({"alpha": (1.0, 0.0), "beta": (1.0, 0.0), "gamma": (1.0, 1.0), "delta": (0.0, 0.0)}),
    }


def _batches(world: SyntheticWorld, state: PersistentState) -> dict[str, list[list[dict[str, Any]]]]:
    pool = world.pool(per_family=8)
    batches: dict[str, list[list[dict[str, Any]]]] = {}
    for family, tasks in pool.items():
        family_batches = []
        for batch_index in (0, 1):
            template = "wording_a" if batch_index == 0 else "wording_b"
            chunk = tasks[batch_index * 4 : (batch_index + 1) * 4]
            rows = []
            for task in chunk:
                episode = world.rollout(state, task)
                episode["source_template"] = template
                rows.append(episode)
            family_batches.append(rows)
        batches[family] = family_batches
    return batches


def _score_factory(world: SyntheticWorld):
    tables = {
        "shadow_select": world.shadow_select,
        "shadow_eval": world.shadow_eval,
        "retention": world.retention,
    }

    def score(state: PersistentState, role: str) -> float | None:
        if role not in tables:
            raise RuntimeError(f"audit requested a probe that is not a training probe: {role}")
        return world.score(state, tables[role])

    return score


def _paired(scores: Mapping[str, Mapping[int, float | None]], left: str, right: str, seeds: Sequence[int]) -> list[float]:
    diffs = []
    for seed in seeds:
        a = scores.get(left, {}).get(seed)
        b = scores.get(right, {}).get(seed)
        if a is None or b is None:
            continue
        diffs.append(float(a) - float(b))
    return diffs


def run_self_check(output_dir: Path | None = None) -> dict[str, Any]:
    cfg = load_config()
    directory = output_dir or (cfg.output_root() / "self_check")
    work = directory / "work"
    work.mkdir(parents=True, exist_ok=True)
    world = SyntheticWorld(tag="c-harness")
    assimilator = SyntheticContentAssimilator(max_entries=int(cfg.raw["assimilator"]["capacity"]["max_entries"]))
    validation_ids = [task.task_id for task in world.shadow_eval + world.retention + world.future]
    manifest = build_manifest(
        assimilator,
        validation_ids=validation_ids,
        resource_limits=cfg.raw["assimilator"]["resource_limits"],
        capacity=cfg.raw["assimilator"]["capacity"],
        hyperparameters={
            "retention_weight": cfg.raw["metric"]["retention_weight"],
            "epsilon": cfg.raw["exploration"]["epsilon"],
            "ridge_alpha": cfg.raw["c2"]["ridge_alpha"],
            "label_route": cfg.raw["label_route"],
        },
    )
    write_manifest(directory / "C_ASSIMILATOR_MANIFEST.json", manifest)

    ledger = BudgetLedger()
    start = make_snapshot(work / "c0-start", {})
    c0_raw = audit_assimilator(
        start,
        world.passive_alpha_batch(),
        assimilator,
        _score_factory(world),
        work_dir=work / "c0",
        mismatch_of=mismatch_of,
        ledger=ledger,
        retention_weight=0.5,
    )
    c0 = judge_c0(c0_raw["arms"], min_gain=0.03, retention_tolerance=0.01)

    histories = _snapshot_histories()
    counts = {"S0": {}, "Sa": {"alpha": 1}, "Sb": {"beta": 1}, "Sboth": {"alpha": 1, "beta": 1}}
    snapshots = []
    for name, skill_counts in counts.items():
        state = make_snapshot(work / "snap" / name, skill_counts)
        state.extra["snapshot_id"] = name
        state.save()
        snapshots.append((name, state))
    neutral = snapshots[0][1]

    def feature_fn(state: PersistentState, family: str) -> dict[str, Any]:
        snapshot_id = str(state.extra.get("snapshot_id") or "S0")
        return pre_acquisition_features(
            state,
            family=family,
            env=world.env,
            families=FAMILIES,
            public={"public_difficulty": 0.5, "public_estimated_tools": 1.0},
            history=histories[snapshot_id],
            declared_envs=ENVS,
        )

    matrix = build_gain_audit(
        snapshots,
        FAMILIES,
        _batches(world, neutral),
        assimilator,
        _score_factory(world),
        feature_fn,
        work_dir=work / "gain",
        ledger=ledger,
        select_ids=[task.task_id for task in world.shadow_select],
        eval_ids=[task.task_id for task in world.shadow_eval],
        future_ids=[task.task_id for task in world.future],
    )
    c1 = judge_c1(matrix["summary"])
    rows = matrix["rows"]
    names = model_feature_names(FAMILIES, ENVS)
    alpha = float(cfg.raw["c2"]["ridge_alpha"])
    min_rows = int(cfg.raw["c2"]["min_rows"])
    state_holdout = grouped_holdout(rows, names, group_key="sequence_id", alpha=alpha, min_rows=min_rows)
    template_holdout = grouped_holdout(rows, names, group_key="source_template", alpha=alpha, min_rows=min_rows)
    family_holdout = unseen_family_advantage(rows, names, alpha=alpha, min_rows=min_rows)
    env_holdout = grouped_holdout(
        rows, env_feature_names(ENVS), group_key="sequence_id", alpha=alpha, min_rows=min_rows
    )
    failure_holdout = grouped_holdout(rows, ("hist_fail",), group_key="sequence_id", alpha=alpha, min_rows=min_rows)
    fitted = fit_gain_predictor(rows, names, alpha=alpha, min_rows=min_rows)
    ledger.add(account=ACCOUNT_COMPUTE, kind="offline_gain_fit", detail="c_train")
    c2 = judge_c2(
        state_advantage=state_holdout.get("mean_advantage"),
        family_excess=family_holdout.get("mean_excess"),
        env_advantage=env_holdout.get("mean_advantage"),
        failure_advantage=failure_holdout.get("mean_advantage"),
        state_groups=str(state_holdout.get("status")),
        margin=float(cfg.raw["c2"]["ablation_margin"]),
        correlation_value=correlation(fitted.get("model"), rows),
    )
    doors = front_door_open(c0, c1, c2)
    weights = fit_best_static(rows)
    logs: list[dict[str, Any]] = []
    sequence_ledger_rows: list[dict[str, Any]] = []
    dev_seeds = (1, 2)
    test_seeds = (3, 4, 5) if doors else (3,)
    n_rounds = 6 if doors else 2
    tasks_per_round = 4 if doors else 2
    feedback = (2, 4, 6) if doors else (2,)
    eval_rounds = (0, 2, 4, 6) if doors else (0, 2)
    sequence_pool = world.pool(per_family=n_rounds * tasks_per_round, templates=("wording_a",))
    start_state = make_snapshot(work / "seq-start", {})

    def run_seeds(seeds: Sequence[int], label: str) -> list[Any]:
        produced: list[Any] = []
        for seed in seeds:
            for name in PILOT_POLICIES:
                policy = make_policy(
                    name,
                    np.random.default_rng(seed * 100 + len(name)),
                    epsilon=float(cfg.raw["exploration"]["epsilon"]),
                    static_weights=weights,
                    model=fitted.get("model"),
                    feature_names=names,
                    train_rows=rows,
                    min_rows=min_rows,
                    alpha=alpha,
                )
                sequence_ledger = BudgetLedger()
                result = run_learning_sequence(
                    start_state,
                    policy,
                    world,
                    assimilator,
                    sequence_pool,
                    work_dir=work / label / str(seed) / name,
                    ledger=sequence_ledger,
                    run_id=f"c-harness-{label}",
                    sequence_seed=seed,
                    families=FAMILIES,
                    declared_envs=ENVS,
                    n_rounds=n_rounds,
                    tasks_per_round=tasks_per_round,
                    feedback_rounds=feedback,
                    eval_rounds=eval_rounds,
                )
                logs.extend(result["logs"])
                produced.extend(result["checkpoints"])
                for row in sequence_ledger.rows:
                    copied = dict(row)
                    copied["sequence_seed"] = seed
                    copied["policy"] = name
                    sequence_ledger_rows.append(copied)
        return produced

    dev_checks = run_seeds(dev_seeds, "dev")
    dev_curves = evaluate_frozen_curves(
        dev_checks,
        future_fn=lambda state: world.score(state, world.future),
        shadow_fn=lambda state: world.score(state, world.shadow_eval),
        retention_fn=lambda state: world.score(state, world.retention),
        ledger=ledger,
    )
    dev_aulc = aulc_by_sequence(dev_curves, policies=PILOT_POLICIES, cost_key="interaction_cost")
    dev_means = {}
    for name in BASELINE_POLICIES:
        values = [value for value in dev_aulc.get(name, {}).values() if value is not None]
        dev_means[name] = None if not values else sum(values) / len(values)
    baseline = freeze_baseline(dev_means, BASELINE_POLICIES)
    test_checks = run_seeds(test_seeds, "test")
    test_curves = evaluate_frozen_curves(
        test_checks,
        future_fn=lambda state: world.score(state, world.future),
        shadow_fn=lambda state: world.score(state, world.shadow_eval),
        retention_fn=lambda state: world.score(state, world.retention),
        ledger=ledger,
    )
    curves = dev_curves + test_curves
    logs = attach_offline_scores(logs, curves)
    cost_scores = aulc_by_sequence(test_curves, policies=PILOT_POLICIES, cost_key="interaction_cost")
    episode_scores = aulc_by_sequence(test_curves, policies=PILOT_POLICIES, cost_key="episode_cost")
    future_scores = final_by_sequence(test_curves, score_key="future_score")
    retention_scores = final_by_sequence(test_curves, score_key="retention")
    pilot = judge_pilot(
        cost_diffs=_paired(cost_scores, M, baseline, test_seeds),
        episode_diffs=_paired(episode_scores, M, baseline, test_seeds),
        final_diffs=_paired(future_scores, M, baseline, test_seeds),
        retention_diffs=_paired(retention_scores, M, baseline, test_seeds),
        baseline=baseline,
        min_aulc=float(cfg.raw["gates"]["min_aulc"]),
        min_final=float(cfg.raw["gates"]["min_final_success"]),
        retention_tolerance=float(cfg.raw["gates"]["retention_tolerance"]),
        wide=float(cfg.raw["gates"]["wide_interval"]),
        n_public_environments=len(cfg.raw["public_environments"]),
        environment_class="synthetic",
    )
    transfer = judge_transfer(None)
    m_costs = [row["interaction_cost"] for row in test_curves if row["policy"] == M]
    base_costs = [row["interaction_cost"] for row in test_curves if row["policy"] == baseline]
    final_gap = _paired(future_scores, M, baseline, test_seeds)
    comparable = bool(final_gap) and max(abs(item) for item in final_gap) <= 0.02
    c_save = 0.0
    if m_costs and base_costs:
        c_save = (sum(base_costs) / len(base_costs)) - (sum(m_costs) / len(m_costs))
    recovered = recovery_point(ledger.interaction_total(), c_save) if comparable else None
    decision = build_decision(
        {
            "c0": c0,
            "c1": c1,
            "c2": c2,
            "pilot": pilot,
            "transfer": transfer,
            "front_door_open": doors,
            "environment_class": "synthetic",
            "n_public_environments": 0,
        }
    )
    decision["recovery_point"] = recovered
    failures = [row for row in rows if row.get("infra_status") != STATUS_OK]
    ledger_rows = []
    for row in list(ledger.rows) + sequence_ledger_rows:
        copied = dict(row)
        copied["run_id"] = "c-harness"
        ledger_rows.append(copied)
    written = write_products(
        directory,
        protocol=cfg.raw,
        manifest=manifest,
        decision=decision,
        assimilator_audit=c0_raw,
        c0_judgment=c0,
        gain_rows=rows,
        visibility=visibility_report(FAMILIES, ENVS),
        curves=curves,
        ledger_rows=ledger_rows,
        logs=logs,
        failures=failures,
    )
    return {
        "decision": decision,
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "pilot": pilot,
        "baseline": baseline,
        "doors": doors,
        "paths": written,
        "n_gain_rows": len(rows),
        "state_advantage": state_holdout.get("mean_advantage"),
        "family_excess": family_holdout.get("mean_excess"),
        "template_advantage": template_holdout.get("mean_advantage"),
        "recovery_point": recovered,
    }
