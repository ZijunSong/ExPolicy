"""CPU tests for the A-v1 protocol. No model server and no LAB docker."""

from __future__ import annotations

from pathlib import Path

import pytest

from expolicy.paper_a.audit import AuditTask, audit_noop
from expolicy.paper_a.budget import request_resample
from expolicy.paper_a.config import ConfigPinError, load_config
from expolicy.paper_a.constants import (
    A0_DISCARD,
    A1_EPISODIC,
    A2_SEMANTIC,
    ACTIONS,
    AUDIT_FAIL,
    AUDIT_INCONCLUSIVE,
    AUDIT_PASS,
    CONTEXT_FEATURES,
    EXPERIENCE_FEATURES,
    STATE_FEATURES,
)
from expolicy.paper_a.crossfit import crossfit_action_selection
from expolicy.paper_a.decision import build_decision, judge_prediction
from expolicy.paper_a.features import LeakageError, apply_simple_rule, predecision_features
from expolicy.paper_a.groups import SplitLeakageError, split_by_source_group
from expolicy.paper_a.inventory import inventory_legacy_matrix
from expolicy.paper_a.matrix import build_utility_matrix, summarize_cell, to_router_records
from expolicy.paper_a.router import _fit_fold_variant, fit_ridge_router, fit_shallow_tree
from expolicy.paper_a.sequence import DecisionView, FixedSelector, run_fixed_acquisition_sequence
from expolicy.r9_lab.assimilators.discard import apply_discard
from expolicy.r9_lab.assimilators.episodic import apply_episodic
from expolicy.r9_lab.assimilators.semantic import apply_semantic
from expolicy.r9_lab.state.persistent_state import PersistentState
from scripts.paper_a import run_a_v1


def _tasks() -> list[AuditTask]:
    rows = []
    for split, n in (("shadow_reward_probe", 4), ("retention_probe", 4)):
        for i in range(n):
            rows.append(AuditTask(f"db:{split}:{i}", split, f"instruction {split} {i}", "db"))
    return rows


def _experience() -> dict:
    return {
        "task_id": "db:acq:1",
        "env": "db",
        "instruction": "add the two account balances",
        "success": True,
        "verifier_outcome": "CORRECT",
        "trajectory": [{"agent": "Action: Answer\nFinal Answer: OK"}],
        "bucket": "acquisition_pool",
        "steps": 2,
        "parser_valid": True,
        "infra_failure": False,
    }


def _eval_constant(y: float):
    def evaluate(state, tasks, *, mode, repeat_id, concurrency, branch):  # noqa: ARG001
        return [{"probe_task_id": task.probe_task_id, "y": y, "infra_status": "ok"} for task in tasks]

    return evaluate


def test_config_pins_match_files_and_lock_actions():
    cfg = load_config()
    assert cfg.protocol_id == "A-v1"
    assert cfg.actions == ACTIONS
    assert cfg.raw["metric"]["subtract_discard"] is False
    raw = dict(cfg.raw)
    raw["actions"] = ["DISCARD", "SKILLOPT_EDIT"]
    broken = type(cfg)(raw=raw, root=cfg.root, config_path=cfg.config_path)
    with pytest.raises(ConfigPinError):
        from expolicy.paper_a.config import _validate_static

        _validate_static(broken)


def test_audit_identical_noop_passes_and_does_not_recenter(tmp_path: Path):
    state = PersistentState.create(tmp_path / "s0")
    report = audit_noop(
        state,
        _tasks(),
        _eval_constant(1.0),
        experiences=[_experience()],
        work_dir=tmp_path / "audit",
        model_identity={"model_id": "Q2"},
        system_prompt="You are a helpful assistant.",
        tool_config={"tools": ["sql"]},
        env_snapshot_id="db-snap",
        repeats=3,
        formal_concurrency=1,
        require_formal_concurrency_replay=True,
    )
    assert report.status == AUDIT_PASS
    assert report.to_dict()["uses_discard_recentering"] is False
    assert "retriever_order" in report.components["fingerprints"]["BASE"]
    assert "skill_hash" in report.components["fingerprints"]["BASE"]
    assert report.penalty["recentered_on_discard"] is False
    assert any(row.get("paired_delta") == 0 for row in report.task_rows)


def test_audit_component_mismatch_and_deterministic_score_gap_fail(tmp_path: Path):
    state = PersistentState.create(tmp_path / "s0")

    def prepare(name, branch, experiences):
        if name == "DISCARD":
            branch.skill_markdown = "mutated"
            branch.save()
        return {"changed": name == "DISCARD"}

    report = audit_noop(
        state,
        _tasks(),
        _eval_constant(1.0),
        experiences=[_experience()],
        work_dir=tmp_path / "audit",
        model_identity={"model_id": "Q2"},
        system_prompt="You are a helpful assistant.",
        tool_config={"tools": ["sql"]},
        env_snapshot_id="db-snap",
        repeats=2,
        formal_concurrency=1,
        prepare_branch=prepare,
    )
    assert report.status == AUDIT_FAIL
    assert report.components["diffs_vs_base"]

    def biased(state, tasks, *, mode, repeat_id, concurrency, branch):  # noqa: ARG001
        y = 0.0 if branch == "DISCARD" else 1.0
        return [{"probe_task_id": task.probe_task_id, "y": y, "infra_status": "ok"} for task in tasks]

    scored = audit_noop(
        PersistentState.create(tmp_path / "s1"),
        _tasks(),
        biased,
        experiences=[_experience()],
        work_dir=tmp_path / "audit2",
        model_identity={"model_id": "Q2"},
        system_prompt="You are a helpful assistant.",
        tool_config={"tools": ["sql"]},
        env_snapshot_id="db-snap",
        repeats=2,
        formal_concurrency=1,
    )
    assert scored.status == AUDIT_FAIL


def test_audit_wide_interval_is_inconclusive_not_a_pass(tmp_path: Path):
    pattern = {
        0: [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        1: [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0, -1.0],
    }

    def evaluate(state, tasks, *, mode, repeat_id, concurrency, branch):  # noqa: ARG001
        rows = []
        shadow_index = 0
        for task in tasks:
            if "shadow" not in task.probe_task_id:
                rows.append({"probe_task_id": task.probe_task_id, "y": 1.0, "infra_status": "ok"})
                continue
            desired = pattern[repeat_id][shadow_index]
            shadow_index += 1
            if branch == "BASE":
                y = 1.0 if desired < 0 else 0.0
            elif desired > 0:
                y = 1.0
            elif desired < 0:
                y = 0.0
            else:
                y = 0.0
            rows.append({"probe_task_id": task.probe_task_id, "y": y, "infra_status": "ok"})
        return rows

    tasks = []
    for i in range(8):
        tasks.append(AuditTask(f"db:shadow:{i}", "shadow_reward_probe", f"shadow {i}", "db"))
    for i in range(4):
        tasks.append(AuditTask(f"db:ret:{i}", "retention_probe", f"retention {i}", "db"))
    report = audit_noop(
        PersistentState.create(tmp_path / "s0"),
        tasks,
        evaluate,
        experiences=[_experience()],
        work_dir=tmp_path / "audit",
        model_identity={"model_id": "Q2"},
        system_prompt="You are a helpful assistant.",
        tool_config={},
        env_snapshot_id="db",
        repeats=2,
        equivalence=0.01,
        formal_concurrency=1,
    )
    assert report.status == AUDIT_INCONCLUSIVE


def test_negative_utility_is_stored_and_not_forced_to_zero():
    items = []
    for i in range(10):
        items.append(
            {
                "probe_task_id": f"s{i}",
                "eval_split": "shadow_reward_probe",
                "y_before": 1.0,
                "y_after": 0.0,
                "infra_status": "ok",
            }
        )
    for i in range(6):
        items.append(
            {
                "probe_task_id": f"r{i}",
                "eval_split": "retention_probe",
                "y_before": 1.0,
                "y_after": 0.0,
                "infra_status": "ok",
            }
        )
    summary = summarize_cell(items)
    assert summary["utility"] < 0
    assert summary["delta_shadow"] == pytest.approx(-1.0)
    assert summary["utility"] != 0


def test_matrix_cache_legacy_and_missing(tmp_path: Path):
    probes = [
        {"probe_task_id": "s", "eval_split": "shadow_reward_probe"},
        {"probe_task_id": "r", "eval_split": "retention_probe"},
    ]
    calls = {"n": 0}

    def evaluate(batch, action, probe_rows):
        calls["n"] += 1
        if action == A2_SEMANTIC:
            raise RuntimeError("backend down")
        return {
            "state_hash_before": "h0",
            "state_hash_after": "h1" if action != A0_DISCARD else "h0",
            "changed": action != A0_DISCARD,
            "accepted": action != A0_DISCARD,
            "items": [
                {
                    "probe_task_id": probe["probe_task_id"],
                    "eval_split": probe["eval_split"],
                    "y_before": 0.0,
                    "y_after": 1.0 if action == A1_EPISODIC else 0.0,
                    "tokens": 3,
                    "tool_calls": 1,
                    "elapsed_s": 0.1,
                    "infra_status": "ok",
                }
                for probe in probe_rows
            ],
        }

    batch = {"batch_id": "db-B00", "env": "db", "task_ids": ["db:1"], "items": [{"task_id": "db:1", "env": "db", "instruction": "alpha 1"}]}
    first = build_utility_matrix([batch], ACTIONS, {"db": probes}, evaluate, cache_dir=tmp_path, config_hash="hash-a", model_hash="m")
    assert calls["n"] == 3
    second = build_utility_matrix([batch], ACTIONS, {"db": probes}, evaluate, cache_dir=tmp_path, config_hash="hash-a", model_hash="m")
    assert calls["n"] == 4  # semantic failure is not a completed cache hit
    semantic = [row for row in second["rows"] if row["action"] == A2_SEMANTIC]
    assert semantic
    assert all(row["utility"] is None for row in semantic)
    assert all(row["infra_status"] == "missing" for row in semantic)
    assert all(row["y_after"] in (None, "") for row in semantic)
    episodic = next(row for row in first["rows"] if row["action"] == A1_EPISODIC)
    assert episodic["utility"] is not None and episodic["utility"] != 0
    other = build_utility_matrix([batch], ACTIONS, {"db": probes}, evaluate, cache_dir=tmp_path, config_hash="hash-b", model_hash="m")
    assert other["legacy"]
    assert all(row["legacy"] is False for row in other["rows"])
    with pytest.raises(ValueError):
        build_utility_matrix([batch], ["SKILLOPT_EDIT"], {"db": probes}, evaluate, cache_dir=tmp_path, config_hash="hash-c", model_hash="m")


def test_source_groups_block_template_and_seed_leakage():
    batches = [
        {"batch_id": "b0", "env": "db", "task_ids": ["db:1"], "items": [{"task_id": "db:1", "env": "db", "instruction": "sum account 12"}]},
        {"batch_id": "b1", "env": "db", "task_ids": ["db:2"], "items": [{"task_id": "db:2", "env": "db", "instruction": "sum account 99"}]},
        {"batch_id": "b1-seed2", "env": "db", "task_ids": ["db:2"], "seed": 9, "items": [{"task_id": "db:2", "env": "db", "instruction": "sum account 99"}]},
        {"batch_id": "b2", "env": "os", "task_ids": ["os:1"], "items": [{"task_id": "os:1", "env": "os", "instruction": "list files"}]},
    ]
    grouped = split_by_source_group(batches, n_outer=3, seed=1)
    assert {row["batch_id"] for row in grouped.batches} == {"b0", "b1", "b2"}
    assert grouped.notes["dropped_duplicates"][0]["reason"] == "seed_is_not_a_new_sample"
    assert grouped.folds["b0"] == grouped.folds["b1"]
    with pytest.raises(SplitLeakageError):
        split_by_source_group(batches, forbidden_task_ids={"db:1"})


def _paired_rows(env: str, batch: str, action: str, shadow_y: float, retention_y: float = 0.0) -> list[dict]:
    rows = []
    for i in range(20):
        rows.append(
            {
                "env": env,
                "batch_id": batch,
                "source_group": batch,
                "action": action,
                "probe_task_id": f"{env}:s:{i}",
                "eval_split": "shadow_reward_probe",
                "y_before": 0.0,
                "y_after": shadow_y,
                "infra_status": "ok",
                "legacy": False,
            }
        )
    for i in range(10):
        rows.append(
            {
                "env": env,
                "batch_id": batch,
                "source_group": batch,
                "action": action,
                "probe_task_id": f"{env}:r:{i}",
                "eval_split": "retention_probe",
                "y_before": 0.0,
                "y_after": retention_y,
                "infra_status": "ok",
                "legacy": False,
            }
        )
    return rows


def test_crossfit_keeps_ties_and_selects_on_the_held_out_half():
    rows = []
    for action, y in ((A0_DISCARD, 0.0), (A1_EPISODIC, 0.0), (A2_SEMANTIC, 0.0)):
        rows.extend(_paired_rows("db", "tie", action, y))
    tied = crossfit_action_selection(rows, n_boot=20, seed=1)
    assert tied["rows"]
    assert all(row["selected_label"] == "TIE" for row in tied["rows"])
    assert all(row["hard_label"] is None for row in tied["rows"])

    contrast = []
    # Zero-padded ids sort in numeric order, so the even half is i = 0,2,4,...
    for i in range(20):
        for action, even_y, odd_y in (
            (A0_DISCARD, 0.0, 0.0),
            (A1_EPISODIC, 1.0, 0.0),
            (A2_SEMANTIC, 0.0, 1.0),
        ):
            contrast.append(
                {
                    "env": "db",
                    "batch_id": "half",
                    "source_group": "half",
                    "action": action,
                    "probe_task_id": f"db:s:{i:02d}",
                    "eval_split": "shadow_reward_probe",
                    "y_before": 0.0,
                    "y_after": even_y if i % 2 == 0 else odd_y,
                    "infra_status": "ok",
                    "legacy": False,
                }
            )
    for i in range(10):
        for action in ACTIONS:
            contrast.append(
                {
                    "env": "db",
                    "batch_id": "half",
                    "source_group": "half",
                    "action": action,
                    "probe_task_id": f"db:r:{i:02d}",
                    "eval_split": "retention_probe",
                    "y_before": 0.0,
                    "y_after": 0.0,
                    "infra_status": "ok",
                    "legacy": False,
                }
            )
    report = crossfit_action_selection(contrast, n_boot=20, seed=1)
    forward = next(row for row in report["rows"] if row["direction"] == "forward")
    assert forward["env_fixed_action"] == A1_EPISODIC
    assert report["naive_hindsight_note"].startswith("descriptive")


def test_crossfit_go_when_each_batch_has_a_real_held_out_gain():
    rows = []
    for env in ("db", "os"):
        for index in range(8):
            batch = f"{env}-B{index:02d}"
            episodic = 0.6 if index < 4 else 0.2
            semantic = 0.2 if index < 4 else 0.6
            for action, score in ((A0_DISCARD, 0.0), (A1_EPISODIC, episodic), (A2_SEMANTIC, semantic)):
                rows.extend(_paired_rows(env, batch, action, score))
    report = crossfit_action_selection(rows, n_boot=40, seed=3)
    assert report["status"] == "A-SPACE-GO"
    assert report["crossfit_selector_gain"] > report["delta_space"]
    assert report["feature_policy_gain"] is None
    assert report["naive_hindsight_gap"] is not None


def test_router_does_not_read_test_labels_or_forbidden_features():
    names = list(CONTEXT_FEATURES + EXPERIENCE_FEATURES + STATE_FEATURES)

    def record(batch_id: str, env: str, group: str, episodic: float, *, capability: float | None = 0.4) -> dict:
        features = {name: 0.0 for name in names}
        features["env_db"] = 1.0 if env == "db" else 0.0
        features["batch_success_rate"] = episodic
        features["capability_shadow_hist"] = capability
        features["capability_retention_hist"] = 0.4 if capability is not None else None
        return {
            "batch_id": batch_id,
            "env": env,
            "source_group": group,
            "features": features,
            "U": {A0_DISCARD: 0.0, A1_EPISODIC: episodic, A2_SEMANTIC: 0.0},
            "delta_shadow": {A0_DISCARD: 0.0, A1_EPISODIC: episodic, A2_SEMANTIC: 0.0},
            "delta_retention": {A0_DISCARD: 0.0, A1_EPISODIC: 0.0, A2_SEMANTIC: 0.0},
        }

    batches = []
    records = []
    for index, group in enumerate(["g0", "g1", "g2", "g3", "g4", "g5"]):
        batch_id = f"db-{index}"
        batches.append(
            {
                "batch_id": batch_id,
                "env": "db",
                "task_ids": [f"db:{index}"],
                "items": [{"task_id": f"db:{index}", "env": "db", "instruction": f"template-{chr(65 + index)} distinct wording"}],
            }
        )
        records.append(record(batch_id, "db", group, 0.2 if index % 2 == 0 else 0.0))
    grouped = split_by_source_group(batches, n_outer=3, seed=0)
    first = fit_ridge_router(records, grouped, variants=["M", "B0", "B3", "B5", "A1"], n_boot=15, seed=0, rules=[{"default": A0_DISCARD}])
    test_ids = {batch_id for batch_id, fold in grouped.folds.items() if fold == 0}
    changed = []
    for row in records:
        copied = {
            **row,
            "U": dict(row["U"]),
            "delta_shadow": dict(row["delta_shadow"]),
            "delta_retention": dict(row["delta_retention"]),
            "features": dict(row["features"]),
        }
        if row["batch_id"] in test_ids:
            copied["U"] = {action: float(value) + 0.3 for action, value in row["U"].items()}
        changed.append(copied)
    second = fit_ridge_router(changed, grouped, variants=["M"], n_boot=15, seed=0, rules=[])
    chosen_1 = {row["batch_id"]: row["chosen_action"] for fold in first["folds"] if fold["test_fold"] == 0 for row in fold["variants"]["M"]["outcomes"]}
    chosen_2 = {row["batch_id"]: row["chosen_action"] for row in second["folds"][0]["variants"]["M"]["outcomes"]}
    assert chosen_1 == chosen_2
    leaked = dict(records[0])
    leaked["features"] = dict(records[0]["features"])
    leaked["features"]["shadow_after"] = 1.0
    with pytest.raises(LeakageError):
        fit_ridge_router([leaked, *records[1:]], grouped, variants=["M"], n_boot=5, seed=0)
    future = dict(records[0])
    future["future_success"] = 1.0
    with pytest.raises(LeakageError):
        fit_ridge_router([future, *records[1:]], grouped, variants=["M"], n_boot=5, seed=0)


def test_fallback_discards_unreliable_env_action_and_a1_keeps_missing_state():
    names = list(CONTEXT_FEATURES + EXPERIENCE_FEATURES + STATE_FEATURES)

    def row(batch_id: str, role: str, group: str, episodic: float, *, capability: float | None = 0.2) -> dict:
        features = {name: 0.1 for name in names}
        features["capability_shadow_hist"] = capability
        features["capability_retention_hist"] = 0.2 if capability is not None else None
        return {
            "batch_id": batch_id,
            "env": "db",
            "source_group": group,
            "role": role,
            "features": features,
            "U": {A0_DISCARD: 0.0, A1_EPISODIC: episodic, A2_SEMANTIC: -0.1},
            "delta_shadow": {A0_DISCARD: 0.0, A1_EPISODIC: episodic, A2_SEMANTIC: -0.1},
            "delta_retention": {action: 0.0 for action in ACTIONS},
        }

    records = [
        row("tr0", "train", "ga", 0.5),
        row("tr1", "train", "gb", 0.5),
        row("va0", "val", "gc", -0.3),
        row("va1", "val", "gd", -0.3),
        row("te0", "test", "ge", 0.5),
        row("miss", "train", "gf", 0.5, capability=None),
    ]
    fitted = _fit_fold_variant(
        records,
        variant="M",
        actions=ACTIONS,
        alpha=1.0,
        n_boot=20,
        seed=1,
        q=0.1,
        threshold_grid=(0.0, 0.01, 0.02, 0.05),
        default_threshold=0.0,
        rules=(),
        min_tree_samples=80,
    )
    assert fitted["env_only"]["db"] == A1_EPISODIC
    assert fitted["fallback"]["db"] == A0_DISCARD
    assert "miss" in fitted["dropped_batches"]
    a1 = _fit_fold_variant(
        records,
        variant="A1",
        actions=ACTIONS,
        alpha=1.0,
        n_boot=10,
        seed=1,
        q=0.1,
        threshold_grid=(0.0, 0.05),
        default_threshold=0.0,
        rules=(),
        min_tree_samples=80,
    )
    assert "miss" not in a1["dropped_batches"]
    assert fitted["exploratory_tree"] is None
    import numpy as np

    tree = fit_shallow_tree(np.column_stack([np.arange(80), np.zeros(80)]), np.arange(80) % 2, min_samples=80)
    assert tree is not None


def test_simple_rule_is_the_frozen_order():
    rules = load_config().raw["simple_rule"]
    low = {name: 0.0 for name in CONTEXT_FEATURES + EXPERIENCE_FEATURES + STATE_FEATURES}
    low["batch_success_rate"] = 0.1
    assert apply_simple_rule(low, rules) == A0_DISCARD
    high = dict(low)
    high["batch_success_rate"] = 0.8
    high["memory_size"] = 0.0
    high["conflict_rate"] = 0.0
    high["memory_similarity"] = 0.0
    assert apply_simple_rule(high, rules) == A1_EPISODIC


def test_sequence_shares_acquisition_and_hides_future(tmp_path: Path):
    s0 = PersistentState.create(tmp_path / "s0")
    seen: list[int] = []

    def apply_fn(state, action, experiences):
        seen.append(id(experiences))
        if action == A0_DISCARD:
            return apply_discard(state, experiences)
        if action == A1_EPISODIC:
            payload = apply_episodic(state, experiences)
            payload["accepted"] = payload["changed"]
            return payload
        payload = apply_semantic(state, experiences)
        payload["accepted"] = payload["changed"]
        return payload

    def evaluate(state, round_index):  # noqa: ARG001
        return {"shadow": 0.4, "retention": 0.4, "future": 0.9, "tokens": 5, "tool_calls": 1, "elapsed_s": 0.2}

    class Watch(FixedSelector):
        def __init__(self, action: str):
            super().__init__(action)
            self.views = []

        def select(self, view: DecisionView) -> str:
            self.views.append(view)
            with pytest.raises(LeakageError):
                _ = view.future_success
            return self.action

    episodic = Watch(A1_EPISODIC)
    rounds = [[_experience() | {"task_id": f"db:acq:{i}"}] for i in range(4)]
    summary = run_fixed_acquisition_sequence(
        s0,
        rounds,
        {"DISCARD": FixedSelector(A0_DISCARD), "M": episodic, "env-only": FixedSelector(A0_DISCARD)},
        apply_fn,
        evaluate,
        work_dir=tmp_path / "seq",
        env="db",
        capability={"source": "calibration", "shadow": 0.2, "retention": 0.2},
        trained_on_states=["S0"],
    )
    assert summary["future_fed_back"] is False
    assert summary["ood_state"] is True
    assert summary["teacher_forced_gain_copied"] is False
    assert summary["claim_sequential"] is False
    discard_hash = PersistentState.load(tmp_path / "seq" / "DISCARD" / "state").content_hash()
    memory_hash = PersistentState.load(tmp_path / "seq" / "M" / "state").content_hash()
    assert discard_hash != memory_hash
    assert any(event.get("future") == 0.9 for event in summary["histories"]["M"])
    # Four rounds, two updating policies that record the list id, plus DISCARD.
    assert len(set(seen)) == 4


def test_capability_probe_source_is_rejected(tmp_path: Path):
    state = PersistentState.create(tmp_path / "s")
    with pytest.raises(LeakageError):
        predecision_features(state, [_experience()], env="db", capability={"source": "shadow_reward_probe", "shadow": 0.9})


def test_decision_names_the_claim_instead_of_pass():
    empty = build_decision()
    assert empty["status"] == "NOT_RUN"
    assert "尚未执行" in empty["claim"]
    go = judge_prediction(
        {
            "diff": 0.04,
            "ci": [0.01, 0.06],
            "mean_delta_retention": 0.0,
            "threshold_validated": True,
            "versus_simple_rule": {"diff": 0.02, "ci": [0.005, 0.03]},
        },
        min_gain=0.02,
        retention_tolerance=0.01,
    )
    assert go["status"] == "GO"
    wide = judge_prediction(
        {"diff": 0.03, "ci": [-0.01, 0.05], "mean_delta_retention": 0.0, "threshold_validated": True, "versus_simple_rule": {"diff": 0.01, "ci": [0.001, 0.02]}},
        min_gain=0.02,
        retention_tolerance=0.01,
    )
    assert wide["status"] == "EVIDENCE_INSUFFICIENT"
    small = judge_prediction(
        {"diff": 0.0, "ci": [-0.005, 0.005], "mean_delta_retention": 0.0, "threshold_validated": True},
        min_gain=0.02,
        retention_tolerance=0.01,
    )
    assert small["status"] == "NO_GO"


def test_resample_budget_allows_one_expansion(tmp_path: Path):
    path = tmp_path / "resample.json"
    assert request_resample(path, budget=1) == 1
    with pytest.raises(RuntimeError):
        request_resample(path, budget=1)


def test_legacy_inventory_does_not_reuse_aggregate_cells(tmp_path: Path):
    cell = tmp_path / "db-B00" / "DISCARD"
    cell.mkdir(parents=True)
    (cell / "DONE.json").write_text('{"action": "DISCARD", "U": -0.049877, "batch_id": "db-B00"}', encoding="utf-8")
    found = inventory_legacy_matrix(tmp_path)
    assert found["splice_into_a_v1"] is False
    assert found["cells"][0]["reusable_as_a_v1_row"] is False


def test_episode_plan_keeps_future_test_out_of_the_matrix():
    from expolicy.paper_a.execute import plan_episode_counts

    plan = plan_episode_counts(load_config())
    assert plan["matrix_episodes"] > plan["audit_episodes"] > 0
    assert plan["probe_before_approval"] == 20
    assert all(info["future_test"] > 0 for info in plan["by_env"].values())
    assert "future_test is counted only in the sequence stage" in plan["note"]


def test_cli_inventory_and_refuses_scoring_without_execute():
    assert run_a_v1.main(["inventory"]) == 0
    assert run_a_v1.main(["audit"]) == 2
    assert run_a_v1.main(["matrix"]) == 2
    assert run_a_v1.main(["sequence", "--execute"]) == 2


def test_router_records_do_not_fill_missing_utility():
    rows = [
        {"batch_id": "b", "env": "db", "action": A0_DISCARD, "utility": None, "delta_shadow": None, "delta_retention": None, "legacy": False, "source_group": "g"},
        {"batch_id": "b", "env": "db", "action": A1_EPISODIC, "utility": 0.2, "delta_shadow": 0.2, "delta_retention": 0.0, "legacy": False, "source_group": "g"},
    ]
    records = to_router_records(rows, {"b": {"batch_success_rate": 0.5}})
    assert records[0]["U"][A0_DISCARD] is None
    assert records[0]["U"][A1_EPISODIC] == 0.2
