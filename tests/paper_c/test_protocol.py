"""CPU tests for the C-v1 acquisition protocol. No model server."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from expolicy.paper_c.assimilator import (
    AssimilatorDrift,
    EpisodicFrozenAssimilator,
    SyntheticContentAssimilator,
    assert_same_algorithm,
)
from expolicy.paper_c.audit import audit_assimilator, build_gain_audit
from expolicy.paper_c.config import ConfigPinError, load_config, validate_config
from expolicy.paper_c.constants import (
    C_ASSIMILATION_FAIL,
    C_COST_FAIL,
    C_INCONCLUSIVE,
    C_NO_HEADROOM,
    C_NOT_RUN,
    C_PASS,
    C_PREDICTION_FAIL,
    C_STATIC_FIRST,
    C_TRANSFER_FAIL,
    ENVS,
    LOG_FIELDS,
    PILOT_POLICIES,
    M,
)
from expolicy.paper_c.controls import rewrite_experiences
from expolicy.paper_c.curves import attach_offline_scores, evaluate_frozen_curves
from expolicy.paper_c.features import (
    LeakageError,
    model_feature_names,
    pre_acquisition_features,
    visibility_report,
)
from expolicy.paper_c.gates import judge_c0, judge_c1, judge_c2, judge_pilot, judge_transfer
from expolicy.paper_c.ledger import BudgetLedger
from expolicy.paper_c.manifest import assert_manifest_intact, build_manifest, write_manifest
from expolicy.paper_c.pilot import run_self_check
from expolicy.paper_c.policies import AcquisitionView, GainPolicy, freeze_baseline
from expolicy.paper_c.predictor import GainPredictor, fit_gain_predictor, permute_gains
from expolicy.paper_c.sequence import CacheReuseError, TrajectoryCache, run_learning_sequence
from expolicy.paper_c.stats import recovery_point
from expolicy.paper_c.utility import acquisition_gain, trapezoid_aulc
from expolicy.paper_c.world import FAMILIES, SyntheticWorld, make_snapshot, mismatch_of
from expolicy.r9_lab.state.persistent_state import PersistentState


def _public() -> dict[str, float]:
    return {"public_difficulty": 0.5, "public_estimated_tools": 1.0}


def _features(state: PersistentState, family: str = "alpha") -> dict:
    return pre_acquisition_features(
        state,
        family=family,
        env="synth",
        families=FAMILIES,
        public=_public(),
        history=None,
        declared_envs=ENVS,
    )


def test_config_locks_epsilon_metric_and_pilot_set():
    cfg = load_config()
    assert cfg.protocol_id == "C-v1"
    assert cfg.raw["exploration"]["epsilon"] == 0.1
    assert tuple(cfg.raw["pilot_policies"]) == PILOT_POLICIES
    raw = dict(cfg.raw)
    raw["exploration"] = dict(raw["exploration"])
    raw["exploration"]["ucb_enabled"] = True
    with pytest.raises(ConfigPinError):
        validate_config(raw)


def test_gain_reports_raw_rates_and_penalizes_only_retention_loss():
    packed = acquisition_gain(
        shadow_before=0.2,
        shadow_after=0.2,
        retention_before=0.8,
        retention_after=0.4,
        retention_weight=0.5,
    )
    assert packed["delta_shadow"] == pytest.approx(0.0)
    assert packed["g"] == pytest.approx(-0.2)
    assert packed["uses_future_test"] is False
    assert trapezoid_aulc([(0.0, 0.0), (1.0, 1.0)])["aulc"] == pytest.approx(0.5)
    cut = trapezoid_aulc([(0.0, 0.0), (2.0, 0.0), (4.0, 1.0)], c_max=2.0)
    assert cut["aulc"] == pytest.approx(0.0)
    assert cut["extrapolated"] is False


def test_pre_acquisition_features_reject_rollout_fields(tmp_path: Path):
    state = make_snapshot(tmp_path / "state", {})
    features = _features(state)
    assert "success" not in features
    assert features["family__alpha"] == 1.0
    assert features["covered__alpha"] == 0.0
    with pytest.raises(LeakageError):
        pre_acquisition_features(
            state,
            family="alpha",
            env="synth",
            families=FAMILIES,
            public={**_public(), "success": True, "trajectory_length": 3},
            history=None,
            declared_envs=ENVS,
        )
    report = visibility_report(FAMILIES, ENVS)
    assert set(report["model_features"]).isdisjoint(report["forbidden_current_rollout"])


def test_content_controls_keep_count_and_change_payload():
    batch = SyntheticWorld().passive_alpha_batch()
    empty = rewrite_experiences(batch, "empty", mismatch_of=mismatch_of)
    wrong = rewrite_experiences(batch, "matched-wrong", mismatch_of=mismatch_of)
    assert len(empty) == len(batch) == len(wrong)
    assert rewrite_experiences(batch, "no-update", mismatch_of=mismatch_of) == []
    assert all(item["family"] == "" for item in empty)
    assert all(item["family"] == "gamma" for item in wrong)


class _CountOnly:
    name = "count"
    version = "0"
    prompt = "count"

    def algorithm_hash(self) -> str:
        return "count-hash"

    def apply(self, state, experiences):
        state.extra["n"] = int(state.extra.get("n") or 0) + len(experiences)
        state.save()
        return {
            "accepted": True,
            "algorithm_hash": "count-hash",
            "online_algorithm_edit": False,
            "teacher_tokens": 0,
            "n": len(experiences),
        }


def test_c0_synthetic_passes_and_count_only_fails(tmp_path: Path):
    world = SyntheticWorld(tag="c0")
    ledger = BudgetLedger()

    def score(state, role):
        table = {"shadow_eval": world.shadow_eval, "retention": world.retention}[role]
        return world.score(state, table)

    audit = audit_assimilator(
        make_snapshot(tmp_path / "start", {}),
        world.passive_alpha_batch(),
        SyntheticContentAssimilator(),
        score,
        work_dir=tmp_path / "arms",
        mismatch_of=mismatch_of,
        ledger=ledger,
    )
    assert judge_c0(audit["arms"], min_gain=0.03, retention_tolerance=0.01)["status"] == C_PASS
    assert audit["arms"]["real"]["n_presented"] == audit["arms"]["empty"]["n_presented"]
    assert audit["arms"]["no-update"]["n_presented"] == 0
    counted = audit_assimilator(
        make_snapshot(tmp_path / "count-start", {}),
        world.passive_alpha_batch(),
        _CountOnly(),
        score,
        work_dir=tmp_path / "count",
        mismatch_of=mismatch_of,
        ledger=BudgetLedger(),
    )
    assert judge_c0(counted["arms"], min_gain=0.03, retention_tolerance=0.01)["status"] == C_ASSIMILATION_FAIL


def test_gain_audit_uses_disjoint_probes_and_leaves_the_source_state(tmp_path: Path):
    world = SyntheticWorld(tag="gain")
    source = make_snapshot(tmp_path / "source", {})
    before = source.content_hash()
    calls: list[str] = []

    def score(state, role):
        calls.append(role)
        table = {
            "shadow_select": world.shadow_select,
            "shadow_eval": world.shadow_eval,
            "retention": world.retention,
        }[role]
        return world.score(state, table)

    batch = []
    for task in world.pool(per_family=4)["alpha"][:4]:
        episode = world.rollout(source, task)
        episode["source_template"] = "wording_a"
        batch.append(episode)
    matrix = build_gain_audit(
        [("S0", source)],
        ["alpha"],
        {"alpha": [batch]},
        SyntheticContentAssimilator(),
        score,
        lambda state, family: _features(state, family),
        work_dir=tmp_path / "matrix",
        ledger=BudgetLedger(),
        select_ids=[task.task_id for task in world.shadow_select],
        eval_ids=[task.task_id for task in world.shadow_eval],
        future_ids=[task.task_id for task in world.future],
    )
    assert "future" not in calls
    assert source.content_hash() == before
    assert matrix["rows"][0]["gain_eval"] == pytest.approx(0.35)
    assert matrix["rows"][0]["split"] == "train_shadow"


def test_predictor_refuses_future_labels_and_small_samples():
    rows = [
        {
            "gain": float(index),
            "split": "train_shadow",
            "env": "synth",
            "family": "alpha",
            "state_id": f"S{index}",
            "features": {"public_difficulty": 0.5},
        }
        for index in range(3)
    ]
    assert fit_gain_predictor(rows, ["public_difficulty"], min_rows=8)["model"] is None
    rows[0]["split"] = "future_test"
    with pytest.raises(LeakageError):
        fit_gain_predictor(rows, ["public_difficulty"], min_rows=1)
    shuffled, changed = permute_gains(
        [{"env": "synth", "gain": 0.0, "features": {}}, {"env": "synth", "gain": 1.0, "features": {}}],
        seed=0,
    )
    assert changed
    assert [row["gain"] for row in shuffled] != [0.0, 1.0]


def test_gate_table_matches_the_registered_failure_names():
    assert judge_c1({"states": []})["status"] == C_NOT_RUN
    assert (
        judge_c1({"states": [{"status": "ok", "best_family": "alpha", "best_eval_gain": 0.0, "uniform_eval_gain": 0.1}]})[
            "status"
        ]
        == C_NO_HEADROOM
    )
    assert (
        judge_c1(
            {
                "states": [
                    {"status": "ok", "best_family": "alpha", "best_eval_gain": 0.4, "uniform_eval_gain": 0.1},
                    {"status": "ok", "best_family": "alpha", "best_eval_gain": 0.4, "uniform_eval_gain": 0.1},
                ]
            }
        )["status"]
        == C_STATIC_FIRST
    )
    assert (
        judge_c2(
            state_advantage=0.2,
            family_excess=0.2,
            env_advantage=0.2,
            failure_advantage=0.0,
            state_groups="ok",
            margin=0.02,
            correlation_value=0.99,
        )["status"]
        == C_PREDICTION_FAIL
    )
    assert (
        judge_c2(
            state_advantage=0.3,
            family_excess=0.2,
            env_advantage=0.0,
            failure_advantage=0.0,
            state_groups="ok",
            margin=0.02,
            correlation_value=0.1,
        )["status"]
        == C_PASS
    )
    assert (
        judge_c2(
            state_advantage=None,
            family_excess=None,
            env_advantage=None,
            failure_advantage=None,
            state_groups="insufficient_groups",
            margin=0.02,
        )["status"]
        == C_INCONCLUSIVE
    )
    assert judge_transfer(None)["status"] == C_NOT_RUN
    assert judge_transfer(False)["status"] == C_TRANSFER_FAIL
    diffs = [0.05, 0.06, 0.07]
    pilot = judge_pilot(
        cost_diffs=diffs,
        episode_diffs=diffs,
        final_diffs=diffs,
        retention_diffs=[0.0, 0.01, -0.005],
        baseline="Coverage",
        min_aulc=0.02,
        min_final=0.02,
        retention_tolerance=0.01,
        wide=0.05,
        n_public_environments=0,
        environment_class="synthetic",
    )
    assert pilot["comparison_status"] == C_PASS
    assert pilot["paper_claim"] == C_NOT_RUN
    cost_fail = judge_pilot(
        cost_diffs=[0.0, 0.001, -0.001],
        episode_diffs=[0.05, 0.06, 0.04],
        final_diffs=diffs,
        retention_diffs=[0.0, 0.0, 0.01],
        baseline="Uniform",
        min_aulc=0.02,
        min_final=0.02,
        retention_tolerance=0.01,
        wide=0.05,
        n_public_environments=2,
        environment_class="public",
    )
    assert cost_fail["comparison_status"] == C_COST_FAIL
    assert (
        judge_pilot(
            cost_diffs=[0.0, 0.0, 0.0],
            episode_diffs=diffs,
            final_diffs=diffs,
            retention_diffs=[0.0, 0.0, 0.0],
            baseline="Uniform",
            min_aulc=0.02,
            min_final=0.02,
            retention_tolerance=0.01,
            wide=0.05,
            n_public_environments=2,
            environment_class="public",
        )["comparison_status"]
        == C_INCONCLUSIVE
    )


def test_baseline_is_frozen_from_development_scores_only():
    assert freeze_baseline({"Uniform": 0.1, "Coverage": 0.4, "M": 0.9}, ["Uniform", "Coverage"]) == "Coverage"


def test_gain_policy_falls_back_without_a_model_and_respects_epsilon_zero():
    rng = np.random.default_rng(0)
    blind = GainPolicy(M, None, ["family__beta"], rng, epsilon=0.0)
    view = AcquisitionView(
        env="synth",
        round_index=1,
        state_hash="s",
        families=("alpha", "beta"),
        remaining={"alpha": 2, "beta": 2},
        features_by_family={"alpha": {"family__beta": 0.0}, "beta": {"family__beta": 1.0}},
        pool_hash="p",
    )
    assert blind.select(view) in {"alpha", "beta"}
    model = GainPredictor(["family__beta"], np.zeros(1), np.ones(1), np.asarray([0.0, 1.0]))
    greedy = GainPolicy(M, model, ["family__beta"], np.random.default_rng(1), epsilon=0.0)
    assert greedy.select(view) == "beta"
    with pytest.raises(LeakageError):
        _ = view.future_score


def test_cache_refuses_cross_state_reuse_and_future_scores_stay_offline(tmp_path: Path):
    cache = TrajectoryCache()
    cache.put("hash-a", "task-1", {"task_id": "task-1", "success": False})
    assert cache.get("hash-a", "task-1")["success"] is False
    with pytest.raises(CacheReuseError):
        cache.get("hash-b", "task-1")
    world = SyntheticWorld(tag="seq")
    ledger = BudgetLedger()
    result = run_learning_sequence(
        make_snapshot(tmp_path / "start", {}),
        GainPolicy(M, None, model_feature_names(FAMILIES, ENVS), np.random.default_rng(0), epsilon=0.0),
        world,
        SyntheticContentAssimilator(),
        world.pool(per_family=2),
        work_dir=tmp_path / "run",
        ledger=ledger,
        run_id="unit",
        sequence_seed=0,
        families=FAMILIES,
        declared_envs=ENVS,
        n_rounds=2,
        tasks_per_round=1,
        feedback_rounds=(2,),
        eval_rounds=(0, 2),
    )
    assert all(row["future_score_offline"] is None for row in result["logs"])
    assert set(LOG_FIELDS).issubset(result["logs"][0])
    interaction = ledger.interaction_total()
    curves = evaluate_frozen_curves(
        result["checkpoints"],
        future_fn=lambda state: world.score(state, world.future),
        shadow_fn=lambda state: world.score(state, world.shadow_eval),
        retention_fn=lambda state: world.score(state, world.retention),
        ledger=ledger,
    )
    assert ledger.interaction_total() == interaction
    assert ledger.overhead_interactions() > 0
    merged = attach_offline_scores(result["logs"], curves)
    filled = [row for row in merged if row["round"] == 2]
    assert filled and filled[0]["future_score_offline"] is not None
    assert result["logs"][0]["future_score_offline"] is None


class _Reject:
    name = "reject"
    version = "0"
    prompt = "reject"

    def algorithm_hash(self) -> str:
        return "reject-hash"

    def apply(self, state, experiences):
        del experiences
        state.extra["dirty"] = True
        state.save()
        return {"accepted": False, "algorithm_hash": "reject-hash", "online_algorithm_edit": False, "teacher_tokens": 0}


def test_rejected_update_rolls_back_and_is_billed(tmp_path: Path):
    world = SyntheticWorld(tag="rollback")
    ledger = BudgetLedger()
    result = run_learning_sequence(
        make_snapshot(tmp_path / "start", {}),
        GainPolicy("M", None, ["public_difficulty"], np.random.default_rng(0), epsilon=0.0),
        world,
        _Reject(),
        world.pool(per_family=1),
        work_dir=tmp_path / "run",
        ledger=ledger,
        run_id="rollback",
        sequence_seed=1,
        families=FAMILIES,
        declared_envs=ENVS,
        n_rounds=1,
        tasks_per_round=1,
        feedback_rounds=(),
        eval_rounds=(0, 1),
    )
    assert result["logs"][0]["assimilation_accepted"] is False
    assert any(row["kind"] == "rollback" for row in ledger.rows)
    assert result["checkpoints"][-1].state.extra.get("dirty") is None


def test_manifest_and_episodic_adapter_stay_frozen(tmp_path: Path):
    assimilator = SyntheticContentAssimilator()
    payload = build_manifest(
        assimilator,
        validation_ids=["shadow:1", "future:1"],
        resource_limits={"max_acquisition_episodes_per_sequence": 24},
        capacity={"max_entries": 256},
        hyperparameters={"epsilon": 0.1},
    )
    path = tmp_path / "C_ASSIMILATOR_MANIFEST.json"
    write_manifest(path, payload)
    stored = assert_manifest_intact(path)
    assert stored["online_algorithm_edit"] is False
    assert stored["not_certified"]["paper_b"]["usable_as_c_assimilator"] is False
    path.write_text(path.read_text(encoding="utf-8").replace("false", "true"), encoding="utf-8")
    with pytest.raises(RuntimeError):
        assert_manifest_intact(path)
    with pytest.raises(AssimilatorDrift):
        assert_same_algorithm(
            "other",
            {"algorithm_hash": stored["assimilator"]["algorithm_hash"], "online_algorithm_edit": False},
        )
    state = PersistentState.create(tmp_path / "epi")
    episodic = EpisodicFrozenAssimilator()
    real = episodic.apply(
        state,
        [
            {
                "task_id": "t1",
                "env": "db",
                "instruction": "keep the verified query",
                "trajectory": [{"agent": "ok"}],
                "success": True,
                "verifier_outcome": "CORRECT",
                "bucket": "acquisition_pool",
            }
        ],
    )
    assert real["n"] == 1
    empty = episodic.apply(
        PersistentState.create(tmp_path / "epi-empty"),
        rewrite_experiences(
            [
                {
                    "task_id": "t2",
                    "env": "db",
                    "family": "alpha",
                    "instruction": "x",
                    "success": True,
                    "bucket": "acquisition_pool",
                }
            ],
            "empty",
            mismatch_of=mismatch_of,
        ),
    )
    assert empty["n"] == 0
    with pytest.raises(LeakageError):
        episodic.apply(state, [{"task_id": "g", "gold_answer": "secret", "success": True, "bucket": "acquisition_pool"}])


def test_recovery_point_is_absent_when_deployment_does_not_save_cost():
    assert recovery_point(10, 3) == 4
    assert recovery_point(10, 0) is None


def test_self_check_writes_the_protocol_products_without_a_paper_claim(tmp_path: Path):
    result = run_self_check(tmp_path / "out")
    assert result["decision"]["paper_claim"] == C_NOT_RUN
    assert result["decision"]["c0_status"] == C_PASS
    text = (tmp_path / "out" / "C_DECISION.md").read_text(encoding="utf-8")
    assert "真实公开环境尚未执行" in text
    for name in (
        "C_PROTOCOL.yaml",
        "C_ASSIMILATOR_MANIFEST.json",
        "C_ASSIMILATOR_AUDIT.md",
        "C_GAIN_MATRIX.csv",
        "C_FEATURE_VISIBILITY_AUDIT.md",
        "C_LEARNING_CURVES.csv",
        "C_COST_LEDGER.jsonl",
        "C_DECISION.md",
    ):
        assert (tmp_path / "out" / name).exists()
    assert result["n_gain_rows"] > 0
    ledger = (tmp_path / "out" / "C_COST_LEDGER.jsonl").read_text(encoding="utf-8")
    assert "evaluation_overhead" in ledger
    assert "acquisition_episode" in ledger
