"""CPU checks for the public C-v1.1 boundary. No model server."""

from __future__ import annotations

from pathlib import Path

import pytest
from expolicy.paper_c.config import load_config
from expolicy.paper_c.features import LeakageError
from expolicy.paper_c.public.baseline import (
    fit_mean_gain_mixture,
    fixed_candidates,
    freeze_best_fixed,
)
from expolicy.paper_c.public.budget import BudgetExhausted, EpisodeBudget
from expolicy.paper_c.public.config import (
    PublicConfigError,
    load_public_config,
    validate_public_config,
)
from expolicy.paper_c.public.controls import make_content_controls
from expolicy.paper_c.public.curves import score_at_common_budget
from expolicy.paper_c.public.e0 import EpisodicE0
from expolicy.paper_c.public.features import public_features
from expolicy.paper_c.public.gates import classify_interval, judge_c0_screen
from expolicy.paper_c.public.runtime import refuse_fake
from expolicy.r9_lab.state.persistent_state import PersistentState


def _success(task_id: str, family: str, action: str = "Action: get_relations(m.1)") -> dict:
    return {
        "task_id": task_id,
        "env": "kg",
        "family": family,
        "instruction": f"what is {family} {task_id}",
        "trajectory": [{"agent": action, "action": action}],
        "success": True,
        "verifier_outcome": "CORRECT",
        "bucket": "acquisition_train",
        "source_verified": True,
    }


def test_c_v1_lock_still_rejects_a_public_environment_list():
    cfg = load_config()
    assert cfg.protocol_id == "C-v1"
    assert cfg.raw["public_environments"] == []


def test_public_config_allows_one_environment_pilot():
    cfg = load_public_config()
    assert cfg.raw["mode"] == "public_pilot"
    assert cfg.raw["active_environment"] == "kg"
    raw = dict(cfg.raw)
    raw["exploration"] = dict(raw["exploration"])
    raw["exploration"]["ucb_enabled"] = True
    with pytest.raises(PublicConfigError):
        validate_public_config(raw)


def test_content_arms_write_the_same_count_when_success_flag_is_rewritten(tmp_path: Path):
    raw = [_success("kg:1", "attribute"), _success("kg:2", "relation", "Action: intersection")]
    built = make_content_controls(raw)
    assert built["counts_match"] is True
    assert built["arms"]["no-update"] == []
    assert all(item["source_verified"] for item in built["arms"]["empty-slots"])
    assert built["arms"]["empty-slots"][0]["instruction"] == "neutral placeholder record"
    assert built["arms"]["query-only"][0]["trajectory"] == [{"agent": "", "action": ""}]
    assert built["arms"]["matched-wrong"][0]["trajectory"][0]["agent"]
    state = PersistentState.create(tmp_path / "mem")
    e0 = EpisodicE0()
    result = e0.apply(state, built["arms"]["empty-slots"])
    assert result["n_written"] == 2
    assert result["accepted"] is True


def test_e0_rejects_future_rows_and_evicts_oldest(tmp_path: Path):
    state = PersistentState.create(tmp_path / "cap")
    e0 = EpisodicE0()
    rows = [_success(f"kg:{index}", "attribute") for index in range(257)]
    result = e0.apply(state, rows)
    assert result["n_evicted"] == 1
    assert len(state.episodic) == 256
    assert state.episodic[0].task_id == "kg:1"
    with pytest.raises(LeakageError):
        e0.apply(state, [_success("future_test:1", "attribute") | {"bucket": "future_test"}])


def test_public_features_use_committed_records_and_reject_rollout_fields(tmp_path: Path):
    state = PersistentState.create(tmp_path / "feat")
    e0 = EpisodicE0()
    e0.apply(state, [_success("kg:1", "attribute")])
    features = public_features(
        state,
        family="attribute",
        families=("attribute", "relation"),
        family_of={"kg:1": "attribute"},
        question="what is the attribute",
        candidate={"instruction_chars": 10, "public_tool_count": 2},
    )
    assert features["covered__attribute"] == 1.0
    assert features["family__relation"] == 0.0
    with pytest.raises(LeakageError):
        public_features(
            state,
            family="attribute",
            families=("attribute",),
            family_of={},
            question="what",
            candidate={"success": True, "public_tool_count": 1},
        )


def test_budget_stopper_refuses_the_extra_episode_and_keeps_the_spent_count():
    budget = EpisodeBudget(1, stage="runtime")
    budget.reserve(1, kind="episode")
    with pytest.raises(BudgetExhausted):
        budget.reserve(1, kind="feedback")
    assert budget.used == 1


def test_fake_environment_is_refused():
    class FakeLabEnv:
        pass

    with pytest.raises(RuntimeError):
        refuse_fake(FakeLabEnv())


def test_common_budget_final_does_not_use_a_later_private_point():
    curves = [
        {"policy": "M", "sequence_seed": 1, "round": 0, "interaction_cost": 0, "future_score": 0.2},
        {"policy": "M", "sequence_seed": 1, "round": 1, "interaction_cost": 10, "future_score": 0.4},
        {"policy": "M", "sequence_seed": 1, "round": 2, "interaction_cost": 30, "future_score": 0.9},
        {"policy": "Coverage", "sequence_seed": 1, "round": 0, "interaction_cost": 0, "future_score": 0.2},
        {"policy": "Coverage", "sequence_seed": 1, "round": 1, "interaction_cost": 10, "future_score": 0.5},
    ]
    scores = score_at_common_budget(curves, policies=("M", "Coverage"))
    assert scores["M"][1] == pytest.approx(0.4)
    assert scores["Coverage"][1] == pytest.approx(0.5)


def test_zero_width_difference_is_not_equivalence():
    summary = {"diff": 0.0, "ci": [0.0, 0.0], "zero_width": True, "population_noise_claim": False, "n": 3}
    assert classify_interval(summary, minimum=0.02) == "C-INCONCLUSIVE"


def test_screen_signal_is_not_a_public_pass():
    arms = {
        "counts_match": True,
        "control_invalid": False,
        "real": {"gain": 0.1, "delta_retention": 0.0},
        "no-update": {"gain": 0.0},
        "empty-slots": {"gain": 0.0},
        "matched-wrong": {"gain": 0.0},
    }
    judgment = judge_c0_screen(arms)
    assert judgment["status"] == "C0-SCREEN-SIGNAL"
    assert judgment["status"] != "C0-PUBLIC-PASS"


def test_mean_gain_mixture_is_named_separately_from_the_frozen_choice():
    rows = [
        {"family": "attribute", "gain_eval": 0.2},
        {"family": "relation", "gain_eval": 0.0},
    ]
    mixture = fit_mean_gain_mixture(rows)
    assert mixture["attribute"] == pytest.approx(1.0)
    candidates = fixed_candidates(["attribute", "relation"], mixture)
    chosen = freeze_best_fixed({"uniform-mixture": 0.1, "mean-gain-mixture": 0.4, "single:attribute": 0.2})
    assert "mean-gain-mixture" in candidates
    assert chosen == "mean-gain-mixture"
