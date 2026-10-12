"""A-v1.1 regressions. These use fake data and never open a model or an environment."""

from __future__ import annotations

import numpy as np

from expolicy.paper_a.audit import _decide_status, stratified_paired_summary
from expolicy.paper_a.stats import interval_claim, paired_flip_interval
from expolicy.paper_a.v11 import (
    PilotRefusal,
    assert_v11_launch,
    cache_usable,
    choose_pilot_batches,
    classify_divergence,
    protocol_hash,
    selection_signal,
    split_probe_tasks,
)
from expolicy.r9_lab.assimilators.semantic import generate_semantic_v11


def test_later_zero_block_does_not_promote_an_earlier_inconclusive_block():
    first = np.zeros(92)
    first[:2] = -1
    serial = [first, np.zeros(92), np.zeros(92)]
    formal = [np.zeros(92), np.zeros(92), np.zeros(92)]
    status, _extra = _decide_status(
        [],
        {"serial": serial, "formal_concurrency": formal},
        equivalence=0.01,
        ci_z=1.96,
        require_formal=True,
        formal_concurrency=1,
    )
    assert status != "PASS"


def test_opposite_directions_stay_visible_in_the_stratified_summary():
    rows = [
        {"mode": "serial", "repeat_id": 0, "branch": "DISCARD", "eval_split": "shadow_reward_probe", "paired_delta": 1.0},
        {"mode": "serial", "repeat_id": 0, "branch": "REJECTED_EDIT", "eval_split": "shadow_reward_probe", "paired_delta": -1.0},
        {"mode": "serial", "repeat_id": 0, "branch": "DISCARD", "eval_split": "retention_probe", "paired_delta": -1.0},
    ]
    summary = stratified_paired_summary(rows)
    means = {(row["branch"], row["eval_split"]): row["mean"] for row in summary}
    assert means[("DISCARD", "shadow_reward_probe")] == 1.0
    assert means[("REJECTED_EDIT", "shadow_reward_probe")] == -1.0
    assert means[("DISCARD", "retention_probe")] == -1.0


def test_zero_width_interval_does_not_claim_population_noise_is_zero():
    claim = interval_claim([0.0, 0.0, 0.0])
    assert claim["population_noise_claim"] is False
    one = interval_claim([0.0])
    assert one["population_noise_claim"] is False
    interval = paired_flip_interval(0, 0, 16)
    assert interval["ci"][0] < 0 < interval["ci"][1]


def test_input_or_tool_divergence_is_not_called_model_noise():
    assert classify_divergence([
        {"reset_hash": "a", "steps": [{"request_hash": "r", "response_hash": "x", "tool_hash": "t"}], "y": 1},
        {"reset_hash": "b", "steps": [{"request_hash": "r", "response_hash": "x", "tool_hash": "t"}], "y": 0},
    ]) == "RESET_OR_INPUT_VARIATION"
    assert classify_divergence([
        {"reset_hash": "a", "steps": [{"request_hash": "r", "response_hash": "x", "tool_hash": "t1"}], "y": 1},
        {"reset_hash": "a", "steps": [{"request_hash": "r", "response_hash": "x", "tool_hash": "t2"}], "y": 0},
    ]) == "TOOL_VARIATION"


def test_changed_cache_key_does_not_reuse_an_existing_file():
    stored = {"cache_key": "old", "infra_status": "ok", "y": 1.0, "complete": True}
    assert cache_usable(stored, "old")
    assert not cache_usable(stored, "new")
    assert not cache_usable({"cache_key": "old", "infra_status": "ok", "y": None, "complete": True}, "old")


def test_missing_eligibility_or_budget_refuses_before_a_service_exists():
    try:
        assert_v11_launch(None, expected_hash="abc", episodes=1, budget=10)
        raised = False
    except PilotRefusal:
        raised = True
    assert raised
    eligibility = {
        "protocol_hash": "abc",
        "environments": ["db"],
        "requested_environments": ["kg"],
        "episodes_used": 0,
    }
    try:
        assert_v11_launch(eligibility, expected_hash="abc", episodes=1, budget=10)
        raised = False
    except PilotRefusal:
        raised = True
    assert raised
    eligibility["requested_environments"] = ["db"]
    eligibility["episodes_used"] = 10
    try:
        assert_v11_launch(eligibility, expected_hash="abc", episodes=1, budget=10)
        raised = False
    except PilotRefusal:
        raised = True
    assert raised


def test_components_are_not_split_between_discovery_and_reserved():
    frozen = {"batches": []}
    for index in range(6):
        frozen["batches"].append(
            {
                "batch_id": f"db-B{index:02d}",
                "env": "db",
                "items": [{"env": "db", "source_group": "shared" if index < 2 else f"g{index}", "instruction": "q"}],
            }
        )
    chosen = choose_pilot_batches(frozen, "db", n_discovery=4, n_reserved=2, seed=20261011)
    discovery = {batch["batch_id"] for batch in chosen["discovery"]}
    assert ("db-B00" in discovery) == ("db-B01" in discovery)


def test_probe_groups_stay_on_one_side():
    records = [
        {"env": "db", "source_group": "same", "instruction": "a"},
        {"env": "db", "source_group": "same", "instruction": "b"},
        {"env": "db", "source_group": "other", "instruction": "c"},
    ]
    split = split_probe_tasks(records, 1, seed=1)
    sides = []
    for name in ("selection", "confirmation"):
        groups = {row["source_group"] for row in split[name]}
        sides.append(groups)
    assert "same" not in sides[0] or "same" not in sides[1]


class _Bad:
    def chat(self, messages, **kwargs):
        class _R:
            text = "not json"
            finish_reason = "stop"
            completion_tokens = 3
        return _R()


def test_semantic_parse_failure_is_recorded_and_does_not_fall_back():
    experiences = [
        {"task_id": "db:1", "instruction": "count rows", "success": True, "trajectory": [{"agent": "SELECT 1"}]},
        {"task_id": "db:2", "instruction": "count more", "success": False, "trajectory": [{"agent": "SELECT 2"}]},
    ]
    entries, meta = generate_semantic_v11(experiences, _Bad(), repair_limit=1)
    assert entries == []
    assert meta["generation_status"] == "OPERATOR_INVALID"
    assert meta["generation_backend"] is None
    assert meta["input_covers_all_experiences"] is True
    assert set(meta["input_task_ids"]) == {"db:1", "db:2"}


def test_selection_signal_needs_two_groups_and_is_not_one_flip():
    assert selection_signal([
        {"source_group": "g1", "winner": "EPISODIC_REPLAY", "n_task_flips": 3, "tie": False},
        {"source_group": "g2", "winner": "SEMANTIC_MEMORY", "n_task_flips": 2, "tie": False},
    ]) == "SIGNAL"
    assert selection_signal([
        {"source_group": "g1", "winner": "EPISODIC_REPLAY", "n_task_flips": 1, "tie": False},
    ]) == "WEAK_SIGNAL"


def test_protocol_hash_changes_when_the_lock_changes():
    assert protocol_hash({"a": 1}) != protocol_hash({"a": 2})
