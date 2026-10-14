"""D-v1.1 contract tests. Population controls stay in test_protocol.py."""

from __future__ import annotations

from expolicy.paper_d.audit import audit_paired_rows, resource_isolation_failures
from expolicy.paper_d.cards import judge_configuration
from expolicy.paper_d.constants import (
    ACTIONABLE_HEADROOM,
    ENV_FIXED_SUFFICIENT,
    MEASUREMENT_INCONCLUSIVE,
    NOT_TESTED,
    OPERATOR_INVALID,
    SPLIT_INSUFFICIENT,
)
from expolicy.paper_d.headroom import crossfit_selection_gain
from expolicy.paper_d.matrix import (
    align_to_plan,
    cell_table,
    charge_executions,
    execution_cache_key,
)
from expolicy.paper_d.policy import fit_simple_observable_policy
from expolicy.paper_d.prospective import validate_prospectively
from expolicy.paper_d.real_config import RealConfigError, load_real_config, validate_real_config
from expolicy.paper_d.real_cost import CostLedger
from expolicy.paper_d.real_execution import RealExecutionRefused, execute_smoke
from expolicy.paper_d.schema import empty_log
from expolicy.paper_d.stats import finite_sample_guarantee_allowed, refitting_bootstrap


def _row(**extra: object) -> dict:
    base = {
        "configuration_id": "cfg",
        "protocol_version": "D-v1.1",
        "model_hash": "m",
        "env_hash": "db",
        "env": "db",
        "state_distribution_id": "dist",
        "state_id": "s0",
        "source_group": "b0",
        "experience_source_group": "b0",
        "batch_id": "b0",
        "action": "A",
        "accepted": True,
        "changed": True,
        "task_id": "t0",
        "probe_role": "future",
        "probe_source_group": "g0",
        "probe_fold": "A",
        "outcome_before": 0.0,
        "outcome_after": 0.0,
        "retention_before": None,
        "retention_after": None,
        "repeat_id": "0",
        "cost": 1.0,
        "infra_status": "ok",
        "measurement_kind": "sample",
        "content_mode": "real",
        "budget": "pilot",
    }
    base.update(extra)
    return empty_log(**base)


def test_prompt_hash_is_paired_per_task_not_globally():
    rows = []
    for task, prompt in (("t0", "prompt-t0"), ("t1", "prompt-t1")):
        for branch in ("BASE", "DISCARD", "REJECTED_EDIT"):
            rows.append(
                {
                    "branch": branch,
                    "configuration_id": "cfg",
                    "state_id": "s0",
                    "batch_id": "b0",
                    "task_id": task,
                    "repeat_id": "0",
                    "prompt_hash": prompt,
                    "model_hash": "m",
                    "outcome": 0.2,
                }
            )
    assert audit_paired_rows(rows) == []
    rows[1]["prompt_hash"] = "changed"
    assert any("initial prompt differs" in reason for reason in audit_paired_rows(rows))


def test_plan_rejects_missing_column_task_duplicate_and_wrong_repeat():
    plan = [
        _row(action="A", task_id="t0"),
        _row(action="B", task_id="t0"),
    ]
    missing_column = align_to_plan([_row(action="A", task_id="t0")], plan)
    assert missing_column["valid"] is False
    assert any("column" in reason for reason in missing_column["reasons"])
    missing_task = align_to_plan(
        [_row(action="A", task_id="t0"), _row(action="B", task_id="t0"), _row(action="A", task_id="t1"), _row(action="B", task_id="t1")],
        plan + [_row(action="A", task_id="t1"), _row(action="B", task_id="t1"), _row(action="A", task_id="t2"), _row(action="B", task_id="t2")],
    )
    assert missing_task["valid"] is False
    duplicated = align_to_plan(plan + [plan[0]], plan)
    assert any("duplicate" in reason for reason in duplicated["reasons"])
    wrong_repeat = align_to_plan([_row(action="A", task_id="t0", repeat_id="1"), _row(action="B", task_id="t0")], plan)
    assert any("repeat" in reason for reason in wrong_repeat["reasons"])


def test_probe_groups_follow_templates_and_ignore_a_new_seed():
    rows = []
    for group in ("g0", "g1"):
        for action in ("A", "B"):
            for role, before_key, after_key in (
                ("future", "outcome_before", "outcome_after"),
                ("retention", "retention_before", "retention_after"),
            ):
                payload = {
                    "action": action,
                    "probe_role": role,
                    "probe_source_group": group,
                    "task_id": f"{role}-{group}",
                    "source_group": "same-batch",
                    "batch_id": "same-batch",
                }
                payload[before_key] = 0.0
                payload[after_key] = 0.2 if role == "future" else 1.0
                rows.append(_row(**payload))
                rows.append(_row(**payload, repeat_id="1"))
    cells = cell_table(rows, retention_weight=0.5)
    assert cells[0]["n_source_groups"] == 2


def test_same_experience_source_stays_in_one_policy_fold():
    records = []
    for state in ("S0", "S_history"):
        for group in ("e0", "e1", "e2", "e3"):
            records.append(
                {
                    "unit_id": f"{group}-{state}",
                    "env": "db",
                    "state_id": state,
                    "source_group": f"{group}-{state}",
                    "experience_source_group": group,
                    "features": {"visible_state": 0.0, "env__db": 1.0},
                    "selection_utilities": {"A": 1.0, "B": 0.0},
                    "evaluation_utilities": {"A": 1.0, "B": 0.0},
                }
            )
    fitted = fit_simple_observable_policy(records, envs=["db"])
    for split in fitted["splits"]:
        assert set(split["train"]).isdisjoint(split["test"])
        assert "e0" in split["train"] or "e0" in split["test"]
        assert not ({"e0"} <= set(split["train"]) and {"e0"} <= set(split["test"]))


def test_execution_id_is_billed_once_and_cache_key_changes_with_decode_config():
    left = _row(execution_id="run-1", cost=4.0, inference_config_hash="decode-a")
    right = dict(left)
    changed = dict(left)
    changed["inference_config_hash"] = "decode-b"
    changed["execution_id"] = "run-2"
    bill = charge_executions([left, right, changed])
    assert bill["n_executions"] == 2
    assert bill["cost"] == 8.0
    assert execution_cache_key(left) != execution_cache_key(changed)


def test_operator_invalid_is_not_a_rejected_noop_and_sample_point_is_not_a_formal_go():
    judgment = judge_configuration(
        measurement_valid=True,
        within_gap=None,
        within_lower_raw=None,
        within_upper=0.2,
        feature_gain=0.1,
        feature_gain_lower=0.0,
        feature_evaluated=True,
        crossfit_gain=0.0,
        population_table=False,
        action_mean_upper={"EPISODIC_REPLAY": 0.0},
        updates_rejected=False,
        content_real=None,
        content_control=None,
        operator_invalid=True,
    )
    assert judgment["primary_status"] == OPERATOR_INVALID
    assert judgment["dimension_detail"]["content_contribution"] == NOT_TESTED
    point_only = judge_configuration(
        measurement_valid=True,
        within_gap=0.2,
        within_lower_raw=0.0,
        within_upper=0.4,
        feature_gain=0.05,
        feature_gain_lower=-0.01,
        feature_evaluated=True,
        crossfit_gain=0.0,
        population_table=False,
        action_mean_upper={"A": 0.1},
        updates_rejected=False,
        content_real=None,
        content_control=None,
    )
    assert point_only["candidate_signal"] is True
    assert ACTIONABLE_HEADROOM not in point_only["dimensions"]
    wide = judge_configuration(
        measurement_valid=True,
        within_gap=0.0,
        within_lower_raw=-0.2,
        within_upper=0.4,
        feature_gain=0.0,
        feature_evaluated=True,
        crossfit_gain=0.0,
        population_table=False,
        action_mean_upper={"A": 0.2},
        updates_rejected=False,
        content_real=0.0,
        content_control=0.0,
        measurement_precision="inconclusive",
    )
    assert wide["primary_status"] == MEASUREMENT_INCONCLUSIVE
    assert ENV_FIXED_SUFFICIENT not in wide["dimensions"]


def test_frozen_probe_fold_requires_both_roles_and_bootstrap_refits():
    rows = [
        _row(probe_fold="A", probe_role="future", task_id="fa", action=action, outcome_after=0.2)
        for action in ("A", "B")
    ]
    assert crossfit_selection_gain(rows)["reason"] == SPLIT_INSUFFICIENT
    draws = []

    def select(sample: dict) -> str:
        draws.append(sample["draw"])
        return "A" if sample["draw"] % 2 == 0 else "B"

    report = refitting_bootstrap(n_boot=4, seed=1, select=select)
    assert report["n_refits"] == 4
    assert report["formal_inference"] is False
    assert draws == [0, 1, 2, 3]
    assert finite_sample_guarantee_allowed(
        equal_groups=True,
        independence_declared=False,
        shared_history=False,
        shared_probe=False,
    ) is False


def test_pending_interval_is_not_a_false_go_and_real_correlation_stays_off():
    from expolicy.paper_d.cards import freeze_diagnostic_card

    judgment = judge_configuration(
        measurement_valid=True,
        within_gap=0.0,
        within_lower_raw=0.0,
        within_upper=0.0,
        feature_gain=0.0,
        feature_evaluated=True,
        crossfit_gain=0.0,
        population_table=True,
        action_mean_upper={"A": 0.0},
        updates_rejected=False,
        content_real=None,
        content_control=None,
    )
    card = freeze_diagnostic_card(
        {**judgment, "primary_status": ACTIONABLE_HEADROOM, "dimensions": [ACTIONABLE_HEADROOM]},
        configuration_id="cfg",
        batch_ids=["diag"],
        task_ids=["diag-task"],
    )
    from expolicy.r9_lab.io_util import sha256_json

    card["experience_source_groups"] = ["e0"]
    card["card_hash"] = sha256_json({key: value for key, value in card.items() if key != "card_hash"})
    report = validate_prospectively(
        [card],
        [
            {
                "configuration_id": "cfg",
                "batch_ids": ["new"],
                "task_ids": ["future"],
                "experience_source_groups": ["e1"],
                "actual_gain": 0.0,
                "actual_gain_lower": -0.1,
                "actual_gain_upper": 0.1,
                "diagnostic_cost": 1.0,
                "training_cost": 1.0,
                "training_avoided": 0.0,
            }
        ],
        report_correlation=False,
    )
    assert report["rows"][0]["pending"] is True
    assert report["rows"][0]["false_go"] is False
    assert report["n_pending"] == 1
    assert report["correlation_reported"] is False
    overlap = validate_prospectively(
        [card],
        [
            {
                "configuration_id": "cfg",
                "batch_ids": ["new"],
                "task_ids": ["future"],
                "experience_source_groups": ["e0"],
                "actual_gain": 0.0,
                "diagnostic_cost": 1.0,
                "training_cost": 1.0,
                "training_avoided": 0.0,
            }
        ],
    )
    assert overlap["rows"][0]["validation_status"] == "invalid"


def test_full_smoke_plan_has_192_cells_per_environment():
    import json
    from pathlib import Path

    from expolicy.paper_d.real_dispatch import smoke_cells
    from expolicy.r9_lab.benchmark.split import load_split

    payload = load_split(Path("outputs/r9_lab/split.json"))
    cells = smoke_cells(payload)
    assert len(cells) == 576
    assert {cell["env"] for cell in cells} == {"db", "os", "kg"}
    assert all(sum(1 for cell in cells if cell["env"] == env) == 192 for env in ("db", "os", "kg"))
    assert json.loads(json.dumps(cells[0]))["sample_index"] == cells[0]["sample_index"]


def test_ledger_survives_reopen_and_blocks_the_cap(tmp_path):
    from expolicy.paper_d.persistent_ledger import PersistentLedger

    path = tmp_path / "ledger.sqlite"
    first = PersistentLedger(path, cap=1)
    assert first.reserve_attempt("a1", "logical", job_kind="vertical") is True
    first.mark_started("a1")
    first.close()
    second = PersistentLedger(path, cap=1)
    assert second.spent() == 1
    assert second.reserve_attempt("a2", "other", job_kind="vertical") is False
    second.close()


def test_real_config_and_execute_do_not_invent_episodes():
    config = load_real_config()
    assert config["protocol_version"] == "D-v1.1"
    assert config["allow_population"] is False
    raw = dict(config)
    raw["retention_tolerance"] = 0.05
    try:
        validate_real_config(raw)
        raised = False
    except RealConfigError:
        raised = True
    assert raised
    dry = execute_smoke(config, execute=False)
    assert dry["episodes"] == 0
    try:
        execute_smoke(config, execute=True)
        refused = False
    except RealExecutionRefused:
        refused = True
    assert refused
    ledger = CostLedger(2)
    assert ledger.reserve("a", 2) is True
    assert ledger.reserve("b", 1) is False
    ledger.commit("a")
    assert resource_isolation_failures({"clone_id": "only-a-name"}) == [
        "writable resource has no write-neighbor-restore evidence"
    ]
