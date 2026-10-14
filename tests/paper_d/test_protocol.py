"""CPU tests for the D-v1 diagnostic protocol. No model server."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from expolicy.paper_d.audit import audit_measurement, audit_paired_rows, valid_noop_bundle
from expolicy.paper_d.cards import freeze_diagnostic_card, judge_configuration
from expolicy.paper_d.config import ConfigPinError, load_config, validate_config
from expolicy.paper_d.constants import (
    ACTIONABLE_HEADROOM,
    CONTENT_UNSUPPORTED,
    DELTA,
    ENV_FIXED_SUFFICIENT,
    HETEROGENEOUS_BUT_UNPREDICTED,
    INCONCLUSIVE,
    MEASUREMENT_INVALID,
    MEASUREMENT_POPULATION,
    MEASUREMENT_SAMPLE,
    PROTOCOL_VERSION,
    STATUS_INFRA,
    STATUS_OK,
    UPDATE_INEFFECTIVE,
)
from expolicy.paper_d.controls import EXPECTED, run_known_structure_controls, two_step_optimum
from expolicy.paper_d.headroom import (
    bound_panel_headroom,
    crossfit_selection_gain,
    estimate_naive_gap,
)
from expolicy.paper_d.inventory import collect_artifact_inventory
from expolicy.paper_d.matrix import build_task_level_matrix, cell_table, charge_cost, unit_id
from expolicy.paper_d.pilot import run_self_check
from expolicy.paper_d.policy import LeakageError, fit_simple_observable_policy
from expolicy.paper_d.prospective import validate_prospectively
from expolicy.paper_d.schema import empty_log, read_jsonl
from expolicy.paper_d.stats import (
    crossed_resample_means,
    evaluation_episodes,
    followup_status,
    population_gaps,
    sequence_cluster,
)
from expolicy.paper_d.utility import legacy_utility_from_repeat, utility_from_expectations
from expolicy.r9_lab.io_util import sha256_file, sha256_json


def _pair(action: str, utility: float, **extra: object) -> list[dict]:
    before, after = (0.0, utility) if utility >= 0 else (-utility, 0.0)
    shared = {
        "configuration_id": "cfg",
        "protocol_version": PROTOCOL_VERSION,
        "model_hash": "m",
        "env_hash": extra.get("env", "db"),
        "env": extra.get("env", "db"),
        "state_distribution_id": "dist",
        "state_id": extra.get("state_id", "s0"),
        "source_group": extra.get("source_group", extra.get("batch_id", "b0")),
        "batch_id": extra.get("batch_id", "b0"),
        "action": action,
        "accepted": extra.get("accepted", True),
        "changed": extra.get("changed", True),
        "repeat_id": extra.get("repeat_id", "0"),
        "cost": 1.0,
        "infra_status": extra.get("infra_status", STATUS_OK),
        "measurement_kind": extra.get("measurement_kind", MEASUREMENT_POPULATION),
        "content_mode": "real",
        "budget": extra.get("budget", "pilot"),
        "source_template": extra.get("source_template", "t0"),
        "prompt_hash": extra.get("prompt_hash", "prompt"),
    }
    tag = f"{shared['state_id']}-{shared['batch_id']}-{shared['source_template']}-{shared['repeat_id']}"
    future = empty_log(
        **shared,
        task_id=str(extra.get("task_id") or f"{tag}-future"),
        probe_role="future",
        outcome_before=before,
        outcome_after=after,
    )
    retention = empty_log(
        **shared,
        task_id=str(extra.get("retention_task_id") or f"{tag}-retention"),
        probe_role="retention",
        retention_before=extra.get("retention_before", 1.0),
        retention_after=extra.get("retention_after", 1.0),
    )
    return [future, retention]


def _panel(spec: list[tuple[str, str, dict[str, float]]], **extra: object) -> list[dict]:
    rows: list[dict] = []
    for batch_id, env, utilities in spec:
        for template in ("t0", "t1"):
            for action, utility in utilities.items():
                rows.extend(
                    _pair(
                        action,
                        utility,
                        batch_id=batch_id,
                        env=env,
                        source_template=template,
                        source_group=batch_id,
                        **extra,
                    )
                )
    return rows


def test_protocol_utility_penalizes_expected_retention_loss_only():
    packed = utility_from_expectations(
        future_before=0.2,
        future_after=0.4,
        retention_before=0.8,
        retention_after=0.7,
        retention_weight=0.5,
    )
    assert packed["gain"] == pytest.approx(0.2)
    assert packed["utility"] == pytest.approx(0.15)
    legacy = [
        legacy_utility_from_repeat(
            future_before=0.0,
            future_after=0.0,
            retention_before=0.5,
            retention_after=0.9,
        )["utility"],
        legacy_utility_from_repeat(
            future_before=0.0,
            future_after=0.0,
            retention_before=0.9,
            retention_after=0.5,
        )["utility"],
    ]
    assert sum(legacy) / 2 == pytest.approx(-0.1)
    assert utility_from_expectations(
        future_before=0.0,
        future_after=0.0,
        retention_before=0.7,
        retention_after=0.7,
    )["utility"] == pytest.approx(0.0)


def test_legacy_repeat_penalty_is_more_negative_than_the_protocol_mean():
    rows = []
    for repeat, before, after in (("0", 0.5, 0.9), ("1", 0.9, 0.5)):
        rows.extend(
            _pair(
                "A",
                0.0,
                repeat_id=repeat,
                task_id="future-shared",
                source_template="t0",
                retention_before=before,
                retention_after=after,
                measurement_kind=MEASUREMENT_SAMPLE,
            )
        )
    # The second template keeps the split valid; retention stays flat there.
    rows.extend(_pair("A", 0.0, source_template="t1", measurement_kind=MEASUREMENT_SAMPLE))
    rows.extend(_pair("B", 0.0, source_template="t0", repeat_id="0", measurement_kind=MEASUREMENT_SAMPLE))
    rows.extend(_pair("B", 0.0, source_template="t0", repeat_id="1", task_id="future-b", measurement_kind=MEASUREMENT_SAMPLE))
    rows.extend(_pair("B", 0.0, source_template="t1", measurement_kind=MEASUREMENT_SAMPLE))
    cells = { (cell["action"]): cell for cell in cell_table(rows, retention_weight=0.5) }
    assert cells["A"]["utility"] == pytest.approx(0.0)
    assert cells["A"]["legacy_utility"] < cells["A"]["utility"]


def test_naive_gap_is_not_an_oracle_and_a_lower_external_value_is_an_audit():
    rows = _panel([("b0", "db", {"A": 0.2, "B": 0.0}), ("b1", "db", {"A": 0.2, "B": 0.0})])
    naive = estimate_naive_gap(rows)
    assert naive["estimate_name"] == "estimate_naive_gap"
    assert naive["is_oracle"] is False
    assert naive["is_credible_upper_bound"] is False
    assert naive["global_gap"] == pytest.approx(0.0)
    units = sorted({unit_id(row) for row in rows})
    flagged = estimate_naive_gap(rows, external_oracle={unit: -0.1 for unit in units})
    assert flagged["below_fixed_requires_audit"] is True
    assert flagged["structural_impossibility"] is False


def test_missing_action_is_not_filled_with_zero():
    rows = _panel([("b0", "db", {"A": -0.2, "B": -0.4}), ("b1", "db", {"A": -0.2})])
    naive = estimate_naive_gap(rows)
    assert naive["valid"] is False
    assert naive["global_gap"] is None
    assert naive["reason"] == "INCOMPLETE_ACTION_SET"


def test_macro_weight_differs_from_uniform_and_within_stays_zero():
    spec = [
        ("b0", "db", {"A": 1.0, "B": 0.0}),
        ("b1", "os", {"A": 0.0, "B": 1.0}),
        ("b2", "os", {"A": 0.0, "B": 1.0}),
        ("b3", "os", {"A": 0.0, "B": 1.0}),
    ]
    rows = _panel(spec)
    uniform = estimate_naive_gap(rows, weight_mode="uniform")
    macro = estimate_naive_gap(rows, weight_mode="macro")
    assert uniform["global_gap"] == pytest.approx(0.25)
    assert macro["global_gap"] == pytest.approx(0.5)
    assert uniform["within_gap"] == pytest.approx(0.0)
    assert macro["within_gap"] == pytest.approx(0.0)


def test_population_gaps_are_nonnegative_and_within_does_not_exceed_global():
    utilities = {}
    envs = {}
    for index in range(6):
        unit = f"u{index}"
        utilities[unit] = {"A": (index % 3) / 3, "B": ((index + 1) % 4) / 5, "C": 0.1}
        envs[unit] = "db" if index < 4 else "os"
    gaps = population_gaps(utilities, envs, weight_mode="macro")
    assert gaps["global_gap"] >= -1e-9
    assert gaps["within_gap"] >= -1e-9
    assert gaps["within_gap"] <= gaps["global_gap"] + 1e-9


def test_crossfit_keeps_a_negative_gain_and_rejects_seed_only_splits():
    rows = []
    rows.extend(_pair("A", 1.0, batch_id="b0", source_template="t0", state_id="u1"))
    rows.extend(_pair("B", 0.0, batch_id="b0", source_template="t0", state_id="u1"))
    rows.extend(_pair("A", 0.0, batch_id="b0", source_template="t1", state_id="u1"))
    rows.extend(_pair("B", 0.0, batch_id="b0", source_template="t1", state_id="u1"))
    rows.extend(_pair("A", 0.0, batch_id="b1", source_template="t0", state_id="u2"))
    rows.extend(_pair("B", 1.0, batch_id="b1", source_template="t0", state_id="u2"))
    rows.extend(_pair("A", 0.4, batch_id="b1", source_template="t1", state_id="u2"))
    rows.extend(_pair("B", 0.0, batch_id="b1", source_template="t1", state_id="u2"))
    # Two batches are one environment. u2's selection half prefers B, which loses on the eval half.
    cross = crossfit_selection_gain(rows)
    assert cross["estimate_name"] == "crossfit_selection_gain"
    assert cross["valid"] is True
    assert cross["clipped"] is False
    assert cross["selection_gain"] < 0.0
    assert cross["min_gain"] < 0.0
    assert cross["global_gap"] is None
    single = []
    for repeat in ("0", "1"):
        single.extend(
            _pair("A", 0.2, task_id="same-future", repeat_id=repeat, source_template=f"seed-{repeat}", batch_id="b0")
        )
        single.extend(
            _pair("B", 0.0, task_id="same-future", repeat_id=repeat, source_template=f"seed-{repeat}", batch_id="b0")
        )
    noise = crossfit_selection_gain(single)
    assert noise["generalization"] is False
    assert noise["selection_gain"] is None


def test_population_intervals_do_not_collapse_unless_the_control_flag_is_set():
    rows = _panel([("b0", "db", {"A": 0.0, "B": 0.0}), ("b1", "db", {"A": 0.0, "B": 0.0})])
    refused = bound_panel_headroom(rows, allow_population=False)
    assert refused["valid"] is False
    allowed = bound_panel_headroom(rows, allow_population=True)
    assert allowed["estimate_name"] == "bound_panel_headroom"
    assert allowed["interval_source"] == "population_table"
    assert allowed["finite_sample_guarantee"] is False
    assert allowed["within_upper"] == pytest.approx(0.0)
    assert allowed["global_gap"] is None


def test_wide_sample_interval_is_not_env_fixed_sufficient():
    rows = _panel(
        [("b0", "db", {"A": 0.0, "B": 0.0}), ("b1", "db", {"A": 0.0, "B": 0.0})],
        measurement_kind=MEASUREMENT_SAMPLE,
    )
    bound = bound_panel_headroom(rows)
    judgment = judge_configuration(
        measurement_valid=True,
        within_gap=None,
        within_lower_raw=bound["within_lower_raw"],
        within_upper=bound["within_upper"],
        feature_gain=0.0,
        feature_evaluated=True,
        crossfit_gain=0.0,
        population_table=False,
        action_mean_upper=bound["action_mean_upper"],
        updates_rejected=False,
        content_real=None,
        content_control=None,
    )
    assert bound["within_upper"] >= DELTA
    assert judgment["primary_status"] == INCONCLUSIVE
    assert ENV_FIXED_SUFFICIENT not in judgment["dimensions"]


def test_known_structure_controls_match_their_literal_expectations():
    result = run_known_structure_controls()
    failed = [(report["name"], report["reasons"]) for report in result["reports"] if not report["passed"]]
    assert failed == []
    by_name = {report["name"]: report for report in result["reports"]}
    assert by_name["exact_tie"]["observed"]["primary"] == ENV_FIXED_SUFFICIENT
    assert by_name["env_only_winner"]["observed"]["naive_global"] == pytest.approx(EXPECTED["env_only_winner"]["global_gap"])
    assert by_name["env_only_winner"]["observed"]["naive_within"] == pytest.approx(0.0)
    assert by_name["observable_crossover"]["observed"]["primary"] == ACTIONABLE_HEADROOM
    assert by_name["hidden_crossover"]["observed"]["primary"] == HETEROGENEOUS_BUT_UNPREDICTED
    assert by_name["hidden_crossover"]["observed"]["crossfit"] > DELTA
    assert by_name["noisy_tie"]["analytical_population_global"] == 0.0
    assert by_name["noisy_tie"]["observed"]["naive_k6"] > by_name["noisy_tie"]["observed"]["naive_k2"] > 0.0
    assert two_step_optimum() == 0.4
    assert by_name["delayed_synergy"]["observed"]["naive_global"] == pytest.approx(0.0)
    assert by_name["delayed_synergy"]["observed"]["long_term_covered"] is False
    assert result["counts_as_real_environment_gain"] is False


def test_policy_does_not_train_on_evaluation_utilities_and_rejects_hidden_state():
    records = []
    for index in range(4):
        records.append(
            {
                "unit_id": f"u{index}",
                "env": "db",
                "source_group": f"g{index}",
                "features": {"visible_state": 0.0, "env__db": 1.0},
                "selection_utilities": {"A": 1.0, "B": 0.0},
                "evaluation_utilities": {"A": 0.0, "B": 1.0},
            }
        )
    fitted = fit_simple_observable_policy(records, envs=["db"])
    assert fitted["used_evaluation_utilities_for_training"] is False
    assert fitted["outer_scored_with"] == "evaluation_utilities"
    assert fitted["splits_disjoint"] is True
    assert {item["action"] for item in fitted["predicted_actions"]} == {"A"}
    for split in fitted["splits"]:
        assert set(split["train"]).isdisjoint(split["test"])
    with pytest.raises(LeakageError):
        fit_simple_observable_policy(
            [{**records[0], "features": {"hidden_state": 1.0}}],
            envs=["db"],
        )


def test_matrix_keeps_failed_updates_and_extreme_foundation_rows():
    rows = _panel([("b0", "db", {"SKILLOPT_EDIT": 0.0, "DISCARD": 0.0})])
    for row in rows:
        if row["action"] == "SKILLOPT_EDIT":
            row["accepted"] = False
            row["changed"] = False
    matrix = build_task_level_matrix(rows)
    assert len(matrix) == len(rows)
    assert any(row["accepted"] is False for row in matrix)
    lows = _panel([("b0", "db", {"A": 0.0, "B": 0.0})])
    highs = _panel([("b1", "os", {"A": 1.0, "B": 1.0})])
    kept = build_task_level_matrix(lows + highs)
    assert len(kept) == len(lows) + len(highs)
    judgment = judge_configuration(
        measurement_valid=True,
        within_gap=0.0,
        within_lower_raw=0.0,
        within_upper=None,
        feature_gain=None,
        feature_evaluated=False,
        crossfit_gain=None,
        population_table=True,
        action_mean_upper=None,
        updates_rejected=True,
        content_real=None,
        content_control=None,
    )
    assert judgment["primary_status"] == UPDATE_INEFFECTIVE
    content = judge_configuration(
        measurement_valid=True,
        within_gap=None,
        within_lower_raw=None,
        within_upper=None,
        feature_gain=None,
        feature_evaluated=False,
        crossfit_gain=None,
        population_table=False,
        action_mean_upper={"A": 0.2},
        updates_rejected=False,
        content_real=0.0,
        content_control=0.0,
    )
    assert CONTENT_UNSUPPORTED in content["dimensions"]
    blocked = judge_configuration(
        measurement_valid=False,
        within_gap=0.5,
        within_lower_raw=0.5,
        within_upper=0.5,
        feature_gain=0.4,
        feature_evaluated=True,
        crossfit_gain=0.4,
        population_table=True,
        action_mean_upper={"A": 0.4},
        updates_rejected=False,
        content_real=None,
        content_control=None,
    )
    assert blocked["primary_status"] == MEASUREMENT_INVALID
    assert blocked["scientific_conclusion_allowed"] is False


def test_partial_infra_failure_does_not_average_the_remaining_tasks():
    rows = []
    for template, status in (("t0", STATUS_OK), ("t1", STATUS_INFRA)):
        rows.extend(_pair("A", 0.5, source_template=template, infra_status=status, measurement_kind=MEASUREMENT_SAMPLE))
        rows.extend(_pair("B", -0.4, source_template=template, infra_status=status, measurement_kind=MEASUREMENT_SAMPLE))
    naive = estimate_naive_gap(rows)
    assert naive["valid"] is False
    assert naive["global_gap"] is None


def test_repeat_is_not_a_new_unit_and_cached_baselines_are_not_double_charged():
    first = _pair("A", 0.1, repeat_id="0")
    second = _pair("A", 0.1, repeat_id="1")
    assert unit_id(first[0]) == unit_id(second[0])
    rows = [
        {**first[0], "cached_baseline": False, "cost": 3.0},
        {**first[0], "cached_baseline": True, "cost": 3.0},
        {**first[0], "cached_baseline": False, "cost": 3.0, "repeat_id": "1"},
        {**first[0], "cached_baseline": True, "cost": 3.0, "prompt_hash": "other"},
    ]
    charged = charge_cost(rows)
    assert charged["cached_hits"] == 1
    assert charged["cost"] == pytest.approx(9.0)
    assert evaluation_episodes(36, 3, 80, 1) == 8640
    assert followup_status(adaptive=True, sequential_preregistered=False)["exploratory_only"] is True
    assert sequence_cluster("s1", "hist") == sequence_cluster("s2", "hist")
    resampled = crossed_resample_means({("g0", "t0"): 0.0, ("g0", "t1"): 1.0, ("g1", "t0"): 0.2, ("g1", "t1"): 0.4}, n_boot=20, seed=0)
    assert resampled["formal_inference"] is False


def test_measurement_audit_fail_closed():
    assert audit_measurement(valid_noop_bundle())["status"] == STATUS_OK
    assert audit_measurement(valid_noop_bundle())["penalty_negative_bias"] is True
    shared = valid_noop_bundle()
    shared["states"][0]["branches"]["DISCARD"]["resources"]["external_db"] = shared["states"][0]["branches"]["BASE"]["resources"]["external_db"]
    assert audit_measurement(shared)["status"] == MEASUREMENT_INVALID
    drifted = valid_noop_bundle()
    drifted["states"][0]["branches"]["BASE"]["model_hash"] = "other-model"
    assert any("version mix" in reason for reason in audit_measurement(drifted)["reasons"])
    relabel = valid_noop_bundle()
    relabel["seed_relabelled_batches"] = ["b0"]
    assert any("seed" in reason for reason in audit_measurement(relabel)["reasons"])
    dropped = valid_noop_bundle()
    dropped["dropped_task_ids"] = ["task-1"]
    assert audit_measurement(dropped)["scientific_conclusion_allowed"] is False
    paired = []
    for branch, outcome in (("BASE", 0.5), ("DISCARD", 0.5), ("REJECTED_EDIT", 0.9)):
        paired.append({"branch": branch, "task_id": "t", "outcome": outcome, "model_hash": "m", "prompt_hash": "p"})
    assert any("offset" in reason for reason in audit_paired_rows(paired))


def test_inventory_requires_a_hash_and_ignores_status_live(tmp_path: Path):
    live = tmp_path / "only_live" / "cell"
    live.mkdir(parents=True)
    (live / "STATUS_LIVE.json").write_text('{"running": true}\n', encoding="utf-8")
    good_dir = tmp_path / "good" / "batch-a" / "DISCARD"
    good_dir.mkdir(parents=True)
    payload = {"U": -0.0137, "shadow": 0.2, "items": [{"task_id": "t"}]}
    (good_dir / "results.json").write_text("[]\n", encoding="utf-8")
    digest = sha256_file(good_dir / "results.json")
    done = {
        "done": True,
        "batch_id": "batch-a",
        "action": "DISCARD",
        "U": -0.0137,
        "shadow": 0.2,
        "items": [{"task_id": "t"}],
        "file_hashes": {"results.json": digest},
    }
    (good_dir / "DONE.json").write_text(json.dumps(done) + "\n", encoding="utf-8")
    bad_dir = tmp_path / "bad" / "batch-b" / "A"
    bad_dir.mkdir(parents=True)
    (bad_dir / "results.json").write_text("nope\n", encoding="utf-8")
    (bad_dir / "DONE.json").write_text(
        '{"done": true, "U": 0.5, "file_hashes": {"results.json": "deadbeef"}}\n',
        encoding="utf-8",
    )
    inventory = collect_artifact_inventory([tmp_path])
    assert inventory["n_status_live_ignored"] == 1
    assert inventory["n_verified"] == 1
    assert inventory["verified"][0]["legacy_values"]["U"] == -0.0137
    assert inventory["verified"][0]["reusable_for_protocol_matrix"] is True
    assert inventory["n_rejected"] == 1
    assert inventory["rejected"][0]["legacy_values"]["U"] == 0.5
    assert inventory["inferred_from_status_live"] is False
    payload["U"] = 99
    assert inventory["verified"][0]["legacy_values"]["U"] == -0.0137


def test_card_freezes_before_training_and_prospective_checks_are_separate():
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
    card = freeze_diagnostic_card(judgment, configuration_id="cfg", batch_ids=["b0"], task_ids=["t0"], policy_trained=False)
    assert card["card_hash"] == sha256_json({key: value for key, value in card.items() if key != "card_hash"})
    with pytest.raises(Exception):
        freeze_diagnostic_card(judgment, configuration_id="cfg", batch_ids=["b0"], task_ids=["t0"], policy_trained=True)
    tampered = dict(card)
    tampered["primary_status"] = ACTIONABLE_HEADROOM
    overlapped = validate_prospectively(
        [card],
        [
            {
                "configuration_id": "cfg",
                "batch_ids": ["b0"],
                "task_ids": ["new"],
                "actual_gain": 0.0,
                "diagnostic_cost": 1.0,
                "training_cost": 2.0,
                "training_avoided": 0.0,
            }
        ],
    )
    assert overlapped["rows"][0]["validation_status"] == "invalid"
    hashed = validate_prospectively(
        [tampered],
        [
            {
                "configuration_id": "cfg",
                "batch_ids": ["b1"],
                "task_ids": ["new"],
                "actual_gain": 0.2,
                "diagnostic_cost": 1.0,
                "training_cost": 1.0,
                "training_avoided": 0.0,
            }
        ],
    )
    assert hashed["rows"][0]["validation_status"] == "invalid"
    go = freeze_diagnostic_card(
        {**judgment, "primary_status": ACTIONABLE_HEADROOM, "dimensions": [ACTIONABLE_HEADROOM]},
        configuration_id="go",
        batch_ids=["diag"],
        task_ids=["diag-task"],
    )
    nogo = card
    report = validate_prospectively(
        [go, nogo],
        [
            {
                "configuration_id": "go",
                "batch_ids": ["new-a"],
                "task_ids": ["future-a"],
                "actual_gain": 0.0,
                "diagnostic_cost": 4.0,
                "training_cost": 5.0,
                "training_avoided": 0.0,
            },
            {
                "configuration_id": "cfg",
                "batch_ids": ["new-b"],
                "task_ids": ["future-b"],
                "actual_gain": 0.1,
                "diagnostic_cost": 6.0,
                "training_cost": 7.0,
                "training_avoided": None,
            },
        ],
    )
    assert report["false_go_rate"] == pytest.approx(1.0)
    assert report["false_nogo_rate"] == pytest.approx(1.0)
    assert report["correlation_reported"] is False
    assert report["significance_claimed"] is False
    assert report["net_savings"] is None
    assert report["diagnostic_cost"] == pytest.approx(10.0)
    with pytest.raises(Exception):
        validate_prospectively([card], [{"configuration_id": "cfg", "savings_from_failures_only": True, "actual_gain": 0.0, "batch_ids": ["z"], "task_ids": ["z"], "diagnostic_cost": 1, "training_cost": 1}])


def test_self_check_writes_the_protocol_products(tmp_path: Path):
    result = run_self_check(tmp_path)
    names = {
        "D_PROTOCOL.md",
        "D_MEASUREMENT_AUDIT.md",
        "D_KNOWN_CONTROLS.csv",
        "D_TASK_MATRIX.parquet",
        "D_HEADROOM_ESTIMATES.csv",
        "D_DIAGNOSTIC_CARDS.jsonl",
        "D_PROSPECTIVE_VALIDATION.csv",
        "D_DECISION.md",
    }
    assert names <= set(result["paths"])
    table = pq.read_table(tmp_path / "D_TASK_MATRIX.parquet")
    assert table.num_rows == sum(len(report["rows"]) for report in result["controls"]["reports"])
    assert table.num_rows > len(result["controls"]["reports"])
    cards = read_jsonl(tmp_path / "D_DIAGNOSTIC_CARDS.jsonl")
    assert len(cards) == 7
    text = (tmp_path / "D_HEADROOM_ESTIMATES.csv").read_text(encoding="utf-8")
    assert "estimate_naive_gap" in text
    assert "crossfit_selection_gain" in text
    assert "bound_panel_headroom" in text
    decision = (tmp_path / "D_DECISION.md").read_text(encoding="utf-8")
    assert "HARNESS_ONLY" in decision
    assert result["prospective"]["case_validation_only"] is True
    assert result["prospective"]["significance_claimed"] is False
    assert result["decision"]["harness_status"] == "PASS"


def test_config_locks_delta():
    cfg = load_config()
    assert cfg.raw["delta"] == DELTA
    raw = dict(cfg.raw)
    raw["delta"] = 0.2
    with pytest.raises(ConfigPinError):
        validate_config(raw)
