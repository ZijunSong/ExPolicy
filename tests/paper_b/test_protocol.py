"""CPU tests for the B-v1 protocol. No model server and no external search index."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
from expolicy.paper_b.config import ConfigPinError, load_config
from expolicy.paper_b.constants import (
    B0,
    B1,
    B2,
    B3,
    B4,
    B5,
    B_GO,
    B_INCONCLUSIVE,
    B_NO_GO_CONTENT,
    B_NO_GO_MECHANISM,
    B_NOT_RUN,
    TEMPLATE_REWRITE,
    M,
)
from expolicy.paper_b.controls import make_experience_controls
from expolicy.paper_b.decision import judge, practical_floor
from expolicy.paper_b.execute import execute_rules
from expolicy.paper_b.induce import build_inducer_prompt, executable_hash, induce_rules
from expolicy.paper_b.invariants import check_unmatched_falls_back
from expolicy.paper_b.models import Budget, Rule
from expolicy.paper_b.pipeline import _collect, run_protocol
from expolicy.paper_b.registry import build_runtime_registry, generic_rules
from expolicy.paper_b.split import audit_splits
from expolicy.paper_b.traces import build_trace_pairs
from expolicy.paper_b.validate import static_errors, validate_rules
from expolicy.paper_b.world import build_world, feasibility_counts


def _comparison(diff: float, lo: float, hi: float) -> dict:
    return {"diff": diff, "ci": [lo, hi], "n": 40, "n_groups": 4}


def _clear_comparisons() -> dict:
    return {
        "M_vs_B1": _comparison(0.10, 0.04, 0.20),
        "M_vs_B3": _comparison(0.10, 0.04, 0.20),
        "M_vs_B5": _comparison(0.08, 0.04, 0.16),
    }


def _families(left: float = 0.05, right: float = 0.08) -> dict:
    return {"entity_time_compare": left, "empty_retrieval_recovery": right}


def test_config_locks_thresholds_and_profile_sizes():
    cfg = load_config()
    assert cfg.protocol_id == "B-v1"
    assert cfg.min_gain == 0.03
    assert cfg.retention_tolerance == 0.01
    assert cfg.counts("pilot")["train"]["entity_time_compare"] == 48
    raw = dict(cfg.raw)
    raw["decision"] = dict(raw["decision"])
    raw["decision"]["min_gain"] = 0.0
    with pytest.raises(ConfigPinError):
        from expolicy.paper_b.config import _validate

        _validate(raw)


def test_practical_floor_rises_when_the_eval_set_is_small():
    assert practical_floor(0.03, 300, 2) == 0.03
    assert practical_floor(0.03, 16, 2) == pytest.approx(2 / 16)


def test_decision_labels_follow_the_preregistered_bounds():
    kwargs = dict(
        floor=0.03,
        retention_delta=0.0,
        retention_tolerance=0.01,
        family_diffs=_families(),
        replacement_changed=True,
        trigger_rate=0.5,
        illegal_rate=0.0,
    )
    assert judge(_clear_comparisons(), **kwargs)["status"] == B_GO

    tied = {
        "M_vs_B1": _comparison(0.0, -0.01, 0.01),
        "M_vs_B3": _comparison(0.0, -0.01, 0.01),
        "M_vs_B5": _comparison(0.0, -0.01, 0.01),
    }
    content_kwargs = dict(kwargs)
    content_kwargs["family_diffs"] = _families(0.0, 0.0)
    assert judge(tied, **content_kwargs)["status"] == B_NO_GO_CONTENT

    mechanism = dict(_clear_comparisons())
    mechanism["M_vs_B5"] = _comparison(0.0, -0.01, 0.01)
    assert judge(mechanism, **kwargs)["status"] == B_NO_GO_MECHANISM

    wide = dict(_clear_comparisons())
    wide["M_vs_B1"] = _comparison(0.05, -0.02, 0.20)
    assert judge(wide, **kwargs)["status"] == B_INCONCLUSIVE

    quiet = dict(kwargs)
    quiet["trigger_rate"] = 0.0
    assert judge(_clear_comparisons(), **quiet)["status"] == B_INCONCLUSIVE
    missing = judge(
        None,
        floor=0.03,
        retention_delta=None,
        retention_tolerance=0.01,
        family_diffs={},
        replacement_changed=False,
        trigger_rate=None,
        illegal_rate=None,
    )
    assert missing["status"] == B_NOT_RUN


def test_static_validation_rejects_hidden_fields_and_free_text():
    registry = build_runtime_registry()
    leaked = Rule(
        rule_id="leak",
        family="entity_time_compare",
        trigger={"all": [{"feature": "gold_answer", "op": "eq", "value": "secret"}]},
        action_template="search_entity_attribute",
    )
    assert static_errors(leaked, registry, [])
    copied = Rule(
        rule_id="copy",
        family="empty_retrieval_recovery",
        trigger={"all": [{"feature": "attribute_kind", "op": "eq", "value": "director"}]},
        action_template="rewrite_empty_query",
        parameters={"rewrite_template": "{entity} biography", "answer": "Dirsecret"},
    )
    assert "rewrite" in static_errors(copied, registry, [])
    blocked = Rule(
        rule_id="blocked",
        family="empty_retrieval_recovery",
        trigger={"all": [{"feature": "attribute_kind", "op": "eq", "value": "director"}]},
        action_template="rewrite_empty_query",
        parameters={"rewrite_template": "{entity} biography"},
        support_trace_ids=["trace-with-Dirsecret-token"],
    )
    assert "blocklist" in static_errors(blocked, registry, ["Dirsecret"])


def test_unmatched_rule_matches_the_base_agent_and_budget_is_enforced():
    registry = build_runtime_registry()
    task = next(task for task in build_world(feasibility_counts(), seed=1) if task.family == "entity_time_compare")
    assert check_unmatched_falls_back(task, registry, registry.budget) == []
    tight_budget = Budget(max_tool_calls=1, max_steps=4, max_tokens=2048, max_interventions=1)
    tight = build_runtime_registry(budget=tight_budget)
    episode = execute_rules(task, generic_rules(), tight, budget=tight_budget)
    assert episode.tool_calls <= 1


def test_split_audit_rejects_shared_entities():
    tasks = build_world(feasibility_counts(), seed=1)
    audit = audit_splits(tasks, full_counts={"near": 100})
    assert audit["ok"]
    assert audit["precision_warning"]
    first = tasks[0]
    tasks.append(
        replace(
            first,
            task_id="other-id",
            split="near",
            db_object="page:different",
            gold_answer="unused",
        )
    )
    assert "entity_overlap" in audit_splits(tasks)["errors"]


def test_pairs_hide_answers_and_mismatch_breaks_the_repair():
    cfg = load_config()
    registry = cfg.registry()
    world = build_world(feasibility_counts(), seed=1)
    train = [task for task in world if task.split == "train"]
    unrelated = [task for task in world if task.split == "unrelated"]
    traces, _, _ = _collect(train, registry, cfg.budget(), ("base", "exploration"))
    other, _, _ = _collect(unrelated, registry, cfg.budget(), ("base",))
    pairs = build_trace_pairs(traces)
    assert pairs
    assert all(pair.alignment == "same_prefix" for pair in pairs)
    for pair in pairs:
        view = pair.inducer_view()
        assert "gold_answer" not in view
        assert "question" not in view
        assert "family" not in view
    prompt = build_inducer_prompt({"pairs": [pair.inducer_view() for pair in pairs]}, registry)
    assert "gold_answer" not in prompt
    assert all(task.gold_answer not in prompt for task in train)
    controls = make_experience_controls(pairs, traces, other)
    assert controls["audit"]["association_broken"]
    assert controls["audit"]["empty_has_actions"] is False
    assert controls["audit"]["unrelated_real"]
    empty = induce_rules(mode="empty", condition=B3, registry=registry, empty_count=len(traces))
    assert executable_hash(empty["rules"]) == executable_hash(generic_rules())
    direct = induce_rules(mode="direct", condition=B5, registry=registry, traces=traces)
    assert direct["rules"]
    assert all(
        len(rule.trigger["all"]) == 1 and rule.trigger["all"][0]["feature"] == "search_count"
        for rule in direct["rules"]
    )


def test_counterexample_replay_tightens_a_broad_rule():
    cfg = load_config()
    registry = cfg.registry()
    world = build_world(feasibility_counts(), seed=1)
    train = [task for task in world if task.split == "train"]
    val = [task for task in world if task.split == "val" and task.family != "retention_other"]
    retention = [task for task in world if task.split == "val" and task.family == "retention_other"]
    broad = Rule(
        rule_id="broad",
        family="empty_retrieval_recovery",
        trigger={"all": [{"feature": "last_observation_empty", "op": "eq", "value": 1}]},
        action_template=TEMPLATE_REWRITE,
        parameters={"rewrite_template": "{entity} biography"},
    )
    validated = validate_rules(
        [broad],
        condition="M",
        train_tasks=train,
        val_tasks=val,
        retention_tasks=retention,
        registry=registry,
        budget=cfg.budget(),
        generic_rules=generic_rules(),
        traces=[],
        counterexamples=True,
        blocklist=[],
        retention_tolerance=0.01,
    )
    assert len(validated["rules"]) == 1
    rule = validated["rules"][0]
    assert rule.revision_count == 1
    features = {atom["feature"] for atom in rule.trigger["all"]}
    assert "has_comparison_cue" in features or "attribute_kind" in features


def test_executor_does_not_copy_the_gold_answer():
    cfg = load_config()
    world = build_world(feasibility_counts(), seed=1)
    recovery = next(task for task in world if task.split == "train" and task.family == "empty_retrieval_recovery")
    hidden = replace(recovery, gold_answer="WRONGANSWER")
    episode = execute_rules(hidden, [], cfg.registry(), budget=cfg.budget())
    assert episode.answer != "WRONGANSWER"
    assert episode.success is False


def _rate(rows: list[dict[str, str]], condition: str, splits: set[str]) -> float:
    subset = [row for row in rows if row["condition"] == condition and row["split"] in splits]
    assert subset
    return sum(int(row["success"]) for row in subset) / len(subset)


def test_feasibility_harness_separates_content_and_mechanism(tmp_path: Path):
    cfg = load_config()
    decision = run_protocol(cfg, tmp_path, profile="feasibility")
    with (tmp_path / "B_TASK_RESULTS.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    target = {"near", "cross"}
    assert _rate(rows, M, target) == 1.0
    assert _rate(rows, B1, target) == 0.5
    assert _rate(rows, B3, target) == 0.5
    assert _rate(rows, B5, target) == 0.5
    assert _rate(rows, B0, target) == 0.0
    assert _rate(rows, B2, target) == 0.0
    assert _rate(rows, B4, target) == 0.0
    for condition in (B0, B1, B2, B3, B4, B5, M):
        assert _rate(rows, condition, {"retention"}) == 1.0
    rules = json.loads((tmp_path / "B_RULES.json").read_text(encoding="utf-8"))
    assert all(rule["action_template"] != TEMPLATE_REWRITE for rule in rules["conditions"]["B3"])
    rewrites = [rule for rule in rules["conditions"]["M"] if rule["action_template"] == TEMPLATE_REWRITE]
    assert rewrites
    assert rewrites[0]["parameters"]["rewrite_template"] == "{entity} biography"
    direct = rules["conditions"]["B5"]
    assert direct
    assert all(atom["feature"] == "search_count" for rule in direct for atom in rule["trigger"]["all"])
    audit = json.loads((tmp_path / "control_audit.json").read_text(encoding="utf-8"))
    assert audit["b1_b3_same_executable"] is True
    assert audit["association_broken"] is True
    blob = (tmp_path / "B_RULES.json").read_text(encoding="utf-8")
    for task in build_world(feasibility_counts(), seed=1):
        if task.family != "entity_time_compare":
            assert task.gold_answer not in blob
    assert decision["paper_claim_allowed"] is False
    assert decision["status"] in {B_GO, B_INCONCLUSIVE, B_NO_GO_CONTENT, B_NO_GO_MECHANISM}
    for name in (
        "B_PROTOCOL.yaml",
        "B_SPLIT_AUDIT.md",
        "B_TRACE_PAIRS.jsonl",
        "B_RULES.json",
        "B_RULE_PROVENANCE.jsonl",
        "B_CONTROL_AUDIT.md",
        "B_TASK_RESULTS.csv",
        "B_TRIGGER_LOG.jsonl",
        "B_COST_LEDGER.jsonl",
        "B_DECISION.md",
    ):
        assert (tmp_path / name).exists()
    ledger = [
        json.loads(line)
        for line in (tmp_path / "B_COST_LEDGER.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["condition"] == "B3" and row["read_experience"] is False for row in ledger)
    assert any(row["condition"] == "M" and row["read_experience"] is True for row in ledger)
