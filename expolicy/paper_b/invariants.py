"""Invariants that the protocol requires before a number can be interpreted."""

from __future__ import annotations

import json

from expolicy.paper_b.execute import execute_rules
from expolicy.paper_b.models import Budget, Rule, Task
from expolicy.paper_b.registry import RuntimeRegistry
from expolicy.paper_b.validate import static_errors


def check_rule_visibility(rules: list[Rule], registry: RuntimeRegistry, blocklist: list[str]) -> list[str]:
    errors = []
    for rule in rules:
        found = static_errors(rule, registry, blocklist)
        if found:
            errors.append(f"{rule.rule_id}:{'|'.join(found)}")
    return errors


def answer_leakage(rules: list[Rule], answers: list[str]) -> list[str]:
    """A frozen rule must not contain a split's answer string."""
    hits = []
    for rule in rules:
        blob = json.dumps(rule.to_dict(), ensure_ascii=False, sort_keys=True).casefold()
        for answer in answers:
            if len(answer) >= 6 and answer.casefold() in blob:
                hits.append(rule.rule_id)
                break
    return hits


def check_unmatched_falls_back(task: Task, registry: RuntimeRegistry, budget: Budget) -> list[str]:
    """A trigger that cannot match must reproduce the base agent's actions."""
    dormant = Rule(
        rule_id="dormant_unmatched",
        family="entity_time_compare",
        trigger={"all": [{"feature": "n_entities", "op": "eq", "value": 99}]},
        action_template="search_entity_attribute",
        parameters={},
    )
    base = execute_rules(task, [], registry, budget=budget)
    held = execute_rules(task, [dormant], registry, budget=budget)
    if base.action_templates != held.action_templates or base.answer != held.answer:
        return ["unmatched_rule_changed_behavior"]
    if held.modified_actions != 0 or held.triggered_rules:
        return ["unmatched_rule_triggered"]
    return []


def check_budget_headroom(budget: Budget) -> list[str]:
    if budget.max_interventions > budget.max_tool_calls:
        return ["rule_budget_exceeds_tool_budget"]
    if budget.max_tool_calls < 1 or budget.max_steps < 1:
        return ["budget_not_positive"]
    return []


def check_confirmation_sealed(train: list[Task], confirmation: list[Task]) -> list[str]:
    train_ids = {task.task_id for task in train}
    train_entities = {task.entity_key for task in train}
    errors = []
    if train_ids & {task.task_id for task in confirmation}:
        errors.append("confirmation_task_overlap")
    if train_entities & {task.entity_key for task in confirmation}:
        errors.append("confirmation_entity_overlap")
    return errors
