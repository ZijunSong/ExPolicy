"""Freeze a rule file, then score every condition with the same scorer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from expolicy.paper_b.artifacts import assert_rules_unchanged, write_rules
from expolicy.paper_b.constants import ROW_FIELDS, SPLIT_RETENTION, STATUS_OK
from expolicy.paper_b.execute import execute_rules
from expolicy.paper_b.induce import executable_hash
from expolicy.paper_b.models import Budget, Rule, Task
from expolicy.paper_b.registry import RuntimeRegistry


def freeze_and_evaluate(
    *,
    tasks: list[Task],
    conditions: dict[str, list[Rule]],
    text_context: dict[str, str],
    registry: RuntimeRegistry,
    budget: Budget,
    rules_path: Path,
    runtime_hash: str,
    pool_hash: dict[str, str],
    inducer_hash: dict[str, str],
    generation_cost: dict[str, int],
    validation_cost: dict[str, int],
) -> dict[str, Any]:
    payload = {
        "runtime_hash": runtime_hash,
        "conditions": {name: [rule.to_dict() for rule in rules] for name, rules in conditions.items()},
    }
    digest = write_rules(rules_path, payload)
    rows: list[dict[str, Any]] = []
    triggers: list[dict[str, Any]] = []
    for condition, rules in conditions.items():
        rule_hash = executable_hash(rules)
        context = text_context.get(condition, "")
        for task in tasks:
            episode = execute_rules(task, rules, registry, budget=budget, text_context=context)
            if episode.tool_calls > budget.max_tool_calls or episode.interventions > budget.max_interventions:
                raise RuntimeError(f"budget exceeded on {condition} {task.task_id}")
            row = {key: None for key in ROW_FIELDS}
            row.update(
                {
                    "experience_pool_hash": pool_hash.get(condition, ""),
                    "inducer_hash": inducer_hash.get(condition, ""),
                    "rule_hash": rule_hash,
                    "runtime_hash": runtime_hash,
                    "condition": condition,
                    "source_group": task.source_group,
                    "task_id": task.task_id,
                    "split": task.split,
                    "family": task.family,
                    "triggered_rules": "|".join(episode.triggered_rules),
                    "modified_actions": episode.modified_actions,
                    "success": int(episode.success),
                    "retention_flag": int(task.split == SPLIT_RETENTION),
                    "tool_calls": episode.tool_calls,
                    "tokens": episode.tokens,
                    "generation_cost": generation_cost.get(condition, 0),
                    "validation_cost": validation_cost.get(condition, 0),
                    "infra_status": episode.infra_status or STATUS_OK,
                }
            )
            rows.append(row)
            triggers.append(
                {
                    "condition": condition,
                    "task_id": task.task_id,
                    "split": task.split,
                    "triggered_rules": list(episode.triggered_rules),
                    "matched_rules": list(episode.matched_rules),
                    "modified_actions": episode.modified_actions,
                    "fallbacks": episode.fallbacks,
                    "action_templates": list(episode.action_templates),
                    "tool_calls": episode.tool_calls,
                    "infra_status": episode.infra_status,
                }
            )
    assert_rules_unchanged(rules_path, digest)
    return {"rows": rows, "triggers": triggers, "rules_digest": digest}
