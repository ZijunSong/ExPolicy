"""Static legality, train replay, and acceptance on tasks that did not propose the rules."""

from __future__ import annotations

import json
from typing import Any

from expolicy.paper_b.constants import (
    FALLBACK_BASE,
    FORBIDDEN_FEATURES,
    FORBIDDEN_SUBSTRINGS,
    TEMPLATE_REWRITE,
    TRIGGER_FEATURES,
)
from expolicy.paper_b.execute import execute_rules
from expolicy.paper_b.induce import executable_hash
from expolicy.paper_b.models import Budget, Episode, Rule, Task, Trace
from expolicy.paper_b.registry import RuntimeRegistry
from expolicy.r9_lab.io_util import sha256_json


def validate_rules(
    candidates: list[Rule],
    *,
    condition: str,
    train_tasks: list[Task],
    val_tasks: list[Task],
    retention_tasks: list[Task],
    registry: RuntimeRegistry,
    budget: Budget,
    generic_rules: list[Rule],
    traces: list[Trace],
    counterexamples: bool,
    blocklist: list[str],
    retention_tolerance: float,
) -> dict[str, Any]:
    """Three gates: static schema, optional train replay, then held-out acceptance."""
    proposed = len(candidates)
    static_ok: list[Rule] = []
    rejected: list[dict[str, Any]] = []
    for rule in candidates:
        errors = static_errors(rule, registry, blocklist)
        if errors:
            rejected.append({"rule_id": rule.rule_id, "stage": "static", "errors": errors})
            continue
        rule.family = registry.template_family(rule.action_template)
        static_ok.append(rule)
    tokens = 0
    if counterexamples:
        base, base_tokens = _base_episodes(train_tasks, registry, budget)
        tokens += base_tokens
        trace_ids = _trace_ids(traces)
        revised, replay_tokens, replay_rejected = _revise(
            static_ok,
            train_tasks,
            base,
            registry,
            budget,
            condition,
            trace_ids,
            registry.max_revisions,
        )
        tokens += replay_tokens
        rejected.extend(replay_rejected)
    else:
        revised = static_ok
    revised = _cap(revised, registry.max_rules_per_family, registry.max_rules)
    accepted, accept_tokens, scores = _accept(
        revised,
        val_tasks,
        retention_tasks,
        generic_rules,
        registry,
        budget,
        retention_tolerance,
    )
    tokens += accept_tokens
    record = f"val_{condition}_{executable_hash(accepted)[:8]}"
    for rule in accepted:
        rule.validation_record = record
    return {
        "rules": accepted,
        "rejected": rejected,
        "accepted": bool(accepted),
        "proposed": proposed,
        "static_rejected": sum(1 for row in rejected if row.get("stage") == "static"),
        "validation_tokens": tokens,
        "validation_record": record,
        **scores,
    }


def static_errors(rule: Rule, registry: RuntimeRegistry, blocklist: list[str]) -> list[str]:
    errors: list[str] = []
    atoms = list(rule.trigger.get("all") or [])
    if not atoms or len(atoms) > registry.max_atoms:
        errors.append("atom_count")
    for atom in atoms:
        feature = str(atom.get("feature") or "")
        if feature in FORBIDDEN_FEATURES or feature not in registry.trigger_features:
            errors.append(f"feature:{feature}")
        if str(atom.get("op") or "") not in registry.ops:
            errors.append("op")
        value = atom.get("value")
        if isinstance(value, str) and len(value) > 40:
            errors.append("value_length")
    if rule.action_template not in registry.templates:
        errors.append("template")
    if rule.fallback != FALLBACK_BASE:
        errors.append("fallback")
    parameters = dict(rule.parameters)
    if rule.action_template == TEMPLATE_REWRITE:
        rewrite = parameters.get("rewrite_template")
        if set(parameters) != {"rewrite_template"} or rewrite not in registry.rewrite_templates:
            errors.append("rewrite")
    elif parameters:
        errors.append("params")
    blob = json.dumps(rule.to_dict(), ensure_ascii=False, sort_keys=True).casefold()
    for bad in FORBIDDEN_SUBSTRINGS:
        if bad.casefold() in blob:
            errors.append(f"forbidden:{bad}")
    for token in blocklist:
        if len(token) >= 6 and token.casefold() in blob:
            errors.append("blocklist")
            break
    return errors


def _base_episodes(tasks: list[Task], registry: RuntimeRegistry, budget: Budget) -> tuple[dict[str, Episode], int]:
    episodes = {}
    tokens = 0
    for task in tasks:
        episode = execute_rules(task, [], registry, budget=budget)
        episodes[task.task_id] = episode
        tokens += episode.tokens
    return episodes, tokens


def _trace_ids(traces: list[Trace]) -> dict[str, str]:
    index: dict[str, str] = {}
    for trace in traces:
        index.setdefault(trace.task_id, trace.trace_id)
    return index


def _revise(
    rules: list[Rule],
    tasks: list[Task],
    base: dict[str, Episode],
    registry: RuntimeRegistry,
    budget: Budget,
    condition: str,
    trace_ids: dict[str, str],
    max_revisions: int,
) -> tuple[list[Rule], int, list[dict[str, Any]]]:
    kept: list[Rule] = []
    rejected: list[dict[str, Any]] = []
    tokens = 0
    for original in rules:
        current = _copy_rule(original)
        seen_counters: list[str] = []
        for attempt in range(max_revisions + 1):
            fires, spent = _replay(current, tasks, base, registry, budget)
            tokens += spent
            supports = [fire for fire in fires if fire["improved"]]
            counters = [fire for fire in fires if fire["counterexample"]]
            if attempt == 0:
                seen_counters = [_trace_id(fire["task_id"], trace_ids) for fire in counters]
            if supports and not counters:
                current.counterexample_trace_ids = seen_counters
                current.support_trace_ids = sorted(
                    set(current.support_trace_ids) | {_trace_id(fire["task_id"], trace_ids) for fire in supports}
                )
                kept.append(current)
                break
            if not supports or attempt >= max_revisions:
                rejected.append(
                    {
                        "rule_id": current.rule_id,
                        "stage": "replay",
                        "errors": ["no_support" if not supports else "counters_remain"],
                    }
                )
                break
            extra = _separating_atom(supports, counters, current, registry.max_atoms)
            if extra is None:
                rejected.append({"rule_id": current.rule_id, "stage": "replay", "errors": ["cannot_separate"]})
                break
            current.trigger["all"].append(extra)
            current.revision_count += 1
            current.rule_id = f"{condition}_{sha256_json(current.executable())[:10]}"
            current.family = registry.template_family(current.action_template)
    return kept, tokens, rejected


def _replay(
    rule: Rule,
    tasks: list[Task],
    base: dict[str, Episode],
    registry: RuntimeRegistry,
    budget: Budget,
) -> tuple[list[dict[str, Any]], int]:
    fires = []
    tokens = 0
    for task in tasks:
        episode = execute_rules(task, [rule], registry, budget=budget)
        tokens += episode.tokens
        if rule.rule_id not in episode.triggered_rules:
            continue
        base_ep = base[task.task_id]
        improved = bool(episode.success) and not base_ep.success
        harmed = (not episode.success) and bool(base_ep.success)
        wasted = (not episode.success) and (not base_ep.success)
        features = {}
        for fire in episode.fire_features:
            if fire["rule_id"] == rule.rule_id:
                features = dict(fire["features"])
                break
        fires.append(
            {
                "task_id": task.task_id,
                "improved": improved,
                "counterexample": harmed or wasted,
                "features": features,
            }
        )
    return fires, tokens


def _separating_atom(
    supports: list[dict[str, Any]],
    counters: list[dict[str, Any]],
    rule: Rule,
    max_atoms: int,
) -> dict[str, Any] | None:
    existing = {str(atom.get("feature")) for atom in rule.trigger.get("all") or []}
    if len(existing) >= max_atoms:
        return None
    for feature in TRIGGER_FEATURES:
        if feature in existing:
            continue
        values = [fire["features"].get(feature) for fire in supports]
        if not values or any(value is None for value in values):
            continue
        if len({json.dumps(value, sort_keys=True) for value in values}) != 1:
            continue
        good = values[0]
        if isinstance(good, str) and len(good) > 40:
            continue
        bad = [fire["features"].get(feature) for fire in counters]
        if bad and all(item != good for item in bad):
            return {"feature": feature, "op": "eq", "value": good}
    return None


def _accept(
    rules: list[Rule],
    val_tasks: list[Task],
    retention_tasks: list[Task],
    generic: list[Rule],
    registry: RuntimeRegistry,
    budget: Budget,
    retention_tolerance: float,
) -> tuple[list[Rule], int, dict[str, float]]:
    generic_target, generic_tokens = _success_rate(val_tasks, generic, registry, budget)
    generic_ret, generic_ret_tokens = _success_rate(retention_tasks, generic, registry, budget)
    target, target_tokens = _success_rate(val_tasks, rules, registry, budget)
    retention, retention_tokens = _success_rate(retention_tasks, rules, registry, budget)
    tokens = generic_tokens + generic_ret_tokens + target_tokens + retention_tokens
    scores = {
        "val_target_success": target,
        "val_retention_success": retention,
        "generic_target_success": generic_target,
        "generic_retention_success": generic_ret,
    }
    if target + 1e-12 < generic_target:
        return [], tokens, scores
    if generic_ret - retention > retention_tolerance + 1e-12:
        return [], tokens, scores
    return rules, tokens, scores


def _success_rate(
    tasks: list[Task],
    rules: list[Rule],
    registry: RuntimeRegistry,
    budget: Budget,
) -> tuple[float, int]:
    if not tasks:
        return 0.0, 0
    episodes = [execute_rules(task, rules, registry, budget=budget) for task in tasks]
    rate = sum(1.0 if episode.success else 0.0 for episode in episodes) / len(episodes)
    return rate, sum(episode.tokens for episode in episodes)


def _cap(rules: list[Rule], per_family: int, total: int) -> list[Rule]:
    grouped: dict[str, list[Rule]] = {}
    for rule in rules:
        grouped.setdefault(rule.family, []).append(rule)
    kept: list[Rule] = []
    for family in sorted(grouped):
        batch = sorted(grouped[family], key=lambda rule: (-len(rule.support_trace_ids), rule.rule_id))
        kept.extend(batch[:per_family])
    kept.sort(key=lambda rule: (-len(rule.support_trace_ids), rule.rule_id))
    return kept[:total]


def _copy_rule(rule: Rule) -> Rule:
    return Rule(
        rule_id=rule.rule_id,
        family=rule.family,
        trigger={"all": [dict(atom) for atom in rule.trigger.get("all") or []]},
        action_template=rule.action_template,
        parameters=dict(rule.parameters),
        fallback=rule.fallback,
        support_trace_ids=list(rule.support_trace_ids),
        counterexample_trace_ids=list(rule.counterexample_trace_ids),
        validation_record=rule.validation_record,
        revision_count=rule.revision_count,
    )


def _trace_id(task_id: str, trace_ids: dict[str, str]) -> str:
    return trace_ids.get(task_id, f"task:{task_id}")
