"""Propose conditional rules from aligned experience, or from a prior when experience is removed."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from expolicy.paper_b.constants import (
    ALIGNED,
    ALIGNMENT_MISMATCHED,
    FALLBACK_BASE,
    FAMILY_COMPARE,
    FAMILY_RECOVERY,
    INDUCER_VERSION,
    TEMPLATE_REWRITE,
    TRIGGER_FEATURES,
)
from expolicy.paper_b.models import Rule, Trace, TracePair
from expolicy.paper_b.registry import RuntimeRegistry, generic_rules
from expolicy.r9_lab.io_util import sha256_json

GenerateFn = Callable[[str, float], str]

INDUCER_HEADER = """Infer executable trigger rules from the packet.
Return a JSON list. Each item has trigger, action_template, and parameters.
Use only catalog features, operators, and templates.
Do not add tools, Python, task ids, questions, or answers.
trigger.all has at most 3 atoms.
"""


def executable_hash(rules: list[Rule]) -> str:
    bodies = [rule.executable() for rule in rules]
    bodies.sort(key=lambda body: sha256_json(body))
    return sha256_json(bodies)


def inducer_hash(mode: str, *, used_model: bool) -> str:
    return sha256_json({"version": INDUCER_VERSION, "mode": mode, "used_model": used_model})


def build_inducer_prompt(packet: dict[str, Any], registry: RuntimeRegistry) -> str:
    safe = {
        "catalog": registry.catalog(),
        "packet": _whitelist_packet(packet),
    }
    return INDUCER_HEADER + json.dumps(safe, ensure_ascii=False, sort_keys=True)


def induce_rules(
    *,
    mode: str,
    condition: str,
    registry: RuntimeRegistry,
    pairs: list[TracePair] | None = None,
    traces: list[Trace] | None = None,
    empty_count: int = 0,
    generate_fn: GenerateFn | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Propose at most one batch of candidates per template. No counterexample filter here."""
    packet = _packet(mode, pairs or [], traces or [], empty_count)
    prompt = build_inducer_prompt(packet, registry)
    rejected: list[dict[str, Any]] = []
    llm_calls = 0
    if generate_fn is not None:
        llm_calls = 1
        raw = generate_fn(prompt, temperature)
        rules, rejected = _rules_from_model(raw, condition, registry)
    elif mode == "empty":
        rules = _clone_prior(condition)
    elif mode == "direct":
        rules = _propose_direct(traces or [], registry, condition)
    elif mode == "mismatched":
        rules = _propose_aligned(pairs or [], registry, condition, {ALIGNMENT_MISMATCHED})
    else:
        rules = _propose_aligned(pairs or [], registry, condition, ALIGNED)
    rules = _cap_batches(rules, registry.max_candidates_per_batch)
    return {
        "rules": rules,
        "rejected": rejected,
        "prompt": prompt,
        "llm_calls": llm_calls,
        "mode": mode,
        "read_experience": mode != "empty",
        "inducer_hash": inducer_hash(mode, used_model=generate_fn is not None),
    }


def _packet(mode: str, pairs: list[TracePair], traces: list[Trace], empty_count: int) -> dict[str, Any]:
    if mode == "empty":
        return {"experience_content_removed": True, "n_experiences": empty_count, "pairs": []}
    if mode == "direct":
        actions = []
        for trace in traces:
            if not trace.success or not trace.steps:
                continue
            for step in trace.steps:
                actions.append(
                    {
                        "action_template": step.action_template,
                        "parameters": dict(step.parameters),
                        "initial_search_count": trace.steps[0].features_before.get("search_count", 0),
                    }
                )
        return {"experience_content_removed": False, "pairs": [], "success_actions": actions}
    allowed = {ALIGNMENT_MISMATCHED} if mode == "mismatched" else ALIGNED
    return {
        "experience_content_removed": False,
        "pairs": [pair.inducer_view() for pair in pairs if pair.alignment in allowed],
    }


def _whitelist_packet(packet: dict[str, Any]) -> dict[str, Any]:
    pairs = []
    for pair in packet.get("pairs") or []:
        if not isinstance(pair, dict):
            continue
        pairs.append(
            {
                "pair_id": pair.get("pair_id"),
                "alignment": pair.get("alignment"),
                "quality": pair.get("quality"),
                "diverge_index": pair.get("diverge_index"),
                "features_before": dict(pair.get("features_before") or {}),
                "success_action": pair.get("success_action"),
                "success_params": dict(pair.get("success_params") or {}),
                "failure_action": pair.get("failure_action"),
                "success_trace_id": pair.get("success_trace_id"),
                "failure_trace_id": pair.get("failure_trace_id"),
            }
        )
    actions = []
    for action in packet.get("success_actions") or []:
        if not isinstance(action, dict):
            continue
        actions.append(
            {
                "action_template": action.get("action_template"),
                "parameters": dict(action.get("parameters") or {}),
                "initial_search_count": action.get("initial_search_count", 0),
            }
        )
    return {
        "experience_content_removed": bool(packet.get("experience_content_removed")),
        "n_experiences": int(packet.get("n_experiences") or 0),
        "pairs": pairs,
        "success_actions": actions,
    }


def _clone_prior(condition: str) -> list[Rule]:
    clones = []
    for rule in generic_rules():
        clones.append(_materialize(rule.executable(), [], condition))
    return clones


def _propose_aligned(
    pairs: list[TracePair],
    registry: RuntimeRegistry,
    condition: str,
    alignments: set[str],
) -> list[Rule]:
    grouped: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        if pair.alignment not in alignments or pair.quality <= 0:
            continue
        if pair.success_action not in registry.templates:
            continue
        atoms = _atoms_from_features(pair.features_before, registry.max_atoms)
        if not atoms:
            continue
        executable = {
            "trigger": {"all": atoms},
            "action_template": pair.success_action,
            "parameters": dict(pair.success_params),
            "fallback": FALLBACK_BASE,
        }
        if not _parameters_ok(executable, registry):
            continue
        slot = grouped.setdefault(sha256_json(executable), {"executable": executable, "supports": []})
        slot["supports"].append(pair.success_trace_id)
    return [_materialize(slot["executable"], slot["supports"], condition) for slot in grouped.values()]


def _propose_direct(traces: list[Trace], registry: RuntimeRegistry, condition: str) -> list[Rule]:
    """Bind every successful action to the initial state, with no alignment and no revision."""
    grouped: dict[str, dict[str, Any]] = {}
    for trace in traces:
        if not trace.success or not trace.steps:
            continue
        initial = trace.steps[0].features_before.get("search_count", 0)
        atom = {"feature": "search_count", "op": "eq", "value": initial}
        for step in trace.steps:
            if step.action_template not in registry.templates:
                continue
            executable = {
                "trigger": {"all": [atom]},
                "action_template": step.action_template,
                "parameters": dict(step.parameters),
                "fallback": FALLBACK_BASE,
            }
            if not _parameters_ok(executable, registry):
                continue
            slot = grouped.setdefault(sha256_json(executable), {"executable": executable, "supports": []})
            slot["supports"].append(trace.trace_id)
    return [_materialize(slot["executable"], slot["supports"], condition) for slot in grouped.values()]


def _atoms_from_features(features: dict[str, Any], max_atoms: int) -> list[dict[str, Any]]:
    atoms = []
    for name in TRIGGER_FEATURES:
        if name not in features:
            continue
        value = features[name]
        if isinstance(value, str) and len(value) > 40:
            continue
        atoms.append({"feature": name, "op": "eq", "value": value})
        if len(atoms) >= max_atoms:
            break
    return atoms


def _parameters_ok(executable: dict[str, Any], registry: RuntimeRegistry) -> bool:
    template = executable["action_template"]
    parameters = executable.get("parameters") or {}
    if template == TEMPLATE_REWRITE:
        rewrite = parameters.get("rewrite_template")
        return set(parameters) == {"rewrite_template"} and rewrite in registry.rewrite_templates
    return parameters == {}


def _materialize(executable: dict[str, Any], supports: list[str], condition: str) -> Rule:
    template = str(executable["action_template"])
    family = FAMILY_RECOVERY if template == TEMPLATE_REWRITE else FAMILY_COMPARE
    return Rule(
        rule_id=f"{condition}_{sha256_json(executable)[:10]}",
        family=family,
        trigger={"all": [dict(atom) for atom in executable["trigger"]["all"]]},
        action_template=template,
        parameters=dict(executable.get("parameters") or {}),
        fallback=FALLBACK_BASE,
        support_trace_ids=sorted(set(supports)),
    )


def _cap_batches(rules: list[Rule], max_candidates: int) -> list[Rule]:
    batches: dict[str, list[Rule]] = defaultdict(list)
    for rule in rules:
        batches[rule.action_template].append(rule)
    kept: list[Rule] = []
    for template in sorted(batches):
        batch = sorted(batches[template], key=lambda rule: (-len(rule.support_trace_ids), rule.rule_id))
        kept.extend(batch[:max_candidates])
    return kept


def _rules_from_model(raw: str, condition: str, registry: RuntimeRegistry) -> tuple[list[Rule], list[dict[str, Any]]]:
    parsed = _extract_json(raw)
    if isinstance(parsed, dict):
        parsed = parsed.get("rules") or [parsed]
    if not isinstance(parsed, list):
        return [], [{"reason": "unparsed"}]
    rules: list[Rule] = []
    rejected: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            rejected.append({"reason": "not_object"})
            continue
        executable = {
            "trigger": item.get("trigger") or {},
            "action_template": item.get("action_template"),
            "parameters": dict(item.get("parameters") or {}),
            "fallback": FALLBACK_BASE,
        }
        template = executable["action_template"]
        if template not in registry.templates or not _parameters_ok(executable, registry):
            rejected.append({"reason": "schema", "action_template": template})
            continue
        atoms = executable["trigger"].get("all") if isinstance(executable["trigger"], dict) else None
        if not isinstance(atoms, list) or not atoms:
            rejected.append({"reason": "trigger"})
            continue
        rules.append(_materialize(executable, [], condition))
    return rules, rejected


def _extract_json(raw: str) -> Any:
    if not raw:
        return None
    start = raw.find("[")
    obj = raw.find("{")
    if start == -1 or (obj != -1 and obj < start):
        start = obj
    if start == -1:
        return None
    try:
        return json.loads(raw[start:])
    except json.JSONDecodeError:
        end = max(raw.rfind("]"), raw.rfind("}"))
        if end <= start:
            return None
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
