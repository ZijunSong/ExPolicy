"""B-v2 induction. Every condition uses the same generator and the full action catalog."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from expolicy.paper_b.models import Rule, Trace, TracePair
from expolicy.paper_b.public_exec import PUBLIC_FEATURES, REWRITE_STRATEGIES
from expolicy.r9_lab.io_util import sha256_json

GenerateFn = Callable[[str, float], str]

HEADER = """Return a JSON list of executable rules.
Each rule has trigger.all, action_template, and parameters.
Use only the catalog features, templates, and rewrite strategies.
Do not add tools, task ids, questions, or answers.
At most 3 trigger atoms.
"""


def catalog() -> dict[str, Any]:
    return {
        "features": list(PUBLIC_FEATURES),
        "templates": ["rewrite_query", "open_top"],
        "rewrite_strategies": list(REWRITE_STRATEGIES),
        "ops": ["eq", "neq", "gte", "lte"],
    }


def induce_v2(
    *,
    mode: str,
    condition: str,
    generate_fn: GenerateFn,
    temperature: float,
    pairs: list[TracePair] | None = None,
    traces: list[Trace] | None = None,
    empty_count: int = 0,
    max_candidates: int = 4,
) -> dict[str, Any]:
    if generate_fn is None:
        raise RuntimeError("B-v2 induction requires a model. It does not clone a handwritten prior.")
    packet = _packet(mode, pairs or [], traces or [], empty_count)
    prompt = HEADER + json.dumps({"catalog": catalog(), "packet": packet}, ensure_ascii=False, sort_keys=True)
    raw = generate_fn(prompt, temperature)
    rules, rejected = _parse(raw, condition, max_candidates)
    return {
        "rules": rules,
        "rejected": rejected,
        "prompt": prompt,
        "llm_calls": 1,
        "mode": mode,
        "read_experience": mode != "empty",
        "inducer_hash": sha256_json({"protocol": "B-v2", "mode": mode, "catalog": catalog()}),
    }


def _packet(mode: str, pairs: list[TracePair], traces: list[Trace], empty_count: int) -> dict[str, Any]:
    if mode == "empty":
        return {"experience_content_removed": True, "n_experiences": empty_count, "pairs": [], "steps": []}
    if mode == "direct":
        steps = []
        for trace in traces:
            for step in trace.steps:
                steps.append(
                    {
                        "features_before": dict(step.features_before),
                        "action_template": step.action_template,
                        "parameters": dict(step.parameters),
                        "history_hash": step.history_hash,
                        "success": trace.success,
                    }
                )
        return {"experience_content_removed": False, "pairs": [], "steps": steps}
    views = []
    allowed = {"mismatched"} if mode == "mismatched" else {"same_prefix", "matched_observation"}
    for pair in pairs:
        if pair.alignment not in allowed:
            continue
        view = pair.inducer_view()
        views.append(view)
    return {"experience_content_removed": False, "pairs": views, "steps": []}


def _parse(raw: str, condition: str, max_candidates: int) -> tuple[list[Rule], list[dict[str, Any]]]:
    parsed = _extract_json(raw)
    if isinstance(parsed, dict):
        parsed = parsed.get("rules") or [parsed]
    if not isinstance(parsed, list):
        return [], [{"reason": "unparsed"}]
    rules: list[Rule] = []
    rejected: list[dict[str, Any]] = []
    for item in parsed:
        if len(rules) >= max_candidates:
            rejected.append({"reason": "over_cap"})
            continue
        if not isinstance(item, dict):
            rejected.append({"reason": "not_object"})
            continue
        template = str(item.get("action_template") or "")
        parameters = dict(item.get("parameters") or {})
        atoms = _atoms(item)
        if template not in {"rewrite_query", "open_top"} or not atoms or len(atoms) > 3:
            rejected.append({"reason": "schema", "action_template": template})
            continue
        if template == "rewrite_query" and parameters.get("strategy") not in REWRITE_STRATEGIES:
            rejected.append({"reason": "strategy"})
            continue
        if template == "open_top" and parameters:
            rejected.append({"reason": "params"})
            continue
        if any(str(atom.get("feature")) not in PUBLIC_FEATURES for atom in atoms):
            rejected.append({"reason": "feature"})
            continue
        executable = {"trigger": {"all": atoms}, "action_template": template, "parameters": parameters, "fallback": "base_agent"}
        rules.append(
            Rule(
                rule_id=f"{condition}_{sha256_json(executable)[:10]}",
                family="recovery",
                trigger={"all": [dict(atom) for atom in atoms]},
                action_template=template,
                parameters=parameters,
            )
        )
    return rules, rejected


def _atoms(item: dict[str, Any]) -> list[dict[str, Any]]:
    trigger = item.get("trigger")
    raw_atoms = None
    if isinstance(trigger, dict):
        raw_atoms = trigger.get("all")
    if raw_atoms is None and isinstance(item.get("trigger.all"), list):
        raw_atoms = item.get("trigger.all")
    atoms = []
    for atom in raw_atoms or []:
        if not isinstance(atom, dict):
            continue
        value = atom.get("value")
        if isinstance(value, bool):
            value = int(value)
        atoms.append({"feature": atom.get("feature"), "op": atom.get("op"), "value": value})
    return atoms


def _extract_json(raw: str) -> Any:
    if not raw:
        return None
    start_list = raw.find("[")
    start_obj = raw.find("{")
    start = start_list if start_list != -1 and (start_obj == -1 or start_list < start_obj) else start_obj
    if start == -1:
        return None
    end = max(raw.rfind("]"), raw.rfind("}"))
    if end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
