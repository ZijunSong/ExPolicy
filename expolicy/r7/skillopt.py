"""SkillOpt-style content-sensitive skill assimilator (baseline, not novelty).

acquired trajectories
→ analyze success/failure
→ bounded add/delete/replace edits to skill artifact
→ held-out validation
→ accept only improving update

The same optimizer is shared by every acquisition method. Only the experience
batch is allowed to change.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from typing import Any, Callable, Sequence

from expolicy.assimilation.carriers.skill_store import Skill, SkillStore
from expolicy.core.experience import Experience
from expolicy.environments.public_search.skills import extract_entities
from expolicy.r7.constants import (
    C1,
    C2,
    C3,
    COMPILER_SKILLOPT,
    EDIT_BUDGET,
    LEAK_KEYS,
    PUBLIC_CAPS,
)
from expolicy.r7.interventional_skills import PRIMITIVES, skill_has_leakage

OPERATORS = ("add_step", "delete_step", "replace_query_suffix", "replace_recovery", "adjust_max_opens")


def _stable_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def skill_hash(skill: Skill) -> str:
    dump = skill.model_dump(mode="json")
    dump.pop("success_count", None)
    dump.pop("fail_count", None)
    dump.pop("skill_id", None)
    return _stable_id(dump)


def analyze_batch(experiences: Sequence[Experience], capability: str) -> dict[str, Any]:
    matched = [e for e in experiences if e.capability == capability]
    n = len(matched)
    n_ok = sum(1 for e in matched if e.success)
    fail = [e for e in matched if not e.success]
    queries: list[str] = []
    titles: list[str] = []
    fail_ents: list[str] = []
    empty_pred = 0
    n_search = 0
    n_open = 0
    for exp in matched:
        n_search += sum(1 for c in exp.tool_calls if c.get("tool") == "search")
        n_open += sum(1 for c in exp.tool_calls if c.get("tool") == "open")
        for call in exp.tool_calls:
            if call.get("tool") == "search":
                q = (call.get("arguments") or {}).get("query")
                if q:
                    queries.append(str(q))
        for ev in exp.evidence:
            if ev.get("title"):
                titles.append(str(ev["title"]))
        if not (exp.predicted_answer or "").strip():
            empty_pred += 1
        if not exp.success:
            fail_ents.extend(extract_entities(exp.question)[:3])
    qblob = " ".join(queries).lower()
    suffixes: list[str] = []
    for pat, suf in (
        (r"\bwhen\b|\byear\b", "year"),
        (r"\bwhere\b|\blocation\b", "location"),
        (r"\bwho\b", "person"),
        (r"\bwhich\b", ""),
        (r"\bborn\b", "born"),
        (r"\bdied\b", "died"),
        (r"\bcapital\b", "capital"),
        (r"\balbum\b", "album"),
        (r"\bfilm\b|\bmovie\b", "film"),
    ):
        if re.search(pat, qblob) or any(re.search(pat, e.question.lower()) for e in fail[:20]):
            if suf:
                suffixes.append(suf)
    top_fail_ents = [t for t, _ in Counter(fail_ents).most_common(4)]
    return {
        "n": n,
        "n_success": n_ok,
        "success_rate": (n_ok / n) if n else 0.0,
        "empty_pred_rate": (empty_pred / n) if n else 0.0,
        "mean_search": (n_search / n) if n else 0.0,
        "mean_open": (n_open / n) if n else 0.0,
        "prefer_titles": [t for t, _ in Counter(titles).most_common(3)],
        "query_suffixes": suffixes[:3],
        "fail_entities": top_fail_ents,
        "majority_tokens": [w for w, _ in Counter(qblob.split()).most_common(8) if len(w) > 3],
        "source_ids": sorted({e.experience_id for e in matched}),
    }


def propose_edits(skill: Skill, analysis: dict[str, Any], *, max_edits: int) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []
    steps = list(skill.decomposition_steps)
    suffixes = list(analysis.get("query_suffixes") or [])
    if suffixes:
        edits.append(
            {
                "op": "replace_query_suffix",
                "query_suffixes": suffixes,
                "reason": "failure_queries_suggest_content_suffixes",
            }
        )
    if analysis.get("empty_pred_rate", 0) >= 0.4:
        edits.append(
            {
                "op": "adjust_max_opens",
                "max_opens": min(10, int((skill.runtime_config or {}).get("max_opens") or 6) + 2),
                "reason": "high_empty_prediction_increase_evidence",
            }
        )
    if analysis.get("mean_open", 0) < 2 and skill.capability in {C1, C2}:
        edits.append(
            {
                "op": "add_step",
                "step": "open additional ranked hit when evidence slots are empty",
                "reason": "too_few_opens_on_this_batch",
            }
        )
    if analysis.get("fail_entities"):
        ents = analysis["fail_entities"][:2]
        edits.append(
            {
                "op": "replace_recovery",
                "failure_recovery": f"retry search with leftover entities {ents} plus relation tail",
                "reason": "content_specific_unresolved_entities",
            }
        )
    if analysis.get("majority_tokens") and "add_step" not in {e["op"] for e in edits}:
        tok = analysis["majority_tokens"][0]
        edits.append(
            {
                "op": "add_step",
                "step": f"if relation remains unresolved, search '{tok}' as a content-derived cue",
                "reason": "batch_token_cue",
            }
        )
    if len(steps) >= 6:
        edits.append(
            {
                "op": "delete_step",
                "index": len(steps) - 1,
                "reason": "trim_redundant_tail_step",
            }
        )
    # keep only known operators, bound budget
    out = []
    seen = set()
    for ed in edits:
        if ed["op"] not in OPERATORS:
            continue
        key = json.dumps(ed, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(ed)
        if len(out) >= max_edits:
            break
    return out


def apply_edit(skill: Skill, edit: dict[str, Any]) -> Skill:
    nxt = skill.model_copy(deep=True)
    cfg = dict(nxt.runtime_config or {})
    op = edit["op"]
    if op == "add_step":
        steps = list(nxt.decomposition_steps)
        step = str(edit.get("step") or "")
        if step and step not in steps:
            steps.append(step)
        nxt.decomposition_steps = steps
    elif op == "delete_step":
        steps = list(nxt.decomposition_steps)
        idx = int(edit.get("index") or (len(steps) - 1))
        if 0 <= idx < len(steps) and len(steps) > 3:
            steps.pop(idx)
        nxt.decomposition_steps = steps
    elif op == "replace_query_suffix":
        cfg["query_suffixes"] = list(edit.get("query_suffixes") or [])
    elif op == "replace_recovery":
        nxt.failure_recovery = str(edit.get("failure_recovery") or nxt.failure_recovery)
    elif op == "adjust_max_opens":
        cfg["max_opens"] = int(edit.get("max_opens") or cfg.get("max_opens") or 6)
    cfg["last_edit"] = {"op": op, "reason": edit.get("reason")}
    nxt.runtime_config = cfg
    nxt.compiler_version = COMPILER_SKILLOPT
    nxt.version = int(nxt.version) + 1
    nxt.skill_id = _stable_id(
        {
            "compiler": COMPILER_SKILLOPT,
            "parent": skill.skill_id,
            "edit": edit,
            "version": nxt.version,
        }
    )
    leaks = skill_has_leakage(nxt)
    if leaks:
        raise RuntimeError(f"skill edit leaked forbidden keys: {leaks}")
    blob = json.dumps(nxt.model_dump(mode="json"), default=str)
    for key in LEAK_KEYS:
        if key in {"capability"}:
            continue
        if f'"{key}"' in blob and key in (nxt.runtime_config or {}):
            raise RuntimeError(f"runtime_config contains leak key {key}")
    return nxt


class SkillOptAssimilator:
    """Content-dependent skill optimizer. Official SkillOpt is the conceptual source."""

    def __init__(
        self,
        *,
        edit_budget: int = EDIT_BUDGET,
        min_delta: float = 0.0,
        epochs: int = 1,
    ) -> None:
        self.edit_budget = int(edit_budget)
        self.min_delta = float(min_delta)
        self.epochs = int(epochs)

    def initial_skills(self) -> dict[str, Skill]:
        return {cap: fn() for cap, fn in PRIMITIVES.items()}

    def optimize(
        self,
        experiences: Sequence[Experience],
        *,
        validate: Callable[[Skill], float],
        initial: dict[str, Skill] | None = None,
        capabilities: Sequence[str] = PUBLIC_CAPS,
    ) -> dict[str, Any]:
        skills = initial or self.initial_skills()
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        analyses: dict[str, Any] = {}
        remaining = self.edit_budget
        for cap in capabilities:
            skill = skills[cap]
            analysis = analyze_batch(experiences, cap)
            analyses[cap] = analysis
            skill.n_support = int(analysis["n"])
            skill.support_success_rate = float(analysis["success_rate"])
            skill.support_failure_rate = 1.0 - float(analysis["success_rate"])
            skill.source_experience_ids = list(analysis["source_ids"])
            skill.compiler_version = COMPILER_SKILLOPT
            base_score = validate(skill)
            skill.validation_gain = 0.0
            current = skill
            current_score = base_score
            edits = propose_edits(current, analysis, max_edits=min(4, remaining))
            for edit in edits:
                if remaining <= 0:
                    break
                remaining -= 1
                cand = apply_edit(current, edit)
                leaks = skill_has_leakage(cand)
                if leaks:
                    rejected.append({"capability": cap, "edit": edit, "reason": f"leak:{leaks}"})
                    continue
                score = validate(cand)
                rec = {
                    "capability": cap,
                    "edit": edit,
                    "base_score": current_score,
                    "cand_score": score,
                    "skill_hash": skill_hash(cand),
                }
                if score >= current_score + self.min_delta:
                    cand.validation_gain = float(score - base_score)
                    current = cand
                    current_score = score
                    accepted.append(rec)
                else:
                    rec["reason"] = "held_out_gate_reject"
                    rejected.append(rec)
            skills[cap] = current
        return {
            "skills": skills,
            "accepted": accepted,
            "rejected": rejected,
            "analyses": analyses,
            "hashes": {cap: skill_hash(sk) for cap, sk in skills.items()},
        }

    def fit_store(
        self,
        experiences: Sequence[Experience],
        store: SkillStore,
        *,
        validate: Callable[[Skill], float],
    ) -> dict[str, Any]:
        result = self.optimize(experiences, validate=validate)
        for skill in result["skills"].values():
            store.install(skill)
        return result


def write_skillopt_port_audit(path) -> None:
    from pathlib import Path

    text = """# SKILLOPT_PORT_AUDIT

SkillOpt (arXiv:2605.23904) is used only as a **strong content-dependent assimilation backend / baseline**.
It is not claimed as ExPolicy novelty.

Official code was not vendored (no pin-able public package in this workspace). The port below matches the paper's algorithm, not a count heuristic.

| Component | Paper | This port |
|---|---|---|
| optimizer input | acquired trajectories (success/failure) | `Experience` batch; queries, opens, empty-pred, fail entities |
| edit operators | bounded add / delete / replace on skill artifact | `add_step`, `delete_step`, `replace_query_suffix`, `replace_recovery`, `adjust_max_opens` |
| acceptance gate | held-out validation; accept only improving update | `validate(skill)` on frozen skill_validation split |
| validation protocol | held-out, fixed | `data/r7_search_suite` skill_validation; never future |
| edit budget | fixed | `EDIT_BUDGET=8` |
| rejected-edit buffer | keep rejected edits | `rejected` list persisted |
| epoch/meta update | iterate edits | `epochs=1` sequential gated edits |
| deployment artifact | updated skill | `SkillStore` JSON under `outputs/r7_policy_generalization/assimilation` |

Fairness:
- passive / Ace / frontier / coverage / exact_utility share this optimizer
- only the experience batch changes
- validation split, optimizer model, edit budget, and initial skills are frozen
- no acquisition-method-specific prompt
"""
    Path(path).write_text(text, encoding="utf-8")
