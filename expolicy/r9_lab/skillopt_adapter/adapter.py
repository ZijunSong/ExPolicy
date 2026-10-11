"""Bounded SkillOpt-style editor for LAB.

Official SkillOpt is ALFWorld-oriented (train.py / eval_only.py / openai_compatible
backend). We try to reuse official apply_edit + evaluate_gate when importable.
If the full trainer contract cannot be used, this module implements a bounded
add/delete/replace editor that:
- uses the same frozen actor as optimizer
- validates on assim_validation only
- rejects if validation does not improve
- never writes rejected edits

This is NOT claimed to be official SkillOpt.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from expolicy.r9_lab.constants import SKILLOPT_ROOT


def try_import_official() -> dict[str, Any]:
    imported: dict[str, Any] = {"available": False, "apply_edit": None, "evaluate_gate": None, "error": None}
    try:
        root = str(SKILLOPT_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from skillopt.evaluation.gate import evaluate_gate
        from skillopt.optimizer.skill import apply_edit

        imported["available"] = True
        imported["apply_edit"] = apply_edit
        imported["evaluate_gate"] = evaluate_gate
    except Exception as exc:  # noqa: BLE001
        imported["error"] = f"{type(exc).__name__}: {exc}"
    return imported


def local_apply_edit(skill: str, edit: dict[str, Any]) -> str:
    op = str(edit.get("op") or "append")
    content = str(edit.get("content") or "").strip()
    target = str(edit.get("target") or "")
    if op in {"append", "add"}:
        if not content:
            return skill
        return skill.rstrip() + "\n\n" + content + "\n"
    if op == "replace":
        if target and target in skill:
            return skill.replace(target, content, 1)
        return skill
    if op == "delete":
        if target and target in skill:
            return skill.replace(target, "", 1)
        return skill
    return skill


def apply_bounded_edit(skill: str, edit: dict[str, Any]) -> tuple[str, str]:
    official = try_import_official()
    if official["available"] and official["apply_edit"] is not None:
        mapped = dict(edit)
        if mapped.get("op") == "add":
            mapped["op"] = "append"
        try:
            updated = official["apply_edit"](skill, mapped)
            return str(updated), "official_apply_edit"
        except Exception:
            pass
    mapped = dict(edit)
    if mapped.get("op") == "add":
        mapped["op"] = "append"
    return local_apply_edit(skill, mapped), "local_bounded_editor"


def propose_edit_heuristic(skill: str, experiences: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successes = [e for e in experiences if e.get("success")]
    failures = [e for e in experiences if not e.get("success")]
    env = str((experiences[0].get("env") if experiences else "") or "lab")
    if successes:
        instr = str(successes[0].get("instruction") or "")[:120]
        return {
            "op": "append",
            "content": f"MUST reuse successful {env} procedure for instructions like: {instr}",
            "target": "",
        }
    if failures:
        instr = str(failures[0].get("instruction") or "")[:120]
        return {
            "op": "append",
            "content": f"NEVER repeat the failed {env} pattern for instructions like: {instr}",
            "target": "",
        }
    return {"op": "append", "content": "", "target": ""}


def propose_edit_llm(skill: str, experiences: Sequence[dict[str, Any]], client: Any) -> dict[str, Any]:
    traces = [
        {
            "task_id": e.get("task_id"),
            "env": e.get("env"),
            "instruction": e.get("instruction"),
            "success": bool(e.get("success")),
            "trajectory": e.get("trajectory") or [],
        }
        for e in experiences
    ]
    prompt = (
        "Propose ONE bounded skill edit as JSON with keys op (add|delete|replace), content, target. "
        "Use only this acquired batch. Do not include gold answers or future tasks.\n"
        f"Current skill:\n{skill[:2000]}\n\nBatch:\n{json.dumps(traces, ensure_ascii=False)[:4000]}"
    )
    try:
        text = client.chat([{"role": "user", "content": prompt}]).text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            row = json.loads(match.group(0))
            op = str(row.get("op") or "append")
            if op not in {"add", "append", "delete", "replace"}:
                op = "append"
            return {"op": op, "content": str(row.get("content") or ""), "target": str(row.get("target") or "")}
    except Exception:
        pass
    return propose_edit_heuristic(skill, experiences)


@dataclass
class SkillOptDecision:
    accepted: bool
    reason: str
    before_score: float
    after_score: float
    before_hash: str
    after_hash: str | None
    edit: dict[str, Any] = field(default_factory=dict)
    backend: str = "bounded_adapter"


def gate_accept(before_score: float, after_score: float) -> bool:
    """Reject if validation does not improve. Matches SkillOpt evaluate_gate: cand > current."""
    official = try_import_official()
    if official["available"] and official["evaluate_gate"] is not None:
        result = official["evaluate_gate"](
            candidate_skill="candidate",
            cand_hard=float(after_score),
            current_skill="current",
            current_score=float(before_score),
            best_skill="current",
            best_score=float(before_score),
            best_step=0,
            global_step=1,
        )
        return str(getattr(result, "action", "")) != "reject"
    return float(after_score) > float(before_score)


def apply_skillopt_edit(
    state: Any,
    experiences: Sequence[dict[str, Any]],
    *,
    validate_fn: Callable[[str], float],
    client: Any = None,
) -> SkillOptDecision:
    """Propose one bounded edit, validate on assim_validation via validate_fn, commit only if improved."""
    from expolicy.r9_lab.state.persistent_state import PersistentState

    assert isinstance(state, PersistentState)
    blocked = [
        e.get("task_id")
        for e in experiences
        if str(e.get("bucket") or "") in {"future_test", "shadow_reward_probe", "retention_probe"}
    ]
    if blocked:
        raise RuntimeError(f"eval/future tasks must not enter SkillOpt update: {blocked}")
    before_hash = state.content_hash()
    before_skill = state.skill_markdown
    before_score = float(validate_fn(before_skill))
    edit = propose_edit_llm(before_skill, experiences, client) if client is not None else propose_edit_heuristic(before_skill, experiences)
    if not str(edit.get("content") or "").strip() and edit.get("op") not in {"delete"}:
        return SkillOptDecision(
            accepted=False,
            reason="empty_edit",
            before_score=before_score,
            after_score=before_score,
            before_hash=before_hash,
            after_hash=before_hash,
            edit=edit,
        )
    candidate, backend = apply_bounded_edit(before_skill, edit)
    after_score = float(validate_fn(candidate))
    if not gate_accept(before_score, after_score):
        # Reject: live state must be unchanged.
        assert state.content_hash() == before_hash
        assert state.skill_markdown == before_skill
        return SkillOptDecision(
            accepted=False,
            reason="validation_did_not_improve",
            before_score=before_score,
            after_score=after_score,
            before_hash=before_hash,
            after_hash=before_hash,
            edit=edit,
            backend=backend,
        )
    state.skill_markdown = candidate
    after_hash = state.save()
    return SkillOptDecision(
        accepted=True,
        reason="validation_improved",
        before_score=before_score,
        after_score=after_score,
        before_hash=before_hash,
        after_hash=after_hash,
        edit=edit,
        backend=backend,
    )
