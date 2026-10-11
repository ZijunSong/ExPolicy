"""A2 SEMANTIC_MEMORY: structured JSON from the acquired batch only.

No gold answers. No future/shadow. Dedup by (scope, trigger, principle).
Retrieval top_k=4. Generator uses the same frozen actor when an LLM client is
provided; otherwise a heuristic extractor (tests).
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from expolicy.r9_lab.assimilators.base import Assimilator
from expolicy.r9_lab.constants import A2_SEMANTIC, SEMANTIC_TOP_K
from expolicy.r9_lab.state.persistent_state import PersistentState, SemanticEntry, token_overlap

FORBIDDEN_GEN = ("gold", "future_test", "shadow_reward", "retention_probe")


def _is_future_or_eval(exp: dict[str, Any]) -> bool:
    bucket = str(exp.get("bucket") or exp.get("split") or "")
    return bucket in {"future_test", "shadow_reward_probe", "retention_probe"}


def heuristic_entries(experiences: Sequence[dict[str, Any]]) -> list[SemanticEntry]:
    entries: list[SemanticEntry] = []
    for exp in experiences:
        env = str(exp.get("env") or "unknown")
        instruction = str(exp.get("instruction") or "")
        trigger = " ".join(instruction.split()[:12])
        success = bool(exp.get("success"))
        principle = (
            f"When facing similar {env} instructions, reuse the verified action pattern from this batch."
            if success
            else f"When facing similar {env} instructions, avoid repeating the failed action pattern."
        )
        failure = "" if success else "observed unsuccessful trajectory; do not invent a gold correction"
        recommended = ""
        if success:
            traj = exp.get("trajectory") or []
            recommended = str((traj[-1] if traj else {}).get("content") or (traj[-1] if traj else {}).get("agent") or "")[:200]
        entries.append(
            SemanticEntry(
                scope=env,
                trigger=trigger or env,
                principle=principle,
                failure_pattern=failure,
                recommended_action_pattern=recommended,
                evidence_task_ids=[str(exp["task_id"])],
            )
        )
    return entries


def llm_entries(experiences: Sequence[dict[str, Any]], client: Any) -> list[SemanticEntry]:
    traces = []
    for exp in experiences:
        traces.append(
            {
                "task_id": exp.get("task_id"),
                "env": exp.get("env"),
                "instruction": exp.get("instruction"),
                "success": bool(exp.get("success")),
                "trajectory": exp.get("trajectory") or [],
            }
        )
    prompt = (
        "From this acquired batch only, emit a JSON list of memory entries with keys "
        "scope, trigger, principle, failure_pattern, recommended_action_pattern, evidence_task_ids. "
        "Do not write gold answers. Do not use future or shadow tasks. "
        "evidence_task_ids must be a subset of the provided task_ids.\n\n"
        f"Batch: {json.dumps(traces, ensure_ascii=False)[:6000]}"
    )
    try:
        text = client.chat([{"role": "user", "content": prompt}]).text
    except Exception:
        return heuristic_entries(experiences)
    low = text.lower()
    if any(bad in low for bad in FORBIDDEN_GEN):
        return heuristic_entries(experiences)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return heuristic_entries(experiences)
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return heuristic_entries(experiences)
    allowed = {str(e["task_id"]) for e in experiences}
    out: list[SemanticEntry] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        evid = [str(x) for x in (row.get("evidence_task_ids") or []) if str(x) in allowed]
        if not evid:
            continue
        out.append(
            SemanticEntry(
                scope=str(row.get("scope") or ""),
                trigger=str(row.get("trigger") or ""),
                principle=str(row.get("principle") or ""),
                failure_pattern=str(row.get("failure_pattern") or ""),
                recommended_action_pattern=str(row.get("recommended_action_pattern") or ""),
                evidence_task_ids=evid,
            )
        )
    return out or heuristic_entries(experiences)


def _dedup_merge(existing: list[SemanticEntry], incoming: Sequence[SemanticEntry]) -> list[SemanticEntry]:
    index = {e.dedup_key(): e for e in existing}
    for e in incoming:
        key = e.dedup_key()
        if key in index:
            seen = set(index[key].evidence_task_ids)
            for tid in e.evidence_task_ids:
                if tid not in seen:
                    index[key].evidence_task_ids.append(tid)
                    seen.add(tid)
        else:
            index[key] = e
    return list(index.values())


class SemanticAssimilator(Assimilator):
    action = A2_SEMANTIC

    def apply(
        self,
        state: PersistentState,
        experiences: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        blocked = [e.get("task_id") for e in experiences if _is_future_or_eval(e)]
        if blocked:
            raise RuntimeError(f"eval/future tasks must not enter semantic update: {blocked}")
        batch_ids = {str(e["task_id"]) for e in experiences}
        client = kwargs.get("client")
        incoming = llm_entries(experiences, client) if client is not None else heuristic_entries(experiences)
        for e in incoming:
            e.evidence_task_ids = [t for t in e.evidence_task_ids if t in batch_ids]
            if not e.evidence_task_ids:
                raise RuntimeError("semantic provenance escaped acquired batch")
        state.semantic = _dedup_merge(state.semantic, incoming)
        for tid in batch_ids:
            if tid not in state.committed_task_ids:
                state.committed_task_ids.append(tid)
        digest = state.save()
        return {
            "action": A2_SEMANTIC,
            "changed": len(incoming) > 0,
            "n": len(incoming),
            "evidence_task_ids": sorted(batch_ids),
            "hash": digest,
        }


def apply_semantic(state: PersistentState, experiences: Sequence[dict[str, Any]], client: Any = None) -> dict[str, Any]:
    return SemanticAssimilator().apply(state, experiences, client=client)


def retrieve_semantic(
    state: PersistentState,
    instruction: str,
    *,
    k: int = SEMANTIC_TOP_K,
    trace: bool = False,
) -> list[SemanticEntry]:
    scored = [(token_overlap(instruction, f"{e.scope} {e.trigger} {e.principle}"), e) for e in state.semantic]
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [e for _, e in scored[:k]]
    if trace:
        state.append_retrieval_trace(
            {"kind": "semantic", "query": instruction[:300], "hit_scopes": [e.scope for e in hits]}
        )
    return hits


def format_semantic_injection(entries: Sequence[SemanticEntry]) -> str:
    if not entries:
        return ""
    lines = ["Structured memory (do not copy gold answers):"]
    for e in entries:
        lines.append(
            f"- scope={e.scope}; trigger={e.trigger}; principle={e.principle}; "
            f"failure={e.failure_pattern}; action_pattern={e.recommended_action_pattern}"
        )
    return "\n".join(lines)
