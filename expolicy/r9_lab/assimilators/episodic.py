"""A1 EPISODIC_REPLAY: verified successful trajectories only; retrieval top_k=4.

Injection matches official PreviousSampleUtilizationCallback spirit: prepend
retrieved trajectories into the first user prompt. Store is OUR jsonl, not a
sliding window of the official callback.
"""

from __future__ import annotations

from typing import Any, Sequence

from expolicy.r9_lab.assimilators.base import Assimilator
from expolicy.r9_lab.constants import A1_EPISODIC, EPISODIC_TOP_K
from expolicy.r9_lab.state.persistent_state import EpisodicRecord, PersistentState, token_overlap

FUTURE_BUCKET_MARKERS = ("future_test",)


def _is_verified_success(exp: dict[str, Any]) -> bool:
    if not bool(exp.get("success")):
        return False
    outcome = str(exp.get("verifier_outcome") or "CORRECT").upper()
    return outcome in {"CORRECT", "SUCCESS", "TRUE", "1"}


def _is_future(exp: dict[str, Any]) -> bool:
    bucket = str(exp.get("bucket") or exp.get("split") or "")
    tid = str(exp.get("task_id") or "")
    return bucket in FUTURE_BUCKET_MARKERS or tid.startswith("future_test:") or "/future_test/" in tid


class EpisodicAssimilator(Assimilator):
    action = A1_EPISODIC

    def apply(
        self,
        state: PersistentState,
        experiences: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        blocked = [e.get("task_id") for e in experiences if _is_future(e)]
        if blocked:
            raise RuntimeError(f"future_test must not enter episodic update: {blocked}")
        n = 0
        written_ids: list[str] = []
        for exp in experiences:
            if not _is_verified_success(exp):
                continue
            rec = EpisodicRecord(
                task_id=str(exp["task_id"]),
                env=str(exp.get("env") or ""),
                instruction=str(exp.get("instruction") or ""),
                trajectory=list(exp.get("trajectory") or []),
                success=True,
                verifier_outcome=str(exp.get("verifier_outcome") or "CORRECT"),
            )
            state.episodic.append(rec)
            if rec.task_id not in state.committed_task_ids:
                state.committed_task_ids.append(rec.task_id)
            written_ids.append(rec.task_id)
            n += 1
        digest = state.save()
        return {"action": A1_EPISODIC, "changed": n > 0, "n": n, "task_ids": written_ids, "hash": digest}


def apply_episodic(state: PersistentState, experiences: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return EpisodicAssimilator().apply(state, experiences)


def retrieve_episodic(
    state: PersistentState,
    instruction: str,
    *,
    k: int = EPISODIC_TOP_K,
    committed_only: bool = True,
    trace: bool = False,
) -> list[EpisodicRecord]:
    pool = list(state.episodic)
    if committed_only:
        allowed = set(state.committed_task_ids)
        pool = [r for r in pool if r.task_id in allowed]
    scored = [(token_overlap(instruction, r.instruction), r) for r in pool]
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [r for _, r in scored[:k]]
    if trace:
        state.append_retrieval_trace(
            {
                "kind": "episodic",
                "query": instruction[:300],
                "hit_task_ids": [r.task_id for r in hits],
            }
        )
    return hits


def format_episodic_injection(records: Sequence[EpisodicRecord]) -> str:
    if not records:
        return ""
    chunks = ["Previous successful trajectories:"]
    for i, rec in enumerate(records, 1):
        steps = []
        for turn in rec.trajectory:
            agent = turn.get("agent") or turn.get("action") or ""
            if agent:
                steps.append(str(agent)[:400])
        chunks.append(f"Question {rec.instruction}:\n" + "\n".join(steps))
    return "\n".join(chunks)
