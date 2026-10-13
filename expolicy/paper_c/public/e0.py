"""E0: frozen episodic memory with a real capacity rule and explicit write counts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from expolicy.paper_c.features import LeakageError
from expolicy.r9_lab.assimilators.episodic import format_episodic_injection, retrieve_episodic
from expolicy.r9_lab.constants import EPISODIC_TOP_K
from expolicy.r9_lab.io_util import sha256_json
from expolicy.r9_lab.runner import set_injection_token_cap
from expolicy.r9_lab.state.persistent_state import EpisodicRecord, PersistentState

_BLOCKED = {"future_test", "future_confirm", "retention_confirm", "shadow_select", "shadow_eval", "retention_probe"}
E0_VERSION = "e0-episodic-1"
E0_PROMPT = (
    "Store source-verified episodic trajectories. Retrieve by token overlap, top_k=4. "
    "Evict the oldest record past 256. Injection uses an 8192-character stand-in for 2048 tokens."
)


def algorithm_hash() -> str:
    return sha256_json(
        {
            "version": E0_VERSION,
            "top_k": EPISODIC_TOP_K,
            "max_entries": 256,
            "eviction": "fifo",
            "injection_char_cap": 8192,
            "prompt": E0_PROMPT,
        }
    )


def _eligible(item: Mapping[str, Any]) -> bool:
    bucket = str(item.get("bucket") or item.get("role") or item.get("split") or "")
    if bucket in _BLOCKED or str(item.get("task_id") or "").startswith("future_test:"):
        return False
    return bool(item.get("source_verified"))


class EpisodicE0:
    """Same retrieval and formatter for every policy. Payload controls happen before apply."""

    name = "episodic_e0"
    version = E0_VERSION
    prompt = E0_PROMPT

    def __init__(self, *, max_entries: int = 256, injection_token_budget: int = 2048):
        if max_entries != 256 or injection_token_budget != 2048 or EPISODIC_TOP_K != 4:
            raise RuntimeError("E0 hyperparameters are frozen")
        self.max_entries = max_entries
        self.injection_token_budget = injection_token_budget
        set_injection_token_cap(injection_token_budget)

    def algorithm_hash(self) -> str:
        return algorithm_hash()

    def prepare_state(self, state: PersistentState) -> None:
        state.semantic = []
        state.skill_markdown = ""

    def apply(self, state: PersistentState, experiences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        self.prepare_state(state)
        for item in experiences:
            if str(item.get("bucket") or "") in _BLOCKED:
                raise LeakageError(f"probe or confirm trajectory cannot be assimilated: {item.get('task_id')}")
        eligible = [item for item in experiences if _eligible(item)]
        written = 0
        for item in eligible:
            if any(record.task_id == str(item.get("task_id")) for record in state.episodic):
                continue
            state.episodic.append(
                EpisodicRecord(
                    task_id=str(item.get("task_id")),
                    env=str(item.get("env") or ""),
                    instruction=str(item.get("instruction") or ""),
                    trajectory=[dict(turn) for turn in item.get("trajectory") or []],
                    success=True,
                    verifier_outcome="SOURCE_VERIFIED",
                )
            )
            task_id = str(item.get("task_id"))
            if task_id not in state.committed_task_ids:
                state.committed_task_ids.append(task_id)
            written += 1
        evicted = 0
        while len(state.episodic) > self.max_entries:
            dropped = state.episodic.pop(0)
            evicted += 1
            if dropped.task_id in state.committed_task_ids:
                state.committed_task_ids.remove(dropped.task_id)
        state.save()
        changed = written > 0 or evicted > 0
        return {
            "action": self.name,
            "accepted": changed,
            "changed": changed,
            "n": written,
            "n_eligible": len(eligible),
            "n_written": written,
            "n_evicted": evicted,
            "algorithm_hash": self.algorithm_hash(),
            "online_algorithm_edit": False,
            "teacher_tokens": 0,
            "actor_tokens": 0,
            "validation_calls": 0,
        }

    def injection_preview(self, state: PersistentState, instruction: str) -> dict[str, Any]:
        hits = retrieve_episodic(state, instruction, k=EPISODIC_TOP_K)
        text = format_episodic_injection(hits)
        from expolicy.r9_lab.runner import _cap_injection

        capped = _cap_injection(text)
        return {
            "hit_ids": [record.task_id for record in hits],
            "text": capped,
            "chars_before": len(text),
            "chars_after": len(capped),
            "uses_agent_field": "Action:" in capped or any(
                (turn.get("agent") or turn.get("action")) for record in hits for turn in record.trajectory
            ),
        }
