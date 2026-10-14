"""REAL, EMPTY and MATCHED_WRONG states. EMPTY is a constructed carrier, not an official trace."""

from __future__ import annotations

from typing import Any, Sequence

from expolicy.paper_d.content_protocol import PROTOCOL
from expolicy.r9_lab.assimilators.discard import apply_discard
from expolicy.r9_lab.assimilators.episodic import apply_episodic, format_episodic_injection, retrieve_episodic
from expolicy.r9_lab.io_util import sha256_json
from expolicy.r9_lab.state.persistent_state import EpisodicRecord, PersistentState, SemanticEntry


def clone_state(state: PersistentState, root: str) -> PersistentState:
    return PersistentState(
        root=root,
        episodic=[EpisodicRecord.from_dict(record.to_dict()) for record in state.episodic],
        semantic=[SemanticEntry.from_dict(entry.to_dict()) for entry in state.semantic],
        skill_markdown=state.skill_markdown,
        extra=dict(state.extra),
        committed_task_ids=list(state.committed_task_ids),
    )


def write_real(base: PersistentState, experiences: Sequence[dict[str, Any]], destination: str) -> dict[str, Any]:
    state = clone_state(base, destination)
    report = apply_episodic(state, experiences)
    return {
        "condition": "REAL",
        "constructed_control": False,
        "written_ids": list(report["task_ids"]),
        "n_written": int(report["n"]),
        "state_hash": state.content_hash(),
        "path": destination,
    }


def write_no_update(base: PersistentState, destination: str) -> dict[str, Any]:
    state = clone_state(base, destination)
    report = apply_discard(state, [])
    state.save()
    return {
        "condition": "NO_UPDATE",
        "constructed_control": False,
        "changed": bool(report["changed"]),
        "state_hash": state.content_hash(),
        "path": destination,
    }


def write_empty(real: PersistentState, new_ids: Sequence[str], destination: str) -> dict[str, Any]:
    """Keep the new slots, replace their task text. Historical entries stay untouched."""
    state = clone_state(real, destination)
    targets = set(new_ids)
    replaced = []
    for record in state.episodic:
        if record.task_id not in targets:
            continue
        record.instruction = str(PROTOCOL["neutral_instruction"])
        record.trajectory = [
            {
                "agent": str(PROTOCOL["neutral_action"]),
                "parsed": "execute",
                "content": str(PROTOCOL["neutral_action"]),
            }
        ]
        replaced.append(record.task_id)
    missing = [task_id for task_id in new_ids if task_id not in replaced]
    if missing:
        raise RuntimeError(f"EMPTY could not find real entries {missing}")
    digest = state.save()
    return {
        "condition": "EMPTY",
        "constructed_control": True,
        "replaced_ids": replaced,
        "state_hash": digest,
        "path": destination,
        "transform": "neutral-instruction-and-action",
    }


def source_markers(experiences: Sequence[dict[str, Any]]) -> list[str]:
    markers = []
    for experience in experiences:
        instruction = " ".join(str(experience.get("instruction") or "").split())
        if len(instruction) >= 40:
            markers.append(instruction[:40])
        elif instruction:
            markers.append(instruction)
    return markers


def injection_for(state: PersistentState, instruction: str) -> dict[str, Any]:
    hits = retrieve_episodic(state, instruction, committed_only=True)
    text = format_episodic_injection(hits)
    return {
        "retrieved_ids": [record.task_id for record in hits],
        "injection": text,
        "injection_hash": sha256_json({"injection": text}),
        "injection_tokens_estimate": len(text) // 4,
        "truncated": False,
    }


def empty_leaks_source(injection: str, experiences: Sequence[dict[str, Any]]) -> list[str]:
    leaked = []
    for marker in source_markers(experiences):
        if marker and marker in injection:
            leaked.append(marker)
    return leaked
