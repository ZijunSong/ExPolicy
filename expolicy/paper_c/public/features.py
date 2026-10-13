"""Pre-acquisition features from committed episodic records, not from synthetic skill counters."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from expolicy.paper_c.features import FORBIDDEN_FEATURE_KEYS, LeakageError
from expolicy.r9_lab.state.persistent_state import PersistentState, token_overlap

PUBLIC_KEYS = {"instruction_chars", "public_tool_count"}


def committed_family_counts(state: PersistentState, family_of: Mapping[str, str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    allowed = set(state.committed_task_ids)
    for record in state.episodic:
        if record.task_id not in allowed:
            continue
        family = family_of.get(record.task_id)
        if family:
            counts[family] += 1
    return dict(counts)


def public_features(
    state: PersistentState,
    *,
    family: str,
    families: Sequence[str],
    family_of: Mapping[str, str],
    question: str,
    history: Mapping[str, Mapping[str, float]] | None = None,
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, float | None]:
    extra = candidate or {}
    banned = [key for key in extra if key in FORBIDDEN_FEATURE_KEYS or key not in PUBLIC_KEYS]
    if banned:
        raise LeakageError(f"current rollout fields are not pre-acquisition features: {banned}")
    tool_count = None if "public_tool_count" not in extra else float(extra["public_tool_count"])
    counts = committed_family_counts(state, family_of)
    memory_texts = [record.instruction for record in state.episodic if record.task_id in set(state.committed_task_ids)]
    overlap = 0.0 if not memory_texts else max(token_overlap(question, text) for text in memory_texts)
    past = (history or {}).get(family) or {}
    features: dict[str, float | None] = {
        "n_committed": float(len(state.committed_task_ids)),
        "capacity_used": float(len(state.episodic) / 256.0),
        "instruction_chars": float(len(question)),
        "memory_overlap": float(overlap),
        "public_tool_count": tool_count,
        "hist_attempts": None if "attempts" not in past else float(past["attempts"]),
        "hist_success_rate": None if "success_rate" not in past else float(past["success_rate"]),
    }
    for name in families:
        features[f"family__{name}"] = 1.0 if name == family else 0.0
        features[f"covered__{name}"] = float(counts.get(name, 0)) if name == family else 0.0
        features[f"interact__{name}"] = float(counts.get(name, 0)) if name == family else 0.0
    return features
