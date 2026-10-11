"""O0–O3 joint oracles. Only the selected (batch, action) is committed.

Q(B,A|S) uses shadow/retention only. future_test is never an oracle input.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from expolicy.r9_lab.constants import ACTIONS
from expolicy.r9_lab.joint_oracle.counterfactual import apply_action
from expolicy.r9_lab.state.clone import clone_state
from expolicy.r9_lab.state.persistent_state import PersistentState


def q_table_from_scores(
    scores: Mapping[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    """scores keyed by (candidate_id, action) from shadow/retention utility only."""
    return dict(scores)


def select_o0_passive_fixed(
    candidates: Sequence[dict[str, Any]],
    q: Mapping[tuple[str, str], float],
    *,
    fixed_action: str,
    rng_index: int = 0,
) -> dict[str, Any]:
    cand = candidates[rng_index % len(candidates)]
    return {"oracle": "O0", "candidate_id": cand["candidate_id"], "action": fixed_action, "Q": q.get((cand["candidate_id"], fixed_action), 0.0)}


def select_o1_passive_oracle_assim(
    candidates: Sequence[dict[str, Any]],
    q: Mapping[tuple[str, str], float],
    *,
    rng_index: int = 0,
) -> dict[str, Any]:
    cand = candidates[rng_index % len(candidates)]
    cid = cand["candidate_id"]
    best_a = max(ACTIONS, key=lambda a: q.get((cid, a), float("-inf")))
    return {"oracle": "O1", "candidate_id": cid, "action": best_a, "Q": q.get((cid, best_a), 0.0)}


def select_o2_oracle_acq_fixed(
    candidates: Sequence[dict[str, Any]],
    q: Mapping[tuple[str, str], float],
    *,
    fixed_action: str,
) -> dict[str, Any]:
    best = max(candidates, key=lambda c: q.get((c["candidate_id"], fixed_action), float("-inf")))
    return {
        "oracle": "O2",
        "candidate_id": best["candidate_id"],
        "action": fixed_action,
        "Q": q.get((best["candidate_id"], fixed_action), 0.0),
    }


def select_o3_oracle_joint(
    candidates: Sequence[dict[str, Any]],
    q: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    best_pair: tuple[str, str] | None = None
    best_q = float("-inf")
    for c in candidates:
        for a in ACTIONS:
            val = q.get((c["candidate_id"], a), float("-inf"))
            if val > best_q:
                best_q = val
                best_pair = (c["candidate_id"], a)
    assert best_pair is not None
    return {"oracle": "O3", "candidate_id": best_pair[0], "action": best_pair[1], "Q": best_q}


def assert_q_has_no_future(q_payload: Mapping[str, Any]) -> None:
    blob = str(q_payload)
    if "future_test" in blob and q_payload.get("uses_future_test"):
        raise RuntimeError("future_test leaked into oracle Q")
    if q_payload.get("used_future_for_selection"):
        raise RuntimeError("future metric used in oracle selection")


def commit_selected(
    live: PersistentState,
    *,
    candidates: Sequence[dict[str, Any]],
    selected_candidate_id: str,
    selected_action: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Only the selected candidate batch is applied to live state."""
    chosen = None
    for c in candidates:
        if c["candidate_id"] == selected_candidate_id:
            chosen = c
            break
    if chosen is None:
        raise KeyError(selected_candidate_id)
    before = live.content_hash()
    payload = apply_action(selected_action, live, chosen["items"], **kwargs)
    return {
        "committed_candidate_id": selected_candidate_id,
        "committed_action": selected_action,
        "not_committed": [c["candidate_id"] for c in candidates if c["candidate_id"] != selected_candidate_id],
        "hash_before": before,
        "hash_after": live.content_hash(),
        "apply": payload,
    }
