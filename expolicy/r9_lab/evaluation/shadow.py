"""Shadow-reward evaluation. Read-only: must not write persistent state."""

from __future__ import annotations

from typing import Any, Sequence

from expolicy.r9_lab.runner import EpisodeResult, run_episode
from expolicy.r9_lab.state.persistent_state import PersistentState


def _persistent_file_fingerprint(state: PersistentState) -> dict[str, str]:
    from expolicy.r9_lab.io_util import sha256_file
    from expolicy.r9_lab.state.persistent_state import EPISODIC_NAME, HASH_NAME, META_NAME, SEMANTIC_NAME, SKILL_NAME

    root = state.root_path()
    out: dict[str, str] = {"content_hash": state.content_hash()}
    for name in (EPISODIC_NAME, SEMANTIC_NAME, SKILL_NAME, META_NAME, HASH_NAME):
        path = root / name
        if path.exists():
            out[name] = sha256_file(path)
    return out


def evaluate_read_only(
    tasks: Sequence[Any],
    client: Any,
    state: PersistentState,
    *,
    lab_env: Any = None,
) -> dict[str, Any]:
    before = _persistent_file_fingerprint(state)
    results: list[EpisodeResult] = []
    for task in tasks:
        results.append(run_episode(task=task, client=client, state=state, lab_env=lab_env))
    after = _persistent_file_fingerprint(state)
    if before != after:
        raise RuntimeError("shadow/retention evaluation mutated persistent state")
    n = len(results) or 1
    success = sum(1 for r in results if r.success) / n
    return {
        "success_rate": success,
        "n": len(results),
        "results": results,
        "state_hash_before": before["content_hash"],
        "state_hash_after": after["content_hash"],
        "wrote_state": False,
    }


def evaluate_shadow(tasks: Sequence[Any], client: Any, state: PersistentState, **kwargs: Any) -> dict[str, Any]:
    payload = evaluate_read_only(tasks, client, state, **kwargs)
    payload["split"] = "shadow_reward_probe"
    payload["used_for_update"] = False
    return payload
