"""Clone / DISCARD isolation checks."""

from __future__ import annotations

from pathlib import Path

from expolicy.r9_lab.assimilators.discard import apply_discard
from expolicy.r9_lab.state.clone import clone_state
from expolicy.r9_lab.state.persistent_state import PersistentState


def assert_clone_independent(src: PersistentState, dest_a: Path, dest_b: Path) -> None:
    a = clone_state(src, dest_a)
    b = clone_state(src, dest_b)
    a.episodic  # noqa: B018
    a.skill_markdown = (a.skill_markdown or "") + "\nCLONE_A_SKILL"
    a.save()
    from expolicy.r9_lab.state.persistent_state import EpisodicRecord

    a.episodic.append(
        EpisodicRecord(task_id="clone-a-only", env="db", instruction="clone a", trajectory=[], success=True)
    )
    a.committed_task_ids.append("clone-a-only")
    a.save()
    b2 = PersistentState.load(dest_b)
    if any(r.task_id == "clone-a-only" for r in b2.episodic):
        raise AssertionError("clone memory not independent")
    if "CLONE_A_SKILL" in b2.skill_markdown:
        raise AssertionError("clone skill not independent")
    live = PersistentState.load(src.root_path())
    if any(r.task_id == "clone-a-only" for r in live.episodic):
        raise AssertionError("clone polluted live memory")


def assert_discard_noop(state: PersistentState) -> None:
    before = state.content_hash()
    apply_discard(state, [{"task_id": "x", "success": True, "instruction": "noop"}])
    after = state.content_hash()
    if before != after:
        raise AssertionError("DISCARD changed state hash")
