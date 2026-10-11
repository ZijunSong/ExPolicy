"""Persistent state: clone isolation, DISCARD no-op, SkillOpt reject, provenance."""

from __future__ import annotations

from pathlib import Path

import pytest

from expolicy.r9_lab.assimilators.discard import apply_discard
from expolicy.r9_lab.assimilators.episodic import apply_episodic, retrieve_episodic
from expolicy.r9_lab.assimilators.semantic import apply_semantic
from expolicy.r9_lab.assimilators.skillopt import apply_skillopt
from expolicy.r9_lab.audit.state_isolation import assert_clone_independent, assert_discard_noop
from expolicy.r9_lab.state.clone import clone_state
from expolicy.r9_lab.state.persistent_state import EpisodicRecord, PersistentState
from tests.r9.fakes import fresh_state


def _exp(task_id: str, *, success: bool = True, bucket: str = "acquisition_pool", env: str = "db") -> dict:
    return {
        "task_id": task_id,
        "env": env,
        "instruction": f"instruction for {task_id}",
        "success": success,
        "verifier_outcome": "CORRECT" if success else "INCORRECT",
        "trajectory": [{"agent": "Action: Answer\nFinal Answer: OK", "content": "OK"}],
        "bucket": bucket,
    }


def test_clone_memory_and_skill_independent(tmp_path: Path):
    src = fresh_state(tmp_path)
    src.skill_markdown = "# base skill\n"
    apply_episodic(src, [_exp("db:acq:1")])
    assert_clone_independent(src, tmp_path / "A", tmp_path / "B")
    a = PersistentState.load(tmp_path / "A")
    b = PersistentState.load(tmp_path / "B")
    assert any(r.task_id == "clone-a-only" for r in a.episodic)
    assert all(r.task_id != "clone-a-only" for r in b.episodic)
    assert "CLONE_A_SKILL" in a.skill_markdown
    assert "CLONE_A_SKILL" not in b.skill_markdown


def test_discard_true_noop(tmp_path: Path):
    state = fresh_state(tmp_path)
    apply_episodic(state, [_exp("db:acq:1")])
    assert_discard_noop(state)
    before = state.content_hash()
    apply_discard(state, [_exp("db:acq:2")])
    assert state.content_hash() == before


def test_rejected_skillopt_edit_does_not_change_live_state(tmp_path: Path):
    state = fresh_state(tmp_path)
    state.skill_markdown = "# original\n"
    state.save()
    before = state.content_hash()
    before_skill = state.skill_markdown

    def validate(_skill: str) -> float:
        return 0.10  # never improves

    out = apply_skillopt(state, [_exp("db:acq:1")], validate_fn=validate)
    assert out["accepted"] is False
    assert state.content_hash() == before
    assert state.skill_markdown == before_skill
    reloaded = PersistentState.load(state.root_path())
    assert reloaded.skill_markdown == before_skill


def test_accepted_skillopt_writes_only_when_validation_improves(tmp_path: Path):
    state = fresh_state(tmp_path)
    state.skill_markdown = "# original\n"
    state.save()

    def validate(skill: str) -> float:
        return 0.2 if "MUST" in skill or "NEVER" in skill else 0.1

    out = apply_skillopt(state, [_exp("db:acq:1")], validate_fn=validate)
    assert out["accepted"] is True
    assert "MUST" in state.skill_markdown or "NEVER" in state.skill_markdown


def test_semantic_provenance_only_from_acquired_batch(tmp_path: Path):
    state = fresh_state(tmp_path)
    batch = [_exp("db:acq:1"), _exp("db:acq:2", success=False)]
    apply_semantic(state, batch)
    allowed = {"db:acq:1", "db:acq:2"}
    for entry in state.semantic:
        assert set(entry.evidence_task_ids) <= allowed
        assert entry.evidence_task_ids


def test_episodic_retrieval_only_committed_experience(tmp_path: Path):
    state = fresh_state(tmp_path)
    apply_episodic(state, [_exp("db:acq:committed")])
    state.episodic.append(
        EpisodicRecord(
            task_id="db:uncommitted",
            env="db",
            instruction="instruction for db:uncommitted extra tokens committed",
            trajectory=[],
            success=True,
        )
    )
    hits = retrieve_episodic(state, "instruction for db:uncommitted extra tokens committed", committed_only=True)
    assert all(h.task_id != "db:uncommitted" for h in hits)
    assert any(h.task_id == "db:acq:committed" for h in retrieve_episodic(state, "instruction for db:acq:committed"))


def test_future_blocked_from_assimilators(tmp_path: Path):
    state = fresh_state(tmp_path)
    fut = _exp("db:future:1", bucket="future_test")
    with pytest.raises(RuntimeError):
        apply_episodic(state, [fut])
    with pytest.raises(RuntimeError):
        apply_semantic(state, [fut])
    with pytest.raises(RuntimeError):
        apply_skillopt(state, [fut], validate_fn=lambda s: 0.0)
