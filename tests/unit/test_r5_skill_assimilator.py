"""Unit tests for R5 skill assimilator causal contract (no GPU)."""

from __future__ import annotations

from pathlib import Path

from expolicy.assimilation.carriers.skill_store import SkillStore, builtin_executable_skills
from expolicy.assimilation.skill_assimilator import COMPILER_VERSION, SkillAssimilator, skill_has_leakage
from expolicy.core.experience import Experience
from expolicy.experiments.common import load_world


def _exp(task, *, success: bool, eid: str, cap: str | None = None) -> Experience:
    exp = Experience(
        experience_id=eid,
        task_id=task.task_id,
        question=task.question,
        capability=cap or task.capability,
        success=success,
        gold_answer=None,
    )
    exp.tool_calls.append({"tool": "search", "arguments": {"query": task.question.split()[0]}, "ok": True})
    exp.evidence.append({"title": "Person file"})
    return exp


def test_fit_requires_eight_and_is_reproducible():
    world = load_world()
    mh = [t for t in world.tasks.values() if t.capability == "multi_hop"]
    assimilator = SkillAssimilator(min_support=8)
    batch = [_exp(t, success=True, eid=f"e{i}") for i, t in enumerate(mh[:8])]
    assert assimilator.fit(batch[:7], "multi_hop") is None
    a = assimilator.fit(batch, "multi_hop")
    b = assimilator.fit(batch, "multi_hop")
    assert a is not None and b is not None
    assert a.skill_id == b.skill_id
    assert a.compiler_version == COMPILER_VERSION
    assert a.n_support == 8
    assert "r2_spec" not in a.provenance


def test_different_batches_differ_and_do_not_leak(tmp_path: Path):
    world = load_world()
    mh = [t for t in world.tasks.values() if t.capability == "multi_hop"]
    xd = [t for t in world.tasks.values() if t.capability == "cross_document_synthesis"]
    assimilator = SkillAssimilator(min_support=8)
    probe_ids = [t.task_id for t in mh[30:42]]
    gold_docs = [d for t in mh[30:42] for d in t.gold_doc_ids]
    a = assimilator.fit(
        [_exp(t, success=True, eid=f"a{i}") for i, t in enumerate(mh[:8])],
        "multi_hop",
        probe_ids=probe_ids,
        gold_doc_ids=gold_docs,
    )
    b = assimilator.fit(
        [_exp(t, success=False, eid=f"b{i}") for i, t in enumerate(mh[8:16])],
        "multi_hop",
        probe_ids=probe_ids,
        gold_doc_ids=gold_docs,
    )
    assert a is not None and b is not None
    assert set(a.source_experience_ids) != set(b.source_experience_ids)
    assert a.skill_id != b.skill_id
    assert skill_has_leakage(a, probe_ids=probe_ids, gold_doc_ids=gold_docs) == []
    store = SkillStore(tmp_path)
    store.install(a)
    assert store.executable_for("multi_hop") is not None
    assert store.executable_for("temporal_reasoning") is None
    assert assimilator.fit([], "multi_hop") is None
    s0 = [t for t in world.tasks.values() if t.capability == "single_hop"][:8]
    wrong = [_exp(t, success=True, eid=f"s{i}", cap="single_hop") for i, t in enumerate(s0)]
    assert assimilator.fit(wrong, "multi_hop") is None
    assert len(builtin_executable_skills()) == 3
    # leftover S4 tasks exist for isolation of 12
    assert len(xd) >= 24
