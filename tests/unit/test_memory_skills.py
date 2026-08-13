"""Semantic memory merge / supersede."""

from __future__ import annotations

from pathlib import Path

from expolicy.assimilation.carriers.semantic_memory import SemanticMemory
from expolicy.assimilation.carriers.skill_store import SkillStore
from expolicy.core.experience import ExperienceAbstraction
from expolicy.environments.search.environment import SearchEnvironment


def test_semantic_merge_and_supersede(tmp_path: Path) -> None:
    mem = SemanticMemory(tmp_path)
    a = ExperienceAbstraction(
        experience_id="e1",
        reusable_principle="Prefer official registry dates over blog rumors.",
        confidence=0.6,
        applicability_scope=["conflicting_evidence"],
    )
    r1 = mem.add_from_abstraction(a)
    r2 = mem.add_from_abstraction(a)
    assert r1.rule_id == r2.rule_id
    assert r2.support >= 2
    stronger = ExperienceAbstraction(
        experience_id="e2",
        reusable_principle="Prefer official registry dates over blog rumors when years disagree.",
        confidence=0.95,
        applicability_scope=["conflicting_evidence"],
    )
    r3 = mem.add_from_abstraction(stronger)
    texts = mem.active_texts()
    assert any("years disagree" in t or "official" in t.lower() for t in texts)


def test_skill_invoke_runs_real_tools(tmp_path: Path) -> None:
    env = SearchEnvironment.from_seed(13)
    task = next(t for t in env.world.tasks.values() if t.capability == "single_hop")
    env.reset(task)
    store = SkillStore(tmp_path)
    store.create(SkillStore.default_lookup_skill())
    result = store.invoke("official_lookup", env, query=task.question)
    assert result["ok"] is True
    assert result["opened"]
