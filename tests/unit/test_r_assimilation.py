"""Unit tests for R1–R2 assimilation repairs (no GPU)."""

from __future__ import annotations

from pathlib import Path

from expolicy.assimilation.carriers.episodic_memory import EpisodicMemory
from expolicy.assimilation.carriers.skill_store import (
    builtin_executable_skills,
    run_skill_on_task,
    validate_skill_schema,
)
from expolicy.core.experience import Experience
from expolicy.environments.search.environment import SearchEnvironment
from expolicy.experiments.common import load_world
from expolicy.experiments.patterns import bridge_relation, normalized_pattern
from expolicy.experiments.probes import build_probe_set, split_experiences_into_batches
from expolicy.models.embedder import Embedder


def test_pattern_and_bridge():
    assert normalized_pattern("t_mh_ceo_012") == "person_employer_ceo"
    assert "employer" in bridge_relation("t_mh_ceo_012")
    assert normalized_pattern("t_tm_old_003") == "org_ceo_before_transition"
    assert normalized_pattern("t_xd_evt_008") == "event_org_and_lead"


def _env() -> SearchEnvironment:
    return SearchEnvironment(load_world())


def test_executable_skills_schema_and_runtime():
    env = _env()
    for skill in builtin_executable_skills():
        assert validate_skill_schema(skill) == []
        tasks = [t for t in env.world.tasks.values() if t.capability == skill.capability]
        assert tasks
        exp = run_skill_on_task(skill, env, tasks[0])
        assert exp.predicted_answer is not None
        assert exp.tool_calls
        assert exp.outcome.get("compliance")


def test_hybrid_episodic_retrieve_prefers_same_pattern(tmp_path: Path):
    env = _env()
    embedder = Embedder()
    mem = EpisodicMemory(tmp_path, embedder)
    mh = next(t for t in env.world.tasks.values() if t.task_id.startswith("t_mh_ceo_"))
    tm = next(t for t in env.world.tasks.values() if t.task_id.startswith("t_tm_old_"))
    mem.add(Experience(task_id=mh.task_id, question=mh.question, capability=mh.capability, success=True))
    mem.add(Experience(task_id=tm.task_id, question=tm.question, capability=tm.capability, success=True))
    other = next(t for t in env.world.tasks.values() if t.task_id.startswith("t_mh_ceo_") and t.task_id != mh.task_id)
    lines, debug = mem.retrieve(
        other.question,
        k=1,
        capability="multi_hop",
        task_pattern="person_employer_ceo",
        mode="hybrid",
        return_debug=True,
    )
    assert debug
    assert debug[0]["capability"] == "multi_hop"
    assert lines and "Warning" in lines[0]
    assert "->" not in lines[0] or "do not" in lines[0].lower() or "never reuse" in lines[0].lower()


def test_probe_builder_holds_out_same_docs():
    env = _env()
    tasks = [t for t in env.world.tasks.values() if t.capability == "multi_hop"][:8]
    exps = [Experience(task_id=t.task_id, question=t.question, capability=t.capability, success=False) for t in tasks]
    probes = build_probe_set(env.world, batch_id="s1_b0", bucket="s1_multi_hop", experiences=exps, n_near=16, n_far=16, n_retention=16, seed=0)
    src_ids = {t.task_id for t in tasks}
    assert all(t.task_id not in src_ids for t in probes.near)
    assert len(probes.near) == 16
    assert len(probes.far) == 16
    assert len(probes.retention) == 16
    batches = split_experiences_into_batches(exps, n_batches=1, batch_size=8, seed=0)
    assert len(batches) == 1 and len(batches[0]) == 8
