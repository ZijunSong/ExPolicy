"""Interventional skills must not emit the final answer themselves."""

from __future__ import annotations

from expolicy.environments.search.environment import SearchEnvironment
from expolicy.r7.interventional_skills import run_interventions, skill_s_bridge, skill_s_compare, skill_s_long_chain
from expolicy.r7.constants import C1, C2, C3


def _helion_env_and_task():
    from expolicy.environments.search.world import World
    from expolicy.experiments.common import ROOT, load_world

    world = load_world(ROOT / "data" / "world" / "helion.json")
    env = SearchEnvironment(world)
    tasks = list(world.tasks.values())
    task = next((t for t in tasks if (t.capability or "") in {C1, "multi_hop", "query_decomposition"}), tasks[0])
    env.reset(task)
    return env, task


def test_run_interventions_returns_no_answer() -> None:
    env, task = _helion_env_and_task()
    for fn in (skill_s_bridge, skill_s_compare, skill_s_long_chain):
        skill = fn()
        env.reset(task)
        out = run_interventions(skill, env, task)
        assert out["answer"] is None
        assert out["replaced_model"] is False
        assert "open" in out["executed"] or "search" in out["executed"]


def test_synthesize_requires_model() -> None:
    from expolicy.r7.interventional_skills import synthesize_with_model

    try:
        synthesize_with_model(None, "q", [], {})
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "policy model" in str(exc)
