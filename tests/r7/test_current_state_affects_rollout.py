from pathlib import Path

from expolicy.closed_loop.agent_state import ClosedLoopRuntime
from expolicy.closed_loop.assimilation.episodic import retrieve_cards
from expolicy.closed_loop.constants import C1
from expolicy.closed_loop.experience import summarize_batch
from expolicy.closed_loop.state import ExPolicyState
from expolicy.environments.public_search.environment import PublicSearchEnvironment, PublicWorld
from tests.r7._closed_loop_fakes import FakeClient, FakeModel, make_task


def test_current_state_affects_rollout(tmp_path: Path):
    task = make_task(0, C1)
    world = PublicWorld()
    world.add_task(task)
    env = PublicSearchEnvironment(world, FakeClient(), top_k=3)
    state = ExPolicyState()
    rt = ClosedLoopRuntime(env=env, model=FakeModel(), root=tmp_path, state=state)
    exp0 = rt.run_task(task)
    assert rt.last_dispatch == "llm+memory"
    tools0 = [a.get("tool") for a in exp0.actions]

    state.active_skills[C1] = True
    exp1 = rt.run_task(task)
    assert rt.last_dispatch == f"frozen_skill:{C1}"
    tools1 = [a.get("tool") for a in exp1.actions]
    assert tools1.count("search") >= 2
    assert tools1 != tools0 or "open" in tools1
    assert exp1.capability_context.get("dispatch") != exp0.capability_context.get("dispatch")

    state.active_skills[C1] = False
    summary = summarize_batch([exp0], capability=C1)
    rt.apply_assimilation("EPISODIC_MEMORY", summary, [exp0])
    cards = retrieve_cards(rt.episodic_dir, task.question, capability=C1, k=3)
    assert cards
    exp2 = rt.run_task(task)
    assert rt.last_dispatch == "llm+memory"
    assert rt.last_memory_injected >= 1
