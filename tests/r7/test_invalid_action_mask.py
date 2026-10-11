from expolicy.closed_loop.agent_state import ClosedLoopRuntime
from expolicy.closed_loop.constants import ACTIVATE_EXISTING_SKILL, C1, DISCARD
from expolicy.closed_loop.state import ExPolicyState
from expolicy.environments.public_search.environment import PublicSearchEnvironment, PublicWorld
from tests.r7._closed_loop_fakes import FakeClient, FakeModel, make_task


def test_invalid_action_mask(tmp_path):
    task = make_task(0, C1)
    world = PublicWorld()
    world.add_task(task)
    env = PublicSearchEnvironment(world, FakeClient())
    state = ExPolicyState()
    rt = ClosedLoopRuntime(env=env, model=FakeModel(), root=tmp_path, state=state)
    valid0 = rt.valid_assim_actions(C1)
    assert ACTIVATE_EXISTING_SKILL in valid0
    assert DISCARD in valid0
    state.active_skills[C1] = True
    valid1 = rt.valid_assim_actions(C1)
    assert ACTIVATE_EXISTING_SKILL not in valid1
    assert DISCARD in valid1
