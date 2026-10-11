import numpy as np

from expolicy.closed_loop.policies.acquisition_linear_ts import LinearThompsonAcquisition
from expolicy.closed_loop.state import ExPolicyState


def test_no_feedback_keeps_prior():
    acq = LinearThompsonAcquisition()
    prior = acq.snapshot()
    rng = np.random.default_rng(1)
    state = ExPolicyState()
    # no-feedback: select is allowed, update is skipped
    acq.select(state, ["bridge_decomposition"], rng, n_rounds=6)
    assert acq.snapshot() == prior
    # if someone calls update, posterior would change — the runner must skip it
    acq.update(state, "bridge_decomposition", 0.3, state, n_rounds=6)
    assert acq.snapshot() != prior
