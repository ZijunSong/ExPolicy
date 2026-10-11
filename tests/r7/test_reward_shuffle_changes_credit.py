import numpy as np

from expolicy.closed_loop.policies.acquisition_linear_ts import LinearThompsonAcquisition
from expolicy.closed_loop.policies.linear_ts import BayesianLinearHead
from expolicy.closed_loop.state import ExPolicyState


def test_reward_shuffle_changes_credit():
    state = ExPolicyState()
    true_r = 0.25
    shuffled = -true_r
    a = LinearThompsonAcquisition()
    b = LinearThompsonAcquisition()
    a.update(state, "bridge_decomposition", true_r, state, n_rounds=6)
    b.update(state, "bridge_decomposition", shuffled, state, n_rounds=6)
    assert a.snapshot() != b.snapshot()
    mu_a = np.array(a.inner.heads["bridge_decomposition"].mean)
    mu_b = np.array(b.inner.heads["bridge_decomposition"].mean)
    assert np.linalg.norm(mu_a - mu_b) > 0

    head = BayesianLinearHead(3)
    phi = np.array([1.0, 0.2, 0.1])
    d1 = head.update(phi, true_r)
    head2 = BayesianLinearHead(3)
    d2 = head2.update(phi, shuffled)
    assert d1["parameter_delta_norm"] > 0
    assert d2["parameter_delta_norm"] > 0
    assert not np.allclose(head.mean, head2.mean)
