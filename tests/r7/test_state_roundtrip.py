from expolicy.closed_loop.state import ExPolicyState


def test_state_roundtrip():
    s = ExPolicyState(round_id=3, budget_used=16, last_batch_capability="bridge_decomposition")
    s.active_skills["comparison_alignment"] = True
    s.acquisition_posterior = {"type": "linear_thompson", "n": 1}
    blob = s.to_dict()
    s2 = ExPolicyState.from_dict(blob)
    assert s2.round_id == 3
    assert s2.active_skills["comparison_alignment"] is True
    assert s2.content_hash() == s.content_hash()
    s2.budget_used = 24
    assert s2.content_hash() != s.content_hash()
