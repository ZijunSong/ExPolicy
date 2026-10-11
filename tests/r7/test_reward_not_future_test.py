import pytest

from expolicy.closed_loop.reward import compute_realized_utility


def test_reward_not_future_test():
    rec = compute_realized_utility(
        shadow_macro_before=0.10,
        shadow_macro_after=0.14,
        retention_before=0.20,
        retention_after=0.19,
        token_cost=50_000,
        tool_cost=40,
    )
    assert rec["uses_future_test"] is False
    assert rec["source"] == "shadow_reward_probe"
    assert rec["delta_shadow"] == pytest.approx(0.04)
    assert rec["reward"] < rec["delta_shadow"]
    assert "future" not in rec["source"]
