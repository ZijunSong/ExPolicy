from pathlib import Path

from expolicy.closed_loop.experience import ExperienceBatchSummary
from expolicy.closed_loop.features import LeakageError, acquisition_features, assimilation_features, scan_payload_for_leakage
from expolicy.closed_loop.policies.linear_ts import LinearThompsonPolicy
from expolicy.closed_loop.state import ExPolicyState


def test_future_test_never_used_for_update():
    src = Path("expolicy/closed_loop/runner/episode.py").read_text()
    # future eval happens after policy update block, never as reward argument
    assert "compute_realized_utility" in src
    assert "shadow_macro_before" in src
    assert "future_tasks" in src
    # update calls do not pass future_tasks
    acq_update = [ln for ln in src.splitlines() if "acq_policy.update" in ln][0]
    assim_update = [ln for ln in src.splitlines() if "assim_policy.update" in ln][0]
    assert "future" not in acq_update
    assert "future" not in assim_update


def test_feature_vectors_reject_forbidden_keys():
    state = ExPolicyState()
    phi = acquisition_features(state, "bridge_decomposition", n_rounds=6)
    assert phi.shape[0] == 11
    summary = ExperienceBatchSummary(
        capability="bridge_decomposition",
        n=2,
        success_rate=0.5,
        failure_rate=0.5,
        mean_reward=0.5,
        mean_searches=2,
        mean_turns=3,
        repeat_query_rate=0,
        premature_answer_rate=0,
        retrieval_miss_rate=0,
        answer_synthesis_failure_rate=0,
        unique_query_ratio=1,
        evidence_completion_rate=1,
        token_cost=1,
        tool_cost=1,
    )
    psi = assimilation_features(state, summary, "DISCARD")
    assert psi.shape[0] == 12
    issues = scan_payload_for_leakage({"reward": 1.0, "delta_shadow": 0.1})
    assert not issues
    leaked = scan_payload_for_leakage({"gold_answer": "Paris"})
    assert leaked
