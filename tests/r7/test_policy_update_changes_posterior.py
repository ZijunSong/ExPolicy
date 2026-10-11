import numpy as np

from expolicy.closed_loop.experience import ExperienceBatchSummary
from expolicy.closed_loop.policies.acquisition_linear_ts import LinearThompsonAcquisition
from expolicy.closed_loop.policies.assimilation_linear_ts import LinearThompsonAssimilation
from expolicy.closed_loop.state import ExPolicyState


def test_policy_update_changes_posterior():
    rng = np.random.default_rng(0)
    state = ExPolicyState()
    acq = LinearThompsonAcquisition()
    before = acq.snapshot()
    dec = acq.select(state, ["bridge_decomposition", "comparison_alignment", "long_chain_search"], rng, n_rounds=6)
    info = acq.update(state, dec.capability, 0.2, state, n_rounds=6)
    after = acq.snapshot()
    assert before != after
    assert float(info["parameter_delta_norm"]) > 0
    assert after["heads"][dec.capability]["n_updates"] == 1

    summary = ExperienceBatchSummary(
        capability="bridge_decomposition",
        n=8,
        success_rate=0.25,
        failure_rate=0.75,
        mean_reward=0.25,
        mean_searches=3.0,
        mean_turns=5.0,
        repeat_query_rate=0.1,
        premature_answer_rate=0.0,
        retrieval_miss_rate=0.0,
        answer_synthesis_failure_rate=0.1,
        unique_query_ratio=0.9,
        evidence_completion_rate=0.8,
        token_cost=100,
        tool_cost=20,
    )
    assim = LinearThompsonAssimilation()
    b2 = assim.snapshot()
    adec = assim.select(state, summary, ["DISCARD", "EPISODIC_MEMORY", "SEMANTIC_MEMORY", "ACTIVATE_EXISTING_SKILL"], rng)
    ainfo = assim.update(state, summary, adec.action, -0.05, state)
    assert assim.snapshot() != b2
    assert float(ainfo["parameter_delta_norm"]) > 0
