from expolicy.closed_loop.assimilation.semantic import _heuristic_note, llm_note
from expolicy.closed_loop.experience import ExperienceBatchSummary
from expolicy.closed_loop.features import acquisition_feature_names, assimilation_feature_names, assert_no_leakage
from expolicy.core.experience import Experience


def test_closed_loop_policy_features_have_no_gold_or_dataset():
    names = acquisition_feature_names() + assimilation_feature_names()
    blob = " ".join(names).lower()
    for bad in ("gold", "future_test", "task_id", "dataset"):
        assert bad not in blob
    assert_no_leakage({n: 0.0 for n in names})


def test_semantic_note_strips_gold():
    summary = ExperienceBatchSummary(
        capability="bridge_decomposition",
        n=1,
        success_rate=0,
        failure_rate=1,
        mean_reward=0,
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
    exp = Experience(task_id="x", question="Who founded Acme?", capability="bridge_decomposition", predicted_answer="Eve")
    note = _heuristic_note("bridge_decomposition", summary, [exp])
    assert "gold" not in note.lower()
    assert "future_test" not in note.lower()
    assert llm_note(None, "bridge_decomposition", [exp]) == ""
