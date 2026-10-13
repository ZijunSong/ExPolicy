"""B-v2 wiring tests. They do not claim the synthetic B-v1 scores are a result."""

from __future__ import annotations

from pathlib import Path

from expolicy.models.base import GenerationResult
from expolicy.paper_b.decision_v2 import (
    CONFIRMED_GAIN,
    INCONCLUSIVE,
    NO_GO_CONTENT,
    PILOT_EXTEND,
    judge_v2,
    retention_upper_harm,
)
from expolicy.paper_b.execute import execute_rules
from expolicy.paper_b.induce_v2 import induce_v2
from expolicy.paper_b.models import Rule, Trace, TraceStep
from expolicy.paper_b.pilot_v2 import fork_task_ids, supplement_task_ids
from expolicy.paper_b.pipeline_v2 import deploy_rules, frozen_b1, load_v2_config
from expolicy.paper_b.public_data import task_from_hotpot
from expolicy.paper_b.public_exec import (
    PublicBudget,
    PublicTask,
    build_messages,
    parse_model_action,
    run_public_episode,
)
from expolicy.paper_b.registry import build_runtime_registry
from expolicy.paper_b.stats import paired_comparison
from expolicy.paper_b.traces import build_trace_pairs
from expolicy.paper_b.world import build_world, feasibility_counts

ROOT = Path(__file__).resolve().parents[2]


class _Recorder:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, state, text_context: str = ""):
        self.seen.append(text_context)
        if state.search_count == 0:
            return "base_search", {"query": state.question}
        return "base_answer", {"answer": "unknown"}


class _Search:
    def search(self, query: str, *, topk: int = 5):
        return [{"document_id": "d1", "title": "Paris", "text": "Paris is a city.", "snippet": "Paris is a city."}]

    def open_id(self, document_id: str):
        return {"document_id": document_id, "title": "Paris", "text": "Paris is the capital of France."}


class _Model:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []

    def generate(self, messages, *, temperature: float, max_new_tokens: int):
        self.calls += 1
        self.messages.append(list(messages))
        if self.calls == 1:
            return GenerationResult(text="<tool>search</tool><arg>paris capital</arg>", prompt_tokens=12, completion_tokens=6)
        return GenerationResult(text="<tool>answer</tool><arg>France</arg>", prompt_tokens=20, completion_tokens=4)


def test_hotpot_row_keeps_supporting_facts_out_of_the_task():
    task = task_from_hotpot(
        {
            "id": "dev_0",
            "question": "Which magazine was started first?",
            "golden_answers": ["Arthur's Magazine"],
            "metadata": {"type": "comparison", "supporting_facts": [["Arthur's Magazine", 0]]},
        },
        "pool",
    )
    assert task is not None
    assert task.family == "retention"
    assert task.gold_answer == "Arthur's Magazine"
    assert not hasattr(task, "supporting_facts")
    assert "supporting_facts" not in task.__dict__


def test_fork_and_supplement_are_capped_before_any_scores():
    records = [
        {"stage": "collect", "task_id": "a", "success": 1, "infra_status": "ok"},
        {"stage": "collect", "task_id": "a", "success": 0, "infra_status": "ok"},
        {"stage": "collect", "task_id": "b", "success": 0, "infra_status": "ok"},
    ]
    assert fork_task_ids(records) == {"a"}
    tasks = [
        PublicTask("b", "q", "g", "train", "bridge", "recovery", "b", cluster="bridge:easy"),
        PublicTask("a", "q2", "g2", "train", "bridge", "recovery", "a", cluster="bridge:hard"),
    ]
    assert supplement_task_ids(records, tasks, 48) == ["b"]
    parsed = task_from_hotpot(
        {"id": "1", "question": "Where?", "golden_answers": ["Paris"], "metadata": {"type": "bridge", "level": "hard"}},
        "pool",
    )
    assert parsed is not None
    assert parsed.cluster == "bridge:hard"


def test_v2_config_is_not_a_renamed_synthetic_harness():
    raw = load_v2_config(ROOT / "configs" / "paper_b_v2.yaml")
    assert raw["protocol_id"] == "B-v2"
    assert raw["harness"] == "public_wikipedia_search"
    assert raw["decision"]["confirmation_opened"] is False
    rules = frozen_b1(raw)
    assert rules
    assert all(rule.action_template == "rewrite_query" for rule in rules)
    assert "biography" not in str([rule.parameters for rule in rules])


def test_policy_receives_text_and_rules_still_run():
    registry = build_runtime_registry()
    task = next(item for item in build_world(feasibility_counts(), seed=1) if item.family == "entity_time_compare")
    recorder = _Recorder()
    rule = Rule(
        rule_id="always_search",
        family="entity_time_compare",
        trigger={"all": [{"feature": "search_count", "op": "eq", "value": 1}]},
        action_template="search_entity_attribute",
    )
    episode = execute_rules(task, [rule], registry, budget=registry.budget, text_context="card", policy=recorder)
    assert episode.triggered_rules == ["always_search"]
    assert "card" in recorder.seen


def test_feature_match_is_not_called_same_prefix_when_history_differs():
    features = {"search_count": 0, "has_comparison_cue": 1}
    def step(template: str, history: str) -> TraceStep:
        return TraceStep(features, template, {}, "obs", history_hash=history, snapshot_hash=history)
    success = Trace("s", "t", "g", "train", "recovery", "a", True, [step("rewrite_query", "h1")])
    failure = Trace("f", "t", "g", "train", "recovery", "b", False, [step("base_search", "h2")])
    pairs = build_trace_pairs([success, failure])
    assert pairs
    assert all(pair.alignment == "matched_observation" for pair in pairs)
    assert all(pair.alignment_basis == "features" for pair in pairs)


def test_same_generator_sees_full_catalog_and_direct_steps():
    calls: list[str] = []

    def generate(prompt: str, temperature: float) -> str:
        calls.append(prompt)
        return json_rule()

    for mode in ("empty", "direct", "aligned", "mismatched"):
        induce_v2(
            mode=mode,
            condition=mode,
            generate_fn=generate,
            temperature=0.0,
            pairs=[],
            traces=[
                Trace(
                    "s",
                    "t",
                    "g",
                    "train",
                    "recovery",
                    "model",
                    True,
                    [TraceStep({"search_count": 1, "new_evidence": 0}, "search", {"query": "paris"}, "obs", history_hash="abc")],
                )
            ],
            empty_count=2,
        )
    assert len(calls) == 4
    assert all("bridge_entity" in prompt and "drop_redundant" in prompt for prompt in calls)
    direct = calls[1]
    assert "history_hash" in direct
    assert "initial_search_count" not in direct
    try:
        induce_v2(mode="empty", condition="B3", generate_fn=None, temperature=0.0)  # type: ignore[arg-type]
    except RuntimeError as exc:
        assert "does not clone" in str(exc)
    else:
        raise AssertionError("missing generator must fail")


def test_rejected_candidates_keep_the_shared_baseline():
    raw = load_v2_config(ROOT / "configs" / "paper_b_v2.yaml")
    baseline = frozen_b1(raw)
    assert deploy_rules("B5", baseline, [], rejected=True) == baseline
    assert deploy_rules("B0", baseline, baseline, rejected=False) == []
    extra = Rule("extra", "recovery", {"all": [{"feature": "search_count", "op": "eq", "value": 1}]}, "open_top")
    merged = deploy_rules("M", baseline, [extra], rejected=False)
    assert merged[0].rule_id == baseline[0].rule_id
    assert extra in merged


def test_decision_text_matches_the_interval_rule():
    good = {"diff": 0.05, "ci": [0.01, 0.09]}
    confirmed = {"diff": 0.08, "ci": [0.04, 0.12]}
    wide = {"diff": 0.05, "ci": [-0.01, 0.12]}
    tied = {"diff": 0.0, "ci": [-0.01, 0.01]}
    kwargs = {"retention_delta": 0.0, "retention_ci": [-0.004, 0.004], "trigger_rate": 0.4}
    extend = judge_v2({"M_vs_B1": good, "M_vs_B3": good, "M_vs_B5": good}, **kwargs)
    assert extend["status"] == PILOT_EXTEND
    proven = judge_v2({"M_vs_B1": confirmed, "M_vs_B3": confirmed, "M_vs_B5": confirmed}, **kwargs)
    assert proven["status"] == CONFIRMED_GAIN
    content = judge_v2({"M_vs_B1": good, "M_vs_B3": tied, "M_vs_B5": good}, **kwargs)
    assert content["status"] == NO_GO_CONTENT
    inconclusive = judge_v2({"M_vs_B1": wide, "M_vs_B3": good, "M_vs_B5": good}, **kwargs)
    assert inconclusive["status"] == INCONCLUSIVE
    harm = retention_upper_harm(100, 0)
    assert harm is not None and harm > 0.02


def test_flat_trigger_key_is_accepted():
    from expolicy.paper_b.induce_v2 import induce_v2

    def generate(prompt: str, temperature: float) -> str:
        return '[{"trigger.all":[{"feature":"new_evidence","op":"eq","value":true}],"action_template":"open_top","parameters":{}}]'

    result = induce_v2(mode="empty", condition="B3", generate_fn=generate, temperature=0.0, empty_count=1)
    assert len(result["rules"]) == 1
    assert result["rules"][0].trigger["all"][0]["value"] == 1


def test_qwen_search_tag_is_a_legal_action():
    parsed = parse_model_action('<search>\n{"query": "Brett Bech sport", "source": "wikipedia"}\n</search>')
    assert parsed == ("search", {"query": "Brett Bech sport"})


def test_public_episode_separates_tools_text_and_fallback():
    task = PublicTask("t", "What is the capital of France?", "Paris", "smoke", "bridge", "recovery", "q")
    model = _Model()
    client = _Search()
    budget = PublicBudget(max_tool_calls=2, max_decision_steps=4, max_completion_tokens=100, max_interventions=2)
    plain = run_public_episode(task, [], client, model, budget, text_context="experience card")
    assert plain.text_in_prompt
    assert plain.model_calls >= 1
    assert plain.tool_calls <= 2
    messages = build_messages(task.question, [], "experience card")
    assert task.gold_answer not in messages[1]["content"]
    assert "experience card" in messages[1]["content"]
    assert "gold_answer" not in messages[1]["content"]
    rule = Rule(
        "smoke_rule",
        "recovery",
        {"all": [{"feature": "search_count", "op": "eq", "value": 0}]},
        "rewrite_query",
        {"strategy": "shorten"},
    )
    ruled = run_public_episode(task, [rule], client, _Model(), budget)
    assert ruled.triggered_rules == ["smoke_rule"]
    assert ruled.actions[0]["arguments"]["strategy"] == "shorten"
    dormant = Rule("nope", "recovery", {"all": [{"feature": "search_count", "op": "eq", "value": 99}]}, "open_top")
    fallback = run_public_episode(task, [dormant], client, _Model(), budget)
    assert fallback.model_calls > 0
    assert fallback.triggered_rules == []


def test_paired_rows_keep_distinct_seeds():
    rows = []
    for seed, success in (("0", 1), ("1", 0)):
        for condition in ("M", "B1"):
            rows.append(
                {
                    "condition": condition,
                    "task_id": "t",
                    "split": "near",
                    "source_group": "bridge",
                    "family": "recovery",
                    "success": success if condition == "M" else 0,
                    "experience_pool_id": "pool",
                    "eval_seed": seed,
                }
            )
    compared = paired_comparison(rows, "M", "B1", {"near"}, n_boot=20, seed=1, alpha=0.05)
    assert compared["n"] == 2


def json_rule() -> str:
    return """[{"trigger":{"all":[{"feature":"search_count","op":"gte","value":1}]},"action_template":"rewrite_query","parameters":{"strategy":"drop_redundant"}}]"""
