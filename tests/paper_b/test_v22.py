"""B-v2.2 checks for rule observations, catalog actions, snapshots, and fragments."""

from __future__ import annotations

import json
from pathlib import Path

from expolicy.models.base import GenerationResult
from expolicy.paper_b.models import Rule
from expolicy.paper_b.public_exec import PublicBudget, PublicTask
from expolicy.paper_b.v21_runtime import run_v21_episode
from expolicy.paper_b.v22_pool import trajectory_pair_count, validated_fragments


class _Docs:
    def __init__(self, snippet: str = "Covered is Christian music.") -> None:
        self.snippet = snippet
        self.queries: list[str] = []

    def search(self, query: str, *, topk: int = 5):
        self.queries.append(query)
        return [{"document_id": "17", "title": "Covered", "snippet": self.snippet, "text": self.snippet}]

    def open_id(self, document_id: str):
        assert document_id == "17"
        return {"document_id": "17", "title": "Covered", "text": "The album Covered: Alive in Asia is Christian."}


class _Empty:
    def search(self, query: str, *, topk: int = 5):
        return []

    def open_id(self, document_id: str):
        return None


class _Script:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.seen: list[str] = []

    def generate(self, messages, *, temperature: float, max_new_tokens: int) -> GenerationResult:
        self.seen.append(json.dumps(messages, ensure_ascii=False))
        return GenerationResult(text=self.outputs.pop(0), prompt_tokens=8, completion_tokens=4, stop_reason="stop")


def _task() -> PublicTask:
    return PublicTask("t", "What genre is Covered?", "Christian", "train", "bridge", "recovery", "q", cluster="bridge:easy")


def _budget() -> PublicBudget:
    return PublicBudget(max_tool_calls=4, max_decision_steps=6, max_completion_tokens=200, max_interventions=2)


def _rule(template: str, strategy: str = "", feature: str = "search_count", value: int = 1) -> Rule:
    return Rule(
        rule_id=f"r-{template}-{strategy or 'none'}",
        family="recovery",
        trigger={"all": [{"feature": feature, "op": "gte", "value": value}]},
        action_template=template,
        parameters={"strategy": strategy} if strategy else {},
        validation_record="test",
    )


def test_rewrite_observation_is_in_the_next_request():
    model = _Script(
        [
            "<tool>search</tool><arg>Covered Alive in Asia genre extra words</arg>",
            "<tool>answer</tool><arg>Christian</arg>",
        ]
    )
    budget = _budget()
    budget.max_interventions = 1
    episode = run_v21_episode(
        _task(),
        _Docs(),
        model,
        budget,
        condition="B1",
        run_id="t",
        checkpoint="Qwen3-14B",
        runtime_hash="h",
        corpus_hash="c",
        rules=[_rule("rewrite_query", "shorten")],
    )
    assert episode["rule_fires"] == 1
    assert "Covered is Christian music." in model.seen[1]
    assert "Covered Alive in Asia genre extra" in model.seen[1]
    assert episode["steps"][1]["request_hash"] != episode["steps"][1]["response_hash"]


def test_open_top_executes_when_legal_and_falls_back_when_not():
    legal = _Script(
        [
            "<tool>search</tool><arg>Covered genre</arg>",
            "<tool>answer</tool><arg>Christian</arg>",
        ]
    )
    budget = _budget()
    budget.max_interventions = 1
    episode = run_v21_episode(
        _task(),
        _Docs(),
        legal,
        budget,
        condition="B1",
        run_id="t",
        checkpoint="Qwen3-14B",
        runtime_hash="h",
        corpus_hash="c",
        rules=[_rule("open_top")],
    )
    assert episode["rule_fires"] == 1
    assert "Alive in Asia" in legal.seen[1]
    assert episode["tool_calls"] == 2
    illegal = _Script(
        [
            "<tool>search</tool><arg>Covered genre</arg>",
            "<tool>answer</tool><arg>Christian</arg>",
        ]
    )
    skipped = run_v21_episode(
        _task(),
        _Empty(),
        illegal,
        _budget(),
        condition="B1",
        run_id="t",
        checkpoint="Qwen3-14B",
        runtime_hash="h",
        corpus_hash="c",
        rules=[_rule("open_top")],
    )
    assert skipped["rule_fires"] == 0
    assert skipped["rule_skips"]
    assert "Alive in Asia" not in illegal.seen[1]


def test_unmatched_rule_matches_the_shared_base_messages():
    outputs = [
        "<tool>search</tool><arg>Covered genre</arg>",
        "<tool>answer</tool><arg>Christian</arg>",
    ]
    plain = _Script(list(outputs))
    held = _Script(list(outputs))
    kwargs = dict(condition="B0", run_id="t", checkpoint="Qwen3-14B", runtime_hash="h", corpus_hash="c")
    run_v21_episode(_task(), _Docs(), plain, _budget(), **kwargs)
    run_v21_episode(_task(), _Docs(), held, _budget(), rules=[_rule("rewrite_query", "shorten", value=99)], **kwargs)
    assert plain.seen == held.seen


def test_snapshot_hash_matches_and_observation_drift_is_reported():
    model = _Script(["<tool>search</tool><arg>Covered genre</arg>", "<tool>answer</tool><arg>Christian</arg>"])
    docs = _Docs()
    original = run_v21_episode(_task(), docs, model, _budget(), condition="B0", run_id="t", checkpoint="Qwen3-14B", runtime_hash="h", corpus_hash="c")
    prefix = [step for step in original["steps"] if step.get("canonical_action") == "search"]
    replay = run_v21_episode(
        _task(),
        _Docs(),
        _Script(["<tool>answer</tool><arg>Christian</arg>"]),
        _budget(),
        condition="B0",
        run_id="t",
        checkpoint="Qwen3-14B",
        runtime_hash="h",
        corpus_hash="c",
        prefix_actions=prefix,
    )
    assert replay["steps"][0]["state_hash"] == prefix[0]["state_hash"]
    drifted = run_v21_episode(
        _task(),
        _Docs("a different snippet"),
        _Script(["<tool>answer</tool><arg>Christian</arg>"]),
        _budget(),
        condition="B0",
        run_id="t",
        checkpoint="Qwen3-14B",
        runtime_hash="h",
        corpus_hash="c",
        prefix_actions=prefix,
    )
    assert drifted["terminal_status"] == "OBSERVATION_MISMATCH"
    assert drifted["model_calls"] == 0


def test_prompt_budget_stops_before_generation():
    model = _Script(["<tool>answer</tool><arg>Christian</arg>"])
    budget = PublicBudget(max_tool_calls=4, max_decision_steps=6, max_completion_tokens=200, max_interventions=2, max_prompt_tokens=1)
    episode = run_v21_episode(_task(), _Docs(), model, budget, condition="B0", run_id="t", checkpoint="Qwen3-14B", runtime_hash="h", corpus_hash="c")
    assert episode["terminal_status"] == "PROMPT_BUDGET"
    assert episode["model_calls"] == 0


def test_empty_pair_is_not_a_validated_fragment_and_combos_do_not_add_tasks():
    base = {"supplement": False, "run_id": "r", "terminal_status": "ANSWERED", "gold_answer": "A", "steps": []}
    rows = [
        {**base, "task_id": "t", "continuation": 0, "official_em": 1.0, "predicted_answer_span": "A"},
        {**base, "task_id": "t", "continuation": 1, "official_em": 0.0, "predicted_answer_span": ""},
        {**base, "task_id": "t", "continuation": 2, "official_em": 1.0, "predicted_answer_span": "A", "supplement": True},
    ]
    assert trajectory_pair_count(rows) == 1
    fragments = validated_fragments(rows, {"t": "What?"})
    assert len(fragments) == 1
    assert fragments[0]["validated"] is False
    rows.append({**base, "task_id": "t", "continuation": 3, "official_em": 1.0, "predicted_answer_span": "A"})
    # continuation 3 is not an initial continuation, so support stays one task.
    assert {item["source_task_ids"][0] for item in validated_fragments(rows, {"t": "What?"})} == {"t"}


def test_v22_pilot_does_not_call_the_old_episode_runner():
    source = Path("/data/ppnm/expolicy/expolicy/paper_b/v22_pilot.py").read_text(encoding="utf-8")
    assert "run_public_episode" not in source
