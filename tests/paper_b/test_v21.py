"""Offline gates for the B-v2.1 parser and Hotpot scorer."""

from __future__ import annotations

from expolicy.models.base import GenerationResult
from expolicy.paper_b.public_exec import PublicBudget, PublicTask
from expolicy.paper_b.v21_parse import fixture_cases, parse_action, query_has_protocol_tag
from expolicy.paper_b.v21_runtime import rescore_record, run_v21_episode
from expolicy.paper_b.v21_score import official_em, official_f1


class _Docs:
    def search(self, query: str, *, topk: int = 5):
        assert not query_has_protocol_tag(query)
        assert "<" not in query
        return [{"document_id": "17", "title": "Covered", "snippet": "Covered is Christian music.", "text": "Covered is Christian music."}]

    def open_id(self, document_id: str):
        assert document_id == "17"
        return {"document_id": "17", "title": "Covered", "text": "The album Covered: Alive in Asia is Christian."}


class _Script:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)

    def generate(self, messages, *, temperature: float, max_new_tokens: int) -> GenerationResult:
        return GenerationResult(text=self.outputs.pop(0), prompt_tokens=8, completion_tokens=4, stop_reason="stop")


def test_fixtures_cover_observed_formats_and_rejections():
    rows = fixture_cases()
    assert len(rows) >= 50
    for row in rows:
        parsed = parse_action(row["text"])
        assert parsed.ok is row["expect_ok"], row["id"]
        if parsed.ok and parsed.tool == "search":
            assert not query_has_protocol_tag(parsed.arguments["query"])
            assert "<" not in parsed.arguments["query"]
        if parsed.ok and row.get("value") and parsed.tool != "find":
            field = {"search": "query", "open": "document_id", "answer": "text"}[parsed.tool]
            assert parsed.arguments[field] == row["value"]


def test_hotpot_em_does_not_treat_a_sentence_as_exact():
    sentence = 'John Cameron Mitchell is best known for originating the title role.'
    assert official_em(sentence, "John Cameron Mitchell") == 0.0
    assert official_f1(sentence, "John Cameron Mitchell") > 0.4
    assert official_em("The director", "director") == 1.0
    assert official_f1("yes", "no") == 0.0


def test_repeated_search_prefix_can_be_replayed():
    from expolicy.paper_b.v21_collect import stall_prefix

    row = {
        "steps": [
            {"canonical_action": "search", "tool_arguments_sent": {"query": "alpha"}, "model_output_raw": "<tool>search</tool><arg>alpha</arg>"},
            {"canonical_action": "search", "tool_arguments_sent": {"query": "alpha"}, "model_output_raw": "<tool>search</tool><arg>alpha</arg>"},
        ]
    }
    assert len(stall_prefix(row) or []) == 2
    task = PublicTask("t", "What genre is Covered?", "Christian", "train", "bridge", "recovery", "q", cluster="bridge:easy")
    budget = PublicBudget(max_tool_calls=4, max_decision_steps=6, max_completion_tokens=200, max_interventions=2)
    episode = run_v21_episode(
        task,
        _Docs(),
        _Script(["<tool>answer</tool><arg>Christian</arg>"]),
        budget,
        condition="B0",
        run_id="test",
        checkpoint="Qwen3-14B",
        runtime_hash="h",
        corpus_hash="c",
        prefix_actions=stall_prefix(row),
    )
    assert episode["model_calls"] == 1
    assert episode["prefix_hash"]
    assert episode["steps"][0]["replayed"] is True
    assert episode["official_em"] == 1.0


def test_answer_survives_write_and_rescore_and_parse_failure_stops():
    task = PublicTask("t", "What genre is Covered?", "Christian", "dev", "bridge", "recovery", "q", cluster="bridge:easy")
    budget = PublicBudget(max_tool_calls=4, max_decision_steps=6, max_completion_tokens=200, max_interventions=2)
    episode = run_v21_episode(
        task,
        _Docs(),
        _Script(
            [
                "<search>\n  <query>Covered Alive in Asia genre</query>\n</search>",
                "<tool>open</tool><arg>17</arg>",
                "<tool>answer</tool><arg>Christian</arg>",
            ]
        ),
        budget,
        condition="B0",
        run_id="test",
        checkpoint="Qwen3-14B",
        runtime_hash="h",
        corpus_hash="c",
    )
    assert episode["official_em"] == 1.0
    assert episode["predicted_answer_raw"] == "Christian"
    assert episode["query_tag_leaks"] == 0
    assert "Christian music" in episode["steps"][1]["observation_sent_to_model"] or "Christian" in episode["steps"][0]["observation_sent_to_model"]
    assert rescore_record(episode)["official_em"] == episode["official_em"]
    failed = run_v21_episode(
        task,
        _Docs(),
        _Script(["hello", "still not an action"]),
        budget,
        condition="B0",
        run_id="test",
        checkpoint="Qwen3-14B",
        runtime_hash="h",
        corpus_hash="c",
    )
    assert failed["terminal_status"] == "PARSE_FAILED"
    assert failed["model_calls"] == 2
    assert failed["corrections"] == 1
