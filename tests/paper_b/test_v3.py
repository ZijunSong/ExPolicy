"""B-v3 reference filters and the one-step replan note."""

from __future__ import annotations

from expolicy.models.base import GenerationResult
from expolicy.paper_b.public_exec import PublicBudget, PublicTask
from expolicy.paper_b.v3_probe import reference_candidates
from expolicy.paper_b.v21_runtime import run_v21_episode


class _Docs:
    def search(self, query: str, *, topk: int = 5):
        return [{"document_id": "17", "title": "Covered", "snippet": "Covered is Christian music.", "text": "Covered is Christian music."}]

    def open_id(self, document_id: str):
        return {"document_id": "17", "title": "Covered", "text": "The album Covered: Alive in Asia is Christian."}


class _Script:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def generate(self, messages, *, temperature: float, max_new_tokens: int) -> GenerationResult:
        self.calls += 1
        return GenerationResult(text=self.outputs.pop(0), prompt_tokens=4, completion_tokens=2, stop_reason="stop")


def _row(task_id: str, continuation: int, em: float, steps: list[dict], gold: str = "Answer") -> dict:
    return {
        "task_id": task_id,
        "continuation": continuation,
        "supplement": False,
        "official_em": em,
        "gold_answer": gold,
        "steps": steps,
    }


def _search(query: str) -> dict:
    return {"canonical_action": "search", "tool_arguments_sent": {"query": query}, "observation_sent_to_model": "", "model_output_raw": query}


def test_gold_only_query_is_not_a_reference_and_task_start_is_formulation():
    rows = [
        _row("known", 0, 1, [_search("known entity relation")]),
        _row("known", 1, 0, [_search("other query")]),
        _row("hidden", 0, 1, [_search("hidden gold token")], gold="hidden gold token"),
        _row("hidden", 1, 0, [_search("other query")], gold="hidden gold token"),
    ]
    found = {item["task_id"]: item for item in reference_candidates(rows, {"known": "known entity relation", "hidden": "what happened"})}
    assert found["known"]["eligible"] is True
    assert found["known"]["label"] == "QUERY_FORMULATION"
    assert found["hidden"]["eligible"] is False


def test_opening_note_changes_prompt_hash_without_an_extra_call():
    task = PublicTask("t", "What genre is Covered?", "Christian", "train", "bridge", "recovery", "q")
    budget = PublicBudget(max_tool_calls=4, max_decision_steps=6, max_completion_tokens=200, max_interventions=1, max_prompt_tokens=32768)
    prefix = [_search("Covered genre")]
    prefix[0]["observation_sent_to_model"] = "id=17\ntitle=Covered\nsnippet=Covered is Christian music."
    model = _Script(["<tool>answer</tool><arg>Christian</arg>"])
    episode = run_v21_episode(
        task,
        _Docs(),
        model,
        budget,
        condition="GENERIC-REPLAN",
        run_id="t",
        checkpoint="Qwen3-14B",
        runtime_hash="h",
        corpus_hash="c",
        prefix_actions=prefix,
        opening_note="Review the question and choose one next action.",
    )
    assert model.calls == 1
    assert episode["base_state_hash"] != episode["policy_prompt_hash"]
    assert episode["official_em"] == 1.0
