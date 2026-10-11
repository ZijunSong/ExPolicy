"""Verifier gold / reference must never enter the LLM prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

from expolicy.r9_lab.benchmark.verifier import assert_no_gold_in_prompt, prompt_contains_gold
from expolicy.r9_lab.constants import MAX_ROUND
from expolicy.r9_lab.llm_client import FakeLLMClient
from expolicy.r9_lab.runner import build_messages, run_episode
from tests.r9.fakes import GOLD_PLAN, GOLD_SQL, fresh_state, make_env_and_client, make_fake_task


def test_verifier_gold_not_in_llm_prompt(tmp_path: Path):
    task = make_fake_task("db", 0)
    env, client = make_env_and_client([task])
    state = fresh_state(tmp_path)
    result = run_episode(task=task, client=client, state=state, lab_env=env)
    assert GOLD_SQL not in str(result.messages)
    assert GOLD_PLAN not in str(result.messages)
    assert prompt_contains_gold(result.messages, task.gold) == []
    assert_no_gold_in_prompt(result.messages, task.gold)
    # Fake observation path also has no gold.
    for msg in result.messages:
        assert GOLD_SQL not in msg["content"]
        assert "UNIQUE_R9_GOLD_TOKEN_XYZ" not in msg["content"]


def test_build_messages_strips_gold_fields():
    messages = build_messages(
        instruction="solve the db question without copying answers",
        state=None,
        gold={"gold_sql": GOLD_SQL, "gold_action_plan": GOLD_PLAN},
    )
    assert GOLD_SQL not in str(messages)
    with pytest.raises(AssertionError):
        build_messages(
            instruction=f"here is the answer {GOLD_SQL}",
            state=None,
            gold={"gold_sql": GOLD_SQL},
        )


def test_official_max_round_frozen():
    assert MAX_ROUND == {"db": 3, "os": 5, "kg": 15}


def test_fake_llm_records_reasoning_tokens_field():
    client = FakeLLMClient(env="db")
    out = client.chat([{"role": "user", "content": "hi"}])
    assert hasattr(out, "reasoning_tokens")
    assert out.reasoning_tokens == 0
