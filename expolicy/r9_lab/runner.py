"""Run one frozen-actor episode against a LAB task (or FakeLabEnv)."""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from expolicy.r9_lab.assimilators.episodic import format_episodic_injection, retrieve_episodic
from expolicy.r9_lab.assimilators.semantic import format_semantic_injection, retrieve_semantic
from expolicy.r9_lab.benchmark.adapter import FakeLabEnv, FakeTask, parse_agent_response
from expolicy.r9_lab.benchmark.verifier import assert_no_gold_in_prompt
from expolicy.r9_lab.constants import EPISODIC_TOP_K, GOLD_FIELD_NAMES, MAX_ROUND, SEMANTIC_TOP_K
from expolicy.r9_lab.state.persistent_state import PersistentState


@dataclass
class EpisodeResult:
    task_id: str
    env: str
    success: bool
    steps: int
    invalid_actions: int
    parser_valid: bool
    tokens: int
    reasoning_tokens: int
    wall_time: float
    infra_failure: bool
    messages: list[dict[str, Any]] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    instruction: str = ""

    def as_experience(self, *, bucket: str = "") -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "env": self.env,
            "instruction": self.instruction,
            "success": self.success,
            "verifier_outcome": "CORRECT" if self.success else "INCORRECT",
            "trajectory": self.trajectory,
            "bucket": bucket,
            "steps": self.steps,
            "invalid_actions": self.invalid_actions,
            "parser_valid": self.parser_valid,
            "tokens": self.tokens,
            "infra_failure": self.infra_failure,
        }


_injection_token_cap: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "a_v11_injection_token_cap", default=None
)


def set_injection_token_cap(limit: int | None) -> None:
    """A-v1.1 carrier cap. None keeps the historical uncapped injection."""
    _injection_token_cap.set(limit)


def _cap_injection(text: str) -> str:
    limit = _injection_token_cap.get()
    if not text or not limit or limit <= 0:
        return text
    # Character fallback when the actor tokenizer is not loaded in this process.
    # 4 characters is a conservative stand-in for one token and never pads.
    return text[: int(limit) * 4]


def inject_state_prefix(state: PersistentState | None, instruction: str) -> str:
    if state is None:
        return ""
    parts: list[str] = []
    e_hits = retrieve_episodic(state, instruction, k=EPISODIC_TOP_K, trace=True)
    s_hits = retrieve_semantic(state, instruction, k=SEMANTIC_TOP_K, trace=True)
    e_text = format_episodic_injection(e_hits)
    s_text = format_semantic_injection(s_hits)
    if e_text:
        parts.append(e_text)
    if s_text:
        parts.append(s_text)
    if state.skill_markdown.strip():
        parts.append("Skill document:\n" + state.skill_markdown.strip())
    return _cap_injection("\n\n".join(parts))


def build_messages(
    *,
    instruction: str,
    state: PersistentState | None,
    history: Sequence[Mapping[str, Any]] | None = None,
    gold: Any = None,
) -> list[dict[str, Any]]:
    prefix = inject_state_prefix(state, instruction)
    # Official PreviousSampleUtilizationCallback replaces
    # {previous_sample_utilization_target_position} in the first user prompt.
    # We prepend retrieved store contents onto the first user message.
    first = (prefix + "\n\n" + instruction).strip() if prefix else instruction
    messages: list[dict[str, Any]] = [{"role": "user", "content": first}]
    if history:
        for turn in history:
            role = str(turn.get("role") or "assistant")
            content = str(turn.get("content") or "")
            messages.append({"role": role, "content": content})
    blob = " ".join(m["content"] for m in messages)
    for banned in GOLD_FIELD_NAMES:
        if banned in first.lower() and banned not in instruction.lower():
            raise AssertionError(f"gold-like field {banned} injected into prompt")
    if gold is not None:
        assert_no_gold_in_prompt(messages, gold)
    return messages


def run_fake_episode(
    env: FakeLabEnv,
    task: FakeTask,
    client: Any,
    *,
    state: PersistentState | None = None,
) -> EpisodeResult:
    t0 = time.time()
    infra_failure = False
    tokens = 0
    reasoning_tokens = 0
    messages: list[dict[str, Any]] = []
    try:
        obs = env.reset(task.task_id)
        gold = env.gold_for(task.task_id)
        messages = build_messages(instruction=task.instruction, state=state, gold=gold)
        max_round = MAX_ROUND.get(task.env, 3)
        for _ in range(max_round):
            result = client.chat(messages)
            tokens += int(getattr(result, "total_tokens", 0) or 0)
            reasoning_tokens += int(getattr(result, "reasoning_tokens", 0) or 0)
            agent_text = result.text
            messages.append({"role": "assistant", "content": agent_text})
            step = env.interact(agent_text)
            if not step["done"]:
                messages.append({"role": "user", "content": step["observation"]})
            else:
                break
        summary = env.complete()
    except Exception:
        infra_failure = True
        summary = {
            "success": False,
            "steps": env.steps,
            "invalid_actions": env.invalid_actions,
            "parser_valid": False,
            "infra_failure": True,
        }
    wall = time.time() - t0
    return EpisodeResult(
        task_id=task.task_id,
        env=task.env,
        success=bool(summary.get("success")),
        steps=int(summary.get("steps") or 0),
        invalid_actions=int(summary.get("invalid_actions") or 0),
        parser_valid=bool(summary.get("parser_valid")),
        tokens=tokens,
        reasoning_tokens=reasoning_tokens,
        wall_time=wall,
        infra_failure=infra_failure or bool(summary.get("infra_failure")),
        messages=messages,
        trajectory=list(env.trajectory),
        instruction=task.instruction,
    )


def run_episode(
    *,
    task: FakeTask,
    client: Any,
    state: PersistentState | None = None,
    lab_env: FakeLabEnv | None = None,
) -> EpisodeResult:
    """Public episode entry. Official LAB Task path is optional; tests use FakeLabEnv."""
    env = lab_env or FakeLabEnv([task])
    return run_fake_episode(env, task, client, state=state)
