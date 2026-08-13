"""End-to-end environment + parser + verifier without an LLM."""

from __future__ import annotations

from expolicy.agents.parser import parse_tool_call
from expolicy.assimilation.verifier import ExperienceVerifier
from expolicy.core.experience import Experience
from expolicy.environments.search.environment import SearchEnvironment
from expolicy.evaluation.metrics import exact_match


def test_manual_tool_loop_can_answer_single_hop() -> None:
    env = SearchEnvironment.from_seed(13)
    task = next(
        t
        for t in env.world.tasks.values()
        if t.capability == "single_hop" and t.split == "validation" and "headquartered" in t.question
    )
    env.reset(task)
    parsed = parse_tool_call(f"<tool>search</tool><arg>{task.question}</arg>")
    search = env.step(parsed.tool, parsed.arguments)
    assert search.ok
    doc_id = search.payload["hits"][0]["document_id"]
    opened = env.step("open", {"document_id": doc_id})
    assert opened.ok
    text = opened.payload["text"]
    # extractive check: gold answer appears in retrieved text for this controlled world
    assert task.gold_answer in text
    exp = Experience(
        task_id=task.task_id,
        question=task.question,
        predicted_answer=task.gold_answer,
        gold_answer=task.gold_answer,
        success=True,
        capability=task.capability,
        tool_calls=[
            {"tool": "search", "arguments": parsed.arguments, "ok": True},
            {"tool": "open", "arguments": {"document_id": doc_id}, "ok": True},
        ],
        actions=[{"tool": "search"}, {"tool": "open"}],
        observations=[{"payload": search.payload}, {"payload": opened.payload}],
        evidence=[{"document_id": doc_id}],
    )
    ver = ExperienceVerifier(env.world).verify(exp)
    assert ver.valid
    assert exact_match(exp.predicted_answer, task.gold_answer)
