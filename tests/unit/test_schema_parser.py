"""Schema, lifecycle, parser, metrics."""

from __future__ import annotations

from expolicy.agents.parser import parse_tool_call
from expolicy.core.experience import Experience, ExperienceCost
from expolicy.core.lifecycle import transition
from expolicy.core.types import ExperienceStatus
from expolicy.core.utility import compute_utility
from expolicy.evaluation.metrics import exact_match, token_f1


def test_experience_json_roundtrip() -> None:
    exp = Experience(task_id="t1", question="Where is Aurel Institute headquartered?", predicted_answer="Aurel")
    data = exp.model_dump_json()
    again = Experience.model_validate_json(data)
    assert again.task_id == "t1"
    assert again.cost.tokens == 0


def test_lifecycle_rejects_illegal() -> None:
    st = transition(ExperienceStatus.ACQUIRED, ExperienceStatus.VERIFIED)
    assert st is ExperienceStatus.VERIFIED
    try:
        transition(ExperienceStatus.ACQUIRED, ExperienceStatus.USED)
        ok = True
    except Exception:
        ok = False
    assert not ok


def test_parser_url_style_search() -> None:
    parsed = parse_tool_call("search?q=Lira+Venn+bio&num_results=5")
    assert parsed.valid
    assert parsed.tool == "search"
    assert "Lira" in parsed.arguments["query"]
    parsed = parse_tool_call("<tool>search</tool>\n<arg>Aurel Institute headquarters</arg>")
    assert parsed.valid
    assert parsed.tool == "search"
    assert "Aurel" in parsed.arguments["query"]
    ans = parse_tool_call("<tool>answer</tool><arg>Aurel</arg>")
    assert ans.tool == "answer"
    assert ans.arguments["text"] == "Aurel"


def test_metrics_and_utility() -> None:
    assert exact_match("Aurel", "aurel")
    assert token_f1("Aurel Institute", "Aurel") > 0
    u = compute_utility(future_gain=0.2, cost=ExperienceCost(tokens=1000, environment_steps=4))
    assert u.utility != 0.0
