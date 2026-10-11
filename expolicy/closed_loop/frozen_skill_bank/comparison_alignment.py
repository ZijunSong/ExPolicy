"""Frozen C2: comparison_alignment. Byte-frozen after R7 start."""

from __future__ import annotations

from typing import Any

from expolicy.closed_loop.constants import C2
from expolicy.closed_loop.frozen_skill_bank.execute import run_frozen_skill
from expolicy.environments.search.world import Task

SKILL_ID = "r7.frozen.comparison_alignment"
CAPABILITY = C2
TOOL_SEQUENCE = ["search", "open", "search", "open", "answer"]
VALIDATED_BEFORE_R7 = True
R7_MUTABLE = False


def spec() -> dict[str, Any]:
    return {
        "skill_id": SKILL_ID,
        "capability": CAPABILITY,
        "tool_sequence": TOOL_SEQUENCE,
        "validated_before_r7": VALIDATED_BEFORE_R7,
        "r7_mutable": R7_MUTABLE,
        "source_path": "expolicy/environments/public_search/skills.py::skill_c2_comparison",
    }


def run(env: Any, task: Task, *, model: Any | None = None):
    return run_frozen_skill(C2, env, task, model=model)
