"""Gold / future / dataset must not leak into R7 skill artifacts. Real classes + artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from expolicy.experiments.common import ROOT
from expolicy.r7.interventional_skills import PRIMITIVES, skill_has_leakage
from expolicy.r7.skillopt import apply_edit, propose_edits, analyze_batch
from expolicy.core.experience import Experience


def test_interventional_primitives_have_no_leak_keys() -> None:
    for cap, fn in PRIMITIVES.items():
        skill = fn()
        issues = skill_has_leakage(skill)
        assert not issues, (cap, issues)
        blob = json.dumps(skill.model_dump(mode="json"))
        assert "gold_answer" not in blob
        assert "future_probe" not in blob
        assert skill.answer_synthesis_rule == "policy_model_only"


def test_r53_count_only_compiler_source_has_no_gold_inputs() -> None:
    src = (ROOT / "expolicy" / "environments" / "public_search" / "skills.py").read_text()
    fn = src.split("def count_only_public_skill")[1].split("def ")[0]
    assert "gold" not in fn.lower()
    assert "future" not in fn.lower()


def test_skillopt_edit_cannot_insert_gold() -> None:
    skill = PRIMITIVES["bridge_decomposition"]()
    experiences = [
        Experience(task_id=f"t{i}", question="Who founded Acme?", capability="bridge_decomposition", success=False)
        for i in range(10)
    ]
    analysis = analyze_batch(experiences, "bridge_decomposition")
    edits = propose_edits(skill, analysis, max_edits=4)
    for ed in edits:
        nxt = apply_edit(skill, ed)
        assert not skill_has_leakage(nxt)
        assert "gold_answer" not in json.dumps(nxt.model_dump(mode="json"))
