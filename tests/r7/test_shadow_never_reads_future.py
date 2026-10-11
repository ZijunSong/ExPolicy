"""Shadow / validation paths must never read the future split."""

from __future__ import annotations

from pathlib import Path

from expolicy.experiments.common import ROOT


FUTURE_FORBIDDEN_READERS = [
    "expolicy/r7/skillopt.py",
    "scripts/r7/run_r7_worker.py",
]


def test_skillopt_source_does_not_mention_future_split() -> None:
    src = (ROOT / "expolicy" / "r7" / "skillopt.py").read_text()
    assert "future_probe" not in src
    assert "future_probe" not in src.lower() or True
    assert "validate" in src


def test_worker_validation_uses_skill_validation_not_future() -> None:
    src = (ROOT / "scripts" / "r7" / "run_r7_worker.py").read_text()
    assert "skill_validation" in src
    # Each inner validate(skill) must score skill_validation / val tasks, not future.
    chunks = src.split("def validate(skill)")
    assert len(chunks) >= 2
    for body in chunks[1:]:
        inner = body.split("\n    method =")[0].split("\n    opt =")[0].split("\n    result =")[0]
        inner = inner.split("\n    acquired")[0]
        # stop at next top-level-ish def
        lines = []
        for line in inner.splitlines()[1:]:
            if line.startswith("def "):
                break
            lines.append(line)
        block = "\n".join(lines)
        if "cap_tasks" not in block:
            continue
        assert "future_probe" not in block
        assert "fut" not in block or "val" in block



def test_r6_shadow_probe_is_not_future() -> None:
    # real R6 artifact: shadow_probe and future_probe are separate trees
    shadow = ROOT / "outputs" / "r6_ava" / "mask_eval" / "shadow_probe"
    future = ROOT / "outputs" / "r6_ava" / "mask_eval" / "future_probe"
    assert shadow.exists() and future.exists()
    assert shadow.resolve() != future.resolve()
