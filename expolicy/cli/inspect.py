"""Inspect saved agents and experiences."""

from __future__ import annotations

import json
from pathlib import Path

from expolicy.core.errors import CheckpointError
from expolicy.core.state import PersistentAgentState
from expolicy.logging.event_logger import iter_jsonl


def inspect_agent(path: str | Path) -> dict:
    state = PersistentAgentState.load(path)
    root = Path(path)
    episodic_n = 0
    rec = root / "episodic_memory" / "records.jsonl"
    if rec.exists():
        episodic_n = sum(1 for line in rec.read_text(encoding="utf-8").splitlines() if line.strip())
    semantic_n = 0
    srec = root / "semantic_memory" / "rules.jsonl"
    if srec.exists():
        semantic_n = sum(1 for line in srec.read_text(encoding="utf-8").splitlines() if line.strip())
    skills = json.loads((root / "skills" / "skills.json").read_text()) if (root / "skills" / "skills.json").exists() else []
    return {
        "version": state.version,
        "model_name": state.model_name,
        "adapter_path": state.adapter_path,
        "harness": state.harness.model_dump(),
        "capability_profile": state.capability_profile.model_dump(),
        "metrics": state.metrics,
        "lineage": [x.model_dump() for x in state.lineage],
        "n_episodic": episodic_n,
        "n_semantic": semantic_n,
        "n_skills": len(skills),
    }


def inspect_experience(run_dir: str | Path, experience_id: str) -> dict:
    path = Path(run_dir) / "experiences.jsonl"
    if not path.exists():
        raise CheckpointError(f"no experiences.jsonl in {run_dir}")
    for row in iter_jsonl(path):
        if row.get("experience_id") == experience_id:
            return row
    raise CheckpointError(f"experience {experience_id} not found")
