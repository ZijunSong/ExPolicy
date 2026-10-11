"""Frozen skill bank loader. Hashes must not change after R7 start."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from expolicy.closed_loop.constants import C1, C2, C3, ROOT
from expolicy.closed_loop.frozen_skill_bank import (
    bridge_decomposition,
    comparison_alignment,
    long_chain_search,
)
from expolicy.closed_loop.io_utils import sha256_file, write_json

BANK_DIR = Path(__file__).resolve().parent
SKILL_FILES = {
    C1: BANK_DIR / "bridge_decomposition.py",
    C2: BANK_DIR / "comparison_alignment.py",
    C3: BANK_DIR / "long_chain_search.py",
}
EXECUTE_FILE = BANK_DIR / "execute.py"
SOURCE_SKILLS = ROOT / "expolicy" / "environments" / "public_search" / "skills.py"

RUNNERS: dict[str, Callable[..., Any]] = {
    C1: bridge_decomposition.run,
    C2: comparison_alignment.run,
    C3: long_chain_search.run,
}
SPECS = {
    C1: bridge_decomposition.spec,
    C2: comparison_alignment.spec,
    C3: long_chain_search.spec,
}


def current_git_head() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def build_manifest() -> dict[str, Any]:
    head = current_git_head()
    skills = []
    for cap, path in SKILL_FILES.items():
        spec = SPECS[cap]()
        skills.append(
            {
                **spec,
                "source_git_head": head,
                "sha256": sha256_file(path),
                "execute_sha256": sha256_file(EXECUTE_FILE),
                "source_sha256": sha256_file(SOURCE_SKILLS) if SOURCE_SKILLS.exists() else "",
                "r7_mutable": False,
                "validated_before_r7": True,
            }
        )
    return {
        "compiler": "r7.frozen_skill.v1",
        "git_head": head,
        "execute_sha256": sha256_file(EXECUTE_FILE),
        "skills": skills,
    }


def write_manifest(path: Path) -> dict[str, Any]:
    manifest = build_manifest()
    write_json(path, manifest)
    return manifest


def assert_frozen(manifest_path: Path) -> None:
    import json

    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = build_manifest()
    if current["execute_sha256"] != expected["execute_sha256"]:
        raise RuntimeError("frozen skill execute.py hash changed")
    exp_map = {s["skill_id"]: s["sha256"] for s in expected["skills"]}
    cur_map = {s["skill_id"]: s["sha256"] for s in current["skills"]}
    if exp_map != cur_map:
        raise RuntimeError(f"frozen skill hashes changed: {exp_map} vs {cur_map}")


def run_skill(capability: str, env: Any, task: Any, *, model: Any | None = None):
    if capability not in RUNNERS:
        raise KeyError(capability)
    return RUNNERS[capability](env, task, model=model)
