"""Third-party working-tree cleanliness. Never modify those checkouts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from expolicy.r9_lab.constants import LAB_ROOT, MEMRL_ROOT, SKILLOPT_ROOT


def git_status_clean(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        return {"path": str(root), "exists": root.exists(), "git": False, "clean": None}
    try:
        out = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:  # noqa: BLE001
        return {"path": str(root), "exists": True, "git": True, "clean": False, "error": str(exc)}
    return {"path": str(root), "exists": True, "git": True, "clean": out.strip() == "", "head": head, "porcelain": out}


def third_party_audit() -> dict[str, Any]:
    return {
        "LifelongAgentBench": git_status_clean(LAB_ROOT),
        "SkillOpt": git_status_clean(SKILLOPT_ROOT),
        "MemRL": git_status_clean(MEMRL_ROOT),
    }
