"""Matrix resume helper: flock + DONE.json so a batch/action is never double-submitted."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from expolicy.r9_lab.io_util import flock, is_done, mark_done, skip_if_done


def run_cell_once(
    cell_dir: Path,
    fn: Callable[[], dict[str, Any]],
    *,
    resume: bool = True,
) -> dict[str, Any]:
    cell_dir.mkdir(parents=True, exist_ok=True)
    with flock(cell_dir / ".lock"):
        if skip_if_done(cell_dir, resume=resume):
            return {"skipped": True, "reason": "already_done", "path": str(cell_dir)}
        payload = fn()
        mark_done(cell_dir, payload if isinstance(payload, dict) else {"ok": True})
        return {"skipped": False, "payload": payload, "path": str(cell_dir)}
