"""One pre-registered resample. A second request is refused."""

from __future__ import annotations

from pathlib import Path

from expolicy.r9_lab.io_util import read_json, write_json


def request_resample(path: Path, *, budget: int = 1) -> int:
    used = 0
    if path.exists():
        used = int(read_json(path).get("used") or 0)
    if used >= budget:
        raise RuntimeError("resample budget is exhausted; do not expand the sample again to chase a positive result")
    used += 1
    write_json(path, {"used": used, "budget": budget})
    return used
