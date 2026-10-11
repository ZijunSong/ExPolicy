"""Split isolation: task ids and questions disjoint across R7 (or frozen R5.3) splits."""

from __future__ import annotations

import json
from pathlib import Path

from expolicy.experiments.common import ROOT

R7 = ROOT / "data" / "r7_search_suite" / "split.json"
R53 = ROOT / "outputs" / "r5_3_external_generalization" / "splits" / "split.json"


def _check(path: Path, keys: tuple[str, ...]) -> None:
    payload = json.loads(path.read_text())
    buckets = {k: payload.get(k) or [] for k in keys}
    ids = {k: {r["task_id"] for r in buckets[k]} for k in keys}
    qs = {k: {str(r.get("question") or "").strip().lower() for r in buckets[k]} for k in keys}
    names = list(keys)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not (ids[a] & ids[b]), (path, a, b, len(ids[a] & ids[b]))
            assert not (qs[a] & qs[b]), (path, a, b, len(qs[a] & qs[b]))


def test_r53_frozen_splits_are_disjoint() -> None:
    assert R53.exists()
    _check(R53, ("calibration", "acquisition_pool", "skill_validation", "future_probe"))


def test_r7_splits_disjoint_if_present() -> None:
    if not R7.exists():
        return
    _check(R7, ("calibration", "acquisition_pool", "skill_validation", "future_probe"))
