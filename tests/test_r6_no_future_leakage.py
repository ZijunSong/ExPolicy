"""Final future ids must never enter selector / shadow evaluator."""

from __future__ import annotations

import json
from pathlib import Path

from expolicy.experiments.r6_ava import R6_ROOT

ROOT = Path("/data/ppnm/expolicy")


def _split():
    path = R6_ROOT / "split_r6_ava.json"
    if not path.exists():
        import subprocess
        import sys

        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "build_r6_ava_split.py")], cwd=str(ROOT))
    return json.loads(path.read_text(encoding="utf-8"))


def test_no_id_or_question_overlap():
    split = _split()
    def ids(key):
        return {r["task_id"] for r in split[key]}
    def qs(key):
        return {str(r.get("question") or "").strip().lower() for r in split[key]}
    c, s, f = ids("candidate_pool"), ids("shadow_probe"), ids("future_probe")
    assert not (c & s) and not (c & f) and not (s & f)
    cq, sq, fq = qs("candidate_pool"), qs("shadow_probe"), qs("future_probe")
    assert not (cq & sq) and not (cq & fq) and not (sq & fq)


def test_future_size_untouched():
    split = _split()
    r54 = json.loads((ROOT / "outputs" / "r5_4_rasa" / "split_r5_4.json").read_text(encoding="utf-8"))
    assert [r["task_id"] for r in split["future_probe"]] == [r["task_id"] for r in r54["future_probe"]]


def test_shadow_is_180_and_candidate_720():
    split = _split()
    assert len(split["candidate_pool"]) == 720
    assert len(split["shadow_probe"]) == 180
    assert len(split["future_probe"]) == 450


def test_ava_source_has_no_future_probe_in_selector():
    text = (ROOT / "expolicy" / "acquisition" / "ava.py").read_text(encoding="utf-8")
    assert "future_probe" not in text
    assert "gold_answer" not in text
