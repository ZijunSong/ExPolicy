"""Frozen R5.4 text router must not be refit in R6."""

from __future__ import annotations

from pathlib import Path

from expolicy.experiments.r5_4 import R54_ROOT, sha256_file


def test_router_pickle_stable():
    path = R54_ROOT / "router" / "text_router.pkl"
    assert path.exists()
    digest = sha256_file(path)
    marker = Path("/data/ppnm/expolicy/outputs/r6_ava/audit/ROUTER_SHA256.txt")
    marker.parent.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text(digest + "\n", encoding="utf-8")
    assert marker.read_text(encoding="utf-8").strip() == digest


def test_r6_code_does_not_call_train_r54_router():
    text = Path("/data/ppnm/expolicy/expolicy/experiments/r6_ava.py").read_text(encoding="utf-8")
    assert "train_r54_router(" not in text
    assert "load_router(R54_ROOT)" in text
