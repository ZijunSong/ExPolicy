"""Pre-registered selection score, tie-break, and frozen manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from expolicy.r9_lab.manifest import ManifestFrozenError, assert_manifest_unmodified, freeze_manifest, load_manifest
from expolicy.r9_lab.selection import (
    NO_FULL_PASS,
    EnvScreenStats,
    evaluate_model,
    select_model,
    selection_score,
)


def _stats(s_db, s_os, s_kg, parser=0.95, infra=0.0):
    return {
        "db": EnvScreenStats(s_db, parser, infra, n=24),
        "os": EnvScreenStats(s_os, parser, infra, n=24),
        "kg": EnvScreenStats(s_kg, parser, infra, n=24),
    }


def test_selection_score_formula():
    # |0.40-0.40|=0, |0.50-0.40|=0.10, |0.30-0.40|=0.10
    # mean=0.20/3, max=0.10, parser=0.95 -> 0.5*(1-0.95)=0.025
    by = _stats(0.40, 0.50, 0.30, parser=0.95)
    mean_dev = (0.0 + 0.10 + 0.10) / 3
    expected = mean_dev + 0.5 * 0.10 + 0.5 * (1 - 0.95)
    assert abs(selection_score(by) - expected) < 1e-12


def test_full_pass_band_and_parser_infra():
    ok = evaluate_model("Q0", _stats(0.10, 0.70, 0.40, parser=0.90, infra=0.05))
    assert ok.full_pass
    bad_band = evaluate_model("Q0", _stats(0.09, 0.40, 0.40))
    assert not bad_band.full_pass
    bad_parser = evaluate_model("Q0", _stats(0.40, 0.40, 0.40, parser=0.89))
    assert not bad_parser.full_pass
    bad_infra = evaluate_model("Q0", _stats(0.40, 0.40, 0.40, infra=0.06))
    assert not bad_infra.full_pass
    ceiling = evaluate_model("Q0", _stats(0.71, 0.40, 0.40))
    assert not ceiling.full_pass


def test_lowest_score_wins():
    q0 = evaluate_model("Q0", _stats(0.50, 0.50, 0.50, parser=0.95))
    q1 = evaluate_model("Q1", _stats(0.40, 0.40, 0.40, parser=0.95))
    out = select_model([q0, q1])
    assert out.status == "Q1"
    assert out.selected is not None


def test_tie_break_q0_over_q1_q2_q3():
    identical = _stats(0.40, 0.40, 0.40, parser=0.95)
    results = [evaluate_model(mid, identical) for mid in ("Q3", "Q2", "Q1", "Q0")]
    out = select_model(results)
    assert out.status == "Q0"
    # reverse insertion order still Q0
    out2 = select_model(list(reversed(results)))
    assert out2.status == "Q0"


def test_no_full_pass_status():
    q0 = evaluate_model("Q0", _stats(0.01, 0.01, 0.01))
    q1 = evaluate_model("Q1", _stats(0.99, 0.99, 0.99))
    out = select_model([q0, q1])
    assert out.status == NO_FULL_PASS
    assert out.selected is None


def test_manifest_freeze_immutability(tmp_path: Path):
    path = tmp_path / "model_manifest.json"
    payload = {
        "selected_model_id": "Q0",
        "hf_repo": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "local_path": "/data/ppnm/models/Qwen3-30B-A3B-Instruct-2507",
        "thinking_mode": "non-thinking-only",
        "dtype": "bfloat16",
        "selection_score": 0.12,
        "screen_split_sha256": "abc",
    }
    freeze_manifest(path, payload)
    stored = load_manifest(path)
    assert stored["frozen"] is True
    assert_manifest_unmodified(path)
    mutated = dict(stored)
    mutated["selected_model_id"] = "Q3"
    with pytest.raises(ManifestFrozenError):
        assert_manifest_unmodified(path, mutated)
    # tamper file
    text = path.read_text()
    path.write_text(text.replace("Q0", "Q3", 1), encoding="utf-8")
    with pytest.raises(ManifestFrozenError):
        assert_manifest_unmodified(path)
