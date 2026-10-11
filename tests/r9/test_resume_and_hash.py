"""Resume must not double-submit; hash guard detects split/model/config change."""

from __future__ import annotations

from pathlib import Path

import pytest

from expolicy.r9_lab.io_util import ResumeHashError, assert_resume_hashes, write_hashes
from expolicy.r9_lab.resume import run_cell_once


def test_resume_does_not_double_submit_batch(tmp_path: Path):
    cell = tmp_path / "matrix" / "db-B00" / "DISCARD"
    submitted: list[str] = []

    def submit() -> dict:
        submitted.append("db-B00:DISCARD")
        return {"batch_id": "db-B00", "action": "DISCARD"}

    first = run_cell_once(cell, submit, resume=True)
    second = run_cell_once(cell, submit, resume=True)
    assert first["skipped"] is False
    assert second["skipped"] is True
    assert submitted == ["db-B00:DISCARD"]
    assert (cell / "DONE.json").exists()


def test_hash_guard_detects_split_model_config_change(tmp_path: Path):
    write_hashes(
        tmp_path,
        {"split_hash": "split-a", "model_hash": "model-a", "config_hash": "cfg-a"},
    )
    assert_resume_hashes(tmp_path, split_hash="split-a", model_hash="model-a", config_hash="cfg-a")
    with pytest.raises(ResumeHashError, match="split"):
        assert_resume_hashes(tmp_path, split_hash="split-b", model_hash="model-a", config_hash="cfg-a")
    with pytest.raises(ResumeHashError, match="model"):
        assert_resume_hashes(tmp_path, split_hash="split-a", model_hash="model-b", config_hash="cfg-a")
    with pytest.raises(ResumeHashError, match="config"):
        assert_resume_hashes(tmp_path, split_hash="split-a", model_hash="model-a", config_hash="cfg-b")


def test_05_and_07_dry_run(tmp_path: Path):
    import importlib.util

    frozen = {
        "batches": [
            {"batch_id": "db-B00", "env": "db", "items": [], "difficulty_mean": 1, "skill_count_mean": 1}
        ]
    }
    (tmp_path / "FROZEN_BATCHES.json").write_text(__import__("json").dumps(frozen), encoding="utf-8")
    path05 = Path("/data/ppnm/expolicy/scripts/r9/05_operator_sanity.py")
    spec = importlib.util.spec_from_file_location("r9_05", path05)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert mod.main(["--dry-run", "--output-root", str(tmp_path)]) == 0
    assert (tmp_path / "audit" / "OPERATOR_SANITY.md").exists()

    path07 = Path("/data/ppnm/expolicy/scripts/r9/07_run_assim_matrix.py")
    spec7 = importlib.util.spec_from_file_location("r9_07", path07)
    mod7 = importlib.util.module_from_spec(spec7)
    assert spec7.loader is not None
    spec7.loader.exec_module(mod7)
    assert mod7.main(["--dry-run", "--output-root", str(tmp_path)]) == 0
    assert (tmp_path / "matrix" / "db-B00" / "DISCARD" / "DONE.json").exists()
