"""Harness patches cannot execute arbitrary keys/code."""

from __future__ import annotations

from pathlib import Path

import pytest

from expolicy.assimilation.carriers.harness_store import HarnessStore
from expolicy.core.errors import HarnessPatchRejected
from expolicy.core.state import HarnessConfig, empty_agent


def test_illegal_harness_key_rejected(tmp_path: Path) -> None:
    store = HarnessStore(tmp_path)
    with pytest.raises(HarnessPatchRejected):
        store.propose(HarnessConfig(), {"shell_command": "rm -rf /"})


def test_legal_patch_increments_version(tmp_path: Path) -> None:
    store = HarnessStore(tmp_path)
    cand = store.propose(HarnessConfig(), {"search_top_k": 8, "min_evidence_before_answer": 2})
    assert cand.version == 1
    assert cand.search_top_k == 8
    state = empty_agent(tmp_path / "agent")
    store.commit(state, cand, {"acc": 0.5})
    assert state.harness.search_top_k == 8
