"""Persistent state checkpoint roundtrip."""

from __future__ import annotations

from pathlib import Path

from expolicy.core.state import empty_agent


def test_agent_checkpoint_roundtrip(tmp_path: Path) -> None:
    state = empty_agent(tmp_path / "agent_v000", model_name="qwen")
    state.metrics = {"accuracy": 0.1}
    state.write_manifest()
    loaded = type(state).load(tmp_path / "agent_v000")
    assert loaded.version == "agent_v000"
    assert loaded.metrics["accuracy"] == 0.1
    child_root = tmp_path / "copies"
    child_root.mkdir()
    dest = tmp_path / "agent_v001"
    child = state.snapshot_into(dest)
    child.version = "agent_v001"
    child.write_manifest()
    assert (dest / "manifest.json").exists()
    assert child.version_index() == 1
