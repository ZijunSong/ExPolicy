from pathlib import Path

from expolicy.closed_loop.frozen_skill_bank import assert_frozen, build_manifest, write_manifest


def test_skill_hash_frozen(tmp_path: Path):
    path = tmp_path / "FROZEN_SKILL_BANK_MANIFEST.json"
    m1 = write_manifest(path)
    assert_frozen(path)
    m2 = build_manifest()
    assert m1["execute_sha256"] == m2["execute_sha256"]
    assert {s["skill_id"]: s["sha256"] for s in m1["skills"]} == {s["skill_id"]: s["sha256"] for s in m2["skills"]}
    for s in m1["skills"]:
        assert s["r7_mutable"] is False
        assert s["validated_before_r7"] is True
