from expolicy.paper_d.calibration_run import planned_cells
from expolicy.paper_d.official_worker import artifact_digest, artifact_digest_matches


def test_calibration_plan_is_384_cells() -> None:
    probes = [
        {"task_id": f"os:{index}", "sample_index": index, "probe_role": "future" if index < 4 else "retention"}
        for index in range(8)
    ]
    cells = planned_cells(probes)
    assert len(cells) == 384
    assert len({(row["state_id"], row["batch_id"], row["task_id"], row["repeat_id"], row["branch"]) for row in cells}) == 384


def test_artifact_digest_ignores_cache_hit_and_accepts_legacy_hash() -> None:
    from expolicy.r9_lab.io_util import sha256_json

    artifact = {"success": True, "cache_hit": False, "result_hash": "placeholder"}
    digest = artifact_digest(artifact)
    artifact["result_hash"] = digest
    assert artifact_digest_matches(artifact, digest)
    legacy = sha256_json({key: value for key, value in artifact.items() if key != "result_hash"})
    assert artifact_digest_matches(artifact, legacy)
