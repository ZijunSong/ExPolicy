from expolicy.paper_d.content_analysis import decide, single_task_dependent, utility_from_deltas
from expolicy.paper_d.content_controls import clone_state, empty_leaks_source, injection_for, write_empty
from expolicy.paper_d.content_plan import build_block, execution_identity, expected_counts
from expolicy.paper_d.content_tasks import select_role_tasks
from expolicy.r9_lab.state.persistent_state import EpisodicRecord, PersistentState


def _task(index: int, role: str) -> dict:
    return {
        "task_id": f"os:{index}",
        "sample_index": str(index),
        "instruction": f"Create unique directory /task{index} with marker {index}",
        "raw_task_hash": f"hash{index}",
        "task_family": "skills:mkdir",
        "probe_role": role,
        "skill_list": ["mkdir"],
    }


def test_plan_counts() -> None:
    counts = expected_counts()
    assert counts == {"main": 768, "history": 256, "case": 32}
    tasks = [_task(index, "future") for index in range(48)]
    assert len(build_block(tasks, block_id="main", state_id="S0")) == 2 * 4 * 48 * 2


def test_known_development_task_is_excluded() -> None:
    records = [
        {"task_id": "os:9", "sample_index": "9", "instruction": "old development task about audit logs", "skill_list": ["ls"]},
        {"task_id": "os:200", "sample_index": "200", "instruction": "brand new backup rotation for /var/custom200", "skill_list": ["cp"]},
    ]
    selected = select_role_tasks(records, n=1, excluded_ids={"os:9"}, blocked_text=[], seed=1, role="future")
    assert selected[0]["task_id"] == "os:200"


def test_missing_retention_makes_utility_unavailable_and_positive_retention_is_not_rewarded() -> None:
    assert utility_from_deltas(0.2, None) is None
    assert utility_from_deltas(0.0, 0.25) == 0.0
    assert utility_from_deltas(0.0, -0.4) == -0.2


def test_one_task_does_not_become_broad_transfer() -> None:
    deltas = [
        {"task_id": "os:1", "family": "a", "delta": 1.0},
        {"task_id": "os:2", "family": "b", "delta": 0.0},
        {"task_id": "os:3", "family": "c", "delta": 0.0},
        {"task_id": "os:4", "family": "d", "delta": 0.0},
    ]
    assert single_task_dependent(deltas)
    label = decide(
        {
            "future": {
                "estimable": True,
                "REAL-NO_UPDATE": {"point": 0.25, "low": 0.0, "high": 0.5},
                "REAL-EMPTY": {"point": 0.25, "low": 0.0, "high": 0.5},
                "REAL-MATCHED_WRONG": {"point": 0.25, "low": 0.0, "high": 0.5},
            },
            "utility": {"REAL-NO_UPDATE": {"point": 0.25, "low": 0.0}},
            "benefits": {"n_benefited": 1, "n_families": 1},
            "single_task_dependent": True,
        }
    )
    assert label == "D_LOCAL_CASE_ONLY"


def test_empty_injection_drops_source_text_and_control_hash_changes_identity(tmp_path) -> None:
    base = PersistentState(root=str(tmp_path / "base"))
    base.episodic = [
        EpisodicRecord(
            task_id="os:1",
            env="os",
            instruction="Create user alice with sudo useradd and a private authorized key",
            trajectory=[{"agent": "sudo useradd alice"}],
            success=True,
        )
    ]
    base.committed_task_ids = ["os:1"]
    real = clone_state(base, str(tmp_path / "real"))
    empty = write_empty(real, ["os:1"], str(tmp_path / "empty"))
    leaked = empty_leaks_source(
        injection_for(_state_from_report(empty), "Create a directory")["injection"],
        [{"instruction": "Create user alice with sudo useradd and a private authorized key"}],
    )
    assert leaked == []
    cell = {"block_id": "main", "batch_id": "os-B1", "condition": "REAL", "task_id": "os:9", "sample_index": "9", "probe_role": "future", "repeat_id": 0}
    assert execution_identity(cell, state_hash="s", control_hash="a") != execution_identity(cell, state_hash="s", control_hash="b")


def _state_from_report(report: dict) -> PersistentState:
    return PersistentState.load(report["path"])
