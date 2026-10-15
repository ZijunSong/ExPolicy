from expolicy.paper_d.confirm_analysis import decide
from expolicy.paper_d.confirm_controls import choose_wrong_bundle, length_ratio
from expolicy.paper_d.confirm_plan import build_plan, expected_count
from expolicy.paper_d.confirm_utility import content_utility_gap, mean_utility, update_utility


def test_content_gap_uses_each_utility_not_the_retention_gap() -> None:
    real = [update_utility(0.0, 0.1)]
    empty = [update_utility(0.0, -0.1)]
    assert real == [0.0]
    assert empty == [-0.05]
    assert content_utility_gap(real, empty) == 0.05


def test_opposite_retention_is_penalized_before_averaging() -> None:
    assert mean_utility([0.0, 0.0], [0.2, -0.2]) == -0.05


def test_missing_retention_blocks_utility() -> None:
    assert update_utility(0.2, None) is None


def test_length_gate_blocks_a_short_wrong_bundle() -> None:
    real = [{"task_id": "os:1", "instruction": "create alpha directory", "trajectory": [{"agent": "alpha " * 50}], "skill_list": ["a"]}]
    donor = [{"task_id": "os:2", "instruction": "create beta directory", "trajectory": [{"agent": "beta"}], "skill_list": ["b"]}]
    result = choose_wrong_bundle(real, donor, ["unrelated question"], lambda text: len(text))
    assert result["status"] == "length_gate_failed"
    assert length_ratio(100, 50) == 0.5


def test_plan_count_for_the_default_confirmation() -> None:
    assert expected_count(48) == 2304
    tasks = [
        {"task_id": f"os:{index}", "sample_index": str(index), "instruction": f"task {index}", "raw_task_hash": "h", "probe_role": "future", "task_family": "x"}
        for index in range(2)
    ]
    assert len(build_plan(tasks, ["C1", "C2"])) == 2 * 4 * 2 * 2


def test_positive_points_without_the_bound_stay_a_candidate() -> None:
    intervals = {
        name: {"point": 0.05, "simultaneous_low": 0.0, "simultaneous_high": 0.1}
        for name in ("U-REAL-NO_UPDATE", "U-REAL-EMPTY", "U-REAL-WRONG")
    }
    label = decide({"controls_valid": True, "bootstrap": {"intervals": intervals}, "robustness_ok": True, "single_batch_or_task": False})
    assert label == "D_CONFIRM_CANDIDATE_REPLICATED"
