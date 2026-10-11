from pathlib import Path

from expolicy.closed_loop.io_utils import read_json
from expolicy.closed_loop.runner.config import R7Config
from expolicy.closed_loop.runner.episode import run_r7_episode
from tests.r7._closed_loop_fakes import FakeClient, FakeModel, tiny_split


def test_round_transition(tmp_path: Path):
    cfg = R7Config(
        condition="expolicy",
        seed=0,
        rounds=2,
        batch_size=2,
        budget=4,
        smoke=True,
        full_eval_at=(0,),
        max_steps=4,
    )
    run_r7_episode(
        cfg,
        output_root=tmp_path,
        resume=False,
        model=FakeModel(),
        split=tiny_split(),
        client=FakeClient(),
    )
    seed = tmp_path / "expolicy" / "seed0"
    r0_before = read_json(seed / "round_00" / "policy_before.json")
    r0_after = read_json(seed / "round_00" / "policy_after.json")
    r1_before = read_json(seed / "round_01" / "policy_before.json")
    upd = read_json(seed / "round_00" / "policy_update.json")
    assert r0_before != r0_after
    assert r1_before == r0_after
    assert float(upd["acquisition"]["parameter_delta_norm"]) > 0
    assert float(upd["assimilation"]["parameter_delta_norm"]) > 0
    rew = read_json(seed / "round_00" / "reward.json")
    assert rew["uses_future_test"] is False
    assert (seed / "round_00" / "experience_batch.jsonl").exists()
    assert read_json(seed / "round_00" / "state_after.json")["round_id"] == 0
    assert read_json(seed / "round_01" / "state_before.json")["budget_used"] >= 2
