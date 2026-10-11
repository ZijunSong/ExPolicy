from pathlib import Path

import pytest

from expolicy.closed_loop.runner.checkpoint import ResumeHashError, assert_resume_allowed, write_hashes
from expolicy.closed_loop.runner.config import R7Config
from expolicy.closed_loop.state import ExPolicyState


def test_resume_hash_guard(tmp_path: Path):
    cfg = R7Config(condition="expolicy", seed=0)
    state = ExPolicyState()
    write_hashes(
        tmp_path,
        {
            "config_hash": cfg.content_hash(),
            "split_hash": "abc",
            "state_hash": state.content_hash(),
            "policy_checkpoint_hash": "def",
        },
    )
    assert_resume_allowed(tmp_path, config=cfg, split_hash="abc", state=state, policy={})
    other = R7Config(condition="expolicy", seed=1)
    with pytest.raises(ResumeHashError):
        assert_resume_allowed(tmp_path, config=other, split_hash="abc", state=state, policy={})
    with pytest.raises(ResumeHashError):
        assert_resume_allowed(tmp_path, config=cfg, split_hash="zzz", state=state, policy={})
