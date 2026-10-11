"""Eval isolation, joint oracle commit rules, future not used for selection."""

from __future__ import annotations

from pathlib import Path

from expolicy.r9_lab.assimilators.episodic import apply_episodic
from expolicy.r9_lab.constants import A0_DISCARD, A1_EPISODIC, A2_SEMANTIC, A3_SKILLOPT
from expolicy.r9_lab.evaluation.future import evaluate_future
from expolicy.r9_lab.evaluation.retention import evaluate_retention
from expolicy.r9_lab.evaluation.shadow import evaluate_shadow
from expolicy.r9_lab.joint_oracle.candidates import generate_candidates
from expolicy.r9_lab.joint_oracle.counterfactual import run_counterfactual
from expolicy.r9_lab.joint_oracle.oracle import (
    assert_q_has_no_future,
    commit_selected,
    select_o3_oracle_joint,
)
from expolicy.r9_lab.llm_client import FakeLLMClient
from expolicy.r9_lab.state.persistent_state import PersistentState
from tests.r9.fakes import fresh_state, make_env_and_client, make_fake_task


def _exp(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "env": "db",
        "instruction": f"instruction {task_id}",
        "success": True,
        "verifier_outcome": "CORRECT",
        "trajectory": [{"agent": "ok"}],
        "bucket": "acquisition_pool",
    }


def test_shadow_and_retention_do_not_write_persistent_state(tmp_path: Path):
    state = fresh_state(tmp_path)
    apply_episodic(state, [_exp("db:acq:1")])
    tasks = [make_fake_task("db", i, split="shadow") for i in range(3)]
    env, client = make_env_and_client(tasks)
    before = state.content_hash()
    evaluate_shadow(tasks, client, state, lab_env=env)
    env2, client2 = make_env_and_client(tasks)
    evaluate_retention(tasks, client2, state, lab_env=env2)
    assert state.content_hash() == before
    reloaded = PersistentState.load(state.root_path())
    assert reloaded.content_hash() == before


def test_counterfactual_branch_does_not_pollute_live(tmp_path: Path):
    live = fresh_state(tmp_path)
    live.skill_markdown = "# live\n"
    live.save()
    before = live.content_hash()
    run_counterfactual(
        live,
        [_exp("db:acq:9")],
        A1_EPISODIC,
        branch_dir=tmp_path / "branch",
    )
    assert live.content_hash() == before
    assert not any(r.task_id == "db:acq:9" for r in live.episodic)
    branch = PersistentState.load(tmp_path / "branch")
    assert any(r.task_id == "db:acq:9" for r in branch.episodic)


def test_only_selected_candidate_batch_is_committed(tmp_path: Path):
    live = fresh_state(tmp_path)
    pool = [_exp(f"db:acq:{i}") for i in range(16)]
    cands = generate_candidates(pool, seed=0, round_id=0, m=4, batch_size=4)
    q = {}
    for c in cands:
        for a in (A0_DISCARD, A1_EPISODIC, A2_SEMANTIC, A3_SKILLOPT):
            q[(c["candidate_id"], a)] = 0.0
    q[(cands[2]["candidate_id"], A1_EPISODIC)] = 1.0
    chosen = select_o3_oracle_joint(cands, q)
    assert chosen["candidate_id"] == cands[2]["candidate_id"]
    result = commit_selected(
        live,
        candidates=cands,
        selected_candidate_id=chosen["candidate_id"],
        selected_action=chosen["action"],
    )
    committed = set(cands[2]["task_ids"])
    live_ids = {r.task_id for r in live.episodic}
    assert live_ids == committed
    for cid in result["not_committed"]:
        other = next(c for c in cands if c["candidate_id"] == cid)
        assert not (set(other["task_ids"]) & live_ids)


def test_final_future_metric_not_used_in_oracle_selection():
    cands = [
        {"candidate_id": "c0", "items": [], "task_ids": []},
        {"candidate_id": "c1", "items": [], "task_ids": []},
    ]
    q = {
        ("c0", A0_DISCARD): 0.1,
        ("c0", A1_EPISODIC): 0.2,
        ("c0", A2_SEMANTIC): 0.0,
        ("c0", A3_SKILLOPT): 0.0,
        ("c1", A0_DISCARD): 0.0,
        ("c1", A1_EPISODIC): 0.0,
        ("c1", A2_SEMANTIC): 0.9,
        ("c1", A3_SKILLOPT): 0.0,
    }
    # A poisoned future score must not be consulted.
    future_scores = {"c0": 99.0, "c1": -99.0}
    chosen = select_o3_oracle_joint(cands, q)
    assert chosen["candidate_id"] == "c1"
    assert chosen["action"] == A2_SEMANTIC
    assert chosen["Q"] == 0.9
    assert chosen["Q"] != future_scores[chosen["candidate_id"]]
    assert_q_has_no_future({"uses_future_test": False, "q": {str(k): v for k, v in q.items()}})


def test_evaluate_future_flagged_not_for_selection(tmp_path: Path):
    state = fresh_state(tmp_path)
    tasks = [make_fake_task("db", 0, split="future_test")]
    env, client = make_env_and_client(tasks)
    payload = evaluate_future(tasks, client, state, lab_env=env)
    assert payload["used_for_oracle_selection"] is False
    assert payload["used_for_update"] is False
