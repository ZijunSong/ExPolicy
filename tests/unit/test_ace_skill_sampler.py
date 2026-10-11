"""Unit tests for Ace-Skill Prioritized Sampler and unique-task gating (CPU only)."""

from __future__ import annotations

import math

import numpy as np

from expolicy.acquisition.ace_skill import AceSkillPrioritizedAcquisition
from expolicy.acquisition.base import make_acquisition
from expolicy.core.capability import CapabilityProfile
from expolicy.core.experience import Experience
from expolicy.core.state import PersistentAgentState
from expolicy.environments.search.world import Task
from expolicy.experiments.r5_2 import (
    activation_times_from_acquired,
    acquisition_distribution,
    unique_task_ids,
)


def _task(i: int, cap: str = "multi_hop") -> Task:
    return Task(
        task_id=f"t_mh_test_{i:03d}",
        question=f"q{i}",
        gold_answer="x",
        capability=cap,
        stage=1,
        gold_doc_ids=[],
        distractor_doc_ids=[],
        hops=1,
        split="acquisition",
    )


def test_make_acquisition_alias():
    p = make_acquisition("ace_skill_prioritized", seed=0)
    assert isinstance(p, AceSkillPrioritizedAcquisition)
    assert p.rho == 0.95 and p.gamma == 0.4 and p.epsilon == 0.1


def test_init_matches_official_zeros_not_paper_text():
    tasks = [_task(i) for i in range(5)]
    pol = AceSkillPrioritizedAcquisition(seed=0)
    pol.bind([t.task_id for t in tasks])
    assert pol.I_alpha is not None and np.allclose(pol.I_alpha, 0.0)
    assert np.allclose(pol.I_beta, 0.0)
    v = pol.proficiency(0)
    assert np.allclose(v, 0.5)
    # After first success, official zeros → v=2/3; paper (0.5,0.5) would be 0.625.
    pol.observe(tasks[0].task_id, 1.0, t=0)
    v1 = float(pol.proficiency(0)[0])
    assert abs(v1 - (2.0 / 3.0)) < 1e-9


def test_weights_formula_and_revisit_across_rounds():
    tasks = [_task(i) for i in range(8)]
    dummy = PersistentAgentState()
    profile = CapabilityProfile()
    pol = AceSkillPrioritizedAcquisition(seed=1)
    pol.bind([t.task_id for t in tasks])
    props0 = pol.propose(dummy, profile, tasks, 4, t=0)
    ids0 = props0[0].task_ids
    assert len(ids0) == 4
    assert len(set(ids0)) == 4
    for tid in ids0:
        pol.observe(tid, 0.0, t=0)
    # Across rounds the same ids may reappear (faithful Ace-Skill).
    seen = []
    for t in range(1, 12):
        props = pol.propose(dummy, profile, tasks, 4, t=t)
        seen.extend(props[0].task_ids)
        for tid in props[0].task_ids:
            pol.observe(tid, 0.0, t=t)
    assert len(seen) > len(set(seen)), "revisit across rounds should occur on a tiny pool"


def test_state_roundtrip_deterministic():
    tasks = [_task(i) for i in range(6)]
    dummy = PersistentAgentState()
    profile = CapabilityProfile()
    a = AceSkillPrioritizedAcquisition(seed=7)
    a.bind([t.task_id for t in tasks])
    props = a.propose(dummy, profile, tasks, 3, t=0)
    for tid in props[0].task_ids:
        a.observe(tid, 1.0, t=0)
    snap = a.state_dict()
    b = AceSkillPrioritizedAcquisition(seed=7)
    b.load_state(snap)
    p1 = a.propose(dummy, profile, tasks, 3, t=1)[0].task_ids
    p2 = b.propose(dummy, profile, tasks, 3, t=1)[0].task_ids
    assert p1 == p2


def test_unique_task_gating_ignores_duplicates():
    exps = []
    for i in range(8):
        exps.append(
            Experience(
                task_id="t_mh_test_000",
                capability="multi_hop",
                success=False,
                run_id="acquire",
            )
        )
    assert len(unique_task_ids(exps, "multi_hop")) == 1
    first = activation_times_from_acquired(exps, k=8)
    assert "multi_hop" not in first
    for i in range(8):
        exps.append(
            Experience(
                task_id=f"t_mh_test_{i:03d}",
                capability="multi_hop",
                success=True,
                run_id="acquire",
            )
        )
    first = activation_times_from_acquired(exps, k=8)
    assert first["multi_hop"] == 16
    dist = acquisition_distribution(exps)
    assert dist["duplicate_rollouts"] == 8
    assert dist["unique_tasks"] == 8
    assert dist["unique_tasks_per_capability"]["multi_hop"] == 8


def test_weight_increases_for_unmastered():
    tasks = [_task(i) for i in range(3)]
    pol = AceSkillPrioritizedAcquisition(seed=0, rho=1.0)
    pol.bind([t.task_id for t in tasks])
    pol.observe(tasks[0].task_id, 1.0, t=0)
    pol.observe(tasks[1].task_id, 0.0, t=0)
    w = pol._compute_weights(0)
    # Failure (low v) should have higher hardness term than success.
    assert w[1] > w[0]
