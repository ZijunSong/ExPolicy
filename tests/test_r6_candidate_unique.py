"""Committed AVA batches must not repeat task ids."""

from __future__ import annotations

import numpy as np

from expolicy.acquisition.ava import ExactAVAAcquisition, balanced_cluster_batches
from expolicy.environments.search.world import Task
from expolicy.state.isolated_clone import EvidenceState


def _tasks(n: int = 64):
    return [
        Task(task_id=f"t{i}", question=f"q{i}", gold_answer="a", capability="bridge_decomposition", stage=1, gold_doc_ids=[], distractor_doc_ids=[], hops=2, split="candidate_pool")
        for i in range(n)
    ]


def test_balanced_batches_have_requested_size():
    tasks = _tasks()
    emb = {t.task_id: np.random.RandomState(0).normal(size=8).astype(np.float32) for t in tasks}
    batches = balanced_cluster_batches(tasks, emb, n_batches=8, batch_size=8, seed=0, round_idx=0)
    assert len(batches) == 8
    assert all(len(b) == 8 for b in batches)


def test_commit_unique_across_rounds():
    tasks = _tasks()
    emb = {t.task_id: np.ones(4, dtype=np.float32) * (i + 1) for i, t in enumerate(tasks)}
    preds = {t.task_id: "bridge_decomposition" for t in tasks}
    ava = ExactAVAAcquisition(seed=0, predicted_cap=preds, embeddings=emb)
    state = EvidenceState(predicted_cap=preds)
    remaining = list(tasks)
    committed = []
    for r in range(4):
        batches = ava.propose_candidates(remaining, round_idx=r)
        scored = ava.score_candidates(state, batches, lambda b, a: 1.0 if a != b else 0.0)
        pick = [tid for tid in ava.select(scored)["task_ids"] if tid not in committed][:8]
        assert len(set(pick)) == len(pick)
        committed.extend(pick)
        state = state.apply_batch(pick, preds)
        remaining = [t for t in remaining if t.task_id not in set(committed)]
    assert len(committed) == len(set(committed))
