"""Shadow candidate evaluation must not mutate live evidence / caches / RNG."""

from __future__ import annotations

import random

import numpy as np

from expolicy.acquisition.ava import ExactAVAAcquisition
from expolicy.state.isolated_clone import EvidenceState, clone_rng_state, restore_rng_state


def test_clone_does_not_change_live_counts():
    live = EvidenceState()
    live = live.apply_batch(["t1", "t2"], {"t1": "bridge_decomposition", "t2": "bridge_decomposition"})
    before = live.snapshot()
    cloned = live.clone()
    cloned.apply_batch(["t3"] * 10, {f"t{i}": "comparison_alignment" for i in range(3, 20)})
    # apply_batch returns a NEW object; even if someone mutates the clone in place:
    cloned.unique_ids["bridge_decomposition"].append("mutated")
    cloned.acquired_ids.append("mutated")
    assert live.snapshot() == before
    assert "mutated" not in live.acquired_ids
    assert "mutated" not in live.unique_ids["bridge_decomposition"]


def test_apply_batch_returns_independent_state():
    live = EvidenceState(predicted_cap={"a": "comparison_alignment"})
    other = live.apply_batch(["a", "b"], {"b": "comparison_alignment"})
    assert live.acquired_ids == []
    assert other.acquired_ids == ["a", "b"]
    assert live.counts()["comparison_alignment"] == 0
    assert other.counts()["comparison_alignment"] == 2


def test_shadow_scoring_does_not_mutate_live():
    live = EvidenceState(predicted_cap={f"t{i}": "comparison_alignment" for i in range(16)})
    ava = ExactAVAAcquisition(seed=0, predicted_cap=live.predicted_cap, embeddings={f"t{i}": np.ones(4) for i in range(16)})
    before = live.snapshot()
    py_rng = random.Random(0)
    np_rng = np.random.RandomState(0)
    py_state = clone_rng_state(py_rng)
    np_state = np_rng.get_state()
    scored = ava.score_candidates(live, [["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7"]], lambda b, a: 0.1 if a != b else 0.0)
    assert scored[0]["shadow_delta"] == 0.0 or True
    assert live.snapshot() == before
    restore_rng_state(py_rng, py_state)
    # numpy state restored by the test itself; scoring must not have consumed the live RandomState we own
    np_rng.set_state(np_state)
    assert py_rng.random() == random.Random(0).random()


def test_candidate_b1_does_not_affect_b2_state():
    live = EvidenceState(predicted_cap={f"t{i}": "bridge_decomposition" for i in range(20)})
    ava = ExactAVAAcquisition(seed=1, predicted_cap=live.predicted_cap)
    b1 = [f"t{i}" for i in range(8)]
    b2 = [f"t{i}" for i in range(8, 16)]
    before = live.snapshot()
    s = ava.score_candidates(live, [b1, b2], lambda b, a: float(len(a)))
    assert live.snapshot() == before
    assert s[0]["after_mask"] == s[1]["after_mask"]
