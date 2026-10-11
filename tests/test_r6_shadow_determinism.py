"""Shadow value of a candidate batch is deterministic given the mask cache."""

from __future__ import annotations

from expolicy.experiments.r6_ava import value_from_cache
from expolicy.state.isolated_clone import EvidenceState


def test_shadow_value_deterministic():
    cache = {
        "none": {"macro": 0.01},
        "comparison_alignment": {"macro": 0.11},
        "bridge_decomposition+comparison_alignment": {"macro": 0.12},
    }
    a = value_from_cache(cache, "none", "comparison_alignment")
    b = value_from_cache(cache, "none", "comparison_alignment")
    assert a == b == 0.10


def test_same_batch_same_mask():
    preds = {f"t{i}": "comparison_alignment" for i in range(16)}
    s = EvidenceState(predicted_cap=preds)
    a = s.apply_batch([f"t{i}" for i in range(8)], preds)
    b = s.apply_batch([f"t{i}" for i in range(8)], preds)
    assert a.mask_key() == b.mask_key()
    assert a.counts() == b.counts()
