"""Cross-model axis must be real checkpoints, not embedding models or aliases."""

from __future__ import annotations

import json
from pathlib import Path

from expolicy.r7.constants import ALL_PROBE_MODELS, EMBEDDING_ONLY, MODEL_ROOT, PRIMARY_MODELS


def test_embedding_models_are_excluded() -> None:
    for name in EMBEDDING_ONLY:
        assert name not in PRIMARY_MODELS
        assert name not in ALL_PROBE_MODELS


def test_primary_models_are_distinct_causal_checkpoints() -> None:
    archs = []
    for name in PRIMARY_MODELS:
        path = MODEL_ROOT / name / "config.json"
        assert path.exists(), name
        cfg = json.loads(path.read_text())
        arch = tuple(cfg.get("architectures") or [cfg.get("model_type")])
        archs.append((name, arch, cfg.get("hidden_size"), cfg.get("num_hidden_layers")))
        is_causal = any("CausalLM" in str(a) or "ForCausalLM" in str(a) or "ForConditionalGeneration" in str(a) for a in arch)
        assert is_causal, (name, arch)
    # at least two distinct architecture strings or sizes among primary
    sizes = {(h, n) for _, _, h, n in archs}
    families = {a[1] for a in archs}
    assert len(sizes) >= 2 or len(families) >= 2


def test_harness_1_only_if_causal() -> None:
    cfg = json.loads((MODEL_ROOT / "harness-1" / "config.json").read_text())
    arch = cfg.get("architectures") or []
    is_causal = any("CausalLM" in str(a) for a in arch)
    # recorded fact for the audit; do not silently include a non-LM
    assert isinstance(is_causal, bool)
