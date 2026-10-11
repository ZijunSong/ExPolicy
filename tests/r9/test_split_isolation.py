"""Split isolation: task-id / exact-question overlap, model_screen isolation, seed."""

from __future__ import annotations

from pathlib import Path

import pytest

from expolicy.r9_lab.audit.leakage import assert_future_not_in_update, assert_model_screen_isolated, future_ids
from expolicy.r9_lab.benchmark.split import (
    build_and_write,
    build_synthetic_catalog,
    construct_split,
)
from expolicy.r9_lab.constants import BUCKETS, ENVS, FORMAL_BUCKETS


def _ids(payload, env, bucket):
    return {r["task_id"] for r in payload["envs"][env][bucket]}


def _qs(payload, env, bucket):
    return {r["instruction"].strip() for r in payload["envs"][env][bucket]}


def test_synthetic_split_task_id_and_question_overlap_is_zero():
    payload = construct_split(build_synthetic_catalog(n_per_env=600), synthetic=True)
    for env in ENVS:
        buckets = list(BUCKETS)
        for i, a in enumerate(buckets):
            for b in buckets[i + 1 :]:
                assert not (_ids(payload, env, a) & _ids(payload, env, b)), (env, a, b)
                assert not (_qs(payload, env, a) & _qs(payload, env, b)), (env, a, b)


def test_model_screen_vs_formal_task_id_overlap_is_zero():
    payload = construct_split(build_synthetic_catalog(n_per_env=600), synthetic=True)
    assert_model_screen_isolated(payload)
    screen = set()
    formal = set()
    for env in ENVS:
        screen |= _ids(payload, env, "model_screen")
        for b in FORMAL_BUCKETS:
            formal |= _ids(payload, env, b)
    assert not (screen & formal)


def test_model_screen_vs_formal_instruction_overlap_is_zero():
    payload = construct_split(build_synthetic_catalog(n_per_env=600), synthetic=True)
    screen_q = set()
    formal_q = set()
    for env in ENVS:
        screen_q |= _qs(payload, env, "model_screen")
        for b in FORMAL_BUCKETS:
            formal_q |= _qs(payload, env, b)
    assert not (screen_q & formal_q)


def test_future_test_never_in_update_path():
    payload = construct_split(build_synthetic_catalog(n_per_env=600), synthetic=True)
    fut = future_ids(payload)
    acq = set()
    for env in ENVS:
        acq |= _ids(payload, env, "acquisition_pool")
        acq |= _ids(payload, env, "calibration")
        acq |= _ids(payload, env, "assim_validation")
    assert not (fut & acq)
    assert_future_not_in_update(acq, payload)
    with pytest.raises(AssertionError):
        assert_future_not_in_update(list(fut)[:1], payload)


def test_random_seed_reproducible():
    catalog = build_synthetic_catalog(n_per_env=600)
    a = construct_split(catalog, seed=20260822, synthetic=True)
    b = construct_split(catalog, seed=20260822, synthetic=True)
    assert a["envs"] == b["envs"]
    c = construct_split(catalog, seed=20260823, synthetic=True)
    assert a["envs"] != c["envs"]


def test_production_fails_without_official_data(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "expolicy.r9_lab.benchmark.split.official_data_path",
        lambda env: tmp_path / f"missing_{env}.json",
    )
    with pytest.raises(FileNotFoundError):
        build_and_write(out_path=tmp_path / "split.json", synthetic=False)
