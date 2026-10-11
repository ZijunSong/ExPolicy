"""Leakage audits: future_test isolation, gold-in-prompt, eval write guards."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from expolicy.r9_lab.benchmark.split import all_bucket_ids
from expolicy.r9_lab.benchmark.verifier import prompt_contains_gold
from expolicy.r9_lab.constants import FORMAL_BUCKETS


def future_ids(split: Mapping[str, Any]) -> set[str]:
    return all_bucket_ids(split, "future_test")


def model_screen_ids(split: Mapping[str, Any]) -> set[str]:
    return all_bucket_ids(split, "model_screen")


def formal_ids(split: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    for bucket in FORMAL_BUCKETS:
        ids |= all_bucket_ids(split, bucket)
    return ids


def assert_model_screen_isolated(split: Mapping[str, Any]) -> None:
    overlap = model_screen_ids(split) & formal_ids(split)
    if overlap:
        raise AssertionError(f"model_screen overlaps formal split: {overlap}")


def assert_future_not_in_update(task_ids: Iterable[str], split: Mapping[str, Any]) -> None:
    forbidden = future_ids(split)
    leaked = set(task_ids) & forbidden
    if leaked:
        raise AssertionError(f"future_test ids entered update path: {leaked}")


def assert_no_gold_in_messages(messages: Sequence[Mapping[str, Any]], gold: Any) -> None:
    hits = prompt_contains_gold(messages, gold)
    if hits:
        raise AssertionError(f"gold leaked into prompt: {hits[:3]}")
