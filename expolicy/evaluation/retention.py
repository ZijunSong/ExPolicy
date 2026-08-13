"""Retention: re-evaluate previous-stage retention splits after an update."""

from __future__ import annotations

from expolicy.evaluation.metrics import aggregate


def retention_delta(before: dict, after: dict) -> dict:
    return {
        "before_accuracy": float(before.get("accuracy") or 0.0),
        "after_accuracy": float(after.get("accuracy") or 0.0),
        "delta": float(after.get("accuracy") or 0.0) - float(before.get("accuracy") or 0.0),
        "n_before": int(before.get("n") or 0),
        "n_after": int(after.get("n") or 0),
    }


def summarize_retention(rows_before: list[dict], rows_after: list[dict]) -> dict:
    return retention_delta(aggregate(rows_before), aggregate(rows_after))
