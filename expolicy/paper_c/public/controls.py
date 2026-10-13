"""Content controls. Source verification is kept separate from the injected payload."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

ARMS = ("no-update", "real", "empty-slots", "matched-wrong", "query-only")
_BLOCKED = {"future_test", "retention_confirm", "future_confirm", "shadow_eval", "shadow_select"}


def _blocked(item: Mapping[str, Any]) -> bool:
    bucket = str(item.get("bucket") or item.get("role") or item.get("split") or "")
    task_id = str(item.get("task_id") or "")
    return bucket in _BLOCKED or task_id.startswith("future_test:")


def verified_successes(experiences: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Screen successes once, before any arm rewrites the payload."""
    kept: list[dict[str, Any]] = []
    for item in experiences:
        if _blocked(item) or item.get("infra_failure"):
            continue
        outcome = str(item.get("verifier_outcome") or "").upper()
        if bool(item.get("success")) and outcome in {"CORRECT", "SUCCESS", "TRUE", "1"}:
            copied = dict(item)
            copied["source_verified"] = True
            kept.append(copied)
    return kept


def _donor_for(item: Mapping[str, Any], donors: Sequence[Mapping[str, Any]], index: int) -> Mapping[str, Any] | None:
    family = str(item.get("family") or "")
    others = [row for row in donors if str(row.get("family") or "") != family and row.get("task_id") != item.get("task_id")]
    if not others:
        others = [row for row in donors if row.get("task_id") != item.get("task_id")]
    if not others:
        return None
    return others[index % len(others)]


def make_content_controls(
    raw_experiences: Sequence[Mapping[str, Any]],
    *,
    donors: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build five arms from one screened success list. Counts must match across writing arms."""
    successes = verified_successes(raw_experiences)
    pool = list(donors) if donors is not None else successes
    real = [dict(item) | {"control_mode": "real", "payload_control": "original"} for item in successes]
    empty = []
    wrong = []
    query = []
    invalid: list[str] = []
    for index, item in enumerate(successes):
        placeholder = dict(item)
        placeholder.update(
            {
                "control_mode": "empty-slots",
                "payload_control": "neutral-placeholder",
                "instruction": "neutral placeholder record",
                "trajectory": [{"agent": "Action: placeholder", "action": "Action: placeholder"}],
                "source_verified": True,
            }
        )
        empty.append(placeholder)
        donor = _donor_for(item, pool, index)
        if donor is None:
            invalid.append(str(item.get("task_id")))
            continue
        mismatched = dict(donor)
        mismatched.update(
            {
                "task_id": item.get("task_id"),
                "control_mode": "matched-wrong",
                "payload_control": "donor-trajectory",
                "source_verified": True,
                "donor_task_id": donor.get("task_id"),
                "family": donor.get("family"),
            }
        )
        actions = []
        for turn in donor.get("trajectory") or []:
            agent = turn.get("agent") or turn.get("action")
            if agent:
                actions.append({"agent": agent, "action": agent})
        if not actions:
            invalid.append(str(item.get("task_id")))
            continue
        mismatched["trajectory"] = actions
        wrong.append(mismatched)
        query_item = dict(item)
        query_item.update(
            {
                "control_mode": "query-only",
                "payload_control": "instruction-without-actions",
                "source_verified": True,
                "trajectory": [{"agent": "", "action": ""}],
            }
        )
        query.append(query_item)
    return {
        "n_raw": len(raw_experiences),
        "n_verified": len(successes),
        "arms": {
            "no-update": [],
            "real": real,
            "empty-slots": empty,
            "matched-wrong": wrong,
            "query-only": query,
        },
        "invalid_donors": invalid,
        "counts_match": len(real) == len(empty) == len(wrong) == len(query) and not invalid,
    }
