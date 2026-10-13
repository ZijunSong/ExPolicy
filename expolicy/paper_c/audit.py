"""C0 content audit and C1 gain matrix. Selection probes and evaluation probes do not overlap."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from expolicy.paper_c.assimilator import FrozenAssimilator, assert_same_algorithm
from expolicy.paper_c.constants import (
    ACCOUNT_INTERACTION,
    CONTENT_MODES,
    MODE_NO_UPDATE,
    STATUS_MISSING,
    STATUS_OK,
)
from expolicy.paper_c.controls import rewrite_experiences
from expolicy.paper_c.ledger import BudgetLedger
from expolicy.paper_c.utility import acquisition_gain
from expolicy.r9_lab.state.clone import clone_state
from expolicy.r9_lab.state.persistent_state import PersistentState

ScoreFn = Callable[[PersistentState, str], float | None]
FeatureFn = Callable[[PersistentState, str], Mapping[str, Any]]


def assert_probe_ids(select_ids: Sequence[str], eval_ids: Sequence[str], future_ids: Sequence[str]) -> None:
    select = set(select_ids)
    evaluation = set(eval_ids)
    future = set(future_ids)
    if select & evaluation:
        raise RuntimeError("selection probe and evaluation probe share task ids")
    if (select | evaluation) & future:
        raise RuntimeError("a training probe shares ids with the future test")


def audit_assimilator(
    start: PersistentState,
    experiences: Sequence[Mapping[str, Any]],
    assimilator: FrozenAssimilator,
    world_score: ScoreFn,
    *,
    work_dir: Path,
    mismatch_of: Callable[[str], str],
    ledger: BudgetLedger,
    retention_weight: float = 0.5,
) -> dict[str, Any]:
    """Same passive sequence, four content arms, one frozen apply()."""
    work_dir.mkdir(parents=True, exist_ok=True)
    expected = assimilator.algorithm_hash()
    arms: dict[str, Any] = {}
    for mode in CONTENT_MODES:
        clone = clone_state(start, work_dir / mode)
        before_shadow = world_score(clone, "shadow_eval")
        before_retention = world_score(clone, "retention")
        payload = rewrite_experiences(experiences, mode, mismatch_of=mismatch_of)
        ledger.add(
            account=ACCOUNT_INTERACTION,
            kind="c0_acquisition",
            interaction_units=float(len(experiences)),
            detail=mode,
        )
        if mode != MODE_NO_UPDATE:
            result = assimilator.apply(clone, payload)
            assert_same_algorithm(expected, result)
            accepted = bool(result["accepted"])
        else:
            accepted = False
            result = {"n": 0, "algorithm_hash": expected}
        after_shadow = world_score(clone, "shadow_eval")
        after_retention = world_score(clone, "retention")
        ledger.add(
            account=ACCOUNT_INTERACTION,
            kind="c0_probe",
            interaction_units=2.0,
            validation_calls=2,
            detail=mode,
        )
        if None in (before_shadow, before_retention, after_shadow, after_retention):
            gain = None
            status = STATUS_MISSING
        else:
            gain = acquisition_gain(
                shadow_before=float(before_shadow),
                shadow_after=float(after_shadow),
                retention_before=float(before_retention),
                retention_after=float(after_retention),
                retention_weight=retention_weight,
            )
            status = STATUS_OK
        arms[mode] = {
            "n_presented": len(payload),
            "accepted": accepted,
            "algorithm_hash": result.get("algorithm_hash"),
            "gain": None if gain is None else gain["g"],
            "delta_shadow": None if gain is None else gain["delta_shadow"],
            "delta_retention": None if gain is None else gain["delta_retention"],
            "raw_shadow_after": None if gain is None else gain["raw_shadow_after"],
            "raw_retention_after": None if gain is None else gain["raw_retention_after"],
            "infra_status": status,
            "state_hash": clone.content_hash(),
        }
    return {"arms": arms, "algorithm_hash": expected}


def build_gain_audit(
    snapshots: Sequence[tuple[str, PersistentState]],
    families: Sequence[str],
    batches: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]],
    assimilator: FrozenAssimilator,
    score: ScoreFn,
    feature_fn: FeatureFn,
    *,
    work_dir: Path,
    ledger: BudgetLedger,
    select_ids: Sequence[str],
    eval_ids: Sequence[str],
    future_ids: Sequence[str],
    retention_weight: float = 0.5,
    source_templates: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Every candidate starts from a clone of the same snapshot and uses the same probes.

    Ranking uses the selection probe. The reported effect uses the disjoint evaluation probe.
    Future ids are checked and never scored here.
    """
    assert_probe_ids(select_ids, eval_ids, future_ids)
    work_dir.mkdir(parents=True, exist_ok=True)
    expected = assimilator.algorithm_hash()
    rows: list[dict[str, Any]] = []
    for snapshot_id, snapshot in snapshots:
        for family in families:
            family_batches = list(batches.get(family) or [])
            for batch_index, batch in enumerate(family_batches):
                dest = work_dir / snapshot_id / family / str(batch_index)
                clone = clone_state(snapshot, dest)
                features = dict(feature_fn(clone, family))
                before = {
                    "select": score(clone, "shadow_select"),
                    "eval": score(clone, "shadow_eval"),
                    "retention": score(clone, "retention"),
                }
                presented = [dict(item) for item in batch]
                for item in presented:
                    ledger.add(
                        account=ACCOUNT_INTERACTION,
                        kind="audit_acquisition",
                        interaction_units=1.0,
                        actor_tokens=int(item.get("actor_tokens") or 0),
                        tool_calls=int(item.get("tool_calls") or 0),
                        detail=f"{snapshot_id}:{family}:{batch_index}",
                    )
                result = assimilator.apply(clone, presented)
                assert_same_algorithm(expected, result)
                after = {
                    "select": score(clone, "shadow_select"),
                    "eval": score(clone, "shadow_eval"),
                    "retention": score(clone, "retention"),
                }
                ledger.add(
                    account=ACCOUNT_INTERACTION,
                    kind="audit_probe",
                    interaction_units=float(len(select_ids) + len(eval_ids) + 2),
                    validation_calls=len(select_ids) + len(eval_ids) + 2,
                    detail=f"{snapshot_id}:{family}",
                )
                complete = all(value is not None for value in (*before.values(), *after.values()))
                if not complete:
                    select_gain = None
                    eval_gain = None
                    status = STATUS_MISSING
                    delta_retention = None
                else:
                    select_pack = acquisition_gain(
                        shadow_before=float(before["select"]),
                        shadow_after=float(after["select"]),
                        retention_before=float(before["retention"]),
                        retention_after=float(after["retention"]),
                        retention_weight=retention_weight,
                    )
                    eval_pack = acquisition_gain(
                        shadow_before=float(before["eval"]),
                        shadow_after=float(after["eval"]),
                        retention_before=float(before["retention"]),
                        retention_after=float(after["retention"]),
                        retention_weight=retention_weight,
                    )
                    select_gain = select_pack["g"]
                    eval_gain = eval_pack["g"]
                    delta_retention = eval_pack["delta_retention"]
                    status = STATUS_OK
                if batch and batch[0].get("source_template"):
                    template = str(batch[0]["source_template"])
                elif source_templates and family in source_templates:
                    template = str(source_templates[family])
                else:
                    template = family
                rows.append(
                    {
                        "sequence_id": snapshot_id,
                        "state_id": snapshot_id,
                        "state_hash": snapshot.content_hash(),
                        "family": family,
                        "batch_index": batch_index,
                        "source_template": template,
                        "env": "synth",
                        "features": features,
                        "gain_select": select_gain,
                        "gain_eval": eval_gain,
                        "gain": eval_gain,
                        "delta_retention": delta_retention,
                        "infra_status": status,
                        "split": "train_shadow",
                    }
                )
    return {"rows": rows, "algorithm_hash": expected, "summary": summarize_gain_rows(rows)}


def summarize_gain_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rank with the selection probe. Quote the evaluation probe, including when they disagree."""
    by_state: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_state.setdefault(str(row["state_id"]), []).append(row)
    states = []
    for state_id, items in sorted(by_state.items()):
        select_means = _family_means(items, "gain_select")
        eval_means = _family_means(items, "gain_eval")
        if not select_means or not eval_means:
            states.append({"state_id": state_id, "status": STATUS_MISSING})
            continue
        ranked = sorted(select_means, key=lambda family: (-float(select_means[family]), family))
        best = ranked[0]
        uniform = sum(eval_means.values()) / len(eval_means)
        states.append(
            {
                "state_id": state_id,
                "status": STATUS_OK,
                "ranked_by_select": ranked,
                "best_family": best,
                "best_eval_gain": eval_means.get(best),
                "uniform_eval_gain": uniform,
                "eval_means": eval_means,
                "select_means": select_means,
            }
        )
    return {"states": states}


def _family_means(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(key)
        if value is None or row.get("infra_status") != STATUS_OK:
            continue
        buckets.setdefault(str(row["family"]), []).append(float(value))
    return {family: sum(values) / len(values) for family, values in buckets.items() if values}

