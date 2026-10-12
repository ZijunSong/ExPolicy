"""Experience controls. Shuffling a bag is not a control; the repair link has to break."""

from __future__ import annotations

from expolicy.paper_b.constants import ALIGNMENT_MISMATCHED, TEMPLATE_REWRITE
from expolicy.paper_b.models import Trace, TracePair


def empty_experiences(traces: list[Trace]) -> list[dict]:
    """Same identifiers and count, with actions, observations, and outcomes removed."""
    rows = []
    for trace in traces:
        rows.append(
            {
                "experience_id": trace.trace_id,
                "steps": [],
                "success": None,
                "features_before": {},
                "action_template": "",
                "observation_redacted": "",
            }
        )
    return rows


def mismatch_pairs(pairs: list[TracePair]) -> list[TracePair]:
    """Bind a state to the repair of the other capability family."""
    rewrites = [pair for pair in pairs if pair.success_action == TEMPLATE_REWRITE and not pair.swapped_from]
    searches = [
        pair
        for pair in pairs
        if pair.success_action != TEMPLATE_REWRITE
        and pair.features_before.get("has_comparison_cue") == 1
        and not pair.swapped_from
    ]
    swapped: list[TracePair] = []
    for index, (left, right) in enumerate(zip(searches, rewrites, strict=False)):
        swapped.append(_swap(left, right, index))
        swapped.append(_swap(right, left, index))
    return swapped


def _swap(target: TracePair, donor: TracePair, index: int) -> TracePair:
    return TracePair(
        pair_id=f"mismatch-{index}-{target.pair_id}",
        alignment=ALIGNMENT_MISMATCHED,
        quality=target.quality,
        task_id=target.task_id,
        source_group=target.source_group,
        family=target.family,
        diverge_index=target.diverge_index,
        features_before=dict(target.features_before),
        success_action=donor.success_action,
        success_params=dict(donor.success_params),
        failure_action=target.failure_action,
        success_trace_id=target.success_trace_id,
        failure_trace_id=target.failure_trace_id,
        observation_redacted=target.observation_redacted,
        swapped_from=target.success_action,
    )


def unrelated_experiences(traces: list[Trace], *, forbidden_task_ids: set[str]) -> list[dict]:
    rows = []
    for trace in traces:
        if trace.task_id in forbidden_task_ids:
            continue
        rows.append(
            {
                "experience_id": trace.trace_id,
                "task_id": trace.task_id,
                "source_group": trace.source_group,
                "success": trace.success,
                "steps": [step.to_dict() for step in trace.steps],
            }
        )
    return rows


def make_experience_controls(
    pairs: list[TracePair],
    real_traces: list[Trace],
    unrelated_traces: list[Trace],
) -> dict:
    real_ids = {trace.task_id for trace in real_traces}
    empty = empty_experiences(real_traces)
    mismatched = mismatch_pairs(pairs)
    unrelated = unrelated_experiences(unrelated_traces, forbidden_task_ids=real_ids)
    broken = [pair for pair in mismatched if pair.swapped_from and pair.swapped_from != pair.success_action]
    schema_ok = all(_pair_schema_ok(pair) for pair in mismatched) if mismatched else False
    unrelated_ids = {row["task_id"] for row in unrelated}
    audit = {
        "n_real_traces": len(real_traces),
        "n_real_pairs": len(pairs),
        "n_empty": len(empty),
        "empty_has_actions": any(row["action_template"] or row["steps"] for row in empty),
        "n_mismatched": len(mismatched),
        "n_association_broken": len(broken),
        "mismatch_schema_ok": schema_ok,
        "association_broken": bool(mismatched) and len(broken) == len(mismatched) and schema_ok,
        "n_unrelated": len(unrelated),
        "unrelated_disjoint": unrelated_ids.isdisjoint(real_ids),
        "unrelated_real": bool(unrelated) and unrelated_ids.isdisjoint(real_ids),
        "reorder_only_is_not_a_control": True,
    }
    return {
        "real_pairs": pairs,
        "empty_experiences": empty,
        "mismatched_pairs": mismatched,
        "unrelated_experiences": unrelated,
        "audit": audit,
    }


def _pair_schema_ok(pair: TracePair) -> bool:
    view = pair.inducer_view()
    required = {"features_before", "success_action", "success_params", "alignment"}
    return required.issubset(view) and isinstance(view["features_before"], dict) and bool(view["success_action"])
