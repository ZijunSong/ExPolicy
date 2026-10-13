"""Build same-prefix and matched-observation pairs. Pairing never reads answers."""

from __future__ import annotations

from expolicy.paper_b.constants import (
    ALIGNMENT_MATCHED,
    ALIGNMENT_SAME,
    TEMPLATE_BASE_ANSWER,
    TEMPLATE_BASE_SEARCH,
)
from expolicy.paper_b.models import Trace, TracePair
from expolicy.r9_lab.io_util import sha256_json


def _feature_hash(features: dict) -> str:
    return sha256_json(features)


def build_trace_pairs(traces: list[Trace]) -> list[TracePair]:
    by_task: dict[str, list[Trace]] = {}
    for trace in traces:
        by_task.setdefault(trace.task_id, []).append(trace)
    pairs: list[TracePair] = []
    used_success: set[str] = set()
    for _task_id, group in sorted(by_task.items()):
        successes = [trace for trace in group if trace.success]
        failures = [trace for trace in group if not trace.success]
        for success in successes:
            for failure in failures:
                emitted = _same_task_pairs(success, failure)
                if emitted:
                    used_success.add(success.trace_id)
                    pairs.extend(emitted)
    pairs.extend(_matched_observation_pairs(traces, used_success))
    pairs.sort(key=lambda pair: pair.pair_id)
    return pairs


def _same_task_pairs(success: Trace, failure: Trace) -> list[TracePair]:
    limit = min(len(success.steps), len(failure.steps))
    earlier_differs = False
    emitted: list[TracePair] = []
    for index in range(limit):
        left = success.steps[index]
        right = failure.steps[index]
        if left.features_before != right.features_before:
            break
        histories_differ = bool(left.history_hash and right.history_hash and left.history_hash != right.history_hash)
        if histories_differ:
            if left.action_template not in {TEMPLATE_BASE_SEARCH, TEMPLATE_BASE_ANSWER} and (
                left.action_template != right.action_template or left.parameters != right.parameters
            ):
                emitted.append(
                    TracePair(
                        pair_id=f"feat:{success.trace_id}:{failure.trace_id}:{index}",
                        alignment=ALIGNMENT_MATCHED,
                        quality=0.5,
                        task_id=success.task_id,
                        source_group=success.source_group,
                        family=success.family,
                        diverge_index=index,
                        features_before=dict(left.features_before),
                        success_action=left.action_template,
                        success_params=dict(left.parameters),
                        failure_action=right.action_template,
                        success_trace_id=success.trace_id,
                        failure_trace_id=failure.trace_id,
                        observation_redacted=left.observation_redacted,
                        prefix_hash=left.history_hash,
                        alignment_basis="features",
                    )
                )
            break
        if left.action_template == right.action_template and left.parameters == right.parameters:
            continue
        if left.action_template in {TEMPLATE_BASE_SEARCH, TEMPLATE_BASE_ANSWER}:
            earlier_differs = True
            continue
        quality = 0.75 if earlier_differs else 1.0
        emitted.append(
            TracePair(
                pair_id=f"{success.trace_id}:{failure.trace_id}:{index}",
                alignment=ALIGNMENT_SAME,
                quality=quality,
                task_id=success.task_id,
                source_group=success.source_group,
                family=success.family,
                diverge_index=index,
                features_before=dict(left.features_before),
                success_action=left.action_template,
                success_params=dict(left.parameters),
                failure_action=right.action_template,
                success_trace_id=success.trace_id,
                failure_trace_id=failure.trace_id,
                observation_redacted=left.observation_redacted,
                prefix_hash=left.history_hash or _feature_hash(left.features_before),
                alignment_basis="history" if left.history_hash else "features",
            )
        )
        earlier_differs = True
    return emitted


def _matched_observation_pairs(traces: list[Trace], used_success: set[str]) -> list[TracePair]:
    index: dict[str, list[tuple[Trace, int]]] = {}
    for trace in traces:
        for step_index, step in enumerate(trace.steps):
            if step.action_template in {TEMPLATE_BASE_SEARCH, TEMPLATE_BASE_ANSWER}:
                continue
            index.setdefault(_feature_hash(step.features_before), []).append((trace, step_index))
    pairs: list[TracePair] = []
    for bucket in index.values():
        for success, s_index in bucket:
            if not success.success or success.trace_id in used_success:
                continue
            step = success.steps[s_index]
            for other, o_index in bucket:
                if other.task_id == success.task_id or other.success:
                    continue
                other_step = other.steps[o_index]
                if other_step.action_template == step.action_template:
                    continue
                if success.source_group != other.source_group:
                    continue
                pairs.append(
                    TracePair(
                        pair_id=f"matched:{success.trace_id}:{other.trace_id}:{s_index}",
                        alignment=ALIGNMENT_MATCHED,
                        quality=0.5,
                        task_id=success.task_id,
                        source_group=success.source_group,
                        family=success.family,
                        diverge_index=s_index,
                        features_before=dict(step.features_before),
                        success_action=step.action_template,
                        success_params=dict(step.parameters),
                        failure_action=other_step.action_template,
                        success_trace_id=success.trace_id,
                        failure_trace_id=other.trace_id,
                        observation_redacted=step.observation_redacted,
                    )
                )
                break
    return pairs
