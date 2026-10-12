"""A-v1.1 pilot helpers. Historical A-v1 reports are inputs, not something this module rewrites."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from expolicy.paper_a.audit import stratified_paired_summary
from expolicy.paper_a.features import template_key
from expolicy.paper_a.stats import interval_claim, paired_flip_interval
from expolicy.r9_lab.benchmark.split import records_in_bucket
from expolicy.r9_lab.io_util import read_json, sha256_json

PROTOCOL_ID = "A-v1.1-pilot"
DIAGNOSTIC_TASKS = ("db:367", "db:414")


class PilotRefusal(RuntimeError):
    """Raised before any model or environment connection."""


def load_v11(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("protocol_id") != PROTOCOL_ID:
        raise PilotRefusal(f"expected {PROTOCOL_ID}")
    if raw.get("pilot", {}).get("train_router") or raw.get("pilot", {}).get("run_sequence"):
        raise PilotRefusal("A-v1.1 pilot does not train a router or run the sequence")
    if raw.get("pilot", {}).get("access_future_test"):
        raise PilotRefusal("future_test is closed")
    if "os" in (raw.get("environments") or []):
        raise PilotRefusal("OS stays out of this pilot")
    for value in (
        raw.get("source_commit"),
        raw.get("split", {}).get("expect_content_sha256"),
        raw.get("frozen_batches", {}).get("expect_sha256"),
    ):
        if not value or str(value).startswith("PLACEHOLDER"):
            raise PilotRefusal("asset hash is missing")
    return raw


def cache_usable(stored: Mapping[str, Any] | None, cache_key: str) -> bool:
    if not stored:
        return False
    if stored.get("cache_key") != cache_key:
        return False
    if stored.get("infra_status") != "ok" or stored.get("y") is None:
        return False
    if not stored.get("complete"):
        return False
    return True


def classify_divergence(traces: Sequence[Mapping[str, Any]]) -> str:
    """Classify the first difference across repeats. Score-only differences are not model noise."""
    usable = [trace for trace in traces if trace.get("steps")]
    if len(usable) < 2:
        return "VARIATION_UNLOCATED"
    first, second = usable[0], usable[1]
    if first.get("reset_hash") != second.get("reset_hash"):
        return "RESET_OR_INPUT_VARIATION"
    steps_a = list(first.get("steps") or [])
    steps_b = list(second.get("steps") or [])
    for left, right in zip(steps_a, steps_b):
        if left.get("request_hash") != right.get("request_hash"):
            return "RESET_OR_INPUT_VARIATION"
        if left.get("response_hash") != right.get("response_hash"):
            return "MODEL_OUTPUT_VARIATION"
        if left.get("tool_hash") != right.get("tool_hash"):
            return "TOOL_VARIATION"
    if first.get("y") != second.get("y"):
        return "VERIFIER_VARIATION"
    return "NO_VARIATION"


def _components(batches: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    parent = {str(batch["batch_id"]): str(batch["batch_id"]) for batch in batches}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        parent[find(left)] = find(right)

    owners: dict[str, str] = {}
    for batch in batches:
        batch_id = str(batch["batch_id"])
        items = list(batch.get("items") or [])
        keys = [template_key(item) for item in items] or [f"id:{batch_id}"]
        for key in keys:
            if key in owners:
                union(batch_id, owners[key])
            else:
                owners[key] = batch_id
        for task_id in batch.get("task_ids") or []:
            marker = f"task:{task_id}"
            if marker in owners:
                union(batch_id, owners[marker])
            else:
                owners[marker] = batch_id
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in batches:
        grouped[find(str(batch["batch_id"]))].append(dict(batch))
    return list(grouped.values())


def choose_pilot_batches(
    frozen: Mapping[str, Any],
    env: str,
    *,
    n_discovery: int,
    n_reserved: int,
    seed: int,
) -> dict[str, Any]:
    """Whole source components stay in one stage. Utility scores are not consulted."""
    batches = [batch for batch in frozen.get("batches") or [] if batch.get("env") == env]
    components = _components(batches)
    rng = random.Random(seed + sum(ord(ch) for ch in env))
    order = list(range(len(components)))
    rng.shuffle(order)
    discovery: list[dict[str, Any]] = []
    reserved: list[dict[str, Any]] = []
    for index in order:
        members = components[index]
        if len(discovery) + len(members) <= n_discovery:
            discovery.extend(members)
        elif len(reserved) + len(members) <= n_reserved:
            reserved.extend(members)
    return {
        "env": env,
        "discovery": discovery,
        "reserved": reserved,
        "n_components_seen": len(components),
        "shortfall": (n_discovery + n_reserved) - (len(discovery) + len(reserved)),
    }


def choose_smoke_tasks(
    split: Mapping[str, Any],
    env: str,
    *,
    n_shadow: int,
    n_retention: int,
    seed: int,
    exclude: set[str],
) -> dict[str, list[dict[str, Any]]]:
    picked: dict[str, list[dict[str, Any]]] = {}
    for bucket, count in (("shadow_reward_probe", n_shadow), ("retention_probe", n_retention)):
        records = [rec for rec in records_in_bucket(split, env, bucket) if rec.task_id not in exclude]
        groups: dict[str, list[Any]] = defaultdict(list)
        for rec in records:
            groups[template_key(rec.to_dict())].append(rec)
        keys = sorted(groups)
        rng = random.Random(seed + sum(ord(ch) for ch in f"{env}:{bucket}"))
        rng.shuffle(keys)
        chosen = []
        while len(chosen) < count and any(groups[key] for key in keys):
            for key in keys:
                if groups[key] and len(chosen) < count:
                    chosen.append(groups[key].pop(0))
        picked[bucket] = [rec.to_dict() for rec in chosen]
    return picked


def split_probe_tasks(
    records: Sequence[Mapping[str, Any]],
    n_selection: int,
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[template_key(rec)].append(rec)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    selection: list[Mapping[str, Any]] = []
    confirmation: list[Mapping[str, Any]] = []
    for key in keys:
        members = groups[key]
        if len(selection) + len(members) <= n_selection:
            selection.extend(members)
        else:
            confirmation.extend(members)
    return {"selection": list(selection), "confirmation": list(confirmation)}


def paired_discard_utility(discard_y: Sequence[float], other_y: Sequence[float]) -> dict[str, float]:
    if len(discard_y) != len(other_y) or not discard_y:
        raise ValueError("paired outcomes must be aligned and non-empty")
    deltas = [float(other) - float(base) for base, other in zip(discard_y, other_y)]
    return {"delta": sum(deltas) / len(deltas), "n": len(deltas), "estimator": "paired_discard_control_v1"}


def selection_signal(batch_winners: Sequence[Mapping[str, Any]]) -> str:
    """batch_winners items: {source_group, winner, n_task_flips, operator_invalid}."""
    if any(row.get("operator_invalid") for row in batch_winners):
        return "WEAK_SIGNAL"
    real = [row for row in batch_winners if int(row.get("n_task_flips") or 0) > 1 and not row.get("tie")]
    groups = {row.get("source_group") for row in real}
    winners = {row.get("winner") for row in real}
    if len(groups) >= 2 and len(winners) >= 2:
        return "SIGNAL"
    return "WEAK_SIGNAL"


def assert_v11_launch(eligibility: Mapping[str, Any] | None, *, expected_hash: str, episodes: int, budget: int) -> None:
    if not eligibility:
        raise PilotRefusal("ELIGIBILITY.json is missing")
    if eligibility.get("protocol_hash") != expected_hash:
        raise PilotRefusal("eligibility hash does not match the locked protocol")
    if eligibility.get("structural_error"):
        raise PilotRefusal(str(eligibility["structural_error"]))
    allowed = set(eligibility.get("environments") or [])
    requested = set(eligibility.get("requested_environments") or allowed)
    if not requested <= allowed:
        raise PilotRefusal(f"environment not eligible: {sorted(requested - allowed)}")
    if int(eligibility.get("episodes_used") or 0) + episodes > budget:
        raise PilotRefusal("episode budget would be exceeded")


def legacy_reinterpretation(audit_root: Path) -> str:
    lines = [
        "# A-v1 测量结果的离线重读",
        "",
        "原 `A_MEASUREMENT_AUDIT.md` 与 `A_DECISION.md` 不改。这里只按分支、题型和重复重读已有 `report.json`。",
        "旧逐题记录没有逐步请求/响应哈希，因此不能离线定位首个分叉。",
        "",
    ]
    if not audit_root.exists():
        lines.append("没有找到审计报告。")
        return "\n".join(lines)
    for path in sorted(audit_root.rglob("report.json")):
        report = read_json(path)
        lines.append(f"## {path.parent.name}")
        lines.append("")
        lines.append(f"原文件状态：`{report.get('status')}`。这不是重读后的全局通过。")
        eq = (report.get("penalty") or {}).get("equivalence") or {}
        claim = interval_claim([eq.get("mean_paired_delta") or 0.0])
        lines.append(
            f"串行描述性均值 {eq.get('mean_paired_delta')}，区间 {eq.get('ci')}。"
            f" 估计器是描述性正态区间，population_noise_claim={claim['population_noise_claim']}。"
        )
        formal = eq.get("formal") or {}
        if formal:
            lines.append(f"后一段复放均值 {formal.get('mean_paired_delta')}，区间 {formal.get('ci')}。并发度配置为 1，不能把它写成新的正式并发证据。")
        stratified = stratified_paired_summary(report.get("task_rows") or [])
        interesting = [row for row in stratified if row["n_pos"] or row["n_neg"]]
        if not interesting:
            lines.append("分项上没有正翻转或负翻转。")
        for row in interesting:
            interval = paired_flip_interval(row["n_pos"], row["n_neg"], row["n"])
            lines.append(
                f"- {row['mode']} repeat {row['repeat_id']} {row['branch']} {row['eval_split']}: "
                f"正 {row['n_pos']}，负 {row['n_neg']}，n={row['n']}，均值 {row['mean']:.4f}，"
                f"保守区间 {interval.get('ci')}"
            )
        lines.append("")
    lines.append("已知翻转题 `db:367`、`db:414` 只出现在 `db-B00` 的串行第 0 次，且两个 no-op 分支同向。没有逐步轨迹，原因标为 `VARIATION_UNLOCATED`。")
    lines.append("")
    return "\n".join(lines)


def code_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def protocol_hash(config: Mapping[str, Any]) -> str:
    return sha256_json(config)
