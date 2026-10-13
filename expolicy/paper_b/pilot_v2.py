"""Collect 96 training tasks and score the seven-group development pilot."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from expolicy.core.errors import BackendUnavailableError
from expolicy.models.openai_compat import OpenAICompatBackend
from expolicy.paper_b.decision_v2 import judge_v2, render_v2_decision
from expolicy.paper_b.features import trigger_matches
from expolicy.paper_b.induce_v2 import induce_v2
from expolicy.paper_b.models import Rule, Trace, TraceStep
from expolicy.paper_b.pipeline_v2 import deploy_rules, frozen_b1, write_public_manifest
from expolicy.paper_b.public_exec import (
    PublicBudget,
    PublicEpisode,
    PublicTask,
    observable_features,
    restore,
    run_public_episode,
)
from expolicy.paper_b.stats import paired_comparison
from expolicy.paper_b.traces import build_trace_pairs
from expolicy.r9_lab.io_util import sha256_json, write_json

CONDITIONS = ("B0", "B1", "B2", "B3", "B4", "B5", "M")
_WRITE_LOCK = threading.Lock()


def fork_task_ids(records: list[dict[str, Any]]) -> set[str]:
    outcomes: dict[str, set[bool]] = defaultdict(set)
    for row in records:
        if row.get("stage") == "collect" and row.get("infra_status") == "ok":
            outcomes[str(row["task_id"])].add(bool(row["success"]))
    return {task_id for task_id, values in outcomes.items() if values == {True, False}}


def supplement_task_ids(records: list[dict[str, Any]], tasks: list[PublicTask], cap: int) -> list[str]:
    forked = fork_task_ids(records)
    seen = {str(row["task_id"]) for row in records if row.get("stage") == "collect"}
    ordered = [task.task_id for task in tasks if task.task_id in seen and task.task_id not in forked]
    return ordered[:cap]


def run_development_pilot(raw: dict[str, Any], manifest: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_public_manifest(out_dir / "B_V2_DATA_MANIFEST.json", manifest)
    budget = PublicBudget(**raw["budget"])
    urls = list(raw["model"]["base_urls"])
    client = _retriever(raw)
    train = _tasks(manifest, "train")
    val = _tasks(manifest, "val")
    dev = _tasks(manifest, "dev")
    ledger_path = out_dir / "B_V2_COST_LEDGER.jsonl"
    collect_path = out_dir / "collect.jsonl"
    collected = _read_jsonl(collect_path)
    _collect(raw, train, client, urls, budget, collect_path, ledger_path, collected, supplement=False)
    collected = _read_jsonl(collect_path)
    forks = fork_task_ids(collected)
    supplemented = any(row.get("supplement") for row in collected)
    if len(forks) < int(raw["collection"]["min_fork_tasks"]) and not supplemented:
        extra_ids = supplement_task_ids(collected, train, int(raw["collection"]["supplement_episodes"]))
        extra = [task for task in train if task.task_id in set(extra_ids)]
        _collect(raw, extra, client, urls, budget, collect_path, ledger_path, collected, supplement=True)
        collected = _read_jsonl(collect_path)
        forks = fork_task_ids(collected)
    traces = _traces_from_records(collected, train)
    pairs = build_trace_pairs(traces)
    pool_id = sha256_json([row["task_id"] for row in collected if row.get("stage") == "collect"])[:16]
    text = _text_experience(traces)
    baseline = frozen_b1(raw)
    induced = _induce_all(raw, traces, pairs, urls[0], len(collected))
    accepted, val_cost = _accept_all(raw, induced, val, baseline, client, urls, budget, out_dir, ledger_path, collected, train)
    deployed = {
        name: deploy_rules(name, baseline, accepted[name]["rules"], rejected=not accepted[name]["accepted"])
        for name in ("B3", "B4", "B5", "M")
    }
    deployed["B0"] = []
    deployed["B1"] = baseline
    deployed["B2"] = []
    rows = _evaluate(raw, dev, deployed, text, client, urls, budget, pool_id, ledger_path, out_dir / "dev.jsonl")
    effects = _effects(rows)
    probe = _replacement_probe(dev[: int(raw["collection"]["probe_snapshots"])], deployed["M"], client, urls[0], budget)
    decision = _decide(raw, rows, effects, probe, forks, pairs, accepted, induced)
    decision["scope_note"] = (
        "这是 B-v2 开发 Pilot，用来决定是否值得打开一次新确认集。"
        "确认集没有评分。结构迁移簇不存在，因为 HotpotQA 没有第二种依赖结构。"
    )
    (out_dir / "B_V2_DECISION.md").write_text(render_v2_decision(decision), encoding="utf-8")
    write_json(out_dir / "decision.json", decision)
    write_json(out_dir / "B_V2_EFFECTS.json", effects)
    (out_dir / "B_V2_SNAPSHOT_AUDIT.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in probe.get("rows") or []),
        encoding="utf-8",
    )
    (out_dir / "B_V2_RUNTIME_AUDIT.md").write_text(_runtime_md(forks, pairs, collected, val_cost), encoding="utf-8")
    (out_dir / "B_V2_BASELINE_AUDIT.md").write_text(_baseline_md(raw, accepted), encoding="utf-8")
    _write_results(out_dir / "B_V2_DEV_RESULTS.csv", rows)
    return decision


def _collect(raw, tasks, client, urls, budget, path, ledger_path, existing, *, supplement: bool) -> None:
    done = {
        (row["task_id"], int(row["continuation"]), bool(row.get("supplement")))
        for row in existing
        if row.get("stage") == "collect"
    }
    jobs = []
    per_task = 1 if supplement else int(raw["collection"]["continuations_per_task"])
    for task in tasks:
        for index in range(per_task):
            key = (task.task_id, index, supplement)
            if key in done:
                continue
            jobs.append((task, index))
    print(f"collect jobs={len(jobs)} supplement={supplement}", flush=True)
    _run_jobs(jobs, raw, client, urls, budget, path, ledger_path, "collect", supplement=supplement, rules=[], text="")


def _evaluate(raw, tasks, deployed, text, client, urls, budget, pool_id, ledger_path, path) -> list[dict[str, Any]]:
    existing = _read_jsonl(path)
    done = {(row["condition"], row["task_id"]) for row in existing}
    jobs = []
    for condition in CONDITIONS:
        for task in tasks:
            if (condition, task.task_id) in done:
                continue
            jobs.append((task, condition, deployed[condition], text if condition == "B2" else ""))
    print(f"dev jobs={len(jobs)}", flush=True)

    def one(url: str, item) -> dict[str, Any]:
        task, condition, rules, context = item
        episode = _episode(raw, task, rules, client, url, budget, context, float(raw["model"]["deploy_temperature"]))
        return _row(task, episode, condition, pool_id, "dev")

    fresh = _parallel(jobs, urls, one)
    for row in fresh:
        _append(path, row)
        _append(ledger_path, {"stage": "dev", **_cost(row)})
    return existing + fresh


def _induce_all(raw, traces, pairs, url, n_records) -> dict[str, dict[str, Any]]:
    model = _model(raw, url)
    mismatched = _mismatch(pairs)
    specs = {
        "B3": ("empty", [], traces, n_records),
        "B4": ("mismatched", mismatched, traces, 0),
        "B5": ("direct", [], _balanced(traces, 40), 0),
        "M": ("aligned", pairs[:40], traces, 0),
    }
    out = {}
    for condition, (mode, pair_rows, trace_rows, empty_count) in specs.items():
        pair_rows = pair_rows[:40]
        out[condition] = induce_v2(
            mode=mode,
            condition=condition,
            generate_fn=_generate(model),
            temperature=0.0,
            pairs=pair_rows,
            traces=trace_rows,
            empty_count=empty_count,
            max_candidates=int(raw["collection"]["max_candidates"]),
        )
        print(f"induce {condition} proposed={len(out[condition]['rules'])} rejected={len(out[condition]['rejected'])}", flush=True)
    return out


def _accept_all(raw, induced, val, baseline, client, urls, budget, out_dir, ledger_path, collected, train):
    target = [task for task in val if task.family == "recovery"]
    retention = [task for task in val if task.family == "retention"]
    base_target = _mean(raw, target, baseline, client, urls, budget, "accept-b1")
    base_ret = _mean(raw, retention, baseline, client, urls, budget, "accept-b1")
    _append(ledger_path, {"stage": "accept_baseline", "episodes": len(target) + len(retention)})
    accepted = {}
    for condition, proposal in induced.items():
        rules = proposal["rules"]
        if condition in {"M", "B4"}:
            rules = _revise(raw, rules, baseline, collected, train, client, urls, budget, ledger_path)
        rate_target = _mean(raw, target, baseline + rules, client, urls, budget, f"accept-{condition}")
        rate_ret = _mean(raw, retention, baseline + rules, client, urls, budget, f"accept-{condition}")
        ok = rate_target + 1e-12 >= base_target and base_ret - rate_ret <= float(raw["decision"]["retention_tolerance"]) + 1e-12
        accepted[condition] = {"accepted": ok and bool(rules), "rules": rules if ok else [], "target": rate_target, "retention": rate_ret}
        print(f"accept {condition} ok={accepted[condition]['accepted']} target={rate_target:.3f} retention={rate_ret:.3f}", flush=True)
    write_json(out_dir / "acceptance.json", {key: {**value, "rules": [rule.to_dict() for rule in value["rules"]]} for key, value in accepted.items()})
    return accepted, {"baseline_target": base_target, "baseline_retention": base_ret}


def _revise(raw, rules, baseline, collected, tasks, client, urls, budget, ledger_path) -> list[Rule]:
    cap = int(raw["collection"]["snapshot_cap"])
    snapshots = []
    by_id = {task.task_id: task for task in tasks}
    for row in collected:
        data = row.get("snapshot")
        task = by_id.get(row.get("task_id"))
        if not data or task is None:
            continue
        snapshots.append((task, _snapshot(data)))
        if len(snapshots) >= cap:
            break
    kept: list[Rule] = []
    for rule in rules[: int(raw["collection"]["max_candidates"])]:
        improved = harmed = 0
        checked = 0
        for task, snapshot in snapshots:
            features = observable_features(restore(snapshot), budget)
            if not trigger_matches(rule.trigger, features, max_atoms=3):
                continue
            if checked >= cap:
                break
            checked += 1
            url = urls[checked % len(urls)]
            base = _continue(raw, task, baseline, client, url, budget, snapshot)
            trial = _continue(raw, task, [*baseline, rule], client, url, budget, snapshot)
            _append(ledger_path, {"stage": "snapshot", "rule_id": rule.rule_id, "task_id": task.task_id})
            if trial.success and not base.success:
                improved += 1
            elif base.success and not trial.success:
                harmed += 1
        if harmed and not improved:
            continue
        kept.append(rule)
    return kept


def _decide(raw, rows, effects, probe, forks, pairs, accepted, induced) -> dict[str, Any]:
    target = {"dev"}
    alpha = 0.05 / 2
    boot = int(raw["decision"]["bootstrap_replicates"])
    seed = int(raw["decision"].get("bootstrap_seed", 20261012))
    comparisons = {
        "M_vs_B1": paired_comparison(rows, "M", "B1", target, n_boot=boot, seed=seed, alpha=alpha),
        "M_vs_B3": paired_comparison(rows, "M", "B3", target, n_boot=boot, seed=seed + 1, alpha=alpha),
        "M_vs_B5": paired_comparison(rows, "M", "B5", target, n_boot=boot, seed=seed + 2, alpha=0.05),
        "M_vs_B2": paired_comparison(rows, "M", "B2", target, n_boot=boot, seed=seed + 3, alpha=0.05),
    }
    retention_rows = [row for row in rows if row["condition"] == "M" and row["family"] == "retention"]
    retention = paired_comparison(
        [row for row in rows if row["family"] == "retention"],
        "M",
        "B0",
        {"dev"},
        n_boot=boot,
        seed=seed + 4,
        alpha=0.05,
    )
    trigger_rows = [row for row in rows if row["condition"] == "M" and row["family"] == "recovery"]
    trigger = sum(1 for row in trigger_rows if row.get("triggered_rules")) / len(trigger_rows) if trigger_rows else 0.0
    judged = judge_v2(
        comparisons,
        floor=float(raw["decision"]["min_gain"]),
        retention_delta=retention.get("diff"),
        retention_ci=retention.get("ci"),
        retention_tolerance=float(raw["decision"]["retention_tolerance"]),
        trigger_rate=trigger,
    )
    judged.update(comparisons)
    judged["retention"] = retention
    judged["effects"] = effects
    judged["replacement"] = probe
    judged["fork_tasks"] = len(forks)
    judged["pairs"] = len(pairs)
    judged["trigger_rate"] = trigger
    judged["acceptance"] = {key: value["accepted"] for key, value in accepted.items()}
    judged["proposed"] = {key: len(value["rules"]) for key, value in induced.items()}
    judged["n_retention_rows"] = len(retention_rows)
    return judged


def _effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    left = {row["task_id"]: row for row in rows if row["condition"] == "M" and row["family"] == "recovery"}
    right = {row["task_id"]: row for row in rows if row["condition"] == "B1" and row["family"] == "recovery"}
    repair = harm = both_ok = both_bad = 0
    for task_id, item in left.items():
        other = right.get(task_id)
        if other is None:
            continue
        if item["success"] and not other["success"]:
            repair += 1
        elif other["success"] and not item["success"]:
            harm += 1
        elif item["success"] and other["success"]:
            both_ok += 1
        else:
            both_bad += 1
    return {"stage": "dev", "versus": "B1", "repair": repair, "harm": harm, "both_success": both_ok, "both_fail": both_bad}


def _replacement_probe(tasks, rules, client, url, budget) -> dict[str, Any]:
    rows = []
    changed = 0
    for task in tasks:
        original = _episode_direct(task, rules, client, url, budget, "", 0.0)
        swapped = [_swap(rule) for rule in rules]
        alternate = _episode_direct(task, swapped, client, url, budget, "", 0.0)
        query_changed = _query(original) != _query(alternate)
        if query_changed:
            changed += 1
        rows.append({"task_id": task.task_id, "original_query": _query(original), "swapped_query": _query(alternate), "query_changed": query_changed})
    return {"changed": changed > 0, "n_changed": changed, "n_tasks": len(tasks), "rows": rows}


def _swap(rule: Rule) -> Rule:
    parameters = dict(rule.parameters)
    if parameters.get("strategy") == "drop_redundant":
        parameters["strategy"] = "shorten"
    elif parameters.get("strategy"):
        parameters["strategy"] = "drop_redundant"
    return Rule(rule.rule_id + "_swap", rule.family, rule.trigger, rule.action_template, parameters, fallback=rule.fallback)


def _query(episode: PublicEpisode) -> str:
    for action in episode.actions:
        if action.get("tool") == "search":
            return str((action.get("arguments") or {}).get("query") or "")
    return ""


def _mean(raw, tasks, rules, client, urls, budget, label: str) -> float:
    if not tasks:
        return 0.0
    episodes = _parallel([(task, rules) for task in tasks], urls, lambda url, item: _episode(raw, item[0], item[1], client, url, budget, "", 0.0))
    print(f"{label} n={len(episodes)}", flush=True)
    return sum(1.0 if episode.success else 0.0 for episode in episodes) / len(episodes)


def _episode(raw, task, rules, client, url, budget, text, temperature) -> PublicEpisode:
    model = _model(raw, url)
    error = "failed"
    for _attempt in range(2):
        try:
            return run_public_episode(
                task,
                rules,
                client,
                model,
                budget,
                text_context=text,
                temperature=temperature,
                max_new_tokens=int(raw["model"]["max_new_tokens"]),
                top_k=int(raw["retriever"]["top_k"]),
            )
        except BackendUnavailableError as exc:
            error = f"failed:{type(exc).__name__}"
    return PublicEpisode(
        task_id=task.task_id,
        answer="",
        success=False,
        tool_calls=0,
        decision_steps=0,
        prompt_tokens=0,
        completion_tokens=0,
        model_calls=0,
        interventions=0,
        invalid_actions=0,
        triggered_rules=[],
        actions=[],
        request_hashes=[],
        text_in_prompt=False,
        thinking_tags=0,
        infra_status=error,
        snapshots=[],
    )


def _continue(raw, task, rules, client, url, budget, snapshot) -> PublicEpisode:
    model = _model(raw, url)
    return run_public_episode(
        task,
        rules,
        client,
        model,
        budget,
        temperature=0.0,
        max_new_tokens=int(raw["model"]["max_new_tokens"]),
        top_k=int(raw["retriever"]["top_k"]),
        initial=snapshot,
    )


def _episode_direct(task, rules, client, url, budget, text, temperature) -> PublicEpisode:
    from expolicy.paper_b.pipeline_v2 import _ThinkingOff

    backend = OpenAICompatBackend("qwen3-14b-r9", base_url=url, api_key="EMPTY", timeout=180.0)
    return run_public_episode(task, rules, client, _ThinkingOff(backend), budget, text_context=text, temperature=temperature, max_new_tokens=256, top_k=5)


def _run_jobs(jobs, raw, client, urls, budget, path, ledger_path, stage, *, supplement, rules, text) -> None:
    temperature = float(raw["model"]["collect_temperature"] if stage == "collect" else raw["model"]["deploy_temperature"])

    def one(url: str, item) -> dict[str, Any]:
        task, index = item
        episode = _episode(raw, task, rules, client, url, budget, text, temperature)
        row = _row(task, episode, "collect", "", stage)
        row["continuation"] = index
        row["supplement"] = supplement
        return row

    for row in _parallel(jobs, urls, one):
        _append(path, row)
        _append(ledger_path, {"stage": stage, "supplement": supplement, **_cost(row)})
        print(f"{stage} {row['task_id']} c={row['continuation']} success={row['success']} tools={row['tool_calls']}", flush=True)


def _parallel(jobs, urls, fn):
    if not jobs:
        return []
    outputs = []
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures = [pool.submit(fn, urls[index % len(urls)], job) for index, job in enumerate(jobs)]
        for future in as_completed(futures):
            outputs.append(future.result())
    return outputs


def _row(task: PublicTask, episode: PublicEpisode, condition: str, pool_id: str, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "condition": condition,
        "task_id": task.task_id,
        "split": task.split,
        "source_group": task.source_group,
        "cluster": task.cluster or task.source_group,
        "family": task.family,
        "success": int(episode.success),
        "tool_calls": episode.tool_calls,
        "decision_steps": episode.decision_steps,
        "prompt_tokens": episode.prompt_tokens,
        "completion_tokens": episode.completion_tokens,
        "model_calls": episode.model_calls,
        "triggered_rules": "|".join(episode.triggered_rules),
        "infra_status": episode.infra_status,
        "experience_pool_id": pool_id,
        "eval_seed": "0",
        "actions": episode.actions,
        "snapshot": _compact_snapshot(episode),
    }


def _cost(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "condition": row.get("condition"),
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "tool_calls": row["tool_calls"],
        "model_calls": row["model_calls"],
        "infra_status": row["infra_status"],
    }


def _traces_from_records(records: list[dict[str, Any]], tasks: list[PublicTask]) -> list[Trace]:
    by_id = {task.task_id: task for task in tasks}
    traces = []
    for row in records:
        if row.get("stage") != "collect" or row.get("infra_status") != "ok":
            continue
        task = by_id.get(row["task_id"])
        if task is None:
            continue
        steps = []
        for action in row.get("actions") or []:
            if not action.get("ok") or action.get("tool") == "answer":
                continue
            steps.append(
                TraceStep(
                    features_before=dict(action.get("features") or {}),
                    action_template=str(action.get("tool") or ""),
                    parameters=dict(action.get("arguments") or {}),
                    observation_redacted="",
                    history_hash=str(action.get("history_hash") or ""),
                    snapshot_hash=str(action.get("history_hash") or ""),
                )
            )
        traces.append(
            Trace(
                trace_id=f"{row['task_id']}:{row['continuation']}:{int(bool(row.get('supplement')))}",
                task_id=task.task_id,
                source_group=task.source_group,
                split=task.split,
                family=task.family,
                continuation=str(row["continuation"]),
                success=bool(row["success"]),
                steps=steps,
            )
        )
    return traces


def _mismatch(pairs) -> list:
    from dataclasses import replace

    swapped = []
    if len(pairs) < 2:
        return swapped
    for index, pair in enumerate(pairs):
        donor = pairs[(index + 1) % len(pairs)]
        if donor.success_action == pair.success_action and donor.success_params == pair.success_params:
            continue
        swapped.append(
            replace(
                pair,
                pair_id=f"mismatch-{pair.pair_id}",
                alignment="mismatched",
                success_action=donor.success_action,
                success_params=dict(donor.success_params),
                swapped_from=pair.success_action,
            )
        )
    return swapped


def _text_experience(traces: list[Trace]) -> str:
    lines = []
    for trace in traces:
        if not trace.success:
            continue
        for step in trace.steps:
            lines.append(f"features={json.dumps(step.features_before, sort_keys=True)} action={step.action_template}")
            if len(lines) >= 40:
                return "\n".join(lines)
    return "\n".join(lines)


def _model(raw: dict[str, Any], url: str):
    from expolicy.paper_b.pipeline_v2 import _ThinkingOff

    backend = OpenAICompatBackend(raw["model"]["name"], base_url=url, api_key=str(raw["model"].get("api_key") or "EMPTY"), timeout=180.0)
    return _ThinkingOff(backend)


def _generate(model):
    def generate(prompt: str, temperature: float) -> str:
        result = model.generate([{"role": "user", "content": prompt}], temperature=temperature, max_new_tokens=512)
        return result.text

    return generate


def _retriever(raw: dict[str, Any]):
    from expolicy.environments.public_search.retriever import RetrieverClient

    return RetrieverClient(raw["retriever"]["base_url"], timeout=120.0)


def _tasks(manifest: dict[str, Any], split: str) -> list[PublicTask]:
    from expolicy.paper_b.public_data import tasks_for_split

    return tasks_for_split(manifest, split)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fields = [
        "condition",
        "task_id",
        "split",
        "source_group",
        "cluster",
        "family",
        "success",
        "tool_calls",
        "decision_steps",
        "prompt_tokens",
        "completion_tokens",
        "model_calls",
        "triggered_rules",
        "infra_status",
        "experience_pool_id",
        "eval_seed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _runtime_md(forks, pairs, collected, val_cost) -> str:
    ok = [row for row in collected if row.get("stage") == "collect" and row.get("infra_status") == "ok"]
    return "\n".join(
        [
            "# B-v2 开发运行审计",
            "",
            f"采集成功轨迹：{len(ok)}",
            f"成败分叉任务：{len(forks)}",
            f"对齐片段：{len(pairs)}",
            f"验证基线：`{json.dumps(val_cost, ensure_ascii=False)}`",
            "",
        ]
    )


def _baseline_md(raw, accepted) -> str:
    lines = ["# B-v2 开发基线", "", "拒绝的候选回到冻结的 B1。", ""]
    for condition, value in accepted.items():
        lines.append(f"- {condition} 接受额外规则：`{value['accepted']}`，条数 {len(value['rules'])}")
    lines.append("")
    return "\n".join(lines)


def _compact_snapshot(episode: PublicEpisode) -> dict[str, Any] | None:
    if not episode.snapshots:
        return None
    snap = episode.snapshots[0]
    return {
        "question": snap.question,
        "queries": list(snap.queries),
        "doc_ids": list(snap.doc_ids),
        "opened": list(snap.opened),
        "hits": [dict(item) for item in list(snap.hits)[:3]],
        "history": [dict(item) for item in list(snap.history)[-3:]],
        "last_observation": snap.last_observation[:400],
        "opened_text": snap.opened_text[-800:],
        "tool_calls": snap.tool_calls,
        "decision_steps": snap.decision_steps,
        "prompt_tokens": snap.prompt_tokens,
        "completion_tokens": snap.completion_tokens,
    }


def _snapshot(data: dict[str, Any]):
    from expolicy.paper_b.public_exec import Snapshot

    return Snapshot(
        question=str(data.get("question") or ""),
        queries=tuple(data.get("queries") or []),
        doc_ids=tuple(data.get("doc_ids") or []),
        opened=tuple(data.get("opened") or []),
        hits=tuple(data.get("hits") or []),
        history=tuple(data.get("history") or []),
        last_observation=str(data.get("last_observation") or ""),
        opened_text=str(data.get("opened_text") or ""),
        tool_calls=int(data.get("tool_calls") or 0),
        decision_steps=int(data.get("decision_steps") or 0),
        prompt_tokens=int(data.get("prompt_tokens") or 0),
        completion_tokens=int(data.get("completion_tokens") or 0),
    )


def _balanced(traces: list[Trace], limit: int) -> list[Trace]:
    success = [trace for trace in traces if trace.success]
    failure = [trace for trace in traces if not trace.success]
    picked: list[Trace] = []
    for left, right in zip(success, failure, strict=False):
        picked.extend([left, right])
        if len(picked) >= limit:
            break
    if not picked:
        picked = traces[:limit]
    return picked[:limit]


def snapshot_features_match(rule: Rule, features: dict[str, int]) -> bool:
    return trigger_matches(rule.trigger, features, max_atoms=3)
