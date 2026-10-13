"""Offline audit of the B-v2.1 pool. No model calls and no gold-based selection."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from expolicy.paper_b.public_exec import REWRITE_STRATEGIES, rewrite_query
from expolicy.paper_b.v21_score import official_em, official_f1

MAX_DECISION = 8
MAX_TOOLS = 8
MIN_LEFT = 4


def dedup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("task_id"), int(row.get("continuation") or 0), bool(row.get("supplement")), row.get("run_id"))
        found[key] = row
    return list(found.values())


def initial_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in dedup_rows(rows) if not row.get("supplement") and int(row.get("continuation") or 0) in {0, 1}]


def trajectory_pair_count(rows: list[dict[str, Any]]) -> int:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in initial_rows(rows):
        grouped[str(row["task_id"])].append(row)
    total = 0
    for items in grouped.values():
        wins = [item for item in items if float(item.get("official_em") or 0) == 1]
        losses = [item for item in items if float(item.get("official_em") or 0) == 0]
        total += len(wins) * len(losses)
    return total


def validated_fragments(rows: list[dict[str, Any]], questions: dict[str, str]) -> list[dict[str, Any]]:
    """One primary fragment per mixed task. Combinations do not add support."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in initial_rows(rows):
        grouped[str(row["task_id"])].append(row)
    fragments = []
    for task_id, items in sorted(grouped.items()):
        wins = sorted((item for item in items if float(item.get("official_em") or 0) == 1), key=lambda item: int(item["continuation"]))
        losses = sorted((item for item in items if float(item.get("official_em") or 0) == 0), key=lambda item: int(item["continuation"]))
        if not wins or not losses:
            continue
        fragments.append(_primary_fragment(wins[0], losses[0], questions.get(task_id, "")))
    return fragments


def select_states(rows: list[dict[str, Any]], *, fork_cap: int = 16, validation_cap: int = 4) -> dict[str, list[dict[str, Any]]]:
    """Gold is not read. Repeat stalls outrank a merely unopened first search."""
    chosen = []
    for row in initial_rows(rows):
        if int(row.get("continuation") or 0) != 0:
            continue
        state = _early_state(row)
        if state is None:
            continue
        chosen.append(state)
    chosen.sort(key=lambda item: (item["priority"], item["task_id"]))
    return {"fork": chosen[:fork_cap], "validation": chosen[fork_cap : fork_cap + validation_cap]}


def pool_audit(rows: list[dict[str, Any]], questions: dict[str, str]) -> dict[str, Any]:
    initial = initial_rows(rows)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in initial:
        by_task[str(row["task_id"])].append(row)
    both_ok = mixed = both_fail = incomplete = 0
    for items in by_task.values():
        continuations = {int(item["continuation"]) for item in items}
        if continuations != {0, 1}:
            incomplete += 1
            continue
        ems = {int(item["official_em"]) for item in items}
        if ems == {1}:
            both_ok += 1
        elif ems == {0, 1}:
            mixed += 1
        else:
            both_fail += 1
    fragments = validated_fragments(rows, questions)
    rescored = 0
    for row in initial:
        again = official_em(row.get("predicted_answer_span") or "", row.get("gold_answer") or "")
        if again != float(row.get("official_em") or 0):
            rescored += 1
    recoveries = sum(1 for row in initial if float(row.get("official_em") or 0) == 1 and _strict_recovery(row))
    loose = sum(1 for row in initial if float(row.get("official_em") or 0) == 1 and _loose_recovery(row))
    states = select_states(rows)
    return {
        "initial_episodes": len(initial),
        "tasks": len(by_task),
        "incomplete_tasks": incomplete,
        "official_em": sum(float(row.get("official_em") or 0) for row in initial),
        "rescore_mismatches": rescored,
        "both_success": both_ok,
        "mixed_tasks": mixed,
        "both_failure": both_fail,
        "trajectory_pair_count": trajectory_pair_count(rows),
        "validated_fragment_count": sum(1 for item in fragments if item["validated"]),
        "catalog_supported_tasks": sorted({item["source_task_ids"][0] for item in fragments if item["validated"]}),
        "answer_span_only": sum(1 for item in fragments if item["pair_kind"] == "answer_span_only"),
        "action_diff_catalog_unsupported": sum(1 for item in fragments if item["pair_kind"] == "action_diff_catalog_unsupported"),
        "system_fault_pairs": sum(1 for item in fragments if item["system_fault"]),
        "strict_recoveries": recoveries,
        "loose_open_or_two_search_successes": loose,
        "fork_states": len(states["fork"]),
        "validation_states": len(states["validation"]),
        "fragments": fragments,
    }


def classify_calibration(row: dict[str, Any]) -> str:
    """Diagnostic label only. It does not change official EM."""
    if float(row.get("official_em") or 0) == 1:
        return "exact"
    predicted = str(row.get("predicted_answer_span") or "")
    gold = str(row.get("gold_answer") or "")
    if not predicted.strip():
        return "format_or_missing"
    lowered = predicted.casefold()
    if any(phrase in lowered for phrase in ("no information", "not mentioned", "not enough", "cannot determine")):
        return "insufficient_evidence"
    f1 = official_f1(predicted, gold)
    if f1 == 0:
        return "wrong_entity_or_fact"
    if len(predicted.split()) > len(gold.split()) + 2:
        return "granularity_or_sentence"
    return "near_paraphrase"


def _primary_fragment(win: dict[str, Any], loss: dict[str, Any], question: str) -> dict[str, Any]:
    win_actions = _actions(win)
    loss_actions = _actions(loss)
    diverge = _diverge(win_actions, loss_actions)
    system_fault = win.get("terminal_status") in {"INFRA", "PARSE_FAILED"} or loss.get("terminal_status") in {"INFRA", "PARSE_FAILED"}
    shared = diverge not in (None, 0)
    features = _features_at(win if diverge is None else win, 0 if diverge is None else diverge)
    catalog = _catalog_hit(question, win_actions, diverge)
    if win_actions == loss_actions:
        kind = "answer_span_only"
    elif catalog["covered"]:
        kind = "catalog_supported"
    else:
        kind = "action_diff_catalog_unsupported"
    validated = kind == "catalog_supported" and not system_fault
    alignment = "same_prefix" if diverge is not None else "same_actions"
    if diverge == 0:
        alignment = "same_prefix"
    if diverge not in (None, 0) and not shared:
        alignment = "matched_observation"
    return {
        "pair_id": f"{win['task_id']}:{win['continuation']}:{loss['continuation']}",
        "source_task_ids": [win["task_id"]],
        "trace_ids": [f"{win['task_id']}:{win['continuation']}", f"{loss['task_id']}:{loss['continuation']}"],
        "alignment_type": alignment,
        "state_hash": "",
        "messages_hash": "",
        "observation_hash": "",
        "budget_left": features.get("tool_budget_left"),
        "features_before": {key: value for key, value in features.items() if key in {"search_count", "repeat_query_count", "repeat_doc_ratio", "new_evidence", "evidence_count", "tool_budget_left"}},
        "success_tool": catalog["tool"],
        "success_parameters": catalog["parameters"],
        "failure_tool": loss_actions[diverge][0] if diverge is not None and diverge < len(loss_actions) else "",
        "failure_parameters": {},
        "terminal_em_labels": {"success": 1, "failure": 0},
        "action_catalog_coverage": catalog["covered"],
        "pair_kind": kind,
        "validated": validated,
        "system_fault": system_fault,
        "diverge_index": diverge,
        "success_answer": win.get("predicted_answer_span"),
        "failure_answer": loss.get("predicted_answer_span"),
        "provenance": "b-v2.1-initial-continuations",
    }


def _catalog_hit(question: str, actions: list[tuple[str, str]], diverge: int | None) -> dict[str, Any]:
    if diverge is None or diverge >= len(actions):
        return {"covered": False, "tool": "", "parameters": {}}
    tool, argument = actions[diverge]
    if tool == "open" and diverge > 0:
        return {"covered": False, "tool": "open", "parameters": {"document_id": argument}}
    if tool != "search":
        return {"covered": False, "tool": tool, "parameters": {}}
    prior = [json.loads(item[1]).get("query", "") for item in actions[:diverge] if item[0] == "search"]
    query = json.loads(argument).get("query", "")
    for strategy in REWRITE_STRATEGIES:
        produced = rewrite_query(strategy, question, prior, "")
        if produced.casefold() == str(query).casefold():
            return {"covered": True, "tool": "rewrite_query", "parameters": {"strategy": strategy}}
    return {"covered": False, "tool": "search", "parameters": {"query_not_registered": True}}


def _actions(row: dict[str, Any]) -> list[tuple[str, str]]:
    found = []
    for step in row.get("steps") or []:
        action = step.get("canonical_action")
        if action in {"search", "open", "find"}:
            found.append((action, json.dumps(step.get("tool_arguments_sent") or {}, sort_keys=True, ensure_ascii=False)))
    return found


def _diverge(left: list[tuple[str, str]], right: list[tuple[str, str]]) -> int | None:
    if left == right:
        return None
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def _features_at(row: dict[str, Any], index: int) -> dict[str, int]:
    queries: list[str] = []
    doc_ids: list[str] = []
    opened = 0
    tools = 0
    seen = 0
    for step in (row.get("steps") or [])[:index]:
        action = step.get("canonical_action")
        if action in {"search", "open", "find"}:
            tools += 1
        if action == "search":
            queries.append(str((step.get("tool_arguments_sent") or {}).get("query") or "").casefold())
            doc_ids.extend(_ids(step.get("observation_sent_to_model") or ""))
        if action == "open":
            opened += 1
        seen += 1
    repeat = len(queries) - len(set(queries)) if queries else 0
    ratio = int(round(100 * (1 - len(set(doc_ids)) / len(doc_ids)))) if doc_ids else 0
    return {
        "search_count": len(queries),
        "repeat_query_count": repeat,
        "repeat_doc_ratio": ratio,
        "new_evidence": 0,
        "evidence_count": opened,
        "tool_budget_left": max(0, MAX_TOOLS - tools),
        "seen_steps": seen,
    }


def _early_state(row: dict[str, Any]) -> dict[str, Any] | None:
    queries: list[str] = []
    seen: list[str] = []
    opened: set[str] = set()
    fallback = None
    decision = 0
    tools = 0
    steps = row.get("steps") or []
    for index, step in enumerate(steps):
        decision += 1
        action = step.get("canonical_action")
        if action in {"search", "open", "find"}:
            tools += 1
        left_decision = MAX_DECISION - decision
        left_tools = MAX_TOOLS - tools
        if action == "open":
            opened.add(str((step.get("tool_arguments_sent") or {}).get("document_id") or ""))
        if action != "search" or left_decision < MIN_LEFT or left_tools < MIN_LEFT:
            continue
        query = str((step.get("tool_arguments_sent") or {}).get("query") or "").casefold()
        ids = _ids(step.get("observation_sent_to_model") or "")
        repeat = bool(query and query in queries)
        repeat_doc = any(item in seen for item in ids)
        queries.append(query)
        seen.extend(ids)
        prefix = steps[: index + 1]
        if repeat:
            return _state(row, prefix, "repeat_query", 0)
        if repeat_doc:
            return _state(row, prefix, "repeat_doc", 1)
        if ids and not opened and fallback is None:
            fallback = _state(row, prefix, "unopened_hits", 2)
    return fallback


def _state(row: dict[str, Any], prefix: list[dict[str, Any]], reason: str, priority: int) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "continuation": int(row["continuation"]),
        "reason": reason,
        "priority": priority,
        "prefix_steps": prefix,
        "cluster": row.get("cluster"),
        "source_group": row.get("source_group"),
    }


def _strict_recovery(row: dict[str, Any]) -> bool:
    """Visible stall, then a changed action, then new evidence. An ordinary open is not enough."""
    queries: list[str] = []
    stalled = False
    for step in row.get("steps") or []:
        action = step.get("canonical_action")
        if action != "search":
            if stalled and action == "open" and _ids(step.get("observation_sent_to_model") or ""):
                return True
            continue
        query = str((step.get("tool_arguments_sent") or {}).get("query") or "").casefold()
        if stalled and query and query not in queries and _ids(step.get("observation_sent_to_model") or ""):
            return True
        if query and query in queries:
            stalled = True
        if query:
            queries.append(query)
    return False


def _loose_recovery(row: dict[str, Any]) -> bool:
    searches = [step for step in row.get("steps") or [] if step.get("canonical_action") == "search"]
    opens = [step for step in row.get("steps") or [] if step.get("canonical_action") == "open"]
    return len(searches) >= 2 and bool(opens)


def _ids(observation: str) -> list[str]:
    found = []
    for line in str(observation).splitlines():
        if line.startswith("id="):
            found.append(line.split("=", 1)[1].strip())
    return found


def render_audit(summary: dict[str, Any], calibration: list[dict[str, Any]]) -> str:
    labels = [f"| {row['task_id']} | {row.get('official_em')} | {classify_calibration(row)} | {str(row.get('predicted_answer_span') or '')[:80]} |" for row in calibration]
    fragments = summary["fragments"]
    pair_lines = [
        f"| {item['pair_id']} | {item['pair_kind']} | {item['alignment_type']} | {item['diverge_index']} | {item['validated']} |"
        for item in fragments
    ]
    return "\n".join(
        [
            "# B-v2.2 已有经验池审计",
            "",
            "本文件只读 B-v2.1 原始记录，没有新的模型调用，也没有用开发标签挑选经验。",
            "",
            f"初始轨迹：{summary['initial_episodes']}，任务：{summary['tasks']}，官方 EM：{summary['official_em']}",
            "",
            f"两条都成功：{summary['both_success']}；一成一败：{summary['mixed_tasks']}；两条都失败：{summary['both_failure']}；缺续行：{summary['incomplete_tasks']}",
            "",
            f"重判不一致：{summary['rescore_mismatches']}",
            "",
            f"轨迹组合数 trajectory_pair_count：{summary['trajectory_pair_count']}",
            "",
            f"可支持注册动作的片段 validated_fragment_count：{summary['validated_fragment_count']}",
            "",
            f"答案措辞差异：{summary['answer_span_only']}；动作不同但目录不能表达：{summary['action_diff_catalog_unsupported']}",
            "",
            f"严格恢复轨迹：{summary['strict_recoveries']}。旧的“两次搜索且 open”计数是 {summary['loose_open_or_two_search_successes']}，不当作恢复。",
            "",
            "## 混合任务",
            "",
            "| 片段 | 类型 | 对齐 | 分叉下标 | 注册动作可表达 |",
            "| --- | --- | --- | --- | --- |",
            *pair_lines,
            "",
            "## 已知证据校准逐题",
            "",
            "官方 EM 不改。下面的标签只解释已保存的预测。",
            "",
            "| 任务 | EM | 标签 | 预测 |",
            "| --- | --- | --- | --- |",
            *labels,
            "",
            f"可重放的训练状态：分叉 {summary['fork_states']}，预留验证 {summary['validation_states']}。选择只用重复查询、重复文档或未打开命中，不用 gold。",
            "",
        ]
    )


def write_pool_audit(
    rows: list[dict[str, Any]],
    questions: dict[str, str],
    calibration: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    summary = pool_audit(rows, questions)
    states = select_states(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "B_V22_EXISTING_POOL_AUDIT.md").write_text(render_audit(summary, calibration), encoding="utf-8")
    _write_jsonl(out_dir / "B_V22_ALIGNED_FRAGMENTS.jsonl", summary["fragments"])
    (out_dir / "fork_states.json").write_text(json.dumps(states, ensure_ascii=False), encoding="utf-8")
    kept = {key: value for key, value in summary.items() if key != "fragments"}
    (out_dir / "pool_audit.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
