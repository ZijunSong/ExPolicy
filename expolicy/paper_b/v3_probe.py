"""B-v3 feasibility probes. Does not rerun the B-v2.2 menu."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

from expolicy.models.openai_compat import OpenAICompatBackend
from expolicy.paper_b.pipeline_v2 import _ThinkingOff
from expolicy.paper_b.public_data import tasks_for_split
from expolicy.paper_b.public_exec import PublicBudget
from expolicy.paper_b.v21_runtime import RUNTIME_VERSION, run_v21_episode
from expolicy.paper_b.v22_pilot import MENU
from expolicy.r9_lab.io_util import sha256_json, sha256_text

RUN_ID = "b-v3"
GENERIC_NOTE = (
    "Review the question and the evidence already shown. "
    "Identify the next missing relation or entity and choose one useful next action. "
    "You may issue a new search query or open an observed document. "
    "Use only the current context. Return one normal tool action."
)
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset({"the", "a", "an", "of", "is", "was", "were", "what", "who", "which", "did", "does", "do", "in", "on", "for", "to", "and", "or"})


def menu_hash() -> str:
    return sha256_json([{"template": template, "parameters": parameters} for template, parameters in MENU])


def reference_candidates(rows: list[dict[str, Any]], questions: dict[str, str]) -> list[dict[str, Any]]:
    """At most one reference action per mixed task. Gold-only strings are rejected."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("supplement") or int(row.get("continuation") or 0) not in {0, 1}:
            continue
        grouped[str(row["task_id"])].append(row)
    found = []
    for task_id, items in sorted(grouped.items()):
        wins = [item for item in items if float(item.get("official_em") or 0) == 1]
        losses = [item for item in items if float(item.get("official_em") or 0) == 0]
        if not wins or not losses:
            continue
        found.append(_one_reference(wins[0], losses[0], questions.get(task_id, "")))
    return found


def _one_reference(win: dict[str, Any], loss: dict[str, Any], question: str) -> dict[str, Any]:
    win_actions = _tool_steps(win)
    loss_actions = _tool_steps(loss)
    diverge = _diverge(win_actions, loss_actions)
    base = {
        "task_id": win["task_id"],
        "eligible": False,
        "reason": "",
        "label": "",
        "prefix_steps": [],
        "forced_action": {},
    }
    if diverge is None:
        base["reason"] = "same_actions_answer_span_only"
        return base
    if diverge >= len(win_actions):
        base["reason"] = "success_has_no_later_tool_action"
        base["label"] = "ANSWER_TIMING"
        return base
    prefix = _prefix_until(win, diverge)
    action, payload = win_actions[diverge]
    visible = _visible(question, prefix)
    gold = str(win.get("gold_answer") or "")
    if action == "search":
        query = str(payload.get("query") or "")
        if not _executable(query, visible, gold):
            base["reason"] = "success_query_not_in_prefix_context"
            base["label"] = _label(prefix)
            return base
        forced = {"tool": "search", "query": query}
    elif action == "open":
        document_id = str(payload.get("document_id") or "")
        if document_id and document_id not in visible:
            base["reason"] = "document_id_not_visible"
            base["label"] = _label(prefix)
            return base
        forced = {"tool": "open", "document_id": document_id}
    else:
        base["reason"] = "unsupported_reference_tool"
        return base
    base.update(eligible=True, reason="visible_next_action", label=_label(prefix), prefix_steps=prefix, forced_action=forced)
    return base


def _label(prefix: list[dict[str, Any]]) -> str:
    searches = [step for step in prefix if step.get("canonical_action") == "search"]
    if not searches:
        return "QUERY_FORMULATION"
    queries = [str((step.get("tool_arguments_sent") or {}).get("query") or "").casefold() for step in searches]
    if len(queries) != len(set(queries)):
        return "RECOVERY"
    if len(searches) == 1 and not any(step.get("canonical_action") == "open" for step in prefix):
        return "NEXT_ACTION"
    return "NEXT_ACTION"


def _executable(query: str, visible: str, gold: str) -> bool:
    tokens = [token for token in _TOKEN.findall(query.casefold()) if token not in _STOP and len(token) > 2]
    if not tokens:
        return False
    visible_text = visible.casefold()
    if any(token not in visible_text for token in tokens):
        return False
    gold_tokens = [token for token in _TOKEN.findall(gold.casefold()) if token not in _STOP and len(token) > 2]
    return not any(token in tokens and token not in visible_text for token in gold_tokens)


def _visible(question: str, prefix: list[dict[str, Any]]) -> str:
    parts = [question]
    for step in prefix:
        parts.append(str(step.get("observation_sent_to_model") or ""))
        parts.append(str(step.get("model_output_raw") or ""))
    return "\n".join(parts)


def _tool_steps(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    found = []
    for step in row.get("steps") or []:
        action = step.get("canonical_action")
        if action in {"search", "open", "find"}:
            found.append((action, dict(step.get("tool_arguments_sent") or {})))
    return found


def _diverge(left: list[tuple[str, dict[str, Any]]], right: list[tuple[str, dict[str, Any]]]) -> int | None:
    if [(name, json.dumps(payload, sort_keys=True)) for name, payload in left] == [(name, json.dumps(payload, sort_keys=True)) for name, payload in right]:
        return None
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def _prefix_until(row: dict[str, Any], tool_index: int) -> list[dict[str, Any]]:
    kept = []
    seen = 0
    for step in row.get("steps") or []:
        if seen >= tool_index:
            break
        kept.append(step)
        if step.get("canonical_action") in {"search", "open", "find"}:
            seen += 1
    return kept


def run_action_space(
    config_path: Path,
    manifest_path: Path,
    answers_path: Path,
    states_path: Path,
    forks_path: Path,
    pool_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["answers_for_scorer_only"] = json.loads(answers_path.read_text(encoding="utf-8"))
    questions = {row["task_id"]: row["question"] for row in manifest["tasks"]}
    states = json.loads(states_path.read_text(encoding="utf-8"))["fork"]
    forks = _read_jsonl(forks_path)
    pool = _read_jsonl(pool_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_scope(out_dir, raw)
    references = reference_candidates(pool, questions)
    (out_dir / "B_V3_REFERENCE_AUDIT.md").write_text(_reference_md(references), encoding="utf-8")
    tasks = {task.task_id: task for task in tasks_for_split(manifest, "train")}
    rows = _run_probes(raw, tasks, states, references, out_dir)
    decision = _decide(states, forks, rows, references)
    (out_dir / "B_V3_ACTION_SPACE_DECISION.md").write_text(_decision_md(decision), encoding="utf-8")
    (out_dir / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    return decision


def _run_probes(raw, tasks, states, references, out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "B_V3_PREFIX_PROBES.jsonl"
    done = {(row.get("task_id"), row.get("probe")) for row in _read_jsonl(path)}
    jobs = []
    for state in states:
        task = tasks.get(state["task_id"])
        if task is None:
            continue
        for probe, note in (("NATIVE", ""), ("GENERIC-REPLAN", GENERIC_NOTE)):
            if (task.task_id, probe) not in done:
                jobs.append((task, state["prefix_steps"], probe, note, None))
    eligible = [item for item in references if item["eligible"]][:8]
    for item in eligible:
        task = tasks.get(item["task_id"])
        if task is None or (item["task_id"], f"REFERENCE-{item['label']}") in done:
            continue
        jobs.append((task, item["prefix_steps"], f"REFERENCE-{item['label']}", "", item["forced_action"]))
    print(f"v3 probe jobs={len(jobs)}", flush=True)
    urls = list(raw["model"]["base_urls"])
    budget = PublicBudget(**{key: raw["budget"][key] for key in ("max_tool_calls", "max_decision_steps", "max_completion_tokens", "max_interventions", "max_prompt_tokens")})
    lock = Lock()

    def work(item: tuple[int, Any]) -> None:
        order, (task, prefix, probe, note, forced) = item
        from expolicy.environments.public_search.retriever import RetrieverClient

        client = RetrieverClient(raw["retriever"]["base_url"], timeout=120.0)
        model = _ThinkingOff(OpenAICompatBackend(raw["model"]["name"], base_url=urls[order % len(urls)], api_key="EMPTY", timeout=180.0))
        record = run_v21_episode(
            task,
            client,
            model,
            budget,
            condition=probe,
            run_id=RUN_ID,
            checkpoint=raw["model"]["path"],
            runtime_hash=sha256_text(RUNTIME_VERSION + probe),
            corpus_hash=sha256_text(raw["retriever"]["corpus"]),
            temperature=0.0,
            max_new_tokens=int(raw["model"]["max_new_tokens"]),
            top_k=int(raw["retriever"]["top_k"]),
            prefix_actions=prefix,
            opening_note=note,
            forced_action=forced,
        )
        record["probe"] = probe
        record["stage"] = "action_space"
        with lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"probe {probe} {task.task_id} em={record['official_em']} terminal={record['terminal_status']}", flush=True)

    if jobs:
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            list(pool.map(work, list(enumerate(jobs))))
    return _read_jsonl(path)


def _decide(states, forks, probes, references) -> dict[str, Any]:
    menu = _menu_table(forks)
    native = {row["task_id"]: row for row in probes if row.get("probe") == "NATIVE"}
    generic = {row["task_id"]: row for row in probes if row.get("probe") == "GENERIC-REPLAN"}
    state_ids = [state["task_id"] for state in states]
    menu_failed = [task_id for task_id in state_ids if menu.get(task_id, {}).get("any_success") is not True]
    native_recover = [task_id for task_id in menu_failed if float(native.get(task_id, {}).get("official_em") or 0) == 1]
    generic_recover = [task_id for task_id in menu_failed if float(generic.get(task_id, {}).get("official_em") or 0) == 1]
    free = sorted(set(native_recover + generic_recover))
    reference_rows = _unique_probes(probes, "REFERENCE")
    reference_success = [row["task_id"] for row in reference_rows if float(row.get("official_em") or 0) == 1 and "RECOVERY" in str(row.get("probe"))]
    formulation_success = [row["task_id"] for row in reference_rows if float(row.get("official_em") or 0) == 1 and "QUERY_FORMULATION" in str(row.get("probe"))]
    generic_explains = bool(generic_recover) and set(native_recover) <= set(generic_recover) and not reference_success
    if len(free) >= 3 and _shared_principle(generic, native, free):
        status = "EXPLORATORY_CONTINUE"
        reason = "至少 3 个训练任务在自由参数续行中恢复，并且新查询有可复核的共同构造。可以进入表达检查。"
    elif len(free) >= 3:
        status = "STOP_B_SEARCH_SETUP"
        reason = "自由参数续行在多个任务上成功，但没有事先可复核的共同构造原则。不进入新程序对照，也不把这些成功写成内容收益。"
    elif generic_explains:
        status = "STOP_B_SEARCH_SETUP"
        reason = "旧菜单失败的状态里，通用重规划只多恢复了个别任务，正常续行没有跨任务恢复。参考重放成功的是起始查询，不能算停滞恢复。停止当前 HotpotQA 搜索设置中的 B。"
    elif reference_success and not free:
        status = "STOP_B_SEARCH_SETUP"
        reason = "参考动作可以重现，但正常续行和通用重规划没有产生它。本轮不自动扩大采样，也不另造第二套机制。"
    else:
        status = "STOP_B_SEARCH_SETUP"
        reason = "无干预续行和通用重规划都没有跨任务恢复，参考动作也没有形成可重复的恢复。停止当前 HotpotQA 搜索设置中的 B。"
    return {
        "protocol": "B-v3-feasibility",
        "status": status,
        "paper_claim_allowed": False,
        "reason": reason,
        "menu_hash": menu_hash(),
        "states": len(state_ids),
        "menu_any_success_tasks": sum(1 for task_id in state_ids if menu.get(task_id, {}).get("any_success")),
        "menu_failed_tasks": menu_failed,
        "native_recover_tasks": native_recover,
        "generic_recover_tasks": generic_recover,
        "free_recover_tasks": free,
        "reference_ran": len(reference_rows),
        "reference_recovery_success": reference_success,
        "formulation_success": formulation_success,
        "eligible_references": sum(1 for item in references if item["eligible"]),
        "stage2_ran": False,
        "unrun": ["slotbank", "induction", "content_screen", "seven_group", "confirmation"],
        "per_state": [
            {
                "task_id": task_id,
                "menu_em": menu.get(task_id, {}).get("ems", {}),
                "native_em": native.get(task_id, {}).get("official_em"),
                "generic_em": generic.get(task_id, {}).get("official_em"),
                "native_terminal": native.get(task_id, {}).get("terminal_status"),
                "generic_terminal": generic.get(task_id, {}).get("terminal_status"),
                "native_f1": native.get(task_id, {}).get("official_f1"),
                "generic_f1": generic.get(task_id, {}).get("official_f1"),
            }
            for task_id in state_ids
        ],
    }


def _shared_principle(generic, native, tasks: list[str]) -> bool:
    """A shared principle needs the same non-empty construction note, which this probe does not invent."""
    del generic, native, tasks
    return False


def _unique_probes(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    found = {}
    for row in rows:
        probe = str(row.get("probe") or "")
        if probe.startswith(prefix):
            found.setdefault((row["task_id"], probe), row)
    return list(found.values())


def _menu_table(forks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in forks:
        grouped[str(row["task_id"])][str(row.get("fork_action"))] = float(row.get("official_em") or 0)
    return {task_id: {"ems": ems, "any_success": any(value == 1 for value in ems.values())} for task_id, ems in grouped.items()}


def _write_scope(out_dir: Path, raw: dict[str, Any]) -> None:
    text = "\n".join(
        [
            "# B-v3 范围",
            "",
            "B-v2.2 的固定恢复目录已经停止，状态是 STOP_CURRENT_COLLECTION。本轮不重跑那 64 条，也不把四种动作合并成一个类别。",
            "",
            f"实际菜单 hash：`{menu_hash()}`",
            "",
            "菜单是 open_top、drop_redundant、entity_plus_relation、shorten。bridge_entity 不在这四项里，本轮不补跑。",
            "",
            f"模型：`{raw['model']['name']}` `{raw['model']['path']}`",
            "",
            f"检索：`{raw['retriever']['base_url']}`",
            "",
            "新增对照只有 NATIVE 和 GENERIC-REPLAN，各 16 条，共用原快照和剩余预算。通用指令在运行前冻结。",
            "",
            "参考动作重放最多 8 条，只在下一查询能从当前已见文本得到时运行。它是诊断，不是学习成绩。",
            "",
        ]
    )
    (out_dir / "B_V3_SCOPE.md").write_text(text, encoding="utf-8")


def _reference_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# B-v3 参考动作审计",
        "",
        "参考动作来自已观察到的成功轨迹。查询里出现前缀文本没有的内容时不运行。",
        "",
        "| 任务 | 可运行 | 类别 | 原因 |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['task_id']} | {row['eligible']} | {row['label']} | {row['reason']} |")
    lines.append("")
    return "\n".join(lines)


def _decision_md(decision: dict[str, Any]) -> str:
    state_lines = [
        f"| {row['task_id']} | {row['menu_em']} | {row['native_em']} | {row['generic_em']} | {row['native_terminal']} | {row['generic_terminal']} |"
        for row in decision["per_state"]
    ]
    return "\n".join(
        [
            "# B-v3 动作空间决定",
            "",
            f"状态：`{decision['status']}`",
            "",
            decision["reason"],
            "",
            f"16 个状态里旧菜单至少一次成功的任务数：{decision['menu_any_success_tasks']}。这不是 4/64 的另一种写法。",
            "",
            f"旧菜单全失败后，NATIVE 恢复任务：{decision['native_recover_tasks'] or '无'}",
            "",
            f"旧菜单全失败后，GENERIC-REPLAN 恢复任务：{decision['generic_recover_tasks'] or '无'}",
            "",
            f"参考重放条数：{decision['reference_ran']}，其中 RECOVERY 类成功：{decision['reference_recovery_success'] or '无'}，QUERY_FORMULATION 成功：{decision.get('formulation_success') or '无'}",
            "",
            "未运行：SlotBank、归纳、20 题内容小试、七组开发和确认集。",
            "",
            "| 任务 | 旧四动作 EM | NATIVE | GENERIC | NATIVE 终止 | GENERIC 终止 |",
            "| --- | --- | --- | --- | --- | --- |",
            *state_lines,
            "",
            "B-v1 仍是 B-INCONCLUSIVE，B-v2 仍是无效测量，B-v2.1 仍是 NO_SIGNAL-PAIRS，B-v2.2 仍是 STOP_CURRENT_COLLECTION。",
            "",
        ]
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
