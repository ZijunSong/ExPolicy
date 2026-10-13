"""B-v2.2 exploratory mechanism run. Episodes go through run_v21_episode only."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

from expolicy.models.openai_compat import OpenAICompatBackend
from expolicy.paper_b.induce_v2 import induce_v2
from expolicy.paper_b.models import Rule, Trace, TracePair, TraceStep
from expolicy.paper_b.pipeline_v2 import _ThinkingOff, frozen_b1
from expolicy.paper_b.public_data import tasks_for_split
from expolicy.paper_b.public_exec import PublicBudget
from expolicy.paper_b.v21_runtime import RUNTIME_VERSION, run_v21_episode
from expolicy.paper_b.v21_score import SCORER_VERSION
from expolicy.paper_b.v22_pool import classify_calibration, write_pool_audit
from expolicy.r9_lab.io_util import sha256_text

RUN_ID = "b-v2.2"
MENU = (
    ("open_top", {}),
    ("rewrite_query", {"strategy": "drop_redundant"}),
    ("rewrite_query", {"strategy": "entity_plus_relation"}),
    ("rewrite_query", {"strategy": "shorten"}),
)


def run_exploratory(config_path: Path, manifest_path: Path, answers_path: Path, pool_path: Path, calibration_path: Path, out_dir: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["answers_for_scorer_only"] = json.loads(answers_path.read_text(encoding="utf-8"))
    questions = {row["task_id"]: row["question"] for row in manifest["tasks"]}
    pool_rows = _read_jsonl(pool_path)
    calibration = _read_jsonl(calibration_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = write_pool_audit(pool_rows, questions, calibration, out_dir)
    states = json.loads((out_dir / "fork_states.json").read_text(encoding="utf-8"))
    ledger = out_dir / "B_V22_COST_LEDGER.jsonl"
    if ledger.exists():
        ledger.unlink()
    aliases_present = any(task.aliases for task in tasks_for_split(manifest, "train")[:20])
    _write_runtime_audit(out_dir, raw, aliases_present, calibration)
    if summary["validated_fragment_count"] == 0 and not states["fork"]:
        return _stop(out_dir, "CATALOG_UNSUPPORTED", "已有成败差异不能表示成注册动作，也没有可重放的早期状态。")
    fork_rows = _run_forks(raw, manifest, states["fork"], out_dir, ledger)
    support = _support(fork_rows)
    (out_dir / "fork_support.json").write_text(json.dumps(support, ensure_ascii=False, indent=2), encoding="utf-8")
    if not any(len(tasks) >= 3 for tasks in support.values()):
        return _stop(
            out_dir,
            "STOP_CURRENT_COLLECTION",
            "历史查询不能映射到注册动作。一次目录分叉之后，没有任何注册动作得到至少 3 个独立任务的成功支持。",
            fork_rows=fork_rows,
            support=support,
        )
    return _induce_and_maybe_continue(raw, manifest, states, fork_rows, support, out_dir, ledger)


def _run_forks(raw, manifest, states, out_dir: Path, ledger: Path) -> list[dict[str, Any]]:
    tasks = {task.task_id: task for task in tasks_for_split(manifest, "train")}
    path = out_dir / "fork_episodes.jsonl"
    done = {(row["task_id"], row.get("fork_action")) for row in _read_jsonl(path)}
    jobs = []
    for state in states:
        task = tasks.get(state["task_id"])
        if task is None:
            continue
        for template, parameters in MENU:
            key = f"{template}:{parameters.get('strategy', '')}"
            if (state["task_id"], key) in done:
                continue
            jobs.append((task, state, template, parameters, key))
    print(f"v22 fork jobs={len(jobs)}", flush=True)
    urls = list(raw["model"]["base_urls"])
    budget = _budget(raw)
    budget.max_interventions = 1
    lock = Lock()

    def work(item: tuple[int, Any]) -> None:
        order, (task, state, template, parameters, key) = item
        from expolicy.environments.public_search.retriever import RetrieverClient

        client = RetrieverClient(raw["retriever"]["base_url"], timeout=120.0)
        model = _ThinkingOff(OpenAICompatBackend(raw["model"]["name"], base_url=urls[order % len(urls)], api_key="EMPTY", timeout=180.0))
        rule = Rule(
            rule_id=f"fork_{key}",
            family="recovery",
            trigger={"all": [{"feature": "search_count", "op": "gte", "value": 1}]},
            action_template=template,
            parameters=dict(parameters),
            validation_record="frozen_menu",
        )
        record = run_v21_episode(
            task,
            client,
            model,
            budget,
            condition="fork",
            run_id=RUN_ID,
            checkpoint=raw["model"]["path"],
            runtime_hash=sha256_text(RUNTIME_VERSION + SCORER_VERSION),
            corpus_hash=sha256_text(raw["retriever"]["corpus"]),
            rules=[rule],
            temperature=0.0,
            max_new_tokens=int(raw["model"]["max_new_tokens"]),
            top_k=int(raw["retriever"]["top_k"]),
            prefix_actions=state["prefix_steps"],
        )
        record["fork_action"] = key
        record["fork_reason"] = state["reason"]
        record["stage"] = "fork"
        with lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            _ledger(ledger, record)
        print(f"fork {task.task_id} {key} em={record['official_em']} terminal={record['terminal_status']} fires={record['rule_fires']}", flush=True)

    if jobs:
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            list(pool.map(work, list(enumerate(jobs))))
    return _read_jsonl(path)


def _support(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    found: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("infra_status") != "ok" or row.get("terminal_status") == "OBSERVATION_MISMATCH":
            continue
        if int(row.get("rule_fires") or 0) < 1 or float(row.get("official_em") or 0) != 1:
            continue
        found[str(row.get("fork_action"))].add(str(row["task_id"]))
    return {key: sorted(values) for key, values in found.items()}


def _induce_and_maybe_continue(raw, manifest, states, fork_rows, support, out_dir: Path, ledger: Path) -> dict[str, Any]:
    traces, pairs = _experience(fork_rows)
    model = _ThinkingOff(OpenAICompatBackend(raw["model"]["name"], base_url=raw["model"]["base_urls"][0], api_key="EMPTY", timeout=180.0))

    def generate(prompt: str, temperature: float) -> str:
        result = model.generate([{"role": "user", "content": prompt}], temperature=temperature, max_new_tokens=800)
        _ledger(ledger, {"stage": "induce", "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens, "tool_calls": 0, "task_id": "", "condition": "induce"})
        return result.text

    proposals = {}
    for condition, mode in (("M", "aligned"), ("B5", "direct"), ("B3", "empty")):
        proposals[condition] = induce_v2(
            mode=mode,
            condition=condition,
            generate_fn=generate,
            temperature=0.0,
            pairs=pairs if mode == "aligned" else [],
            traces=traces if mode == "direct" else [],
            empty_count=len(traces),
            max_candidates=4,
        )
        proposals[condition]["rules"] = [rule.to_dict() for rule in proposals[condition]["rules"][:2]]
        print(f"induce {condition} rules={len(proposals[condition]['rules'])} rejected={len(proposals[condition]['rejected'])}", flush=True)
    (out_dir / "B_V22_PROPOSALS.json").write_text(json.dumps({key: {"rules": value["rules"], "rejected": value["rejected"], "mode": value["mode"]} for key, value in proposals.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    _provenance(out_dir, proposals, support)
    if not proposals["M"]["rules"]:
        return _stop(out_dir, "UPDATE_NOT_ESTABLISHED", "目录分叉产生了跨任务成功，但 M 没有提出可执行规则。不运行空规则的接受或开发评分。", fork_rows=fork_rows, support=support)
    return _validate(raw, manifest, states, proposals, out_dir, ledger, fork_rows, support)


def _validate(raw, manifest, states, proposals, out_dir, ledger, fork_rows, support) -> dict[str, Any]:
    train_tasks = {task.task_id: task for task in tasks_for_split(manifest, "train")}
    val_tasks = tasks_for_split(manifest, "val")
    dev_tasks = tasks_for_split(manifest, "dev")
    baseline = frozen_b1(raw)
    kept = {name: [_rule_from(item) for item in payload["rules"]] for name, payload in proposals.items()}
    train_rows = _eval_sets(
        raw,
        [train_tasks[item["task_id"]] for item in states["validation"] if item["task_id"] in train_tasks],
        {"B1": baseline, "M": baseline + kept["M"], "B3": baseline + kept.get("B3", []), "B5": baseline + kept.get("B5", [])},
        out_dir / "train_validation.jsonl",
        ledger,
        "train_validation",
        prefixes={item["task_id"]: item["prefix_steps"] for item in states["validation"]},
    )
    if _repairs(train_rows, "M", "B1") <= 0:
        return _stop(out_dir, "UPDATE_NOT_ESTABLISHED", "训练侧重放没有出现 M 相对共同 B1 的净修复。不把未接受的规则送去独立接受集。", fork_rows=fork_rows, support=support, train_rows=train_rows)
    acceptance_tasks = _take(val_tasks, "bridge", 16) + _take(val_tasks, "comparison", 8)
    acceptance = _eval_sets(
        raw,
        acceptance_tasks,
        {"B1": baseline, "M": baseline + kept["M"], "B3": baseline + kept.get("B3", []), "B5": baseline + kept.get("B5", [])},
        out_dir / "B_V22_ACCEPTANCE.jsonl",
        ledger,
        "acceptance",
    )
    _write_acceptance(out_dir, acceptance)
    if _repairs(acceptance, "M", "B1") <= 0:
        return _stop(out_dir, "ACCEPTANCE_FAIL", "独立接受集上 M 没有相对 B1 的净修复。", fork_rows=fork_rows, support=support, train_rows=train_rows, acceptance=acceptance)
    dev_set = _take(dev_tasks, "bridge", 20) + _take(dev_tasks, "comparison", 10)
    dev = _eval_sets(
        raw,
        dev_set,
        {"B0": [], "B1": baseline, "M": baseline + kept["M"], "B3": baseline + kept.get("B3", []), "B5": baseline + kept.get("B5", [])},
        out_dir / "B_V22_DEV_RESULTS.jsonl",
        ledger,
        "dev",
    )
    return _finish_dev(out_dir, dev, fork_rows, support, train_rows, acceptance)


def _eval_sets(raw, tasks, rule_sets, path: Path, ledger: Path, stage: str, prefixes: dict[str, list] | None = None) -> list[dict[str, Any]]:
    done = {(row.get("task_id"), row.get("condition"), row.get("stage")) for row in _read_jsonl(path)}
    jobs = [(task, name, rules) for name, rules in rule_sets.items() for task in tasks if (task.task_id, name, stage) not in done]
    print(f"v22 {stage} jobs={len(jobs)}", flush=True)
    urls = list(raw["model"]["base_urls"])
    budget = _budget(raw)
    lock = Lock()

    def work(item: tuple[int, Any]) -> None:
        order, (task, name, rules) = item
        from expolicy.environments.public_search.retriever import RetrieverClient

        client = RetrieverClient(raw["retriever"]["base_url"], timeout=120.0)
        model = _ThinkingOff(OpenAICompatBackend(raw["model"]["name"], base_url=urls[order % len(urls)], api_key="EMPTY", timeout=180.0))
        record = run_v21_episode(
            task,
            client,
            model,
            budget,
            condition=name,
            run_id=RUN_ID,
            checkpoint=raw["model"]["path"],
            runtime_hash=sha256_text(RUNTIME_VERSION),
            corpus_hash=sha256_text(raw["retriever"]["corpus"]),
            rules=rules,
            temperature=0.0,
            max_new_tokens=int(raw["model"]["max_new_tokens"]),
            top_k=int(raw["retriever"]["top_k"]),
            prefix_actions=(prefixes or {}).get(task.task_id),
        )
        record["stage"] = stage
        with lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            _ledger(ledger, record)
        print(f"{stage} {name} {task.task_id} em={record['official_em']} fires={record['rule_fires']}", flush=True)

    if jobs:
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            list(pool.map(work, list(enumerate(jobs))))
    return _read_jsonl(path)


def _experience(rows: list[dict[str, Any]]) -> tuple[list[Trace], list[TracePair]]:
    traces = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("terminal_status") == "OBSERVATION_MISMATCH" or int(row.get("rule_fires") or 0) < 1:
            continue
        forced = next((step for step in row.get("steps") or [] if str(step.get("source") or "").startswith("rule:")), None)
        if forced is None:
            continue
        action, _, strategy = str(row.get("fork_action") or "").partition(":")
        step = TraceStep(
            features_before=dict(forced.get("features_before") or {}),
            action_template=action or "rewrite_query",
            parameters={"strategy": strategy} if strategy else {},
            observation_redacted="rule_observation",
            history_hash=str(row.get("prefix_hash") or ""),
        )
        trace = Trace(
            trace_id=sha256_text(f"{row['task_id']}:{row.get('fork_action')}")[:12],
            task_id=row["task_id"],
            source_group=str(row.get("source_group") or ""),
            split="train",
            family=str(row.get("family") or ""),
            continuation=str(row.get("fork_action")),
            success=float(row.get("official_em") or 0) == 1,
            steps=[step],
        )
        traces.append(trace)
        by_task[row["task_id"]].append(row)
    pairs = []
    for task_id, items in by_task.items():
        wins = [item for item in items if float(item.get("official_em") or 0) == 1]
        losses = [item for item in items if float(item.get("official_em") or 0) == 0]
        if not wins or not losses:
            continue
        win, loss = wins[0], losses[0]
        win_action, _, win_strategy = str(win.get("fork_action")).partition(":")
        loss_action, _, _loss_strategy = str(loss.get("fork_action")).partition(":")
        forced = next(step for step in win["steps"] if str(step.get("source") or "").startswith("rule:"))
        pairs.append(
            TracePair(
                pair_id=sha256_text(task_id + win_action + loss_action)[:12],
                alignment="same_prefix",
                quality=1.0,
                task_id=task_id,
                source_group=str(win.get("source_group") or ""),
                family=str(win.get("family") or ""),
                diverge_index=0,
                features_before=dict(forced.get("features_before") or {}),
                success_action=win_action,
                success_params={"strategy": win_strategy} if win_strategy else {},
                failure_action=loss_action,
                success_trace_id=sha256_text(win["task_id"] + str(win.get("fork_action")))[:12],
                failure_trace_id=sha256_text(loss["task_id"] + str(loss.get("fork_action")))[:12],
                prefix_hash=str(win.get("prefix_hash") or ""),
                alignment_basis="history",
            )
        )
    return traces, pairs


def _repairs(rows: list[dict[str, Any]], left: str, right: str) -> int:
    by = defaultdict(dict)
    for row in rows:
        if row.get("infra_status") == "ok":
            by[row["task_id"]][row["condition"]] = float(row.get("official_em") or 0)
    repairs = 0
    for item in by.values():
        if left in item and right in item and item[left] == 1 and item[right] == 0:
            repairs += 1
    return repairs


def _take(tasks, group: str, count: int):
    chosen = sorted((task for task in tasks if task.source_group == group), key=lambda task: task.task_id)
    return chosen[:count]


def _rule_from(item: dict[str, Any]) -> Rule:
    return Rule(
        rule_id=item["rule_id"],
        family=item.get("family") or "recovery",
        trigger=item["trigger"],
        action_template=item["action_template"],
        parameters=dict(item.get("parameters") or {}),
        validation_record="v22",
    )


def _write_runtime_audit(out_dir: Path, raw: dict[str, Any], aliases_present: bool, calibration: list[dict[str, Any]]) -> None:
    labels = Counterish(calibration)
    text = "\n".join(
        [
            "# B-v2.2 规则运行审计",
            "",
            f"运行版本：`{RUNTIME_VERSION}`。新的重放、候选和评估都调用 `run_v21_episode`。",
            "",
            "规则动作和模型动作共用 observation 写入。`open_top` 会打开当前尚未打开的第一条 hit；没有 hit 时回退到基线，并记录 skip。",
            "",
            "请求 hash 在追加 observation 之前计算。快照重放会累计已消耗的决策和 token，观察不一致时停止并记 OBSERVATION_MISMATCH。",
            "",
            f"抽查的训练任务 aliases 是否非空：`{aliases_present}`。主分只使用记录里的 gold。",
            "",
            f"模型：`{raw['model']['name']}`，检索：`{raw['retriever']['base_url']}`。",
            "",
            "已知证据校准的逐题标签：",
            "",
            *[f"- {task_id}: {label}" for task_id, label in labels],
            "",
        ]
    )
    (out_dir / "B_V22_RULE_RUNTIME_AUDIT.md").write_text(text, encoding="utf-8")


def Counterish(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(str(row["task_id"]), classify_calibration(row)) for row in rows]


def _provenance(out_dir: Path, proposals: dict[str, Any], support: dict[str, list[str]]) -> None:
    rows = []
    for condition, payload in proposals.items():
        for rule in payload["rules"]:
            rows.append({"condition": condition, "rule": rule, "fork_support": support})
    _write_jsonl(out_dir / "B_V22_RULE_PROVENANCE.jsonl", rows)


def _write_acceptance(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary = {
        "episodes": len(rows),
        "repairs_m_vs_b1": _repairs(rows, "M", "B1"),
        "repairs_m_vs_b3": _repairs(rows, "M", "B3"),
        "repairs_m_vs_b5": _repairs(rows, "M", "B5"),
        "harms_m_vs_b1": _repairs(rows, "B1", "M"),
    }
    (out_dir / "B_V22_ACCEPTANCE.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _finish_dev(out_dir, dev, fork_rows, support, train_rows, acceptance) -> dict[str, Any]:
    m_b1 = _repairs(dev, "M", "B1")
    m_b5 = _repairs(dev, "M", "B5")
    b5_b1 = _repairs(dev, "B5", "B1")
    if m_b1 > 0 and m_b5 > 0 and b5_b1 < m_b1:
        status = "exploratory_continue"
        reason = "开发筛查里 M 相对 B1 有净修复，而且 B5 没有完整解释这份增量。这仍是小样本探索，不是正式 GO。"
    elif m_b1 > 0 and m_b5 <= 0:
        status = "DIRECT_SUFFICIENT"
        reason = "M 的开发修复没有超过公平的直接归纳 B5。"
    else:
        status = "stop"
        reason = "五组开发筛查没有出现可核验的 M 净修复。"
    return _stop(out_dir, status, reason, fork_rows=fork_rows, support=support, train_rows=train_rows, acceptance=acceptance, dev=dev)


def _stop(out_dir: Path, status: str, reason: str, **payload) -> dict[str, Any]:
    for name in ("B_V22_PROPOSALS.json", "B_V22_ACCEPTANCE.json"):
        path = out_dir / name
        if not path.exists():
            path.write_text("{}\n", encoding="utf-8")
    for name in ("B_V22_RULE_PROVENANCE.jsonl", "B_V22_DEV_RESULTS.jsonl", "B_V22_COST_LEDGER.jsonl"):
        path = out_dir / name
        if not path.exists():
            path.write_text("", encoding="utf-8")
    decision = {
        "protocol": "B-v2.2-exploratory",
        "status": status,
        "paper_claim_allowed": False,
        "rules_updated": status == "exploratory_continue",
        "model_saw_rule_observation": True,
        "reason": reason,
        "unrun": _unrun(status),
        "support": payload.get("support") or {},
        "fork_episodes": len(payload.get("fork_rows") or []),
        "train_episodes": len(payload.get("train_rows") or []),
        "acceptance_episodes": len(payload.get("acceptance") or []),
        "dev_episodes": len(payload.get("dev") or []),
    }
    (out_dir / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "B_V22_DECISION.md").write_text(_decision_md(decision), encoding="utf-8")
    return decision


def _unrun(status: str) -> list[str]:
    if status in {"CATALOG_UNSUPPORTED", "STOP_CURRENT_COLLECTION", "blocked"}:
        return ["induction", "train_validation", "acceptance", "dev_screen", "seven_group", "confirmation"]
    if status == "UPDATE_NOT_ESTABLISHED":
        return ["acceptance", "dev_screen", "seven_group", "confirmation"]
    if status == "ACCEPTANCE_FAIL":
        return ["dev_screen", "seven_group", "confirmation"]
    return ["seven_group", "confirmation"]


def _decision_md(decision: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# B-v2.2 决策",
            "",
            f"状态：`{decision['status']}`",
            "",
            decision["reason"],
            "",
            f"规则是否实际更新：`{decision['rules_updated']}`",
            "",
            "模型是否能看到规则观察：运行时修复后可以。离线测试检查下一次请求包含新的 query 或正文。",
            "",
            f"目录分叉条数：{decision['fork_episodes']}",
            "",
            f"未运行阶段：{', '.join(decision['unrun'])}",
            "",
            "B-v1 仍是 B-INCONCLUSIVE。B-v2 的 NO_GO 仍是 INVALID_RUNTIME_EVALUATION。B-v2.1 仍是 B-V2.1-NO_SIGNAL-PAIRS，没有改写成通过。",
            "",
            "本轮不声称论文方法成立。",
            "",
        ]
    )


def _budget(raw: dict[str, Any]) -> PublicBudget:
    return PublicBudget(**{key: raw["budget"][key] for key in ("max_tool_calls", "max_decision_steps", "max_completion_tokens", "max_interventions", "max_prompt_tokens")})


def _ledger(path: Path, row: dict[str, Any]) -> None:
    payload = {
        "stage": row.get("stage"),
        "task_id": row.get("task_id"),
        "condition": row.get("condition"),
        "fork_action": row.get("fork_action"),
        "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "tool_calls": row.get("tool_calls"),
        "official_em": row.get("official_em"),
        "terminal_status": row.get("terminal_status"),
        "rule_fires": row.get("rule_fires"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
