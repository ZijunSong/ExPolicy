"""B-v2.1 training recollection. Old B-v2 traces are not reused."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from expolicy.models.openai_compat import OpenAICompatBackend
from expolicy.paper_b.pipeline_v2 import _ThinkingOff, load_v2_config
from expolicy.paper_b.public_data import tasks_for_split
from expolicy.paper_b.public_exec import PublicBudget
from expolicy.paper_b.v21_runtime import PARSER_VERSION, run_v21_episode
from expolicy.paper_b.v21_score import SCORER_VERSION
from expolicy.r9_lab.io_util import sha256_text

RUN_ID = "b-v2.1-stage-b"


def run_stage_b(config_path: Path, manifest_path: Path, answers_path: Path, out_dir: Path) -> dict[str, Any]:
    raw = load_v2_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["answers_for_scorer_only"] = json.loads(answers_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = sorted(tasks_for_split(manifest, "train"), key=lambda task: task.task_id)
    path = out_dir / "B_V21_RAW_EPISODES.jsonl"
    done = _done(path)
    jobs = [(task, index) for task in tasks for index in (0, 1) if (task.task_id, index) not in done]
    print(f"stage-b jobs={len(jobs)} already={len(done)}", flush=True)
    urls = list(raw["model"]["base_urls"])
    budget = PublicBudget(**{key: raw["budget"][key] for key in ("max_tool_calls", "max_decision_steps", "max_completion_tokens", "max_interventions", "max_prompt_tokens")})
    runtime_hash = sha256_text(PARSER_VERSION + SCORER_VERSION)
    corpus_hash = sha256_text(raw["retriever"]["corpus"] + raw["retriever"]["index"])
    checkpoint = raw["model"]["path"]
    temperature = float(raw["model"]["collect_temperature"])

    _run_pool(jobs, urls, raw, budget, runtime_hash, corpus_hash, checkpoint, temperature, path, supplement=False)
    rows = _read(path)
    report = coverage_report(rows)
    if report["gate"] != "pass":
        extra = _supplement_jobs(rows, tasks, _done(path))
        print(f"supplement jobs={len(extra)}", flush=True)
        _run_pool(extra, urls, raw, budget, runtime_hash, corpus_hash, checkpoint, temperature, path, supplement=True)
        rows = _read(path)
        report = coverage_report(rows)
    (out_dir / "B_V21_PAIR_COVERAGE.md").write_text(_coverage_md(report), encoding="utf-8")
    (out_dir / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("episodes", "official_em", "fork_tasks", "aligned_fragments", "gate")}, ensure_ascii=False), flush=True)
    return report


def _run_pool(jobs, urls, raw, budget, runtime_hash, corpus_hash, checkpoint, temperature, path: Path, *, supplement: bool) -> None:
    write_lock = Lock()

    def work(item: tuple[int, Any]) -> None:
        order, job = item
        task, index, prefix = job if supplement else (job[0], job[1], None)
        url = urls[order % len(urls)]
        from expolicy.environments.public_search.retriever import RetrieverClient

        client = RetrieverClient(raw["retriever"]["base_url"], timeout=120.0)
        model = _ThinkingOff(OpenAICompatBackend(raw["model"]["name"], base_url=url, api_key="EMPTY", timeout=180.0))
        record = run_v21_episode(
            task,
            client,
            model,
            budget,
            condition="B0",
            run_id=RUN_ID,
            checkpoint=checkpoint,
            runtime_hash=runtime_hash,
            corpus_hash=corpus_hash,
            temperature=temperature,
            max_new_tokens=int(raw["model"]["max_new_tokens"]),
            top_k=int(raw["retriever"]["top_k"]),
            prefix_actions=prefix,
        )
        record.update(
            {
                "stage": "collect",
                "continuation": index,
                "supplement": supplement,
                "temperature": temperature,
                "success": record["official_em"] == 1.0,
                "base_url": url,
            }
        )
        with write_lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        print(
            f"collect {task.task_id} c={index} em={record['official_em']} terminal={record['terminal_status']} tools={record['tool_calls']} supplement={supplement}",
            flush=True,
        )

    if jobs:
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            list(pool.map(work, list(enumerate(jobs))))


def _supplement_jobs(rows: list[dict[str, Any]], tasks, done: set[tuple[str, int]]) -> list[tuple[Any, int, list[dict[str, Any]]]]:
    by_id = {task.task_id: task for task in tasks}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("stage") == "collect" and row.get("infra_status") == "ok" and not row.get("supplement"):
            grouped[str(row["task_id"])].append(row)
    forked = set()
    for task_id, items in grouped.items():
        ems = {float(item.get("official_em") or 0) for item in items}
        if 1.0 in ems and 0.0 in ems:
            forked.add(task_id)
    chosen = []
    for task_id in sorted(grouped):
        if task_id in forked or task_id not in by_id:
            continue
        prefix = next((stall_prefix(row) for row in sorted(grouped[task_id], key=lambda item: int(item["continuation"])) if stall_prefix(row)), None)
        if prefix is None:
            continue
        chosen.append((by_id[task_id], prefix))
        if len(chosen) >= 24:
            break
    jobs = []
    for task, prefix in chosen:
        for index in (2, 3):
            if (task.task_id, index) not in done:
                jobs.append((task, index, prefix))
    return jobs


def stall_prefix(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    """First repeated search, while at least two tool calls remain. Does not read gold."""
    seen: list[str] = []
    prefix: list[dict[str, Any]] = []
    for step in row.get("steps") or []:
        if step.get("canonical_action") not in {"search", "open", "find"}:
            continue
        prefix.append(step)
        if step.get("canonical_action") != "search":
            continue
        query = str((step.get("tool_arguments_sent") or {}).get("query") or "").casefold()
        if query and query in seen and len(prefix) <= 6:
            return prefix
        if query:
            seen.append(query)
    return None


def coverage_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("stage") == "collect" and row.get("infra_status") == "ok"]
    base = [row for row in ok if not row.get("supplement")]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base:
        grouped[str(row["task_id"])].append(row)
    fragments = _pairs(grouped, same_prefix=False)
    supplement_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ok:
        if row.get("supplement") and row.get("prefix_hash"):
            supplement_groups[(str(row["task_id"]), str(row["prefix_hash"]))].append(row)
    fragments.extend(_pairs({key[0] + ":" + key[1]: items for key, items in supplement_groups.items()}, same_prefix=True))
    clusters = sorted({item["cluster"] for item in fragments if item.get("cluster")})
    groups = sorted({item["source_group"] for item in fragments if item.get("source_group")})
    recovered = sum(1 for row in ok if row.get("official_em") == 1.0 and _recovered(row))
    gate = len(fragments) >= 12 and len({item["task_id"] for item in fragments}) >= 8 and len(clusters) > 1
    supplemented = any(row.get("supplement") for row in ok)
    return {
        "episodes": len(ok),
        "tasks": len(grouped),
        "official_em": sum(float(row.get("official_em") or 0) for row in base),
        "supplement_episodes": sum(1 for row in ok if row.get("supplement")),
        "fork_tasks": len({item["task_id"] for item in fragments}),
        "aligned_fragments": len(fragments),
        "clusters": clusters,
        "source_groups": groups,
        "natural_recoveries": recovered,
        "snapshot_resume": supplemented,
        "snapshot_note": (
            "补采从第一次重复检索处重放同一工具前缀，再各采样两条续行。没有从头重跑冒充局部重放。"
            if supplemented
            else "同题成败片段不足，且没有可重放的重复检索前缀，因此没有做补采。"
        ),
        "gate": "pass" if gate else "B-V2.1-NO_SIGNAL-PAIRS",
    }


def _pairs(grouped: dict[str, list[dict[str, Any]]], *, same_prefix: bool) -> list[dict[str, Any]]:
    fragments = []
    for items in grouped.values():
        wins = [item for item in items if item.get("official_em") == 1.0]
        losses = [item for item in items if item.get("official_em") == 0.0]
        for win in wins:
            for loss in losses:
                fragments.append(
                    {
                        "task_id": win["task_id"],
                        "cluster": win.get("cluster"),
                        "source_group": win.get("source_group"),
                        "success_continuation": win.get("continuation"),
                        "failure_continuation": loss.get("continuation"),
                        "same_prefix": same_prefix,
                    }
                )
    return fragments


def _recovered(row: dict[str, Any]) -> bool:
    queries = []
    for step in row.get("steps") or []:
        if step.get("canonical_action") == "search":
            queries.append(str((step.get("tool_arguments_sent") or {}).get("query") or "").casefold())
    return len(queries) >= 2 and (len(queries) != len(set(queries)) or any(step.get("canonical_action") == "open" for step in row.get("steps") or []))


def _coverage_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# B-v2.1 训练成败覆盖",
            "",
            f"有效轨迹：{report['episodes']}，其中补采 {report['supplement_episodes']}，从头采集官方 EM：{report['official_em']}",
            "",
            f"同题成败片段：{report['aligned_fragments']}，独立任务：{report['fork_tasks']}",
            "",
            f"来源：{', '.join(report['source_groups']) or '无'}",
            "",
            f"问题结构：{', '.join(report['clusters']) or '无'}",
            "",
            f"自然恢复轨迹：{report['natural_recoveries']}",
            "",
            report["snapshot_note"],
            "",
            f"数据门：`{report['gate']}`",
            "",
        ]
    )


def _done(path: Path) -> set[tuple[str, int]]:
    found = set()
    for row in _read(path):
        if row.get("stage") == "collect":
            found.add((str(row["task_id"]), int(row["continuation"])))
    return found


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
