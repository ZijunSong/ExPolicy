"""B-v2.1 stage A calibration. Confirmation questions are not run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from expolicy.models.openai_compat import OpenAICompatBackend
from expolicy.paper_b.pipeline_v2 import _ThinkingOff, frozen_b1, load_v2_config
from expolicy.paper_b.public_data import tasks_for_split
from expolicy.paper_b.public_exec import PublicBudget, PublicTask
from expolicy.paper_b.v21_parse import fixture_cases, parse_action, query_has_protocol_tag
from expolicy.paper_b.v21_runtime import PARSER_VERSION, run_v21_episode
from expolicy.paper_b.v21_score import SCORER_VERSION, score_answer
from expolicy.r9_lab.io_util import sha256_text, write_json

RUN_ID = "b-v2.1-stage-a"


def select_stage_a(manifest: dict[str, Any]) -> dict[str, list[PublicTask]]:
    """ID order only. Confirmation splits are not selected."""
    dev = tasks_for_split(manifest, "dev")
    bridge = sorted((task for task in dev if task.family == "recovery"), key=lambda task: task.task_id)
    comparison = sorted((task for task in dev if task.family == "retention"), key=lambda task: task.task_id)
    search = bridge[:24] + comparison[:16]
    train = sorted(tasks_for_split(manifest, "train"), key=lambda task: task.task_id)
    return {"evidence": train[:10], "search": search, "b1": search[:12] + search[24:32]}


def run_stage_a(config_path: Path, manifest_path: Path, answers_path: Path, out_dir: Path, *, git: dict[str, Any]) -> dict[str, Any]:
    raw = load_v2_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["answers_for_scorer_only"] = json.loads(answers_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fixtures = fixture_cases()
    _write_jsonl(out_dir / "B_V21_PARSER_FIXTURES.jsonl", fixtures)
    fixture_ok = _fixtures_pass(fixtures)
    groups = select_stage_a(manifest)
    titles = _supporting_titles()
    client = _retriever(raw)
    model = _ThinkingOff(OpenAICompatBackend(raw["model"]["name"], base_url=raw["model"]["base_urls"][0], api_key="EMPTY", timeout=180.0))
    budget = PublicBudget(**{key: raw["budget"][key] for key in ("max_tool_calls", "max_decision_steps", "max_completion_tokens", "max_interventions", "max_prompt_tokens")})
    runtime_hash = sha256_text(PARSER_VERSION + SCORER_VERSION)
    corpus_hash = sha256_text(raw["retriever"]["corpus"] + raw["retriever"]["index"])
    checkpoint = raw["model"]["path"]
    rows: list[dict[str, Any]] = []
    for task in groups["evidence"]:
        evidence = _evidence_for(task, titles.get(task.task_id) or [], client)
        rows.append(_one(task, client, model, budget, "calibration", evidence, [], raw, runtime_hash, corpus_hash, checkpoint))
    b1_ids = {task.task_id for task in groups["b1"]}
    rules = frozen_b1(raw)
    for task in groups["search"]:
        rows.append(_one(task, client, model, budget, "B0", "", [], raw, runtime_hash, corpus_hash, checkpoint))
        if task.task_id in b1_ids:
            rows.append(_one(task, client, model, budget, "B1", "", rules, raw, runtime_hash, corpus_hash, checkpoint))
    _write_jsonl(out_dir / "B_V21_RAW_EPISODES.jsonl", rows)
    _write_jsonl(out_dir / "B_V21_CALIBRATION_RESULTS.jsonl", [row for row in rows if row["condition"] == "calibration"])
    audit = _audit(rows, fixture_ok)
    (out_dir / "B_V21_RESCORE_AUDIT.md").write_text(_rescore_md(rows, audit), encoding="utf-8")
    (out_dir / "B_V2_AUDIT_ADDENDUM.md").write_text(_addendum(), encoding="utf-8")
    decision = _decision(audit)
    (out_dir / "B_V21_DECISION.md").write_text(_decision_md(decision, audit), encoding="utf-8")
    write_json(
        out_dir / "B_V21_RUNTIME_MANIFEST.json",
        {
            "protocol_id": "B-v2.1",
            "run_id": RUN_ID,
            "git": git,
            "call_path": "expolicy.paper_b.v21_runtime.run_v21_episode",
            "previous_call_path": "expolicy.paper_b.public_exec.run_public_episode",
            "parser_version": PARSER_VERSION,
            "scorer_version": SCORER_VERSION,
            "checkpoint": checkpoint,
            "model_name": raw["model"]["name"],
            "base_url": raw["model"]["base_urls"][0],
            "retriever": raw["retriever"]["base_url"],
            "corpus_hash": corpus_hash,
            "runtime_hash": runtime_hash,
            "budget": raw["budget"],
            "confirmation_opened": False,
            "confirmation_exposure": "B_V2_DATA_MANIFEST.json already stored confirmation question text on 2026-10-12. Those questions are not unseen and were not run in B-v2.1.",
            "selected_ids": {name: [task.task_id for task in tasks] for name, tasks in groups.items()},
        },
    )
    write_json(out_dir / "decision.json", decision)
    return decision


def _one(task, client, model, budget, condition, evidence, rules, raw, runtime_hash, corpus_hash, checkpoint) -> dict[str, Any]:
    record = run_v21_episode(
        task,
        client,
        model,
        budget,
        condition=condition,
        run_id=RUN_ID,
        checkpoint=checkpoint,
        runtime_hash=runtime_hash,
        corpus_hash=corpus_hash,
        rules=rules,
        evidence=evidence,
        require_evidence=condition == "calibration",
        temperature=0.0,
        max_new_tokens=int(raw["model"]["max_new_tokens"]),
        top_k=int(raw["retriever"]["top_k"]),
    )
    print(
        f"{condition} {task.task_id} em={record['official_em']} terminal={record['terminal_status']} tools={record['tool_calls']}",
        flush=True,
    )
    return record


def _fixtures_pass(rows: list[dict[str, Any]]) -> bool:
    if len(rows) < 50:
        return False
    for row in rows:
        parsed = parse_action(row["text"])
        if bool(parsed.ok) != bool(row["expect_ok"]):
            return False
        if parsed.ok and parsed.tool == "search" and query_has_protocol_tag(parsed.arguments["query"]):
            return False
        if parsed.ok and row.get("value") and row["tool"] != "find":
            field = "text" if parsed.tool == "answer" else ("document_id" if parsed.tool == "open" else "query")
            if parsed.arguments.get(field) != row["value"]:
                return False
    return True


def _audit(rows: list[dict[str, Any]], fixture_ok: bool) -> dict[str, Any]:
    reread = []
    mismatches = 0
    for row in rows:
        again = score_answer(row["predicted_answer_span"], row["gold_answer"])
        same = again["official_em"] == row["official_em"] and abs(again["official_f1"] - row["official_f1"]) < 1e-9
        mismatches += int(not same)
        reread.append(same)
        if row["condition"] != "calibration" and row["gold_answer"] and row["gold_answer"] in json.dumps(row["steps"]):
            # gold may appear inside a retrieved passage. That is not a prompt-field leak.
            pass
    calibration = [row for row in rows if row["condition"] == "calibration"]
    b0 = [row for row in rows if row["condition"] == "B0"]
    answered = [row for row in rows if row["terminal_status"] == "ANSWERED"]
    leaks = sum(int(row["query_tag_leaks"]) for row in rows)
    answer_saved = all(bool(row["predicted_answer_raw"]) and bool(row["predicted_answer_span"]) for row in answered)
    first = sum(1 for row in b0 if row["first_parse_ok"]) / len(b0) if b0 else 0.0
    em = sum(row["official_em"] for row in calibration)
    open_missing = sum(int(row["open_text_missing"]) for row in rows)
    return {
        "fixture_ok": fixture_ok,
        "fixture_n": 54 if fixture_ok else 0,
        "query_tag_leaks": leaks,
        "answer_saved": answer_saved,
        "rescore_mismatches": mismatches,
        "calibration_em": em,
        "calibration_n": len(calibration),
        "b0_first_parse_rate": first,
        "b0_n": len(b0),
        "b0_em": sum(row["official_em"] for row in b0),
        "open_text_missing": open_missing,
        "episodes": len(rows),
    }


def _decision(audit: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    measurement = (
        audit["fixture_ok"]
        and audit["query_tag_leaks"] == 0
        and audit["answer_saved"]
        and audit["rescore_mismatches"] == 0
        and audit["open_text_missing"] == 0
        and audit["b0_first_parse_rate"] >= 0.98
        and audit.get("calibration_evidence_missing", 0) == 0
    )
    if audit["calibration_em"] < 8:
        reasons.append(
            f"已知证据诊断的官方 EM 是 {audit['calibration_em']}/{audit['calibration_n']}，没有达到建议的 8。"
            "分数保留原值。近义、粒度、事实和格式原因只能来自逐题审计，不能由这段判断自动生成。"
        )
    if not measurement:
        reasons.append("执行器验收没有通过，不能把分数解释成经验规则无效或模型能力结论。")
        if audit["b0_first_parse_rate"] < 0.98:
            reasons.append(f"B0 首次可解析动作率 {audit['b0_first_parse_rate']:.3f}，低于 0.98。")
        if audit["query_tag_leaks"]:
            reasons.append(f"检索词仍有协议标签 {audit['query_tag_leaks']} 次。")
        if not audit["answer_saved"] or audit["rescore_mismatches"]:
            reasons.append("答案没有完整保存，或写盘后不能复判。")
        status = "B-V2.1-BLOCKED-RUNTIME"
    elif audit["b0_em"] < 4:
        reasons.append("执行链可复判，但 40 道正常搜索的官方 EM 仍少于 4。当前设置没有足够成功信号，不开始 192 条重采。")
        status = "B-V2.1-NO_SIGNAL"
    else:
        reasons.append("执行器验收通过，并且正常搜索有可用的正确样本。可以进入训练重采。")
        status = "B-V2.1-BASELINE-PASS"
    return {
        "status": status,
        "measurement_valid": measurement,
        "update_valid": False,
        "coverage_adequate": False,
        "method_status": "not_run",
        "paper_claim_allowed": False,
        "reasons": reasons,
        "audit": audit,
    }


def _supporting_titles() -> dict[str, list[str]]:
    from expolicy.environments.public_search.datasets import load_hotpotqa

    found: dict[str, list[str]] = {}
    for table in load_hotpotqa().values():
        for row in table:
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            facts = meta.get("supporting_facts") or {}
            titles = facts.get("title") if isinstance(facts, dict) else []
            found[str(row.get("id") or row.get("_id"))] = [str(title) for title in titles if title]
    return found


def _evidence_for(task: PublicTask, titles: list[str], client) -> str:
    """Calibration only: paste the Hotpot supporting paragraphs, not a search policy."""
    del client
    text = _raw_supporting_paragraphs(task.question, titles)
    if text:
        return text
    return ""


_RAW_CONTEXT: dict[str, list[tuple[str, str]]] | None = None


def _raw_context_index() -> dict[str, list[tuple[str, str]]]:
    global _RAW_CONTEXT
    if _RAW_CONTEXT is not None:
        return _RAW_CONTEXT
    from expolicy.paper_b.public_data import normalize_question

    index: dict[str, list[tuple[str, str]]] = {}
    for path in (
        Path("/data/ppnm/tmp/HotpotQA_raw/hotpot_dev_distractor_v1.json"),
        Path("/data/ppnm/tmp/HotpotQA_raw/hotpot_train_v1.1.json"),
    ):
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            key = normalize_question(str(row.get("question") or ""))
            if not key or key in index:
                continue
            paragraphs = []
            for item in row.get("context") or []:
                if not isinstance(item, list) or len(item) < 2:
                    continue
                sentences = item[1] if isinstance(item[1], list) else [item[1]]
                body = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())
                if body:
                    paragraphs.append((str(item[0]), body))
            index[key] = paragraphs
    _RAW_CONTEXT = index
    return index


def _raw_supporting_paragraphs(question: str, titles: list[str]) -> str:
    from expolicy.paper_b.public_data import normalize_question

    wanted = {title.casefold() for title in titles if title}
    if not wanted:
        return ""
    chunks = []
    for title, body in _raw_context_index().get(normalize_question(question), []):
        if title.casefold() in wanted:
            chunks.append(f"{title}\n{body}")
    return "\n\n".join(chunks)


def _retriever(raw: dict[str, Any]):
    from expolicy.environments.public_search.retriever import RetrieverClient

    return RetrieverClient(raw["retriever"]["base_url"], timeout=120.0)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _rescore_md(rows: list[dict[str, Any]], audit: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# B-v2.1 复判",
            "",
            f"条数：{len(rows)}",
            f"重读后 EM/F1 不一致：{audit['rescore_mismatches']}",
            f"明确回答的原文与 span 一致：`{audit['answer_saved']}`",
            f"发出的 query 含协议标签：{audit['query_tag_leaks']}",
            "",
        ]
    )


def _addendum() -> str:
    return "\n".join(
        [
            "# B-v2 审计附录",
            "",
            "历史文件保持原样：B-v1 是 B-INCONCLUSIVE，B-v2 开发 Pilot 是 B-V2-NO_GO-CONTENT。",
            "",
            "B-v2 的方法证据状态是 `INVALID_RUNTIME_EVALUATION`。那次 NO_GO 不能用来否定经验内容、不能证明各组等效，也不能用来评价 Qwen3-14B 的 HotpotQA 能力。",
            "",
            "原因是当时的解析器把嵌套 query 标签送进检索器，不认识裸 JSON 动作，并且没有把答案原文写入日志。",
            "",
        ]
    )


def rerun_calibration(config_path: Path, manifest_path: Path, answers_path: Path, out_dir: Path) -> dict[str, Any]:
    """One allowed repair: replace only the 10 evidence episodes. Search rows stay."""
    raw = load_v2_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["answers_for_scorer_only"] = json.loads(answers_path.read_text(encoding="utf-8"))
    prior_path = out_dir / "B_V21_RAW_EPISODES.jsonl"
    prior = [json.loads(line) for line in prior_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [row for row in prior if row["condition"] != "calibration"]
    round0 = out_dir / "B_V21_CALIBRATION_ROUND0.jsonl"
    if not round0.exists():
        _write_jsonl(round0, [row for row in prior if row["condition"] == "calibration"])
    groups = select_stage_a(manifest)
    titles = _supporting_titles()
    client = _retriever(raw)
    model = _ThinkingOff(OpenAICompatBackend(raw["model"]["name"], base_url=raw["model"]["base_urls"][0], api_key="EMPTY", timeout=180.0))
    budget = PublicBudget(**{key: raw["budget"][key] for key in ("max_tool_calls", "max_decision_steps", "max_completion_tokens", "max_interventions", "max_prompt_tokens")})
    runtime_hash = sha256_text(PARSER_VERSION + SCORER_VERSION + "evidence-context-2")
    corpus_hash = sha256_text(raw["retriever"]["corpus"] + raw["retriever"]["index"])
    checkpoint = raw["model"]["path"]
    fresh = []
    for task in groups["evidence"]:
        evidence = _evidence_for(task, titles.get(task.task_id) or [], client)
        fresh.append(_one(task, client, model, budget, "calibration", evidence, [], raw, runtime_hash, corpus_hash, checkpoint))
    rows = fresh + kept
    fixtures = fixture_cases()
    fixture_ok = _fixtures_pass(fixtures)
    _write_jsonl(prior_path, rows)
    _write_jsonl(out_dir / "B_V21_CALIBRATION_RESULTS.jsonl", fresh)
    audit = _audit(rows, fixture_ok)
    audit["repair_round"] = 1
    audit["calibration_evidence_missing"] = sum(1 for row in fresh if int(row.get("evidence_chars") or 0) == 0)
    (out_dir / "B_V21_RESCORE_AUDIT.md").write_text(_rescore_md(rows, audit), encoding="utf-8")
    decision = _decision(audit)
    if audit["calibration_evidence_missing"]:
        decision["measurement_valid"] = False
        decision["status"] = "B-V2.1-BLOCKED-RUNTIME"
        decision["reasons"].append(f"有 {audit['calibration_evidence_missing']} 道校准题没有送入支持段落。")
    decision["reasons"].insert(0, "本轮是允许的一次公共修复：校准提示不再给可抄写的占位答案，并把 Hotpot 支持段落原文送入模型。正常搜索轨迹保持第一轮。")
    (out_dir / "B_V21_DECISION.md").write_text(_decision_md(decision, audit), encoding="utf-8")
    write_json(out_dir / "decision.json", decision)
    return decision


def _decision_md(decision: dict[str, Any], audit: dict[str, Any]) -> str:
    reasons = "\n".join(f"- {reason}" for reason in decision["reasons"])
    return "\n".join(
        [
            "# B-v2.1 决策",
            "",
            f"状态：`{decision['status']}`",
            "",
            f"measurement_valid：`{decision['measurement_valid']}`",
            "",
            f"已知证据 EM：{audit['calibration_em']}/{audit['calibration_n']}",
            "",
            f"B0 首次解析率：{audit['b0_first_parse_rate']:.3f}，B0 EM：{audit['b0_em']}/{audit['b0_n']}",
            "",
            "## 理由",
            "",
            reasons,
            "",
        ]
    )
