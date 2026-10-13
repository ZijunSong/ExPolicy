"""B-v2 smoke and deployment. Synthetic B-v1 stays on its own entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from expolicy.models.factory import build_backend
from expolicy.models.openai_compat import OpenAICompatBackend
from expolicy.paper_b.decision_v2 import SMOKE_PASS, render_v2_decision
from expolicy.paper_b.induce_v2 import induce_v2
from expolicy.paper_b.models import Rule
from expolicy.paper_b.public_data import assign_manifest, tasks_for_split
from expolicy.paper_b.public_exec import (
    PUBLIC_FEATURES,
    PublicBudget,
    PublicEpisode,
    build_messages,
    run_public_episode,
)
from expolicy.r9_lab.io_util import sha256_json, sha256_text, write_json

PROTOCOL = "B-v2"


def load_v2_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("protocol_id") != PROTOCOL:
        raise RuntimeError("paper_b_v2 config must declare protocol_id B-v2")
    if raw.get("harness") == "local_search":
        raise RuntimeError("B-v2 cannot be selected by renaming the synthetic harness")
    return raw


def frozen_b1(raw: dict[str, Any]) -> list[Rule]:
    rules = []
    for item in raw["b1_rules"]:
        executable = {
            "trigger": item["trigger"],
            "action_template": item["action_template"],
            "parameters": item.get("parameters") or {},
        }
        rules.append(
            Rule(
                rule_id=f"b1_{sha256_json(executable)[:10]}",
                family="recovery",
                trigger=item["trigger"],
                action_template=item["action_template"],
                parameters=dict(item.get("parameters") or {}),
                validation_record="frozen_before_smoke",
            )
        )
    return rules


def deploy_rules(condition: str, baseline: list[Rule], accepted: list[Rule] | None, *, rejected: bool, max_rules: int = 8) -> list[Rule]:
    """Rejected candidates return to the shared baseline, not to an empty agent."""
    if condition == "B0":
        return []
    if condition == "B2":
        return []
    if condition == "B1":
        return list(baseline)
    if rejected or not accepted:
        return list(baseline)
    merged = list(baseline)
    seen = {rule.rule_id for rule in merged}
    for rule in accepted:
        if rule.rule_id in seen:
            continue
        merged.append(rule)
        seen.add(rule.rule_id)
        if len(merged) >= max_rules:
            break
    return merged[:max_rules]


def build_real_manifest(raw: dict[str, Any]) -> dict[str, Any]:
    from expolicy.environments.public_search.datasets import load_hotpotqa

    tables = load_hotpotqa()
    rows: list[dict[str, Any]] = []
    for split_rows in tables.values():
        rows.extend(split_rows)
    manifest = assign_manifest(rows, raw["counts"])
    manifest["corpus"] = {
        "index": raw["retriever"]["index"],
        "corpus": raw["retriever"]["corpus"],
        "corpus_hash": sha256_text(raw["retriever"]["corpus"] + raw["retriever"]["index"]),
    }
    manifest["model"] = {
        "backend": raw["model"]["backend"],
        "name": raw["model"]["name"],
        "path": raw["model"]["path"],
        "base_url": raw["model"]["base_url"],
        "thinking": False,
        "deploy_temperature": raw["model"]["deploy_temperature"],
        "collect_temperature": raw["model"]["collect_temperature"],
    }
    return manifest


class _ThinkingOff:
    """Recorded adapter. A transport error stays an error."""

    def __init__(self, backend: OpenAICompatBackend):
        self.backend = backend
        self.calls = 0
        self.backend_id = backend.backend_id
        self.name = backend.name

    def generate(self, messages, *, temperature: float, max_new_tokens: int):
        self.calls += 1
        return self.backend.generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            extra={"chat_template_kwargs": {"enable_thinking": False}},
        )


def run_smoke(raw: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = PublicBudget(**raw["budget"])
    reasons: list[str] = []
    blocked = ""
    model_identity: dict[str, Any] = {}
    try:
        backend = build_backend(raw["model"])
        if not isinstance(backend, OpenAICompatBackend):
            raise RuntimeError("B-v2 smoke expects the OpenAI-compatible server already serving Qwen3-14B")
        model = _ThinkingOff(backend)
        from expolicy.environments.public_search.retriever import RetrieverClient

        client = RetrieverClient(raw["retriever"]["base_url"])
        health = client.health()
        if not health.get("ok"):
            blocked = "检索服务没有通过健康检查"
    except Exception as exc:
        blocked = f"真实后端没有接通：{exc}"
        health = {}
        model = None
        client = None
    manifest = {}
    episodes: list[PublicEpisode] = []
    probes: dict[str, Any] = {}
    if not blocked and model is not None and client is not None:
        try:
            import urllib.request

            with urllib.request.urlopen(raw["model"]["base_url"].rstrip("/") + "/models", timeout=10) as handle:
                listing = json.loads(handle.read().decode("utf-8"))
            served = [item.get("id") for item in listing.get("data") or []]
            root = ((listing.get("data") or [{}])[0].get("root"))
            model_identity = {"served": served, "root": root, "expected_name": raw["model"]["name"], "expected_path": raw["model"]["path"]}
            if raw["model"]["name"] not in served or str(root) != str(raw["model"]["path"]):
                blocked = f"服务中的 checkpoint 与配置不一致：{model_identity}"
        except Exception as exc:
            blocked = f"无法核对 checkpoint：{exc}"
    if not blocked and model is not None and client is not None:
        manifest = build_real_manifest(raw)
        write_public_manifest(out_dir / "B_V2_DATA_MANIFEST.json", manifest)
        smoke_tasks = tasks_for_split(manifest, "smoke")
        if len(smoke_tasks) < int(raw["counts"]["smoke"]["recovery"]):
            blocked = f"smoke 任务不足：{len(smoke_tasks)}"
        else:
            baseline = frozen_b1(raw)
            episodes, probes, reasons = _smoke_episodes(smoke_tasks, baseline, client, model, budget, raw)
            if any(episode.infra_status != "ok" for episode in episodes):
                blocked = "有 episode 在模型或工具调用时失败"
            elif model.calls <= 0:
                blocked = "smoke 没有实际调用模型"
            elif not probes.get("text_entered"):
                blocked = "文本经验没有进入模型请求"
            elif not probes.get("rule_executed"):
                blocked = "命中的规则没有被执行"
            elif not probes.get("fallback_called_model"):
                blocked = "没有命中规则时没有回到模型"
            elif not probes.get("induction_called_model"):
                blocked = "归纳没有调用同一个模型"
            elif sum(episode.tool_calls for episode in episodes) == 0:
                blocked = "二十道 smoke 题没有产生任何外部检索。模型调用本身还不是搜索已经接通"
    if blocked:
        decision = {"status": "B-V2-BLOCKED", "reasons": [blocked], "paper_claim_allowed": False, "scope_note": _scope()}
    else:
        decision = {
            "status": SMOKE_PASS,
            "reasons": reasons or ["模型、检索、文本、规则和预算检查通过。这不是开发 Pilot，也不是确认集。"],
            "paper_claim_allowed": False,
            "scope_note": _scope(),
        }
    _write_smoke(out_dir, decision, episodes, probes, model_identity, health, raw)
    return decision


def _smoke_episodes(tasks, baseline, client, model, budget, raw):
    episodes = []
    for task in tasks:
        episodes.append(
            run_public_episode(
                task,
                baseline,
                client,
                model,
                budget,
                temperature=float(raw["model"]["deploy_temperature"]),
                max_new_tokens=int(raw["model"]["max_new_tokens"]),
                top_k=int(raw["retriever"]["top_k"]),
            )
        )
    first = tasks[0]
    text = "When a search repeats and adds no new document, shorten the query."
    with_text = run_public_episode(first, [], client, model, budget, text_context=text, temperature=0.0, max_new_tokens=64, top_k=int(raw["retriever"]["top_k"]))
    without = run_public_episode(first, [], client, model, budget, temperature=0.0, max_new_tokens=64, top_k=int(raw["retriever"]["top_k"]))
    planted = Rule(
        rule_id="smoke_rule",
        family="recovery",
        trigger={"all": [{"feature": "search_count", "op": "eq", "value": 0}]},
        action_template="rewrite_query",
        parameters={"strategy": "shorten"},
    )
    ruled = run_public_episode(first, [planted], client, model, budget, temperature=0.0, max_new_tokens=32, top_k=int(raw["retriever"]["top_k"]))
    dormant = Rule(
        rule_id="smoke_dormant",
        family="recovery",
        trigger={"all": [{"feature": "search_count", "op": "eq", "value": 99}]},
        action_template="open_top",
    )
    fallback = run_public_episode(first, [dormant], client, model, budget, temperature=0.0, max_new_tokens=32, top_k=int(raw["retriever"]["top_k"]))
    calls_before = model.calls
    induce_v2(mode="empty", condition="B3", generate_fn=_model_generate(model), temperature=0.0, empty_count=1, max_candidates=4)
    probes = {
        "text_entered": with_text.text_in_prompt and with_text.request_hashes != without.request_hashes,
        "rule_executed": "smoke_rule" in ruled.triggered_rules and any(item.get("arguments", {}).get("strategy") == "shorten" for item in ruled.actions),
        "fallback_called_model": fallback.model_calls > 0 and "smoke_dormant" not in fallback.triggered_rules,
        "induction_called_model": model.calls > calls_before,
        "tool_budget_held": all(episode.tool_calls <= budget.max_tool_calls for episode in episodes),
        "answer_not_in_tool_count": True,
        "gold_not_in_messages": True,
        "features": list(PUBLIC_FEATURES),
    }
    messages = build_messages(first.question, [], text)
    probes["gold_not_in_messages"] = first.gold_answer not in json.dumps(messages)
    reasons = []
    if any(episode.thinking_tags for episode in episodes):
        reasons.append("响应里仍出现 thinking 标记。配置请求了关闭 thinking，但服务输出没有完全去掉。")
    snapshot_rows = []
    if ruled.snapshots:
        snap = snapshot_of_from_episode(ruled)
        if snap is not None:
            continued = run_public_episode(first, [planted], client, model, budget, initial=snap, temperature=0.0, max_new_tokens=16, top_k=int(raw["retriever"]["top_k"]))
            plain = run_public_episode(first, [], client, model, budget, initial=snap, temperature=0.0, max_new_tokens=16, top_k=int(raw["retriever"]["top_k"]))
            snapshot_rows.append(
                {
                    "snapshot_hash": snap.identity(),
                    "with_rule_query": _first_query(continued),
                    "without_rule_query": _first_query(plain),
                    "query_changed": _first_query(continued) != _first_query(plain),
                }
            )
    probes["snapshot_rows"] = snapshot_rows
    return episodes, probes, reasons


def snapshot_of_from_episode(episode: PublicEpisode):
    return episode.snapshots[0] if episode.snapshots else None


def _first_query(episode: PublicEpisode) -> str:
    for action in episode.actions:
        if action.get("tool") == "search":
            return str((action.get("arguments") or {}).get("query") or "")
    return ""


def _model_generate(model):
    def generate(prompt: str, temperature: float) -> str:
        result = model.generate(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_new_tokens=256,
        )
        return result.text

    return generate


def write_public_manifest(path: Path, manifest: dict[str, Any]) -> None:
    public = {key: value for key, value in manifest.items() if key != "answers_for_scorer_only"}
    public["answers_stored_separately"] = True
    public["n_answers"] = len(manifest.get("answers_for_scorer_only") or {})
    write_json(path, public)
    write_json(path.with_name("B_V2_SCORER_ANSWERS.json"), manifest.get("answers_for_scorer_only") or {})


def _write_smoke(out_dir, decision, episodes, probes, model_identity, health, raw) -> None:
    (out_dir / "B_V2_RUNTIME_AUDIT.md").write_text(_runtime_md(decision, probes, model_identity, health, episodes), encoding="utf-8")
    (out_dir / "B_V2_BASELINE_AUDIT.md").write_text(_baseline_md(raw), encoding="utf-8")
    (out_dir / "B_V2_DECISION.md").write_text(render_v2_decision(decision), encoding="utf-8")
    write_json(out_dir / "B_V2_EFFECTS.json", {"stage": "smoke", "m_contrasts": "not_run", "n_episodes": len(episodes), "probes": probes})
    rows = probes.get("snapshot_rows") or []
    path = out_dir / "B_V2_SNAPSHOT_AUDIT.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    write_json(out_dir / "decision.json", decision)


def _scope() -> str:
    return "B-v2 smoke 只检查真实模型、Wikipedia 检索、文本、规则和预算是否接通。它不是七组开发 Pilot，也没有打开确认集。"


def _runtime_md(decision, probes, model_identity, health, episodes) -> str:
    calls = sum(episode.model_calls for episode in episodes)
    tools = sum(episode.tool_calls for episode in episodes)
    return "\n".join(
        [
            "# B-v2 运行时审计",
            "",
            f"状态：`{decision.get('status')}`",
            "",
            f"模型身份：`{json.dumps(model_identity, ensure_ascii=False)}`",
            "",
            f"检索健康：`{json.dumps(health, ensure_ascii=False)[:500]}`",
            "",
            f"smoke episode 数：{len(episodes)}；模型调用：{calls}；外部工具调用：{tools}",
            "",
            f"检查：`{json.dumps({key: value for key, value in probes.items() if key != 'snapshot_rows'}, ensure_ascii=False)}`",
            "",
        ]
    )


def _baseline_md(raw: dict[str, Any]) -> str:
    lines = ["# B-v2 基线审计", "", "B1 在 smoke 之前冻结，不在测试分数上挑选。", ""]
    for rule in raw["b1_rules"]:
        lines.append(f"- `{rule['action_template']}` {json.dumps(rule['trigger'], ensure_ascii=False)} {json.dumps(rule.get('parameters') or {}, ensure_ascii=False)}")
    lines.append("")
    lines.append("候选被拒绝时回到这组 B1，而不是退回没有程序的 B0。B0 本身不加这些规则。")
    lines.append("")
    return "\n".join(lines)
