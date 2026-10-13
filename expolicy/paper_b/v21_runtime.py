"""B-v2.1 episode loop. Parse, execute, score, and keep the raw record."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from expolicy.models.base import GenerationResult
from expolicy.paper_b.public_exec import (
    PublicBudget,
    PublicState,
    PublicTask,
    SearchClient,
    observable_features,
    rewrite_query,
)
from expolicy.paper_b.v21_parse import ParsedAction, parse_action, query_has_protocol_tag
from expolicy.paper_b.v21_score import SCORER_VERSION, score_answer
from expolicy.r9_lab.io_util import sha256_text

PARSER_VERSION = "b-v2.1-parser-1"
SNIPPET_CHARS = 280
OPEN_CHARS = 1200
CORRECTION = (
    "The previous output was not one valid action. Submit exactly one block and nothing else:\n"
    "<tool>search</tool><arg>keywords</arg>\n"
    "or <tool>open</tool><arg>document_id</arg>\n"
    "or <tool>find</tool><arg>{\"document_id\":\"id\",\"pattern\":\"word\"}</arg>\n"
    "or <tool>answer</tool><arg>short span</arg>\n"
    "Do not put tags inside the argument. Do not add a second action."
)


@dataclass
class V21Episode:
    task_id: str
    condition: str
    predicted_answer_raw: str
    predicted_answer_span: str
    official_em: float
    official_f1: float
    terminal_status: str
    prompt_tokens: int
    completion_tokens: int
    tool_calls: int
    decision_steps: int
    model_calls: int
    corrections: int
    rule_fires: int
    first_parse_ok: bool
    query_tag_leaks: int
    open_text_missing: int
    evidence_chars: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)
    infra_status: str = "ok"

    def record(self, *, run_id: str, task: PublicTask, pool_id: str, checkpoint: str, runtime_hash: str, corpus_hash: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "task_id": self.task_id,
            "condition": self.condition,
            "source_group": task.source_group,
            "cluster": task.cluster,
            "family": task.family,
            "experience_pool_id": pool_id,
            "eval_seed": "0",
            "checkpoint": checkpoint,
            "parser_version": PARSER_VERSION,
            "scorer_version": SCORER_VERSION,
            "runtime_hash": runtime_hash,
            "corpus_hash": corpus_hash,
            "predicted_answer_raw": self.predicted_answer_raw,
            "predicted_answer_span": self.predicted_answer_span,
            "gold_answer": task.gold_answer,
            "official_em": self.official_em,
            "official_f1": self.official_f1,
            "terminal_status": self.terminal_status,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tool_calls": self.tool_calls,
            "decision_steps": self.decision_steps,
            "model_calls": self.model_calls,
            "corrections": self.corrections,
            "rule_fires": self.rule_fires,
            "first_parse_ok": self.first_parse_ok,
            "query_tag_leaks": self.query_tag_leaks,
            "open_text_missing": self.open_text_missing,
            "evidence_chars": self.evidence_chars,
            "infra_status": self.infra_status,
            "steps": self.steps,
        }


def system_prompt(*, evidence: str = "") -> str:
    if evidence:
        return (
            "Answer the question using only the evidence in the user message. "
            "Do not search and do not add facts that are not in the evidence. "
            "Output exactly one answer action and no other text. "
            "The argument is the shortest span copied from the evidence."
        )
    return (
        "You are a Wikipedia search agent. Use only the evidence you have been shown. "
        "Emit exactly one action and no other text.\n"
        "<tool>search</tool><arg>keywords</arg>\n"
        "<tool>open</tool><arg>document_id from the search results</arg>\n"
        "<tool>answer</tool><arg>the shortest answer span</arg>\n"
        "The argument must not contain tags."
    )


def run_v21_episode(
    task: PublicTask,
    client: SearchClient,
    model: Any,
    budget: PublicBudget,
    *,
    condition: str,
    run_id: str,
    checkpoint: str,
    runtime_hash: str,
    corpus_hash: str,
    pool_id: str = "",
    rules: list[Any] | None = None,
    evidence: str = "",
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    top_k: int = 5,
    require_evidence: bool = False,
    prefix_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    state = PublicState(question=task.question)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt(evidence=evidence)},
        {
            "role": "user",
            "content": task.question
            if not evidence
            else f"Question: {task.question}\n\nEvidence:\n{evidence}",
        },
    ]
    steps: list[dict[str, Any]] = []
    answer_raw = ""
    answer_span = ""
    terminal = "BUDGET"
    corrections = 0
    rule_fires = 0
    first_parse_ok = False
    first_model = True
    leaks = 0
    open_missing = 0
    infra = "ok"
    if require_evidence and not str(evidence).strip():
        terminal = "EVIDENCE_MISSING"
    prefix_hash = _replay_prefix(state, client, messages, steps, prefix_actions or [], top_k)
    try:
        while terminal != "EVIDENCE_MISSING" and state.decision_steps < budget.max_decision_steps and state.completion_tokens < budget.max_completion_tokens:
            if not evidence and rules:
                fired = _try_rule(state, rules, client, budget, top_k, steps, task)
                if fired:
                    rule_fires += 1
                    if state.answered:
                        answer_raw = state.answer
                        answer_span = state.answer
                        terminal = "ANSWERED"
                        break
                    continue
            result = model.generate(messages, temperature=temperature, max_new_tokens=max_new_tokens)
            state.model_calls += 1
            state.decision_steps += 1
            state.prompt_tokens += int(result.prompt_tokens)
            state.completion_tokens += int(result.completion_tokens)
            parsed = parse_action(result.text)
            if first_model:
                first_parse_ok = parsed.ok
                first_model = False
            if not parsed.ok:
                step = _step_log(len(steps), result, parsed, "", {}, "", "", messages)
                steps.append(step)
                if corrections == 0:
                    corrections += 1
                    messages.append({"role": "assistant", "content": result.text})
                    messages.append({"role": "user", "content": CORRECTION})
                    continue
                terminal = "PARSE_FAILED"
                break
            if parsed.tool == "answer":
                submitted = str(parsed.arguments.get("text") or "")
                if not answer_raw:
                    answer_raw = submitted
                if len(submitted.split()) > 8 and corrections < 1:
                    corrections += 1
                    messages.append({"role": "assistant", "content": result.text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Rewrite only the answer span you already stated. "
                                "Do not add facts or a full sentence. Use one answer action."
                            ),
                        }
                    )
                    steps.append(_step_log(len(steps), result, parsed, "answer", {"text": submitted}, "", submitted, messages))
                    continue
                answer_span = submitted
                state.answer = answer_span
                state.answered = True
                steps.append(_step_log(len(steps), result, parsed, "answer", {"text": answer_span}, "", answer_span, messages))
                terminal = "ANSWERED"
                break
            if parsed.tool in {"search", "open", "find"} and state.tool_calls >= budget.max_tool_calls:
                terminal = "BUDGET"
                steps.append(_step_log(len(steps), result, parsed, parsed.tool, parsed.arguments, "", "tool budget exhausted", messages))
                break
            sent, raw_tool, observation, leaked, missing = _execute(state, client, parsed, top_k)
            leaks += leaked
            open_missing += missing
            state.tool_calls += 1
            state.history.append({"tool": parsed.tool, "arguments": sent, "observation": observation[:1500]})
            messages.append({"role": "assistant", "content": result.text})
            messages.append({"role": "user", "content": observation})
            steps.append(_step_log(len(steps), result, parsed, parsed.tool, sent, raw_tool, observation, messages))
        else:
            if terminal == "BUDGET" and state.answered:
                terminal = "ANSWERED"
    except Exception as exc:
        infra = f"failed:{type(exc).__name__}"
        terminal = "INFRA"
        steps.append({"step": len(steps), "parse_status": "infra", "error": str(exc)[:300]})
    scored = score_answer(answer_span, task.gold_answer, task.aliases)
    episode = V21Episode(
        task_id=task.task_id,
        condition=condition,
        predicted_answer_raw=answer_raw,
        predicted_answer_span=answer_span,
        official_em=scored["official_em"],
        official_f1=scored["official_f1"],
        terminal_status=terminal,
        prompt_tokens=state.prompt_tokens,
        completion_tokens=state.completion_tokens,
        tool_calls=state.tool_calls,
        decision_steps=state.decision_steps,
        model_calls=state.model_calls,
        corrections=corrections,
        rule_fires=rule_fires,
        first_parse_ok=first_parse_ok,
        query_tag_leaks=leaks,
        open_text_missing=open_missing,
        evidence_chars=len(evidence),
        steps=steps,
        infra_status=infra,
    )
    record = episode.record(
        run_id=run_id,
        task=task,
        pool_id=pool_id,
        checkpoint=checkpoint,
        runtime_hash=runtime_hash,
        corpus_hash=corpus_hash,
    )
    record["prefix_hash"] = prefix_hash
    return record


def _replay_prefix(state, client, messages, steps, prefix_actions: list[dict[str, Any]], top_k: int) -> str:
    if not prefix_actions:
        return ""
    for step in prefix_actions:
        tool = str(step.get("canonical_action") or "")
        if tool not in {"search", "open", "find"}:
            continue
        parsed = ParsedAction(True, tool, dict(step.get("tool_arguments_sent") or {}), "replay")
        sent, raw_tool, observation, _leaked, _missing = _execute(state, client, parsed, top_k)
        state.tool_calls += 1
        state.decision_steps += 1
        state.history.append({"tool": tool, "arguments": sent, "observation": observation[:1500]})
        messages.append({"role": "assistant", "content": str(step.get("model_output_raw") or "")})
        messages.append({"role": "user", "content": observation})
        logged = _step_log(len(steps), GenerationResult(text=str(step.get("model_output_raw") or ""), stop_reason="replay"), parsed, tool, sent, raw_tool, observation, messages)
        logged["replayed"] = True
        steps.append(logged)
    payload = [
        {"tool": step.get("canonical_action"), "arguments": step.get("tool_arguments_sent")}
        for step in prefix_actions
        if step.get("canonical_action") in {"search", "open", "find"}
    ]
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _try_rule(state, rules, client, budget, top_k, steps, task) -> bool:
    from expolicy.paper_b.features import trigger_matches

    if state.interventions >= budget.max_interventions or state.tool_calls >= budget.max_tool_calls:
        return False
    features = observable_features(state, budget)
    matching = [rule for rule in rules if trigger_matches(rule.trigger, features, max_atoms=3)]
    matching.sort(key=lambda rule: (-len(rule.trigger.get("all") or []), rule.rule_id))
    for rule in matching:
        if rule.action_template != "rewrite_query":
            continue
        strategy = str(rule.parameters.get("strategy") or "")
        try:
            query = rewrite_query(strategy, state.question, state.queries, state.opened_text)
        except ValueError:
            continue
        if query_has_protocol_tag(query):
            continue
        parsed = ParsedAction(True, "search", {"query": query, "strategy": strategy}, "rule")
        fake = GenerationResult(text="", stop_reason="rule")
        sent, raw_tool, observation, _leaked, _missing = _execute(state, client, parsed, top_k)
        state.tool_calls += 1
        state.interventions += 1
        state.decision_steps += 1
        state.history.append({"tool": "search", "arguments": sent, "observation": observation[:1500]})
        steps.append(_step_log(len(steps), fake, parsed, "search", sent, raw_tool, observation, []))
        return True
    return False


def _execute(state: PublicState, client: SearchClient, parsed: ParsedAction, top_k: int):
    leaks = 0
    missing = 0
    if parsed.tool == "search":
        query = str(parsed.arguments["query"])
        if query_has_protocol_tag(query):
            leaks += 1
        hits = client.search(query, topk=top_k) or []
        state.queries.append(query)
        state.hits = hits
        before = set(state.doc_ids)
        for hit in hits:
            doc_id = str(hit.get("document_id") or "")
            if doc_id:
                state.doc_ids.append(doc_id)
        state.last_new_doc = int(any(str(hit.get("document_id") or "") not in before for hit in hits))
        raw = json.dumps(hits, ensure_ascii=False)[:4000]
        observation = _format_hits(hits)
        return {"query": query}, raw, observation, leaks, missing
    if parsed.tool == "open":
        doc_id = str(parsed.arguments["document_id"])
        doc = client.open_id(doc_id)
        text = ""
        title = ""
        if doc:
            text = str(doc.get("text") or "")
            title = str(doc.get("title") or "")
            if doc_id not in state.opened:
                state.opened.append(doc_id)
                state.last_new_doc = 1
            state.opened_text = (state.opened_text + "\n" + text[:OPEN_CHARS])[-4000:]
        if not text.strip():
            missing += 1
        raw = json.dumps({"document_id": doc_id, "title": title, "text": text[:OPEN_CHARS]}, ensure_ascii=False)
        observation = f"id={doc_id}\ntitle={title}\ntext={text[:OPEN_CHARS]}"
        return {"document_id": doc_id}, raw, observation, leaks, missing
    doc_id = str(parsed.arguments.get("document_id") or "")
    pattern = str(parsed.arguments.get("pattern") or "")
    doc = client.open_id(doc_id) if doc_id else None
    text = str((doc or {}).get("text") or "")
    index = text.lower().find(pattern.lower()) if pattern else -1
    snippet = text[max(0, index - 80) : index + 200] if index >= 0 else ""
    raw = json.dumps({"document_id": doc_id, "pattern": pattern, "index": index, "snippet": snippet}, ensure_ascii=False)
    observation = f"id={doc_id}\npattern={pattern}\nsnippet={snippet}"
    return {"document_id": doc_id, "pattern": pattern}, raw, observation, leaks, missing


def _format_hits(hits: list[dict[str, Any]]) -> str:
    lines = []
    for hit in hits[:5]:
        snippet = str(hit.get("snippet") or hit.get("text") or "")[:SNIPPET_CHARS]
        lines.append(
            f"id={hit.get('document_id')}\ntitle={hit.get('title')}\nsnippet={snippet}"
        )
    return "\n\n".join(lines) if lines else "no hits"


def _step_log(index, result: GenerationResult, parsed: ParsedAction, tool, sent, raw_tool, observation, messages) -> dict[str, Any]:
    request = json.dumps(messages, ensure_ascii=False) if messages else ""
    return {
        "step": index,
        "model_output_raw": result.text,
        "finish_reason": result.stop_reason,
        "request_hash": sha256_text(request),
        "response_hash": sha256_text(result.text or ""),
        "parse_status": "ok" if parsed.ok else parsed.reason,
        "schema_valid": parsed.ok,
        "canonical_action": parsed.tool if parsed.ok else "",
        "tool_arguments_sent": sent,
        "tool_result_raw": raw_tool[:4000] if isinstance(raw_tool, str) else raw_tool,
        "observation_sent_to_model": observation[:2000],
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }


def rescore_record(record: dict[str, Any]) -> dict[str, float]:
    return score_answer(str(record.get("predicted_answer_span") or ""), str(record.get("gold_answer") or ""))
