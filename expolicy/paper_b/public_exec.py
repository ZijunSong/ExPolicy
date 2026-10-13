"""B-v2 search episode. Rules and the shared model policy run on the same tools."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from expolicy.models.base import GenerationResult
from expolicy.paper_b.features import trigger_matches
from expolicy.paper_b.models import Rule
from expolicy.r9_lab.io_util import sha256_json, sha256_text

PUBLIC_FEATURES: tuple[str, ...] = (
    "search_count",
    "repeat_query_count",
    "repeat_doc_ratio",
    "new_evidence",
    "evidence_count",
    "tool_budget_left",
)
REWRITE_STRATEGIES: tuple[str, ...] = (
    "drop_redundant",
    "entity_plus_relation",
    "bridge_entity",
    "shorten",
)
EXTERNAL_TOOLS = frozenset({"search", "open", "find"})
_STOP = frozenset(
    {"the", "a", "an", "of", "is", "was", "were", "what", "who", "which", "did", "does", "do", "in", "on", "for", "to"}
)
_SPAN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")
_TOOL = re.compile(r"<tool>\s*(\w+)\s*</tool>", re.I)
_ARG = re.compile(r"<arg>\s*(.*?)\s*</arg>", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)

HIDDEN_KEYS = frozenset(
    {"gold", "gold_answer", "supporting_titles", "supporting_facts", "capability", "family", "aliases", "answer"}
)


class SearchClient(Protocol):
    def search(self, query: str, *, topk: int = 5) -> list[dict[str, Any]]: ...

    def open_id(self, document_id: str) -> dict[str, Any] | None: ...


class ChatModel(Protocol):
    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> GenerationResult: ...


@dataclass
class PublicBudget:
    max_tool_calls: int = 8
    max_decision_steps: int = 12
    max_completion_tokens: int = 8192
    max_interventions: int = 2
    max_prompt_tokens: int = 32768


@dataclass
class PublicTask:
    task_id: str
    question: str
    gold_answer: str
    split: str
    source_group: str
    family: str
    entity_key: str
    dataset: str = "hotpotqa"
    aliases: tuple[str, ...] = ()
    cluster: str = ""


@dataclass
class PublicState:
    question: str
    queries: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    opened: list[str] = field(default_factory=list)
    hits: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    last_observation: str = ""
    last_new_doc: int = 0
    opened_text: str = ""
    tool_calls: int = 0
    decision_steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    answer: str = ""
    answered: bool = False
    model_calls: int = 0
    interventions: int = 0
    invalid_actions: int = 0
    thinking_tags: int = 0


@dataclass(frozen=True)
class Snapshot:
    question: str
    queries: tuple[str, ...]
    doc_ids: tuple[str, ...]
    opened: tuple[str, ...]
    hits: tuple[dict[str, Any], ...]
    history: tuple[dict[str, Any], ...]
    last_observation: str
    opened_text: str
    tool_calls: int
    decision_steps: int
    prompt_tokens: int
    completion_tokens: int

    def identity(self) -> str:
        return sha256_json(
            {
                "question": self.question,
                "queries": list(self.queries),
                "doc_ids": list(self.doc_ids),
                "opened": list(self.opened),
                "history": list(self.history),
            }
        )


def observable_features(state: PublicState, budget: PublicBudget) -> dict[str, int]:
    repeat = 0
    if state.queries:
        repeat = len(state.queries) - len({item.casefold() for item in state.queries})
    ratio = 0
    if state.doc_ids:
        ratio = int(round(100 * (1 - len(set(state.doc_ids)) / len(state.doc_ids))))
    return {
        "search_count": len(state.queries),
        "repeat_query_count": repeat,
        "repeat_doc_ratio": ratio,
        "new_evidence": int(state.last_new_doc),
        "evidence_count": len(state.opened),
        "tool_budget_left": max(0, budget.max_tool_calls - state.tool_calls),
    }


def build_messages(question: str, history: Sequence[dict[str, Any]], text_context: str) -> list[dict[str, str]]:
    """Model-visible messages. Gold, family, and supporting titles are not fields here."""
    safe_history = []
    for item in history:
        safe_history.append({key: value for key, value in item.items() if key not in HIDDEN_KEYS})
    user = {"question": question, "history": safe_history}
    if text_context:
        user["text_experience"] = text_context
    return [
        {
            "role": "system",
            "content": (
                "You are a Wikipedia search agent. Ground the answer in retrieved passages. "
                "Emit exactly one XML tool block: search, open, find, or answer."
            ),
        },
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def parse_model_action(text: str) -> tuple[str, dict[str, Any]] | None:
    body = text or ""
    if "<think>" in body:
        body = _THINK.sub("", body)
    tool = _TOOL.search(body)
    if tool:
        name = tool.group(1).lower()
        arg = _ARG.search(body)
        raw = arg.group(1).strip() if arg else ""
        return _action_from_parts(name, raw)
    for name in ("search", "open", "find", "answer"):
        match = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", body, flags=re.S | re.I)
        if match:
            return _action_from_parts(name, match.group(1).strip())
    return None


def _action_from_parts(name: str, raw: str) -> tuple[str, dict[str, Any]] | None:
    payload: dict[str, Any] = {}
    if raw.startswith("{"):
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                payload = loaded
        except json.JSONDecodeError:
            payload = {}
    if name == "search":
        query = str(payload.get("query") or raw)
        return "search", {"query": query}
    if name == "open":
        return "open", {"document_id": str(payload.get("document_id") or raw)}
    if name == "find":
        return "find", {
            "document_id": str(payload.get("document_id") or ""),
            "pattern": str(payload.get("pattern") or raw),
        }
    if name == "answer":
        return "answer", {"answer": str(payload.get("answer") or raw)}
    return None


def rewrite_query(strategy: str, question: str, queries: Sequence[str], opened_text: str) -> str:
    if strategy not in REWRITE_STRATEGIES:
        raise ValueError(f"unregistered rewrite {strategy}")
    last = queries[-1] if queries else question
    if strategy == "drop_redundant":
        words = [word for word in re.findall(r"[A-Za-z0-9]+", last) if word.casefold() not in _STOP]
        return " ".join(words[:12]) or last
    if strategy == "entity_plus_relation":
        entities = _SPAN.findall(question)[:2]
        relation = [word for word in re.findall(r"[A-Za-z]+", question) if word.casefold() not in _STOP][:3]
        return " ".join([*entities, *relation]) or last
    if strategy == "bridge_entity":
        known = set(_SPAN.findall(question))
        for span in _SPAN.findall(opened_text):
            if span not in known:
                return span
        return last
    words = last.split()
    return " ".join(words[:6]) or last


def snapshot_of(state: PublicState) -> Snapshot:
    return Snapshot(
        question=state.question,
        queries=tuple(state.queries),
        doc_ids=tuple(state.doc_ids),
        opened=tuple(state.opened),
        hits=tuple(state.hits),
        history=tuple(state.history),
        last_observation=state.last_observation,
        opened_text=state.opened_text,
        tool_calls=state.tool_calls,
        decision_steps=state.decision_steps,
        prompt_tokens=state.prompt_tokens,
        completion_tokens=state.completion_tokens,
    )


def restore(snapshot: Snapshot) -> PublicState:
    return PublicState(
        question=snapshot.question,
        queries=list(snapshot.queries),
        doc_ids=list(snapshot.doc_ids),
        opened=list(snapshot.opened),
        hits=[dict(item) for item in snapshot.hits],
        history=[dict(item) for item in snapshot.history],
        last_observation=snapshot.last_observation,
        opened_text=snapshot.opened_text,
        tool_calls=snapshot.tool_calls,
        decision_steps=snapshot.decision_steps,
        prompt_tokens=snapshot.prompt_tokens,
        completion_tokens=snapshot.completion_tokens,
    )


@dataclass
class PublicEpisode:
    task_id: str
    answer: str
    success: bool
    tool_calls: int
    decision_steps: int
    prompt_tokens: int
    completion_tokens: int
    model_calls: int
    interventions: int
    invalid_actions: int
    triggered_rules: list[str]
    actions: list[dict[str, Any]]
    request_hashes: list[str]
    text_in_prompt: bool
    thinking_tags: int
    infra_status: str
    snapshots: list[Snapshot]


def run_public_episode(
    task: PublicTask,
    rules: list[Rule],
    client: SearchClient,
    model: ChatModel | None,
    budget: PublicBudget,
    *,
    text_context: str = "",
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    top_k: int = 5,
    initial: Snapshot | None = None,
) -> PublicEpisode:
    state = restore(initial) if initial is not None else PublicState(question=task.question)
    triggered: list[str] = []
    actions: list[dict[str, Any]] = []
    request_hashes: list[str] = []
    text_in_prompt = False
    infra = "ok"
    snapshots: list[Snapshot] = []
    try:
        while not state.answered and state.decision_steps < budget.max_decision_steps:
            if state.completion_tokens >= budget.max_completion_tokens:
                break
            features = observable_features(state, budget)
            applied = _apply_rule(state, rules, client, budget, features, triggered, actions, top_k)
            if applied:
                snapshots.append(snapshot_of(state))
                continue
            if model is None:
                infra = "no_model"
                break
            messages = build_messages(task.question, state.history, text_context)
            blob = json.dumps(messages, ensure_ascii=False)
            text_in_prompt = text_in_prompt or (bool(text_context) and text_context in blob)
            result = model.generate(messages, temperature=temperature, max_new_tokens=max_new_tokens)
            state.model_calls += 1
            state.prompt_tokens += int(result.prompt_tokens)
            state.completion_tokens += int(result.completion_tokens)
            state.decision_steps += 1
            if "<think>" in (result.text or ""):
                state.thinking_tags += 1
            parsed = parse_model_action(result.text)
            request_hashes.append(sha256_text(blob))
            if parsed is None:
                state.invalid_actions += 1
                actions.append({"source": "model", "tool": "", "ok": False})
                continue
            tool, arguments = parsed
            _execute_tool(
                state,
                client,
                tool,
                arguments,
                budget,
                top_k,
                actions,
                source="model",
                features=features,
                history_hash=sha256_json(state.history),
            )
            snapshots.append(snapshot_of(state))
    except Exception as exc:
        infra = f"failed:{type(exc).__name__}"
        if "gold answer" in str(exc):
            raise
    success = _scored(state.answer, task)
    return PublicEpisode(
        task_id=task.task_id,
        answer=state.answer,
        success=success,
        tool_calls=state.tool_calls,
        decision_steps=state.decision_steps,
        prompt_tokens=state.prompt_tokens,
        completion_tokens=state.completion_tokens,
        model_calls=state.model_calls,
        interventions=state.interventions,
        invalid_actions=state.invalid_actions,
        triggered_rules=triggered,
        actions=actions,
        request_hashes=request_hashes,
        text_in_prompt=text_in_prompt,
        thinking_tags=state.thinking_tags,
        infra_status=infra,
        snapshots=snapshots,
    )


def _apply_rule(
    state: PublicState,
    rules: list[Rule],
    client: SearchClient,
    budget: PublicBudget,
    features: dict[str, int],
    triggered: list[str],
    actions: list[dict[str, Any]],
    top_k: int,
) -> bool:
    if state.interventions >= budget.max_interventions:
        return False
    matching = [rule for rule in rules if trigger_matches(rule.trigger, features, max_atoms=3)]
    matching.sort(key=lambda rule: (-len(rule.trigger.get("all") or []), rule.rule_id))
    for rule in matching:
        try:
            tool, arguments = _rule_action(rule, state)
        except ValueError:
            continue
        if tool in EXTERNAL_TOOLS and state.tool_calls >= budget.max_tool_calls:
            continue
        state.interventions += 1
        state.decision_steps += 1
        triggered.append(rule.rule_id)
        _execute_tool(
            state,
            client,
            tool,
            arguments,
            budget,
            top_k,
            actions,
            source=f"rule:{rule.rule_id}",
            features=features,
            history_hash=sha256_json(state.history),
        )
        return True
    return False


def _rule_action(rule: Rule, state: PublicState) -> tuple[str, dict[str, Any]]:
    if rule.action_template == "rewrite_query":
        strategy = str(rule.parameters.get("strategy") or "")
        query = rewrite_query(strategy, state.question, state.queries, state.opened_text)
        return "search", {"query": query, "strategy": strategy}
    if rule.action_template == "open_top":
        for hit in state.hits:
            doc_id = str(hit.get("document_id") or "")
            if doc_id and doc_id not in state.opened:
                return "open", {"document_id": doc_id}
        raise ValueError("no hit to open")
    raise ValueError(f"unregistered template {rule.action_template}")


def _execute_tool(
    state: PublicState,
    client: SearchClient,
    tool: str,
    arguments: dict[str, Any],
    budget: PublicBudget,
    top_k: int,
    actions: list[dict[str, Any]],
    *,
    source: str,
    features: dict[str, Any] | None = None,
    history_hash: str = "",
) -> None:
    if tool == "answer":
        state.answer = str(arguments.get("answer") or "")
        state.answered = True
        actions.append(
            {
                "source": source,
                "tool": "answer",
                "arguments": {},
                "ok": True,
                "features": features or {},
                "history_hash": history_hash,
            }
        )
        return
    if tool not in EXTERNAL_TOOLS:
        state.invalid_actions += 1
        actions.append({"source": source, "tool": tool, "ok": False})
        return
    if state.tool_calls >= budget.max_tool_calls:
        state.invalid_actions += 1
        actions.append({"source": source, "tool": tool, "ok": False, "error": "tool_budget"})
        return
    state.tool_calls += 1
    state.last_new_doc = 0
    if tool == "search":
        query = str(arguments.get("query") or "")
        hits = client.search(query, topk=top_k) or []
        before = set(state.doc_ids)
        state.queries.append(query)
        state.hits = hits
        for hit in hits:
            doc_id = str(hit.get("document_id") or "")
            if doc_id:
                state.doc_ids.append(doc_id)
        state.last_new_doc = int(any(str(hit.get("document_id") or "") not in before for hit in hits))
        observation = json.dumps({"query": query, "n_hits": len(hits), "titles": [hit.get("title") for hit in hits[:3]]}, ensure_ascii=False)
    elif tool == "open":
        doc_id = str(arguments.get("document_id") or "")
        doc = client.open_id(doc_id)
        text = ""
        title = ""
        if doc:
            text = str(doc.get("text") or "")[:1200]
            title = str(doc.get("title") or "")
            if doc_id not in state.opened:
                state.opened.append(doc_id)
                state.last_new_doc = 1
            state.opened_text = (state.opened_text + "\n" + text)[-4000:]
        observation = json.dumps({"document_id": doc_id, "title": title, "text": text[:500]}, ensure_ascii=False)
    else:
        observation = json.dumps({"pattern": arguments.get("pattern"), "document_id": arguments.get("document_id")}, ensure_ascii=False)
    state.last_observation = observation
    state.history.append({"tool": tool, "arguments": {key: value for key, value in arguments.items() if key != "answer"}, "observation": observation[:800]})
    actions.append(
        {
            "source": source,
            "tool": tool,
            "arguments": dict(arguments),
            "ok": True,
            "features": features or {},
            "history_hash": history_hash,
        }
    )


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower()).split())


def _same_answer(prediction: str, gold: str) -> bool:
    return bool(_normalize(gold)) and _normalize(prediction) == _normalize(gold)


def _scored(answer: str, task: PublicTask) -> bool:
    if _same_answer(answer, task.gold_answer):
        return True
    return any(_same_answer(answer, alias) for alias in task.aliases)

