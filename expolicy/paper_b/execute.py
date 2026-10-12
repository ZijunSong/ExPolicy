"""Execute registered templates under a shared budget. Rules cannot add tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from expolicy.paper_b.constants import (
    STATUS_INFRA,
    STATUS_OK,
    TEMPLATE_BASE_ANSWER,
    TEMPLATE_BASE_SEARCH,
    TEMPLATE_REWRITE,
    TEMPLATE_SEARCH,
)
from expolicy.paper_b.features import (
    attribute_phrase,
    comparison_direction,
    contains_term,
    exact_match,
    extract_base_answer,
    extract_slot,
    infer_attribute_kind,
    is_year,
    observable_features,
    parse_entities,
    redact_observation,
    trigger_matches,
)
from expolicy.paper_b.models import Budget, Document, Episode, Rule, Task, TraceStep
from expolicy.paper_b.registry import RuntimeRegistry
from expolicy.r9_lab.io_util import sha256_text


class IllegalAction(RuntimeError):
    pass


Policy = Callable[["AgentState"], tuple[str, dict[str, Any]]]


@dataclass
class AgentState:
    question: str
    evidence: dict[str, str] = field(default_factory=dict)
    search_count: int = 0
    step_count: int = 0
    last_observation: str = ""
    last_query: str = ""
    tool_calls: int = 0
    tokens: int = 0
    interventions: int = 0
    answer: str = ""
    answered: bool = False
    infra_status: str = STATUS_OK

    def features(self) -> dict[str, Any]:
        return observable_features(
            self.question,
            evidence=self.evidence,
            search_count=self.search_count,
            step_count=self.step_count,
            last_observation=self.last_observation,
        )


class SearchEnv:
    """Word-boundary search over the task corpus. The corpus is not a feature."""

    def __init__(self, task: Task):
        self._docs: tuple[Document, ...] = task.corpus

    def search(self, query: str) -> str:
        for doc in self._docs:
            if all(contains_term(query, term) for term in doc.terms):
                return doc.text
        return ""


def token_cost(query: str, observation: str) -> int:
    return 8 + len(query) // 4 + len(observation) // 4


def base_policy(state: AgentState) -> tuple[str, dict[str, Any]]:
    if state.search_count == 0:
        return TEMPLATE_BASE_SEARCH, {"query": state.question}
    extracted = extract_base_answer(state.last_observation)
    if extracted:
        return TEMPLATE_BASE_ANSWER, {"answer": extracted}
    return TEMPLATE_BASE_ANSWER, {"answer": "unknown"}


def exploration_policy(state: AgentState) -> tuple[str, dict[str, Any]]:
    """Preregistered second continuation. It is data collection, not the inducer."""
    features = state.features()
    if features["has_comparison_cue"] == 1 and features["n_entities"] >= 2 and features["evidence_filled"] < features["n_entities"]:
        return TEMPLATE_SEARCH, {}
    if features["attribute_kind"] == "director" and features["last_observation_empty"] == 1:
        return TEMPLATE_REWRITE, {"rewrite_template": "{entity} biography"}
    if features["attribute_kind"] == "director" and features["search_count"] == 0:
        return TEMPLATE_SEARCH, {}
    return base_policy(state)


def _charge(state: AgentState, query: str, observation: str, budget: Budget) -> None:
    if state.tool_calls >= budget.max_tool_calls:
        raise IllegalAction("tool_budget")
    cost = token_cost(query, observation)
    if state.tokens + cost > budget.max_tokens:
        raise IllegalAction("token_budget")
    state.tool_calls += 1
    state.tokens += cost
    state.search_count += 1
    state.last_query = query
    state.last_observation = observation


def _search(state: AgentState, env: SearchEnv, query: str, budget: Budget) -> str:
    if state.tool_calls >= budget.max_tool_calls:
        raise IllegalAction("tool_budget")
    observation = env.search(query)
    _charge(state, query, observation, budget)
    return observation


def _store(state: AgentState, entity: str, observation: str) -> None:
    value = extract_slot(observation)
    if entity and value:
        state.evidence[entity] = value


def apply_action(
    state: AgentState,
    env: SearchEnv,
    template: str,
    parameters: dict[str, Any],
    budget: Budget,
    *,
    allowed_rewrites: tuple[str, ...],
) -> str:
    """Apply one legal action. Raises IllegalAction before any tool call when illegal."""
    if template == TEMPLATE_BASE_SEARCH:
        query = str(parameters.get("query") or state.question)
        _search(state, env, query, budget)
        return template
    if template == TEMPLATE_BASE_ANSWER:
        if state.tool_calls >= budget.max_tool_calls:
            raise IllegalAction("tool_budget")
        answer = str(parameters.get("answer") or "unknown")
        cost = token_cost(answer, "")
        if state.tokens + cost > budget.max_tokens:
            raise IllegalAction("token_budget")
        state.tool_calls += 1
        state.tokens += cost
        state.answer = answer
        state.answered = True
        return template
    if template == TEMPLATE_SEARCH:
        entities = [entity for entity in parse_entities(state.question) if entity not in state.evidence]
        if not entities or state.tool_calls >= budget.max_tool_calls:
            raise IllegalAction("search_precondition")
        phrase = attribute_phrase(infer_attribute_kind(state.question))
        for entity in entities:
            if state.tool_calls >= budget.max_tool_calls:
                break
            observation = _search(state, env, f"{entity} {phrase}", budget)
            _store(state, entity, observation)
        return template
    if template == TEMPLATE_REWRITE:
        rewrite = str(parameters.get("rewrite_template") or "")
        entities = parse_entities(state.question)
        if (
            state.features()["last_observation_empty"] != 1
            or not entities
            or rewrite not in allowed_rewrites
            or state.tool_calls >= budget.max_tool_calls
        ):
            raise IllegalAction("rewrite_precondition")
        entity = entities[-1]
        query = rewrite.format(entity=entity, attribute=attribute_phrase(infer_attribute_kind(state.question)))
        observation = _search(state, env, query, budget)
        _store(state, entity, observation)
        return template
    raise IllegalAction("unknown_template")


def _maybe_finalize(state: AgentState, budget: Budget) -> None:
    if state.answered or state.tool_calls >= budget.max_tool_calls:
        return
    answer = ""
    if has_two_years(state):
        answer = _compare_years(state)
    elif infer_attribute_kind(state.question) == "director":
        for value in state.evidence.values():
            if value and not is_year(value):
                answer = value
                break
    if not answer:
        return
    cost = token_cost(answer, "")
    if state.tokens + cost > budget.max_tokens:
        return
    state.tool_calls += 1
    state.tokens += cost
    state.answer = answer
    state.answered = True


def has_two_years(state: AgentState) -> bool:
    years = [value for value in state.evidence.values() if is_year(value)]
    return len(years) >= 2


def _compare_years(state: AgentState) -> str:
    dated = [(entity, int(value)) for entity, value in state.evidence.items() if is_year(value)]
    if len(dated) < 2:
        return ""
    dated.sort(key=lambda item: item[1])
    if comparison_direction(state.question) == "later":
        return dated[-1][0]
    return dated[0][0]


def execute_rules(
    task: Task,
    rules: list[Rule],
    registry: RuntimeRegistry,
    *,
    budget: Budget | None = None,
    text_context: str = "",
    policy: Policy | None = None,
    score: bool = True,
) -> Episode:
    """Run one episode. `text_context` is visible to a learned base agent; the frozen base ignores it."""
    limit = budget or registry.budget
    state = AgentState(question=task.question)
    env = SearchEnv(task)
    decide = policy or base_policy
    steps: list[TraceStep] = []
    triggered: list[str] = []
    matched: list[str] = []
    fire_features: list[dict[str, Any]] = []
    templates: list[str] = []
    modified = 0
    fallbacks = 0
    try:
        while (
            not state.answered
            and state.step_count < limit.max_steps
            and state.tool_calls < limit.max_tool_calls
            and state.tokens < limit.max_tokens
        ):
            features = state.features()
            applied = ""
            applied_params: dict[str, Any] = {}
            if policy is None and state.interventions < limit.max_interventions:
                matching = [rule for rule in rules if trigger_matches(rule.trigger, features, max_atoms=registry.max_atoms)]
                matching.sort(key=lambda rule: (-len(rule.trigger.get("all") or []), rule.rule_id))
                for rule in matching:
                    matched.append(rule.rule_id)
                    try:
                        apply_action(
                            state,
                            env,
                            rule.action_template,
                            rule.parameters,
                            limit,
                            allowed_rewrites=registry.rewrite_templates,
                        )
                    except IllegalAction:
                        fallbacks += 1
                        continue
                    applied = rule.action_template
                    applied_params = dict(rule.parameters)
                    state.interventions += 1
                    modified += 1
                    triggered.append(rule.rule_id)
                    fire_features.append({"rule_id": rule.rule_id, "features": features, "task_id": task.task_id})
                    break
            if not applied:
                template, params = decide(state)
                apply_action(
                    state,
                    env,
                    template,
                    params,
                    limit,
                    allowed_rewrites=registry.rewrite_templates,
                )
                applied = template
                applied_params = dict(params)
            templates.append(applied)
            _maybe_finalize(state, limit)
            steps.append(
                TraceStep(
                    features_before=features,
                    action_template=applied,
                    parameters=applied_params,
                    observation_redacted=redact_observation(state.last_observation),
                )
            )
            state.step_count += 1
    except IllegalAction:
        state.infra_status = STATUS_INFRA
    success = exact_match(state.answer, task.gold_answer) if score else False
    return Episode(
        task_id=task.task_id,
        answer=state.answer,
        success=success,
        steps=steps,
        triggered_rules=triggered,
        matched_rules=matched,
        modified_actions=modified,
        tool_calls=state.tool_calls,
        tokens=state.tokens,
        interventions=state.interventions,
        fallbacks=fallbacks,
        infra_status=state.infra_status,
        action_templates=templates,
        text_context_sha256=sha256_text(text_context),
        fire_features=fire_features,
    )
