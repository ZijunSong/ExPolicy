"""LLM search agent: plan → search → open → find → synthesize.

Every model call, tool call, and observation is recorded on the Experience.
"""

from __future__ import annotations

import time
from typing import Any

from expolicy.agents.base import BaseAgent
from expolicy.agents.parser import parse_tool_call
from expolicy.agents.prompts import SYSTEM_PROMPT, build_user_prompt, observation_message
from expolicy.core.experience import Experience, ExperienceCost
from expolicy.core.state import HarnessConfig, PersistentAgentState
from expolicy.core.types import FailureType
from expolicy.environments.search.environment import SearchEnvironment
from expolicy.environments.search.world import Task
from expolicy.logging.event_logger import EventLogger
from expolicy.models.base import ModelBackend


class LLMSearchAgent(BaseAgent):
    def __init__(
        self,
        model: ModelBackend,
        env: SearchEnvironment,
        *,
        harness: HarnessConfig | None = None,
        logger: EventLogger | None = None,
        memory_retriever: Any | None = None,
        skill_store: Any | None = None,
    ) -> None:
        self.model = model
        self.env = env
        self.harness = harness or HarnessConfig()
        self.logger = logger
        self.memory_retriever = memory_retriever
        self.skill_store = skill_store

    def run_task(self, task: Task, state: PersistentAgentState | None = None) -> Experience:
        harness = state.harness if state is not None else self.harness
        started = time.perf_counter()
        self.env.reset(task)
        memories, rules, skills = self._context(task, state)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    task.question,
                    memories=memories,
                    rules=rules,
                    skills=skills,
                    harness_notes=self._harness_notes(harness),
                ),
            },
        ]
        experience = Experience(
            task_id=task.task_id,
            question=task.question,
            gold_answer=task.gold_answer,
            capability=task.capability,
            stage=task.stage,
            split=task.split,
            capability_context={"capability": task.capability, "stage": task.stage, "hops": task.hops},
            initial_state={"question": task.question, "tools": self.env.available_tools()},
            agent_version=state.version if state else None,
        )
        if self.logger:
            self.logger.emit("TASK_STARTED", {"task_id": task.task_id, "question": task.question})

        predicted = ""
        opened_docs: list[str] = []
        n_consistency = max(1, harness.n_consistency)
        samples: list[str] = []

        for step in range(harness.max_steps):
            result = self.model.generate(
                messages,
                max_new_tokens=256,
                temperature=harness.temperature,
            )
            experience.cost = experience.cost.add(
                ExperienceCost(
                    tokens=result.tokens,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    latency=result.latency,
                )
            )
            parsed = parse_tool_call(result.text)
            action = {"step": step, "raw": result.text, "tool": parsed.tool, "arguments": parsed.arguments}
            experience.actions.append(action)
            if self.logger:
                self.logger.emit("ACTION", action)
            if not parsed.valid:
                obs = {"error": parsed.error or "invalid tool call", "raw": result.text[:500]}
                experience.observations.append(obs)
                messages.append({"role": "assistant", "content": result.text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Invalid tool call. Output exactly:\n"
                            "<tool>search</tool>\n<arg>entity name keywords</arg>\n"
                            "or <tool>open</tool><arg>doc_0000</arg>\n"
                            "or <tool>answer</tool><arg>short span</arg>"
                        ),
                    }
                )
                continue
            if parsed.tool == "answer":
                if harness.min_evidence_before_answer > 0 and not opened_docs:
                    obs = {"error": "answer rejected: open at least one document first"}
                    experience.observations.append(obs)
                    messages.append({"role": "assistant", "content": result.text})
                    messages.append({"role": "user", "content": "You must open evidence before answering. Search or open a document."})
                    continue
                predicted = str(parsed.arguments.get("text") or "").strip()
                if n_consistency > 1 and harness.temperature > 0:
                    extra = self.model.generate(
                        messages,
                        max_new_tokens=64,
                        temperature=max(harness.temperature, 0.4),
                        n=n_consistency,
                    )
                    samples = list(extra.raw.get("samples") or [extra.text])
                experience.observations.append({"tool": "answer", "text": predicted})
                break

            if parsed.tool == "invoke_skill" and self.skill_store is not None:
                skill_obs = self.skill_store.invoke(
                    str(parsed.arguments.get("skill") or ""),
                    self.env,
                    query=task.question,
                )
                experience.tool_calls.append({"tool": "invoke_skill", "arguments": parsed.arguments, "ok": True})
                experience.observations.append(skill_obs)
                messages.append({"role": "assistant", "content": result.text})
                messages.append({"role": "user", "content": observation_message("invoke_skill", skill_obs)})
                continue

            if parsed.tool == "search":
                parsed.arguments["query"] = parsed.arguments.get("query") or task.question
            tool_result = self.env.step(parsed.tool, parsed.arguments)
            experience.cost.environment_steps += 1
            experience.cost.tool_calls += 1
            experience.tool_calls.append(
                {
                    "tool": parsed.tool,
                    "arguments": parsed.arguments,
                    "ok": tool_result.ok,
                    "latency": tool_result.latency,
                }
            )
            if self.logger:
                self.logger.emit("TOOL_CALL", {"tool": parsed.tool, "ok": tool_result.ok})
            payload = tool_result.payload if tool_result.ok else {"error": tool_result.error}
            if parsed.tool == "open" and tool_result.ok:
                opened_docs.append(str(payload.get("document_id") or ""))
                experience.evidence.append(
                    {
                        "document_id": payload.get("document_id"),
                        "source_type": payload.get("source_type"),
                        "reliability": payload.get("reliability"),
                        "title": payload.get("title"),
                    }
                )
            experience.observations.append({"tool": parsed.tool, "payload": payload})
            if self.logger:
                self.logger.emit("OBSERVATION", {"tool": parsed.tool, "ok": tool_result.ok})
            messages.append({"role": "assistant", "content": result.text})
            messages.append({"role": "user", "content": observation_message(parsed.tool, payload)})

        wall = time.perf_counter() - started
        experience.cost.latency = wall
        experience.predicted_answer = predicted
        success = _normalize_answer(predicted) == _normalize_answer(task.gold_answer) or _soft_match(
            predicted, task.gold_answer, task.aliases
        )
        experience.success = success
        experience.reward = 1.0 if success else 0.0
        experience.outcome = {
            "predicted": predicted,
            "gold": task.gold_answer,
            "success": success,
            "opened_docs": opened_docs,
            "n_steps": len(experience.actions),
            "samples": samples,
        }
        if samples:
            uniq = { _normalize_answer(s) for s in samples if s }
            experience.uncertainty = 1.0 - (1.0 / max(len(uniq), 1))
        if not success:
            experience.failure_type = infer_failure(experience)
        if self.logger:
            self.logger.emit(
                "TASK_FINISHED",
                {"task_id": task.task_id, "success": success, "predicted": predicted, "gold": task.gold_answer},
            )
            self.logger.emit("EXPERIENCE_CREATED", {"experience_id": experience.experience_id})
        return experience

    def _context(
        self, task: Task, state: PersistentAgentState | None
    ) -> tuple[list[str], list[str], list[str]]:
        memories: list[str] = []
        rules: list[str] = []
        skills: list[str] = []
        if self.memory_retriever is not None and state is not None:
            memories = self.memory_retriever.retrieve(task.question, k=5)
            rules = self.memory_retriever.rules(k=8)
        if self.skill_store is not None:
            skills = self.skill_store.summaries()
        return memories, rules, skills

    def _harness_notes(self, harness: HarnessConfig) -> str:
        return (
            f"max_steps={harness.max_steps}; top_k={harness.search_top_k}; "
            f"require_citation={harness.require_citation}; packing={harness.context_packing}"
        )


def _normalize_answer(text: str) -> str:
    return " ".join((text or "").lower().replace(";", " ").split())


def _soft_match(pred: str, gold: str, aliases: list[str]) -> bool:
    np = _normalize_answer(pred)
    if not np:
        return False
    if _normalize_answer(gold) in np or np in _normalize_answer(gold):
        return True
    return any(_normalize_answer(alias) and _normalize_answer(alias) in np for alias in aliases)


def infer_failure(experience: Experience) -> FailureType:
    tools = [call.get("tool") for call in experience.tool_calls]
    if "search" not in tools:
        return FailureType.PLANNING
    opened = [ev.get("document_id") for ev in experience.evidence]
    if not opened:
        return FailureType.RETRIEVAL
    if any(not call.get("ok", True) for call in experience.tool_calls):
        return FailureType.TOOL
    if experience.predicted_answer:
        return FailureType.ANSWER_SYNTHESIS
    return FailureType.REASONING
