"""Closed-loop agent runtime: current persistent state actually drives rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from expolicy.agents.llm_agent import LLMSearchAgent
from expolicy.closed_loop.assimilation.activate_skill import apply_activate
from expolicy.closed_loop.assimilation.discard import apply_discard
from expolicy.closed_loop.assimilation.episodic import apply_episodic, retrieve_cards
from expolicy.closed_loop.assimilation.semantic import active_notes, apply_semantic
from expolicy.closed_loop.constants import (
    ACTIVATE_EXISTING_SKILL,
    ASSIM_ACTIONS,
    DISCARD,
    EPISODIC_MEMORY,
    SEMANTIC_MEMORY,
)
from expolicy.closed_loop.experience import ExperienceBatchSummary
from expolicy.closed_loop.frozen_skill_bank import run_skill
from expolicy.closed_loop.state import ExPolicyState
from expolicy.core.experience import Experience
from expolicy.core.state import HarnessConfig, PersistentAgentState, empty_agent
from expolicy.environments.public_search.environment import PUBLIC_SYSTEM_PROMPT, PublicSearchEnvironment
from expolicy.environments.search.world import Task
from expolicy.logging.event_logger import EventLogger


class MemoryFacade:
    def __init__(self, episodic_dir: Path, semantic_dir: Path) -> None:
        self.episodic_dir = episodic_dir
        self.semantic_dir = semantic_dir

    def retrieve(self, query: str, k: int = 3, **kwargs: Any):
        cap = kwargs.get("capability")
        cards = retrieve_cards(self.episodic_dir, query, capability=cap, k=k)
        debug = [{"n": len(cards), "capability": cap}]
        if kwargs.get("return_debug"):
            return cards, debug
        return cards

    def rules(self, k: int = 8) -> list[str]:
        return active_notes(self.semantic_dir, k=k)


class ClosedLoopRuntime:
    def __init__(
        self,
        *,
        env: PublicSearchEnvironment,
        model: Any,
        root: Path,
        state: ExPolicyState,
        logger: EventLogger | None = None,
        max_steps: int = 8,
    ) -> None:
        self.env = env
        self.model = model
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = state
        self.episodic_dir = self.root / "episodic_memory"
        self.semantic_dir = self.root / "semantic_memory"
        self.episodic_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        harness = HarnessConfig(max_steps=max_steps, search_top_k=5, min_evidence_before_answer=1, temperature=0.0)
        self.agent_state: PersistentAgentState = empty_agent(self.root / "agent_v000", model_name=getattr(model, "name", "") or "")
        self.agent_state.harness = harness
        facade = MemoryFacade(self.episodic_dir, self.semantic_dir)
        self.llm = LLMSearchAgent(
            model,
            env,
            harness=harness,
            logger=logger,
            memory_retriever=facade,
            use_executable_skill=False,
            system_prompt=PUBLIC_SYSTEM_PROMPT,
        )
        self.last_dispatch = ""
        self.last_memory_injected = 0

    def persistent_state_hash(self) -> str:
        payload = {
            "active_skills": dict(self.state.active_skills),
            "episodic": sorted(p.name for p in self.episodic_dir.glob("*")),
            "semantic": sorted(p.name for p in self.semantic_dir.glob("*")),
            "epi_n": sum(self.state.episodic_memory_size.values()),
            "sem_n": sum(self.state.semantic_memory_size.values()),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def valid_assim_actions(self, capability: str) -> list[str]:
        actions = list(ASSIM_ACTIONS)
        if self.state.active_skills.get(capability):
            actions = [a for a in actions if a != ACTIVATE_EXISTING_SKILL]
        return actions

    def run_task(self, task: Task) -> Experience:
        cap = task.capability
        if cap and self.state.active_skills.get(cap):
            self.last_dispatch = f"frozen_skill:{cap}"
            self.last_memory_injected = 0
            exp = run_skill(cap, self.env, task, model=self.model)
            exp.capability_context = dict(exp.capability_context or {})
            exp.capability_context["dispatch"] = self.last_dispatch
            exp.capability_context["persistent_state_hash"] = self.persistent_state_hash()
            return exp
        self.last_dispatch = "llm+memory"
        exp = self.llm.run_task(task, state=self.agent_state)
        injected = 0
        if getattr(self.llm, "last_retrieval", None):
            injected = int(self.llm.last_retrieval.get("n") or 0)
        self.last_memory_injected = injected
        exp.capability_context = dict(exp.capability_context or {})
        exp.capability_context["dispatch"] = self.last_dispatch
        exp.capability_context["memory_injected"] = injected
        exp.capability_context["persistent_state_hash"] = self.persistent_state_hash()
        return exp

    def apply_assimilation(
        self,
        action: str,
        summary: ExperienceBatchSummary,
        experiences: list[Experience],
    ) -> dict[str, Any]:
        if action == DISCARD:
            return apply_discard(self.state, experiences)
        if action == EPISODIC_MEMORY:
            return apply_episodic(self.episodic_dir, self.state, experiences)
        if action == SEMANTIC_MEMORY:
            return apply_semantic(self.semantic_dir, self.state, summary, experiences, model=self.model)
        if action == ACTIVATE_EXISTING_SKILL:
            return apply_activate(self.state, summary.capability)
        raise ValueError(f"unknown assimilation action {action}")
