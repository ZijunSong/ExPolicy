"""Persistent agent wrapping LLM search + stores."""

from __future__ import annotations

from pathlib import Path

from expolicy.agents.llm_agent import LLMSearchAgent
from expolicy.assimilation.carriers.episodic_memory import EpisodicMemory
from expolicy.assimilation.carriers.semantic_memory import SemanticMemory
from expolicy.assimilation.carriers.skill_store import SkillStore
from expolicy.core.experience import Experience
from expolicy.core.state import PersistentAgentState, empty_agent
from expolicy.environments.search.environment import SearchEnvironment
from expolicy.environments.search.world import Task
from expolicy.logging.event_logger import EventLogger
from expolicy.models.base import ModelBackend
from expolicy.models.embedder import Embedder


class MemoryFacade:
    def __init__(self, episodic: EpisodicMemory, semantic: SemanticMemory) -> None:
        self.episodic = episodic
        self.semantic = semantic

    def retrieve(self, query: str, k: int = 5, **kwargs):
        return self.episodic.retrieve(query, k=k, **kwargs)

    def rules(self, k: int = 8) -> list[str]:
        return self.semantic.active_texts(k=k)


class PersistentAgent:
    def __init__(
        self,
        state: PersistentAgentState,
        model: ModelBackend,
        env: SearchEnvironment,
        *,
        embedder: Embedder | None = None,
        logger: EventLogger | None = None,
    ) -> None:
        self.state = state
        self.model = model
        self.env = env
        self.logger = logger
        self.embedder = embedder
        self.episodic = EpisodicMemory(state.episodic_dir(), embedder) if embedder else None
        self.semantic = SemanticMemory(state.semantic_dir())
        self.skills = SkillStore(state.skills_dir())
        if not self.skills.skills:
            self.skills.create(SkillStore.default_lookup_skill())
        facade = MemoryFacade(self.episodic, self.semantic) if self.episodic else None
        self.search_agent = LLMSearchAgent(
            model,
            env,
            harness=state.harness,
            logger=logger,
            memory_retriever=facade,
            skill_store=self.skills,
        )

    @classmethod
    def create(
        cls,
        root: Path,
        model: ModelBackend,
        env: SearchEnvironment,
        *,
        embedder: Embedder | None = None,
        logger: EventLogger | None = None,
        model_name: str = "",
    ) -> PersistentAgent:
        state = empty_agent(root, model_name=model_name)
        return cls(state, model, env, embedder=embedder, logger=logger)

    def run_task(self, task: Task) -> Experience:
        self.search_agent.harness = self.state.harness
        return self.search_agent.run_task(task, state=self.state)
