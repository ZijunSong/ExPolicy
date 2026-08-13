"""Search environment: real retrieval tools over a local corpus."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from expolicy.core.errors import EnvironmentError
from expolicy.environments.search.bm25 import BM25Index, SearchHit
from expolicy.environments.search.world import Document, Task, World, build_world


class ToolResult(BaseModel):
    tool: str
    ok: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    latency: float = 0.0


class BaseEnvironment(ABC):
    @abstractmethod
    def reset(self, task: Task) -> dict[str, Any]: ...

    @abstractmethod
    def available_tools(self) -> list[str]: ...

    @abstractmethod
    def step(self, tool: str, arguments: dict[str, Any]) -> ToolResult: ...


class SearchEnvironment(BaseEnvironment):
    """Local reproducible BM25 search. Search results are never fabricated."""

    def __init__(self, world: World, index: BM25Index | None = None):
        if not world.documents:
            raise EnvironmentError("world has no documents")
        self.world = world
        self.index = index or BM25Index.from_world(world)
        self.current_task: Task | None = None
        self.opened: set[str] = set()
        self.search_log: list[dict[str, Any]] = []
        self.n_steps = 0

    @classmethod
    def from_seed(cls, seed: int = 13) -> SearchEnvironment:
        return cls(build_world(seed))

    def reset(self, task: Task) -> dict[str, Any]:
        if task.task_id not in self.world.tasks:
            raise EnvironmentError(f"unknown task {task.task_id}")
        self.current_task = task
        self.opened = set()
        self.search_log = []
        self.n_steps = 0
        return {
            "task_id": task.task_id,
            "question": task.question,
            "capability": task.capability,
            "tools": self.available_tools(),
        }

    def available_tools(self) -> list[str]:
        return ["search", "open", "find"]

    def step(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        self.n_steps += 1
        try:
            if tool == "search":
                payload = self._search(str(arguments.get("query", "")))
            elif tool == "open":
                payload = self._open(str(arguments.get("document_id", "")))
            elif tool == "find":
                payload = self._find(
                    str(arguments.get("document_id", "")),
                    str(arguments.get("pattern", "")),
                )
            else:
                raise EnvironmentError(f"unknown tool {tool}")
            return ToolResult(tool=tool, ok=True, payload=payload, latency=time.perf_counter() - started)
        except Exception as exc:
            return ToolResult(
                tool=tool,
                ok=False,
                error=str(exc),
                latency=time.perf_counter() - started,
            )

    def _search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        hits: list[SearchHit] = self.index.search(query, top_k=top_k)
        serialized = [
            {
                "document_id": hit.document_id,
                "title": hit.title,
                "score": hit.score,
                "rank": hit.rank,
                "snippet": hit.snippet,
                "source_type": hit.source_type,
                "reliability": hit.reliability,
                "year": hit.year,
            }
            for hit in hits
        ]
        self.search_log.append({"query": query, "n_hits": len(serialized)})
        return {"query": query, "hits": serialized}

    def _open(self, document_id: str) -> dict[str, Any]:
        doc: Document = self.index.open(document_id)
        self.opened.add(document_id)
        return {
            "document_id": doc.document_id,
            "title": doc.title,
            "text": doc.text,
            "source_type": doc.source_type,
            "reliability": doc.reliability,
            "year": doc.year,
            "city": doc.city,
        }

    def _find(self, document_id: str, pattern: str) -> dict[str, Any]:
        matches = self.index.find(document_id, pattern)
        self.opened.add(document_id)
        return {"document_id": document_id, "pattern": pattern, "matches": matches}

    def get_task(self, task_id: str) -> Task:
        if task_id not in self.world.tasks:
            raise EnvironmentError(f"unknown task {task_id}")
        return self.world.tasks[task_id]
