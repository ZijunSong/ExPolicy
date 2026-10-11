"""Public search environment: agent query → local retriever → open/read → answer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from expolicy.core.errors import EnvironmentError
from expolicy.environments.public_search.retriever import RetrieverClient
from expolicy.environments.search.environment import BaseEnvironment, ToolResult
from expolicy.environments.search.world import Task

PUBLIC_SYSTEM_PROMPT = """You are a Wikipedia search agent.
You must ground every answer in retrieved passages. Never invent document ids or facts.

Emit exactly one tool call per turn, using this XML format and nothing else:

<tool>search</tool>
<arg>entity or relation keywords</arg>

<tool>open</tool>
<arg>document_id</arg>

<tool>find</tool>
<arg>{"document_id": "document_id", "pattern": "keyword"}</arg>

<tool>answer</tool>
<arg>short span</arg>

Rules:
1. First action must be search.
2. Then open the most relevant document_id from the hits.
3. For multi-hop questions, search again using a bridge entity found in opened evidence.
4. For comparison questions, search each entity independently before comparing.
5. The final answer is a short span: a name, number, date, yes/no, or short phrase.
6. Do not output URLs, JSON wrappers, or commentary — only one XML tool block.
"""


@dataclass
class PublicWorld:
    tasks: dict[str, Task] = field(default_factory=dict)
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_task(self, task: Task) -> None:
        self.tasks[task.task_id] = task


class PublicSearchEnvironment(BaseEnvironment):
    """Search-R1 style local Wikipedia retrieval. Results are never fabricated."""

    def __init__(self, world: PublicWorld, client: RetrieverClient, *, top_k: int = 5) -> None:
        self.world = world
        self.client = client
        self.top_k = top_k
        self.current_task: Task | None = None
        self.opened: set[str] = set()
        self.search_log: list[dict[str, Any]] = []
        self.doc_cache: dict[str, dict[str, Any]] = {}
        self.n_steps = 0

    def reset(self, task: Task) -> dict[str, Any]:
        if task.task_id not in self.world.tasks:
            self.world.add_task(task)
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

    def _cache(self, doc: dict[str, Any]) -> dict[str, Any]:
        did = str(doc.get("document_id") or doc.get("id") or "")
        if did:
            self.doc_cache[did] = doc
            self.world.documents[did] = doc
        return doc

    def _search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        hits = self.client.search(query, topk=top_k or self.top_k)
        serialized = []
        for rank, hit in enumerate(hits):
            doc = self._cache(hit)
            serialized.append(
                {
                    "document_id": doc["document_id"],
                    "title": doc.get("title") or "",
                    "score": float(doc.get("score") or 0.0),
                    "rank": rank,
                    "snippet": (doc.get("snippet") or doc.get("text") or "")[:280],
                    "source_type": "wikipedia",
                    "reliability": 0.9,
                    "year": 2018,
                }
            )
        self.search_log.append({"query": query, "n_hits": len(serialized)})
        return {"query": query, "hits": serialized}

    def _open(self, document_id: str) -> dict[str, Any]:
        doc = self.doc_cache.get(document_id)
        if doc is None:
            fetched = self.client.open_id(document_id)
            if fetched is None:
                raise EnvironmentError(f"unknown document {document_id}")
            doc = self._cache(fetched)
        self.opened.add(document_id)
        return {
            "document_id": doc["document_id"],
            "title": doc.get("title") or "",
            "text": doc.get("text") or doc.get("contents") or "",
            "source_type": "wikipedia",
            "reliability": 0.9,
            "year": 2018,
        }

    def _find(self, document_id: str, pattern: str) -> dict[str, Any]:
        doc = self._open(document_id)
        text = doc.get("text") or ""
        idx = text.lower().find(pattern.lower()) if pattern else -1
        snippet = ""
        if idx >= 0:
            snippet = text[max(0, idx - 80) : idx + 200]
        return {"document_id": document_id, "pattern": pattern, "index": idx, "snippet": snippet}
