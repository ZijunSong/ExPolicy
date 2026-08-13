"""Tool-level wrappers used by the search environment."""

from __future__ import annotations

from typing import Any

from expolicy.environments.search.environment import SearchEnvironment, ToolResult


def search(env: SearchEnvironment, query: str) -> ToolResult:
    return env.step("search", {"query": query})


def open_document(env: SearchEnvironment, document_id: str) -> ToolResult:
    return env.step("open", {"document_id": document_id})


def find(env: SearchEnvironment, document_id: str, pattern: str) -> ToolResult:
    return env.step("find", {"document_id": document_id, "pattern": pattern})
