"""Parse ReAct-style tool calls from model text. No invented tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_TOOL_BLOCK = re.compile(
    r"<tool>\s*(?P<tool>[a-zA-Z_]+)\s*</tool>\s*<arg(?:s)?>(?P<arg>.*?)</arg(?:s)?>",
    re.DOTALL | re.IGNORECASE,
)
_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_FUNC = re.compile(
    r"(?P<tool>search|open|find|answer|invoke_skill)\s*\(\s*(?P<arg>.*)\)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_URL_SEARCH = re.compile(r"search\?q=([^&\s]+)", re.IGNORECASE)
_DOC_ID = re.compile(r"doc_\d+", re.IGNORECASE)
_KV_SEARCH = re.compile(r"(?:^|\n)\s*(?:action\s*:)?\s*search\s*[:=\-]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_KV_OPEN = re.compile(r"(?:^|\n)\s*open\s*[:=\-]\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)
_KV_ANSWER = re.compile(
    r"(?:^|\n)\s*(?:final\s+answer|answer)\s*[:=\-]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class ParsedCall:
    tool: str
    arguments: dict[str, Any]
    raw: str
    valid: bool
    error: str | None = None


def parse_tool_call(text: str) -> ParsedCall:
    raw = (text or "").strip()
    if not raw:
        return ParsedCall("invalid", {}, raw, False, "empty model output")

    match = _TOOL_BLOCK.search(raw)
    if match:
        tool = match.group("tool").strip().lower()
        arg = match.group("arg").strip()
        return _coerce(tool, arg, raw)

    url = _URL_SEARCH.search(raw)
    if url:
        from urllib.parse import unquote_plus

        query = unquote_plus(url.group(1)).replace("+", " ").strip()
        return ParsedCall("search", {"query": query}, raw, bool(query))

    json_match = _JSON_BLOCK.search(raw)
    if json_match:
        try:
            payload = json.loads(json_match.group(1))
            tool = str(payload.get("tool") or payload.get("name") or "").lower()
            args = payload.get("arguments") or payload.get("args") or {}
            if isinstance(args, str):
                return _coerce(tool, args, raw)
            if isinstance(args, dict):
                return ParsedCall(tool, args, raw, tool in {"search", "open", "find", "answer", "invoke_skill"})
        except json.JSONDecodeError:
            pass

    for line in reversed(raw.splitlines()):
        func = _FUNC.search(line.strip())
        if func:
            return _coerce(func.group("tool").lower(), func.group("arg").strip().strip("\"'"), raw)

    kv_search = _KV_SEARCH.search(raw)
    if kv_search:
        return _coerce("search", kv_search.group(1).strip().strip("\"'"), raw)
    kv_open = _KV_OPEN.search(raw)
    if kv_open:
        return _coerce("open", kv_open.group(1).strip(), raw)
    doc = _DOC_ID.search(raw)
    if doc and re.search(r"\bopen\b", raw, re.I):
        return ParsedCall("open", {"document_id": doc.group(0).lower()}, raw, True)

    # last-ditch: if the model dumped a final answer without a tool tag
    if "final answer" in raw.lower():
        after = raw.lower().split("final answer", 1)[1]
        after = after.split(":", 1)[-1].strip()
        return ParsedCall("answer", {"text": after[:500]}, raw, True)
    kv_answer = _KV_ANSWER.search(raw)
    if kv_answer:
        return _coerce("answer", kv_answer.group(1).strip(), raw)

    return ParsedCall("invalid", {}, raw, False, "could not parse a tool call")


def _coerce(tool: str, arg: str, raw: str) -> ParsedCall:
    tool = tool.lower()
    arg = arg.strip()
    if tool == "search":
        query = _maybe_json_field(arg, "query") or arg
        return ParsedCall("search", {"query": query}, raw, bool(query))
    if tool == "open":
        doc_id = _maybe_json_field(arg, "document_id") or arg
        return ParsedCall("open", {"document_id": doc_id}, raw, bool(doc_id))
    if tool == "find":
        if arg.startswith("{"):
            try:
                payload = json.loads(arg)
                return ParsedCall(
                    "find",
                    {
                        "document_id": str(payload.get("document_id", "")),
                        "pattern": str(payload.get("pattern", "")),
                    },
                    raw,
                    True,
                )
            except json.JSONDecodeError:
                pass
        if "," in arg:
            doc_id, pattern = arg.split(",", 1)
            return ParsedCall("find", {"document_id": doc_id.strip(), "pattern": pattern.strip()}, raw, True)
        return ParsedCall("find", {"document_id": arg, "pattern": arg}, raw, bool(arg))
    if tool == "answer":
        text = _maybe_json_field(arg, "text") or arg
        return ParsedCall("answer", {"text": text}, raw, bool(text))
    if tool == "invoke_skill":
        name = _maybe_json_field(arg, "skill") or arg
        return ParsedCall("invoke_skill", {"skill": name}, raw, bool(name))
    return ParsedCall(tool, {"raw": arg}, raw, False, f"unknown tool {tool}")


def _maybe_json_field(arg: str, field: str) -> str | None:
    stripped = arg.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            if field in payload:
                return str(payload[field])
        except json.JSONDecodeError:
            return None
    return None
