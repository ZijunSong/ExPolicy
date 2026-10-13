"""B-v2.1 action parser. One action is accepted. Conflicting actions are rejected."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

ALLOWED_TOOLS = frozenset({"search", "open", "find", "answer"})
_FENCE = re.compile(r"```(?:xml|json)?\s*(.*?)```", re.I | re.S)
_THINK = re.compile(r"<think>.*?</think>", re.I | re.S)
_TOOL_ARG = re.compile(
    r"<tool>\s*(?P<tool>[a-zA-Z_]+)\s*</tool>\s*<arg(?:s)?>(?P<arg>.*?)</arg(?:s)?>",
    re.I | re.S,
)
_NAMED = {
    name: re.compile(rf"<{name}\b[^>]*>(?P<body>.*?)</{name}>", re.I | re.S)
    for name in ("search", "open", "find", "answer")
}
_WRAPPER = re.compile(
    r"^\s*<(?P<tag>query|arg|args|arguments)>\s*(?P<body>.*?)\s*</(?P=tag)>\s*$",
    re.I | re.S,
)
_PROTOCOL = re.compile(r"</?(?:tool|arg|args|arguments|query|search|open|find|answer|think)\b", re.I)


@dataclass(frozen=True)
class ParsedAction:
    ok: bool
    tool: str
    arguments: dict[str, Any]
    reason: str


def parse_action(text: str) -> ParsedAction:
    """Return one canonical action, or a rejection. Never guess among conflicts."""
    raw = text or ""
    hidden = " ".join(_THINK.findall(raw))
    outside = _THINK.sub(" ", raw)
    if hidden.strip() and _actions_in(hidden):
        return ParsedAction(False, "", {}, "action_inside_think")
    fenced = [block.strip() for block in _FENCE.findall(outside) if block.strip()]
    bodies = fenced or [outside.strip()]
    found: list[ParsedAction] = []
    for body in bodies:
        found.extend(_actions_in(body))
    usable = [item for item in found if item.ok]
    if len(usable) != 1 or len({item.tool for item in found}) != 1 or len(found) != 1:
        reason = "no_action" if not found else "conflicting_or_invalid_actions"
        return ParsedAction(False, "", {}, reason)
    return usable[0]


def query_has_protocol_tag(query: str) -> bool:
    return _PROTOCOL.search(query or "") is not None


def _actions_in(body: str) -> list[ParsedAction]:
    found: list[ParsedAction] = []
    for match in _TOOL_ARG.finditer(body):
        found.append(_from_tool(match.group("tool"), match.group("arg")))
    if found:
        return found
    named = [(name, pattern.search(body)) for name, pattern in _NAMED.items()]
    named = [(name, match) for name, match in named if match]
    if len(named) > 1:
        return [ParsedAction(False, "", {}, "conflicting_or_invalid_actions") for _ in named]
    if len(named) == 1:
        name, match = named[0]
        return [_from_tool(name, match.group("body"))]
    payload = _one_json_object(body)
    if payload is None:
        return []
    return [_from_json(payload)]


def _one_json_object(body: str) -> dict[str, Any] | None:
    text = body.strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start or text[:start].strip():
            return None
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _from_json(payload: dict[str, Any]) -> ParsedAction:
    tool = str(payload.get("tool") or "").strip().lower()
    arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else payload
    if not isinstance(arguments, dict):
        return ParsedAction(False, tool, {}, "missing_arguments")
    return _from_tool(tool, arguments)


def _from_tool(tool: str, raw_arg: str | dict[str, Any]) -> ParsedAction:
    name = tool.strip().lower()
    if name not in ALLOWED_TOOLS:
        return ParsedAction(False, name, {}, "unknown_tool")
    if isinstance(raw_arg, dict):
        fields = raw_arg
    else:
        text = raw_arg.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError:
                return ParsedAction(False, name, {}, "bad_json")
            if not isinstance(loaded, dict):
                return ParsedAction(False, name, {}, "bad_json")
            fields = loaded
        else:
            fields = {"_raw": _unwrap_complete(text), "_wrapped_query": _used_query_wrapper(text)}
    if name == "search":
        query = _field(fields, "query")
        if query is None or query_has_protocol_tag(query) or "<" in query or ">" in query:
            return ParsedAction(False, name, {}, "dirty_or_missing_query")
        return ParsedAction(True, name, {"query": query}, "ok")
    if name == "open":
        if fields.get("_wrapped_query"):
            return ParsedAction(False, name, {}, "open_used_query_tag")
        doc_id = _field(fields, "document_id")
        if doc_id is None or query_has_protocol_tag(doc_id) or "<" in doc_id:
            return ParsedAction(False, name, {}, "missing_document_id")
        return ParsedAction(True, name, {"document_id": doc_id}, "ok")
    if name == "find":
        if "_raw" in fields:
            return ParsedAction(False, name, {}, "missing_find_fields")
        doc_id = _field(fields, "document_id")
        pattern = _field(fields, "pattern")
        if doc_id is None or pattern is None:
            return ParsedAction(False, name, {}, "missing_find_fields")
        return ParsedAction(True, name, {"document_id": doc_id, "pattern": pattern}, "ok")
    text = _answer_text(fields)
    if text is None:
        return ParsedAction(False, name, {}, "missing_answer_text")
    return ParsedAction(True, name, {"text": text}, "ok")


def _field(fields: dict[str, Any], key: str) -> str | None:
    if key in fields and fields[key] is not None:
        value = str(fields[key]).strip()
        return value or None
    if "_raw" in fields:
        return str(fields["_raw"]).strip() or None
    return None


def _answer_text(fields: dict[str, Any]) -> str | None:
    present = [key for key in ("text", "answer") if key in fields and str(fields[key]).strip()]
    if not present and "_raw" in fields:
        value = str(fields["_raw"]).strip()
        return value or None
    if len(present) > 1 and str(fields["text"]).strip() != str(fields["answer"]).strip():
        return None
    if not present:
        return None
    return str(fields[present[0]]).strip()


def _unwrap_complete(text: str) -> str:
    current = text.strip()
    for _ in range(3):
        match = _WRAPPER.fullmatch(current)
        if not match:
            break
        current = match.group("body").strip()
    return current


def _used_query_wrapper(text: str) -> bool:
    current = text.strip()
    for _ in range(3):
        match = _WRAPPER.fullmatch(current)
        if not match:
            return False
        if match.group("tag").lower() == "query":
            return True
        current = match.group("body").strip()
    return False


def fixture_cases() -> list[dict[str, Any]]:
    """At least 50 offline cases. Legal samples parse; illegal samples reject."""
    rows: list[dict[str, Any]] = []
    legal_xml = [
        ("<tool>search</tool><arg>John Cameron Mitchell director</arg>", "search", "John Cameron Mitchell director"),
        ("<tool>open</tool><arg>doc-12</arg>", "open", "doc-12"),
        ("<tool>answer</tool><arg>John Cameron Mitchell</arg>", "answer", "John Cameron Mitchell"),
        ("<search><query>Paris capital</query></search>", "search", "Paris capital"),
        ("<search>\n  <query>Bispebjerg Cemetery Copenhagen</query>\n</search>", "search", "Bispebjerg Cemetery Copenhagen"),
        ("```xml\n<search>\n  <query>Covered Alive in Asia genre</query>\n</search>\n```", "search", "Covered Alive in Asia genre"),
        ('<search>{"query": "Raider Klan producer"}</search>', "search", "Raider Klan producer"),
        ('{"tool":"search","arguments":{"query":"John Cameron Mitchell director"}}', "search", "John Cameron Mitchell director"),
        ('{"tool":"answer","arguments":{"text":"John Cameron Mitchell"}}', "answer", "John Cameron Mitchell"),
        ('{"tool":"answer","arguments":{"answer":"Christian"}}', "answer", "Christian"),
        ("```json\n{\"tool\":\"open\",\"arguments\":{\"document_id\":\"17\"}}\n```", "open", "17"),
        ('<find>{"document_id":"17","pattern":"population"}</find>', "find", "17"),
    ]
    for index, (text, tool, value) in enumerate(legal_xml):
        rows.append({"id": f"legal-{index:02d}", "text": text, "expect_ok": True, "tool": tool, "value": value})
    for index in range(20):
        query = f"entity {index} relation"
        rows.append(
            {
                "id": f"legal-nested-{index:02d}",
                "text": f"<search><query>{query}</query></search>",
                "expect_ok": True,
                "tool": "search",
                "value": query,
            }
        )
    for index in range(12):
        rows.append(
            {
                "id": f"legal-json-{index:02d}",
                "text": json.dumps({"tool": "search", "arguments": {"query": f"plain query {index}"}}),
                "expect_ok": True,
                "tool": "search",
                "value": f"plain query {index}",
            }
        )
    illegal = [
        ("<tool>search</tool><arg><query>Paris</query></arg><tool>answer</tool><arg>Paris</arg>", "conflicting"),
        ("<search><query>one</query></search><answer>two</answer>", "conflicting"),
        ("<tool>search</tool><arg></arg>", "missing"),
        ("<tool>unknown</tool><arg>x</arg>", "unknown"),
        ("<search><query><b>Paris</b></query></search>", "dirty"),
        ("just a sentence about John Cameron Mitchell", "none"),
        ('{"tool":"answer","arguments":{"text":"A","answer":"B"}}', "conflict_fields"),
        ("<open><query>not a document id</query></open>", "open_query"),
        ("<think><search><query>hidden</query></search></think><answer>visible</answer>", "think_and_answer"),
        ("<tool>find</tool><arg>only-one-field</arg>", "find_missing"),
    ]
    for index, (text, reason) in enumerate(illegal):
        rows.append({"id": f"illegal-{index:02d}", "text": text, "expect_ok": False, "reason": reason})
    return rows
