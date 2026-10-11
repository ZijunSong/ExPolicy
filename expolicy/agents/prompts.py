"""Prompts for the search agent. Memory/skills/harness are injected, never answers."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a research search agent over the Helion Archive.
You must ground every answer in retrieved documents. Never invent document ids or facts.

Emit exactly one tool call per turn, using this XML format and nothing else:

<tool>search</tool>
<arg>Lira Venn birth city</arg>

<tool>open</tool>
<arg>doc_0001</arg>

<tool>find</tool>
<arg>{"document_id": "doc_0001", "pattern": "born"}</arg>

<tool>answer</tool>
<arg>Aurel</arg>

Example:
User: In which city was Lira Venn born?
Assistant:
<tool>search</tool>
<arg>Lira Venn born</arg>

Rules:
1. First action must be search.
2. Then open the most relevant document_id from the hits.
3. Prefer official and encyclopedia sources over blogs.
4. If sources conflict, trust official/encyclopedia, then answer.
5. The final answer is a short span: a name, city, year, product, or 'X; Y' pair.
6. Do not output URLs, JSON wrappers, or commentary — only one XML tool block.
"""


def build_user_prompt(
    question: str,
    *,
    memories: list[str] | None = None,
    rules: list[str] | None = None,
    skills: list[str] | None = None,
    harness_notes: str = "",
) -> str:
    parts = [f"Question: {question}"]
    if memories:
        parts.append("Episodic procedure cards (do NOT copy stored answers or document ids):")
        for i, mem in enumerate(memories[:3], 1):
            parts.append(f"Memory {i}:\n{mem}")
    if rules:
        parts.append("Semantic rules:")
        parts.extend(f"- {r}" for r in rules[:8])
    if skills:
        parts.append("Reusable skills you may follow:")
        parts.extend(f"- {s}" for s in skills[:4])
    if harness_notes:
        parts.append(f"Runtime policy: {harness_notes}")
    parts.append("Emit the next tool call now.")
    return "\n".join(parts)


def observation_message(tool: str, payload: dict) -> str:
    import json

    rendered = json.dumps(payload, ensure_ascii=False)
    if len(rendered) > 3500:
        rendered = rendered[:3500] + " …[truncated]"
    return f"Tool `{tool}` returned:\n{rendered}\nEmit the next tool call."
