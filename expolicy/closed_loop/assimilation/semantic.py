"""SEMANTIC_MEMORY: compress a batch into a capability note. Never includes gold/future."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from expolicy.closed_loop.constants import SEMANTIC_MEMORY
from expolicy.closed_loop.experience import ExperienceBatchSummary
from expolicy.closed_loop.state import ExPolicyState
from expolicy.core.experience import Experience


FORBIDDEN = ("gold", "future_test", "future_probe", "dataset")


def _heuristic_note(capability: str, summary: ExperienceBatchSummary, experiences: Sequence[Experience]) -> str:
    queries = []
    for exp in experiences:
        for call in exp.tool_calls:
            if call.get("tool") == "search":
                q = str((call.get("arguments") or {}).get("query") or "").strip()
                if q:
                    queries.append(q)
    uniq = list(dict.fromkeys(queries))[:8]
    return (
        f"For {capability} tasks: run search/open hops before answering. "
        f"Batch success_rate={summary.success_rate:.2f}, repeat_query_rate={summary.repeat_query_rate:.2f}, "
        f"premature_answer_rate={summary.premature_answer_rate:.2f}, mean_searches={summary.mean_searches:.1f}. "
        f"Recent query patterns: {uniq}. "
        "Do not copy previous answers; extract the missing bridge/entity from opened evidence."
    )


def llm_note(model: Any, capability: str, experiences: Sequence[Experience]) -> str:
    if model is None:
        return ""
    traces = []
    for exp in experiences[:8]:
        tools = [str(c.get("tool")) for c in exp.tool_calls]
        queries = [str((c.get("arguments") or {}).get("query") or "") for c in exp.tool_calls if c.get("tool") == "search"]
        traces.append(
            {
                "capability": exp.capability,
                "question": exp.question,
                "tools": tools,
                "queries": queries[:6],
                "success": bool(exp.success),
                "predicted_empty": not bool((exp.predicted_answer or "").strip()),
            }
        )
    prompt = (
        "Write one short reusable procedure note for a search agent. "
        "Use only the batch traces. Do not mention gold answers, datasets, or future tests. "
        "Do not restate any answer span.\n\n"
        f"Capability: {capability}\n"
        f"Traces: {json.dumps(traces, ensure_ascii=False)[:4000]}"
    )
    try:
        result = model.generate(
            [{"role": "user", "content": prompt}],
            max_new_tokens=120,
            temperature=0.0,
        )
        text = (result.text or "").strip()
        low = text.lower()
        if any(bad in low for bad in FORBIDDEN):
            return ""
        return text.splitlines()[0][:400]
    except Exception:
        return ""


def apply_semantic(
    store_dir: Path,
    state: ExPolicyState,
    summary: ExperienceBatchSummary,
    experiences: Sequence[Experience],
    *,
    model: Any | None = None,
) -> dict[str, Any]:
    store_dir.mkdir(parents=True, exist_ok=True)
    note = llm_note(model, summary.capability, experiences) or _heuristic_note(summary.capability, summary, experiences)
    rec = {
        "capability": summary.capability,
        "text": note,
        "timestamp": time.time(),
        "n": summary.n,
        "success_rate": summary.success_rate,
    }
    path = store_dir / "notes.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    cap = summary.capability
    state.semantic_memory_size[cap] = int(state.semantic_memory_size.get(cap) or 0) + 1
    return {"action": SEMANTIC_MEMORY, "changed": True, "n": 1, "note": note, "path": str(path)}


def active_notes(store_dir: Path, *, capability: str | None = None, k: int = 8) -> list[str]:
    path = store_dir / "notes.jsonl"
    if not path.exists():
        return []
    notes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if capability and rec.get("capability") not in {capability, None}:
            # still usable as general rule
            pass
        notes.append(str(rec.get("text") or ""))
    return [n for n in notes if n][-k:]
