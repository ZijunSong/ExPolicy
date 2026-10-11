"""EPISODIC_MEMORY: write trajectory summaries (never gold) and retrieve into prompts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from expolicy.closed_loop.constants import EPISODIC_MEMORY
from expolicy.closed_loop.state import ExPolicyState
from expolicy.core.experience import Experience


def _card(exp: Experience) -> dict[str, Any]:
    queries = [str((c.get("arguments") or {}).get("query") or "") for c in exp.tool_calls if c.get("tool") == "search"]
    n_open = sum(1 for c in exp.tool_calls if c.get("tool") == "open")
    titles = [str(ev.get("title") or "") for ev in exp.evidence if ev.get("title")]
    return {
        "experience_id": exp.experience_id,
        "capability": exp.capability,
        "question": exp.question,
        "predicted_answer": exp.predicted_answer,
        "success": bool(exp.success),
        "n_search": len(queries),
        "n_open": n_open,
        "queries": queries[:6],
        "opened_titles": titles[:6],
        "procedure": "search then open evidence before answering; do not copy stored answers",
        "timestamp": time.time(),
    }


def apply_episodic(
    store_dir: Path,
    state: ExPolicyState,
    experiences: Sequence[Experience],
) -> dict[str, Any]:
    store_dir.mkdir(parents=True, exist_ok=True)
    path = store_dir / "records.jsonl"
    n = 0
    with path.open("a", encoding="utf-8") as handle:
        for exp in experiences:
            card = _card(exp)
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
            cap = str(exp.capability or state.last_batch_capability or "")
            if cap:
                state.episodic_memory_size[cap] = int(state.episodic_memory_size.get(cap) or 0) + 1
            n += 1
    return {"action": EPISODIC_MEMORY, "changed": n > 0, "n": n, "path": str(path)}


def load_cards(store_dir: Path) -> list[dict[str, Any]]:
    path = store_dir / "records.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def retrieve_cards(store_dir: Path, query: str, *, capability: str | None = None, k: int = 3) -> list[str]:
    cards = load_cards(store_dir)
    q = (query or "").lower()
    q_toks = set(q.split())
    scored = []
    for card in cards:
        if capability and card.get("capability") and card.get("capability") != capability:
            cap_bonus = 0.0
        else:
            cap_bonus = 1.0 if capability and card.get("capability") == capability else 0.2
        text = (card.get("question") or "") + " " + " ".join(card.get("queries") or [])
        toks = set(text.lower().split())
        overlap = len(q_toks & toks) / max(1, len(q_toks))
        scored.append((cap_bonus + overlap, card))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for _, card in scored[:k]:
        titles = ", ".join(card.get("opened_titles") or [])
        queries = "; ".join(card.get("queries") or [])
        out.append(
            "Procedure card (do NOT copy any stored answer): "
            f"capability={card.get('capability')}; prior_queries={queries}; "
            f"opened_titles={titles}; success_hint={bool(card.get('success'))}; "
            f"{card.get('procedure')}"
        )
    return out
