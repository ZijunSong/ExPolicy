"""Self-contained frozen skill executor. Copied from R5.3 public skills; not imported at runtime.

All tool calls go through SearchEnv. No gold lookup, no task-id answers.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

from expolicy.closed_loop.constants import C1, C2, C3
from expolicy.core.experience import Experience, ExperienceCost
from expolicy.environments.search.world import Task
from expolicy.evaluation.metrics import exact_match, normalize

COMPILER_VERSION = "r7.frozen_skill.v1"

_CAP_SPAN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-zA-Z0-9]+){0,4})\b")
_QUOTED = re.compile(r"[\"']([^\"']{2,80})[\"']")
_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_AND_ENT = re.compile(r"(.+?)\s+(?:and|or|vs\.?|versus)\s+(.+?)(?:\s+\?|$)", re.I)
STOP = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "which", "what",
    "who", "when", "where", "how", "did", "does", "was", "were", "is", "are",
    "this", "that", "with", "from", "by", "at",
}


def extract_entities(question: str) -> list[str]:
    found: list[str] = []
    for m in _QUOTED.finditer(question.strip()):
        found.append(m.group(1).strip())
    for m in _CAP_SPAN.finditer(question.strip()):
        span = m.group(1).strip()
        if span.lower() not in STOP and len(span) > 2:
            found.append(span)
    seen: set[str] = set()
    out: list[str] = []
    for item in found:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out[:6]


def split_comparison_entities(question: str) -> tuple[str, str]:
    ents = extract_entities(question)
    m = _AND_ENT.search(question.replace("?", " ?"))
    if m:
        a, b = m.group(1).strip(" .,"), m.group(2).strip(" ?,.")
        a_ents = extract_entities(a) or [a.split(",")[-1].strip()]
        b_ents = extract_entities(b) or [b.split(",")[-1].strip()]
        return a_ents[-1], b_ents[0]
    if len(ents) >= 2:
        return ents[0], ents[1]
    if ents:
        return ents[0], ents[0]
    toks = [t for t in re.findall(r"[A-Za-z][A-Za-z]+", question) if t.lower() not in STOP]
    if len(toks) >= 2:
        return toks[0], toks[1]
    return question, question


def relation_tail(question: str, primary: str) -> str:
    q = question
    if primary:
        q = re.sub(re.escape(primary), " ", q, flags=re.I)
    q = re.sub(r"\b(who|what|which|where|when|how|whom)\b", " ", q, flags=re.I)
    q = re.sub(r"[?]", " ", q)
    toks = [t for t in q.split() if t.lower() not in STOP]
    return " ".join(toks[:8]).strip()


def extract_bridge(text: str, question: str, visited: set[str]) -> str | None:
    q_norm = set(normalize(question).split())
    titles_and_caps = []
    for m in _CAP_SPAN.finditer(text[:1500]):
        span = m.group(1).strip()
        if normalize(span) in visited:
            continue
        if set(normalize(span).split()) <= q_norm:
            continue
        if span.lower() in STOP:
            continue
        titles_and_caps.append(span)
    return titles_and_caps[0] if titles_and_caps else None


def heuristic_answer(question: str, opened: list[dict[str, Any]]) -> str:
    blob = "\n".join((d.get("title") or "") + ". " + (d.get("text") or "") for d in opened)
    q = question.lower()
    if any(w in q for w in ("year", "when", "which year")):
        years = _YEAR.findall(blob)
        if years:
            return years[0]
    if any(w in q for w in ("how many", "how much", "population", "number")):
        nums = _NUMBER.findall(blob)
        if nums:
            return nums[0]
    kws = [t for t in normalize(question).split() if t not in STOP]
    best = ""
    best_score = 0
    for sent in re.split(r"(?<=[.!?])\s+", blob):
        ns = normalize(sent)
        score = sum(1 for k in kws if k in ns)
        if score > best_score and len(sent.split()) <= 20:
            caps = extract_entities(sent)
            if caps:
                best = caps[0]
                best_score = score
    return best


def llm_synthesize(model: Any, question: str, opened: list[dict[str, Any]]) -> str:
    if model is None:
        return ""
    evidence = []
    for doc in opened[:6]:
        title = doc.get("title") or ""
        text = (doc.get("text") or "")[:900]
        evidence.append(f"[{title}] {text}")
    prompt = (
        "Answer the question using only the evidence. "
        "Return a short span (name, number, date, yes/no, or short phrase). "
        "No explanation.\n\n"
        f"Question: {question}\n\nEvidence:\n" + "\n\n".join(evidence)
    )
    try:
        result = model.generate(
            [{"role": "user", "content": prompt}],
            max_new_tokens=32,
            temperature=0.0,
        )
        text = (result.text or "").strip().splitlines()[0].strip()
        text = re.sub(r"^(answer\s*:\s*)", "", text, flags=re.I)
        return text.strip().strip('"')
    except Exception:
        return ""


def execute_skill(
    capability: str,
    env: Any,
    *,
    task: Task,
    synthesizer: Callable[..., str] | None = None,
    max_opens: int = 6,
    max_hops: int | None = None,
) -> dict[str, Any]:
    question = task.question
    hops = int(max_hops if max_hops is not None else (4 if capability == C3 else 2))
    opened: list[dict[str, Any]] = []
    searches: list[dict[str, Any]] = []
    visited_entities: set[str] = set()
    visited_docs: set[str] = set()
    evidence_chain: list[str] = []
    planned = {
        C1: ["search", "open", "search", "open", "answer"],
        C2: ["search", "open", "search", "open", "answer"],
        C3: ["search", "open", "search", "open", "search", "open", "answer"],
    }[capability]
    executed: list[str] = []

    def do_search(q: str) -> dict[str, Any]:
        result = env.step("search", {"query": q})
        payload = result.payload if result.ok else {"error": result.error, "hits": []}
        payload = dict(payload)
        payload["query"] = q
        searches.append(payload)
        executed.append("search")
        return payload

    def do_open(hits: list[dict[str, Any]], *, prefer: str = "") -> dict[str, Any] | None:
        ranked = list(hits or [])
        if prefer:
            ranked = sorted(
                ranked,
                key=lambda h: (0 if prefer.lower() in (h.get("title") or "").lower() else 1, int(h.get("rank") or 99)),
            )
        for hit in ranked:
            did = str(hit.get("document_id") or "")
            if not did or did in visited_docs:
                continue
            result = env.step("open", {"document_id": did})
            if not result.ok:
                continue
            payload = result.payload
            visited_docs.add(did)
            opened.append(payload)
            evidence_chain.append(payload.get("title") or did)
            executed.append("open")
            return payload
        return None

    if capability == C2:
        a, b = split_comparison_entities(question)
        state = {"entity_a": a, "entity_b": b}
        visited_entities.update({normalize(a), normalize(b)})
        for ent in (a, b):
            hop = do_search(ent)
            do_open(hop.get("hits") or [], prefer=ent.split()[0] if ent else "")
        answer = heuristic_answer(question, opened)
    elif capability == C3:
        ents = extract_entities(question)
        primary = ents[0] if ents else " ".join(question.split()[:6])
        remaining = relation_tail(question, primary)
        state = {"subgoal": remaining, "primary": primary}
        current = primary
        for hop_i in range(hops):
            if normalize(current) in visited_entities and hop_i > 0:
                break
            visited_entities.add(normalize(current))
            hop = do_search(f"{current} {remaining}".strip())
            doc = do_open(hop.get("hits") or [], prefer=current.split()[0])
            if not doc:
                hop = do_search(current)
                doc = do_open(hop.get("hits") or [])
            nxt = extract_bridge((doc or {}).get("text") or "", question, visited_entities)
            if not nxt:
                break
            current = nxt
            remaining = relation_tail(question, current)
            state["remaining_subgoal"] = remaining
        answer = heuristic_answer(question, opened)
    else:
        ents = extract_entities(question)
        primary = ents[0] if ents else " ".join(question.split()[:8])
        remaining = relation_tail(question, primary)
        state = {"primary": primary, "relation": remaining}
        visited_entities.add(normalize(primary))
        hop1 = do_search(primary)
        doc1 = do_open(hop1.get("hits") or [], prefer=primary.split()[0] if primary else "")
        bridge = extract_bridge((doc1 or {}).get("text") or "", question, visited_entities)
        if not bridge and ents[1:]:
            bridge = ents[1]
        state["bridge"] = bridge
        if bridge:
            visited_entities.add(normalize(bridge))
            hop2 = do_search(f"{bridge} {remaining}".strip())
            do_open(hop2.get("hits") or [], prefer=bridge.split()[0])
        if len(opened) < 2:
            rec = do_search(f"{primary} {remaining}".strip())
            do_open(rec.get("hits") or [])
            executed.append("recovery")
        answer = heuristic_answer(question, opened)

    if synthesizer is not None:
        llm_ans = synthesizer(question, opened)
        if llm_ans:
            answer = llm_ans
    executed.append("answer")
    return {
        "ok": bool(answer) and bool(opened),
        "answer": answer,
        "opened": opened,
        "searches": searches,
        "state": {
            **state,
            "visited_entities": sorted(visited_entities),
            "visited_documents": sorted(visited_docs),
            "evidence_chain": evidence_chain,
        },
        "compliance": {"planned": planned, "executed": executed},
        "evidence_complete": len(opened) >= (3 if capability == C3 else 2),
        "n_open": len(opened),
        "n_search": len(searches),
        "max_opens": max_opens,
    }


def run_frozen_skill(
    capability: str,
    env: Any,
    task: Task,
    *,
    model: Any | None = None,
) -> Experience:
    env.reset(task)
    started = time.perf_counter()

    def _syn(question: str, opened: list[dict[str, Any]]) -> str:
        return llm_synthesize(model, question, opened)

    result = execute_skill(capability, env, task=task, synthesizer=_syn if model is not None else None)
    predicted = str(result.get("answer") or "").strip()
    exp = Experience(
        task_id=task.task_id,
        question=task.question,
        gold_answer=task.gold_answer,
        capability=task.capability,
        stage=task.split and 0 or 0,
        split=task.split,
        predicted_answer=predicted,
        capability_context={"skill": capability, "executable": True, "frozen_r7": True},
    )
    for payload in result.get("searches") or []:
        exp.tool_calls.append({"tool": "search", "arguments": {"query": payload.get("query")}, "ok": True})
        exp.observations.append({"tool": "search", "payload": payload})
        exp.actions.append({"tool": "search", "arguments": {"query": payload.get("query")}})
    for payload in result.get("opened") or []:
        exp.tool_calls.append({"tool": "open", "arguments": {"document_id": payload.get("document_id")}, "ok": True})
        exp.observations.append({"tool": "open", "payload": payload})
        exp.actions.append({"tool": "open", "arguments": {"document_id": payload.get("document_id")}})
        exp.evidence.append(
            {"document_id": payload.get("document_id"), "title": payload.get("title"), "source_type": "wikipedia"}
        )
    exp.actions.append({"tool": "answer", "arguments": {"text": predicted}})
    gold = task.gold_answer or ""
    success = exact_match(predicted, gold) or any(exact_match(predicted, a) for a in (task.aliases or []))
    exp.success = success
    exp.reward = 1.0 if success else 0.0
    exp.outcome = {
        "predicted": predicted,
        "success": success,
        "skill": capability,
        "compliance": result.get("compliance"),
        "evidence_complete": result.get("evidence_complete"),
        "dispatch": f"frozen_skill:{capability}",
    }
    exp.cost = ExperienceCost(
        environment_steps=len(exp.tool_calls),
        tool_calls=len(exp.tool_calls),
        latency=time.perf_counter() - started,
    )
    return exp
