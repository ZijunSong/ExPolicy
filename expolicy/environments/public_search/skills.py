"""Capability-specific executable search procedures for public Wikipedia QA.

Control-flow only. No Helion entity schema, field names, or template strings.
Gold entities are never hard-coded.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Callable

from expolicy.assimilation.carriers.skill_store import Skill, SkillStore
from expolicy.core.experience import Experience, ExperienceCost
from expolicy.environments.search.world import Task
from expolicy.evaluation.metrics import exact_match, normalize

C1 = "bridge_decomposition"
C2 = "comparison_alignment"
C3 = "long_chain_search"
PUBLIC_CAPS = (C1, C2, C3)
COMPILER_VERSION = "r5.3.public_skill.v1"

_CAP_SPAN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-zA-Z0-9]+){0,4})\b")
_QUOTED = re.compile(r"[\"']([^\"']{2,80})[\"']")
_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_COMPARE = re.compile(
    r"\b(?:compare|which|who (?:is|was|are|were) (?:older|younger|taller|larger|first|earlier)|or )\b",
    re.I,
)
_AND_ENT = re.compile(
    r"(.+?)\s+(?:and|or|vs\.?|versus)\s+(.+?)(?:\s+\?|$)",
    re.I,
)
STOP = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "which", "what",
    "who", "when", "where", "how", "did", "does", "was", "were", "is", "are",
    "this", "that", "with", "from", "by", "at",
}


def _stable_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def skill_c1_bridge() -> Skill:
    return Skill(
        name="bridge_search_procedure",
        capability=C1,
        executable=True,
        preconditions=["needs_bridge_entity"],
        decomposition_steps=[
            "identify primary entity / relation",
            "first-hop search",
            "read candidate evidence",
            "extract bridge entity",
            "search bridge + unresolved relation",
            "collect supporting evidence",
            "synthesize answer",
        ],
        tool_sequence=["search", "open", "search", "open", "answer"],
        state_variables=["primary", "bridge", "relation", "visited_entities", "visited_documents", "evidence_chain"],
        evidence_requirements=["hop1_passage", "hop2_passage"],
        stopping_condition="bridge_and_relation_bound",
        answer_synthesis_rule="answer only from opened evidence after a bridge hop",
        verifier="two_hops_open",
        failure_recovery="if bridge missing, open next hit and retry first-hop entity variants",
        provenance=["r5.3_public", COMPILER_VERSION],
    )


def skill_c2_comparison() -> Skill:
    return Skill(
        name="comparison_alignment_procedure",
        capability=C2,
        executable=True,
        preconditions=["two_entities_to_compare"],
        decomposition_steps=[
            "parse entity A and entity B",
            "search entity A",
            "search entity B",
            "extract the same comparison attribute",
            "check unit / date / qualifier",
            "compare",
            "output answer",
        ],
        tool_sequence=["search", "open", "search", "open", "answer"],
        state_variables=["entity_a", "entity_b", "attribute", "value_a", "value_b"],
        evidence_requirements=["entity_a_passage", "entity_b_passage"],
        stopping_condition="both_entities_opened",
        answer_synthesis_rule="compare aligned attributes from independently retrieved evidence",
        verifier="two_entities_open",
        failure_recovery="if one entity missing, search the leftover name plus the comparison attribute",
        provenance=["r5.3_public", COMPILER_VERSION],
    )


def skill_c3_long_chain() -> Skill:
    return Skill(
        name="long_chain_search_procedure",
        capability=C3,
        executable=True,
        preconditions=["needs_at_least_three_hops"],
        decomposition_steps=[
            "initialize unresolved subgoal",
            "search/open",
            "extract next-hop entity",
            "update evidence state",
            "check final target",
            "continue if unresolved",
            "stop at hop/tool budget",
            "synthesize",
        ],
        tool_sequence=["search", "open", "search", "open", "search", "open", "answer"],
        state_variables=["subgoal", "visited_entities", "visited_documents", "evidence_chain", "remaining_subgoal"],
        evidence_requirements=["chain_of_at_least_three_hops"],
        stopping_condition="final_target_or_budget",
        answer_synthesis_rule="synthesize from the ordered evidence chain without repeating visited entities",
        verifier="chain_open",
        failure_recovery="skip already visited titles; try next ranked hit",
        provenance=["r5.3_public", COMPILER_VERSION],
    )


PRIMITIVES = {
    C1: skill_c1_bridge,
    C2: skill_c2_comparison,
    C3: skill_c3_long_chain,
}


def count_only_public_skill(cap: str, *, n_support: int, min_support: int, version: int) -> Skill:
    skill = PRIMITIVES[cap]()
    skill.status = "active"
    skill.executable = True
    skill.n_support = int(n_support)
    skill.version = int(version)
    skill.compiler_version = COMPILER_VERSION
    skill.runtime_config = {
        "primitive_library": "r5_3_public_v1",
        "fit_mode": "count_only",
        "n_support_used": n_support,
        "max_opens": 6,
        "max_hops": 4 if cap == C3 else 2,
    }
    skill.activation_reason = (
        f"count_only unique_task_ids={n_support}>={min_support}; capability={cap}; duplicates do not count toward K"
    )
    skill.provenance = [f"compiler:{COMPILER_VERSION}", f"primitive:{skill.name}", "fit_mode:count_only"]
    skill.source_experience_ids = []
    skill.skill_id = _stable_id(
        {"compiler": COMPILER_VERSION, "capability": cap, "version": version, "fit_mode": "count_only", "n_support": n_support}
    )
    return skill


def extract_entities(question: str) -> list[str]:
    q = question.strip()
    found: list[str] = []
    for m in _QUOTED.finditer(q):
        found.append(m.group(1).strip())
    for m in _CAP_SPAN.finditer(q):
        span = m.group(1).strip()
        if span.lower() not in STOP and len(span) > 2:
            found.append(span)
    # de-dup preserve order
    seen = set()
    out = []
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
        # keep the trailing entity-like tail of each side
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
    # yes/no comparison
    if q.startswith("are ") or q.startswith("is ") or q.startswith("was ") or q.startswith("were "):
        return ""
    # last-title fallback is too naive; prefer a capitalized span near question keywords
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


def execute_public_skill(
    skill: Skill,
    env,
    *,
    task: Task,
    synthesizer: Callable[..., str] | None = None,
) -> dict[str, Any]:
    question = task.question
    cfg = dict(skill.runtime_config or {})
    max_opens = int(cfg.get("max_opens") or 6)
    max_hops = int(cfg.get("max_hops") or (4 if skill.capability == C3 else 2))
    opened: list[dict[str, Any]] = []
    searches: list[dict[str, Any]] = []
    visited_entities: set[str] = set()
    visited_docs: set[str] = set()
    evidence_chain: list[str] = []
    compliance = {"planned": list(skill.tool_sequence), "executed": []}

    def do_search(q: str) -> dict[str, Any]:
        result = env.step("search", {"query": q})
        payload = result.payload if result.ok else {"error": result.error, "hits": []}
        searches.append(payload)
        compliance["executed"].append("search")
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
            compliance["executed"].append("open")
            if len(opened) >= max_opens:
                return payload
            return payload
        return None

    if skill.capability == C2:
        a, b = split_comparison_entities(question)
        state = {"entity_a": a, "entity_b": b, "visited_entities": [a, b]}
        visited_entities.update({normalize(a), normalize(b)})
        for ent in (a, b):
            hop = do_search(ent)
            do_open(hop.get("hits") or [], prefer=ent.split()[0] if ent else "")
        answer = heuristic_answer(question, opened)
    elif skill.capability == C3:
        ents = extract_entities(question)
        primary = ents[0] if ents else " ".join(question.split()[:6])
        remaining = relation_tail(question, primary)
        state = {"subgoal": remaining, "primary": primary}
        current = primary
        for _hop in range(max_hops):
            if normalize(current) in visited_entities and _hop > 0:
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
            compliance["executed"].append("recovery")
        answer = heuristic_answer(question, opened)

    if synthesizer is not None:
        llm_ans = synthesizer(question, opened)
        if llm_ans:
            answer = llm_ans
    elif not answer:
        answer = heuristic_answer(question, opened)

    ok = bool(answer) and bool(opened)
    if ok:
        skill.success_count += 1
    else:
        skill.fail_count += 1
    return {
        "ok": ok,
        "skill": skill.name,
        "answer": answer,
        "opened": opened,
        "searches": searches,
        "state": {**state, "visited_entities": sorted(visited_entities), "visited_documents": sorted(visited_docs), "evidence_chain": evidence_chain},
        "compliance": compliance,
        "evidence_complete": len(opened) >= (3 if skill.capability == C3 else 2),
        "n_open": len(opened),
        "n_search": len(searches),
        "procedure_compliance": float(len(set(compliance["executed"]) & {"search", "open"}) >= 2),
    }


def run_public_skill_on_task(
    skill: Skill,
    env,
    task: Task,
    *,
    model: Any | None = None,
) -> Experience:
    env.reset(task)
    started = time.perf_counter()

    def _syn(question: str, opened: list[dict[str, Any]]) -> str:
        return llm_synthesize(model, question, opened)

    result = execute_public_skill(skill, env, task=task, synthesizer=_syn if model is not None else None)
    predicted = str(result.get("answer") or "").strip()
    exp = Experience(
        task_id=task.task_id,
        question=task.question,
        gold_answer=task.gold_answer,
        capability=task.capability,
        stage=task.stage,
        split=task.split,
        predicted_answer=predicted,
        capability_context={"skill": skill.name, "executable": True, "public": True},
    )
    for payload in result.get("searches") or []:
        exp.tool_calls.append({"tool": "search", "arguments": {"query": payload.get("query")}, "ok": True})
        exp.observations.append({"tool": "search", "payload": payload})
        exp.actions.append({"tool": "search", "arguments": {"query": payload.get("query")}})
    for payload in result.get("opened") or []:
        exp.tool_calls.append({"tool": "open", "arguments": {"document_id": payload.get("document_id")}, "ok": True})
        exp.observations.append({"tool": "open", "payload": payload})
        exp.actions.append({"tool": "open", "arguments": {"document_id": payload.get("document_id")}})
        exp.evidence.append({"document_id": payload.get("document_id"), "title": payload.get("title"), "source_type": "wikipedia"})
    exp.actions.append({"tool": "answer", "arguments": {"text": predicted}})
    gold = task.gold_answer or ""
    success = exact_match(predicted, gold) or any(exact_match(predicted, a) for a in (task.aliases or []))
    exp.success = success
    exp.reward = 1.0 if success else 0.0
    exp.outcome = {
        "predicted": predicted,
        "gold": gold,
        "success": success,
        "skill": skill.name,
        "compliance": result.get("compliance"),
        "evidence_complete": result.get("evidence_complete"),
    }
    exp.cost = ExperienceCost(
        environment_steps=len(exp.tool_calls),
        tool_calls=len(exp.tool_calls),
        latency=time.perf_counter() - started,
    )
    return exp


def install_count_only(store: SkillStore, acquired: list[Experience], *, min_support: int) -> tuple[SkillStore, dict[str, int]]:
    first: dict[str, int] = {}
    seen: dict[str, set[str]] = {cap: set() for cap in PUBLIC_CAPS}
    for i, exp in enumerate(acquired, 1):
        cap = exp.capability if exp.capability in seen else None
        if not cap or not exp.task_id:
            continue
        seen[cap].add(str(exp.task_id))
        if cap not in first and len(seen[cap]) >= min_support:
            first[cap] = i
    versions: dict[str, int] = {}
    for cap in PUBLIC_CAPS:
        n_unique = len(seen[cap])
        if n_unique < min_support:
            continue
        versions[cap] = versions.get(cap, 0) + 1
        store.install(count_only_public_skill(cap, n_support=n_unique, min_support=min_support, version=versions[cap]))
    return store, first
