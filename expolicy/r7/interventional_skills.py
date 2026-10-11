"""Backend 1: interventional executable skills.

Skills may only change search/open/evidence state. The policy model always
generates the final answer. Gold spans, dataset names, and task ids are never
used for dispatch or answer extraction.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Callable

from expolicy.assimilation.carriers.skill_store import Skill, SkillStore
from expolicy.core.experience import Experience, ExperienceCost
from expolicy.environments.public_search.skills import (
    extract_bridge,
    extract_entities,
    relation_tail,
    split_comparison_entities,
)
from expolicy.environments.search.world import Task
from expolicy.evaluation.metrics import exact_match, normalize
from expolicy.r7.constants import C1, C2, C3, COMPILER_INTERVENTIONAL, PUBLIC_CAPS

_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")


def _stable_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def skill_s_bridge(*, version: int = 1, runtime_config: dict[str, Any] | None = None) -> Skill:
    cfg = {
        "max_opens": 6,
        "max_hops": 2,
        "force_bridge_search": True,
        "return_control_to_model": True,
        **(runtime_config or {}),
    }
    skill = Skill(
        name="S_bridge",
        capability=C1,
        executable=True,
        status="active",
        preconditions=["needs_bridge_entity"],
        decomposition_steps=[
            "detect partial bridge evidence",
            "extract bridge entity from opened text",
            "search second-hop evidence",
            "return evidence slots to the policy model",
        ],
        tool_sequence=["search", "open", "search", "open"],
        state_variables=["primary", "bridge", "relation", "evidence_slots"],
        evidence_requirements=["hop1_passage", "hop2_passage"],
        stopping_condition="bridge_and_relation_bound_or_budget",
        answer_synthesis_rule="policy_model_only",
        verifier="two_hops_open",
        failure_recovery="if bridge missing, open next hit and retry first-hop entity variants",
        provenance=["r7_interventional", COMPILER_INTERVENTIONAL],
        compiler_version=COMPILER_INTERVENTIONAL,
        version=version,
        runtime_config=cfg,
        activation_reason="interventional_bridge; does_not_emit_final_answer",
    )
    skill.skill_id = _stable_id(
        {"compiler": COMPILER_INTERVENTIONAL, "name": skill.name, "version": version, "cfg": cfg}
    )
    return skill


def skill_s_compare(*, version: int = 1, runtime_config: dict[str, Any] | None = None) -> Skill:
    cfg = {
        "max_opens": 6,
        "force_both_operands": True,
        "return_control_to_model": True,
        **(runtime_config or {}),
    }
    skill = Skill(
        name="S_compare",
        capability=C2,
        executable=True,
        status="active",
        preconditions=["two_entities_to_compare"],
        decomposition_steps=[
            "identify comparison operands from the question text",
            "retrieve evidence for operand A",
            "retrieve evidence for operand B",
            "align attribute/time/unit slots",
            "return structured evidence to the policy model",
        ],
        tool_sequence=["search", "open", "search", "open"],
        state_variables=["entity_a", "entity_b", "attribute", "value_a", "value_b"],
        evidence_requirements=["entity_a_passage", "entity_b_passage"],
        stopping_condition="both_entities_opened",
        answer_synthesis_rule="policy_model_only",
        verifier="two_entities_open",
        failure_recovery="if one entity missing, search leftover name plus comparison attribute",
        provenance=["r7_interventional", COMPILER_INTERVENTIONAL],
        compiler_version=COMPILER_INTERVENTIONAL,
        version=version,
        runtime_config=cfg,
        activation_reason="interventional_compare; does_not_emit_final_answer",
    )
    skill.skill_id = _stable_id(
        {"compiler": COMPILER_INTERVENTIONAL, "name": skill.name, "version": version, "cfg": cfg}
    )
    return skill


def skill_s_long_chain(*, version: int = 1, runtime_config: dict[str, Any] | None = None) -> Skill:
    cfg = {
        "max_opens": 8,
        "max_hops": 4,
        "return_control_to_model": True,
        **(runtime_config or {}),
    }
    skill = Skill(
        name="S_long_chain",
        capability=C3,
        executable=True,
        status="active",
        preconditions=["needs_at_least_three_hops"],
        decomposition_steps=[
            "maintain unresolved evidence slots",
            "iteratively search the next missing relation",
            "stop on evidence sufficiency or budget",
            "return evidence graph to the policy model",
        ],
        tool_sequence=["search", "open", "search", "open", "search", "open"],
        state_variables=["subgoal", "visited_entities", "evidence_chain", "remaining_subgoal"],
        evidence_requirements=["chain_of_at_least_three_hops"],
        stopping_condition="final_target_or_budget",
        answer_synthesis_rule="policy_model_only",
        verifier="chain_open",
        failure_recovery="skip already visited titles; try next ranked hit",
        provenance=["r7_interventional", COMPILER_INTERVENTIONAL],
        compiler_version=COMPILER_INTERVENTIONAL,
        version=version,
        runtime_config=cfg,
        activation_reason="interventional_long_chain; does_not_emit_final_answer",
    )
    skill.skill_id = _stable_id(
        {"compiler": COMPILER_INTERVENTIONAL, "name": skill.name, "version": version, "cfg": cfg}
    )
    return skill


PRIMITIVES: dict[str, Callable[..., Skill]] = {
    C1: skill_s_bridge,
    C2: skill_s_compare,
    C3: skill_s_long_chain,
}


def install_interventional(store: SkillStore, *, runtime_by_cap: dict[str, dict[str, Any]] | None = None) -> SkillStore:
    runtime_by_cap = runtime_by_cap or {}
    for cap, fn in PRIMITIVES.items():
        store.install(fn(runtime_config=runtime_by_cap.get(cap)))
    return store


def _do_search(env, q: str, searches: list, executed: list) -> dict[str, Any]:
    result = env.step("search", {"query": q})
    payload = result.payload if result.ok else {"error": result.error, "hits": [], "query": q}
    payload = dict(payload)
    payload.setdefault("query", q)
    searches.append(payload)
    executed.append("search")
    return payload


def _do_open(
    env,
    hits: list[dict[str, Any]],
    *,
    prefer: str,
    visited_docs: set[str],
    opened: list[dict[str, Any]],
    evidence_chain: list[str],
    executed: list[str],
    max_opens: int,
) -> dict[str, Any] | None:
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


def run_interventions(skill: Skill, env, task: Task) -> dict[str, Any]:
    """Change action/evidence state only. Never produce a final answer."""
    question = task.question
    cfg = dict(skill.runtime_config or {})
    max_opens = int(cfg.get("max_opens") or 6)
    max_hops = int(cfg.get("max_hops") or (4 if skill.capability == C3 else 2))
    extra_suffixes = list(cfg.get("query_suffixes") or [])
    opened: list[dict[str, Any]] = []
    searches: list[dict[str, Any]] = []
    visited_entities: set[str] = set()
    visited_docs: set[str] = set()
    evidence_chain: list[str] = []
    executed: list[str] = []
    slots: dict[str, Any] = {}

    def search(q: str) -> dict[str, Any]:
        return _do_search(env, q, searches, executed)

    def open_hits(hits: list[dict[str, Any]], prefer: str = "") -> dict[str, Any] | None:
        if len(opened) >= max_opens:
            return None
        return _do_open(
            env,
            hits,
            prefer=prefer,
            visited_docs=visited_docs,
            opened=opened,
            evidence_chain=evidence_chain,
            executed=executed,
            max_opens=max_opens,
        )

    if skill.capability == C2:
        a, b = split_comparison_entities(question)
        slots = {"entity_a": a, "entity_b": b}
        visited_entities.update({normalize(a), normalize(b)})
        attr = relation_tail(question, a)
        slots["attribute"] = attr
        for ent in (a, b):
            q = f"{ent} {attr}".strip() if attr else ent
            hop = search(q)
            doc = open_hits(hop.get("hits") or [], prefer=ent.split()[0] if ent else "")
            if extra_suffixes and (doc is None or len((doc.get("text") or "")) < 40):
                for suf in extra_suffixes[:2]:
                    hop = search(f"{ent} {suf}".strip())
                    open_hits(hop.get("hits") or [], prefer=ent.split()[0] if ent else "")
    elif skill.capability == C3:
        ents = extract_entities(question)
        primary = ents[0] if ents else " ".join(question.split()[:6])
        remaining = relation_tail(question, primary)
        slots = {"subgoal": remaining, "primary": primary}
        current = primary
        for hop_i in range(max_hops):
            if len(opened) >= max_opens:
                break
            if normalize(current) in visited_entities and hop_i > 0:
                break
            visited_entities.add(normalize(current))
            hop = search(f"{current} {remaining}".strip())
            doc = open_hits(hop.get("hits") or [], prefer=current.split()[0])
            if not doc:
                hop = search(current)
                doc = open_hits(hop.get("hits") or [])
            nxt = extract_bridge((doc or {}).get("text") or "", question, visited_entities)
            if not nxt:
                break
            current = nxt
            remaining = relation_tail(question, current)
            slots["remaining_subgoal"] = remaining
        slots["unresolved"] = remaining
    else:
        ents = extract_entities(question)
        primary = ents[0] if ents else " ".join(question.split()[:8])
        remaining = relation_tail(question, primary)
        slots = {"primary": primary, "relation": remaining}
        visited_entities.add(normalize(primary))
        hop1 = search(primary)
        doc1 = open_hits(hop1.get("hits") or [], prefer=primary.split()[0] if primary else "")
        bridge = extract_bridge((doc1 or {}).get("text") or "", question, visited_entities)
        if not bridge and ents[1:]:
            bridge = ents[1]
        slots["bridge"] = bridge
        if bridge:
            visited_entities.add(normalize(bridge))
            hop2 = search(f"{bridge} {remaining}".strip())
            open_hits(hop2.get("hits") or [], prefer=bridge.split()[0])
        if len(opened) < 2:
            rec = search(f"{primary} {remaining}".strip())
            open_hits(rec.get("hits") or [])
            executed.append("recovery")
        for suf in extra_suffixes[:1]:
            if len(opened) >= max_opens:
                break
            hop = search(f"{primary} {suf}".strip())
            open_hits(hop.get("hits") or [])

    return {
        "opened": opened,
        "searches": searches,
        "state": {
            **slots,
            "visited_entities": sorted(visited_entities),
            "visited_documents": sorted(visited_docs),
            "evidence_chain": evidence_chain,
        },
        "executed": executed,
        "n_open": len(opened),
        "n_search": len(searches),
        "answer": None,
        "replaced_model": False,
    }


def synthesize_with_model(model: Any, question: str, opened: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    """Hand control back to the current backbone. Never extract a gold span."""
    if model is None:
        raise RuntimeError("interventional skill requires a policy model for final answer")
    evidence = []
    for doc in opened[:8]:
        title = doc.get("title") or ""
        text = (doc.get("text") or "")[:900]
        evidence.append(f"[{title}] {text}")
    slots = []
    for key in ("primary", "bridge", "entity_a", "entity_b", "attribute", "relation", "remaining_subgoal"):
        if state.get(key):
            slots.append(f"{key}={state[key]}")
    prompt = (
        "You are the policy model. A search skill collected evidence but must NOT answer for you.\n"
        "Answer the question using only the evidence. Return a short span "
        "(name, number, date, yes/no, or short phrase). No explanation.\n\n"
        f"Question: {question}\n"
        f"Evidence slots: {'; '.join(slots) if slots else 'none'}\n\n"
        "Evidence:\n" + ("\n\n".join(evidence) if evidence else "(no documents opened)")
    )
    result = model.generate(
        [{"role": "user", "content": prompt}],
        max_new_tokens=64,
        temperature=0.0,
    )
    text = (result.text or "").strip().splitlines()
    answer = text[0].strip() if text else ""
    answer = re.sub(r"^(answer\s*:\s*)", "", answer, flags=re.I).strip().strip('"')
    # strip thinking-model leftovers if a final line exists
    if "</think>" in (result.text or ""):
        after = (result.text or "").split("</think>")[-1].strip()
        if after:
            answer = after.splitlines()[0].strip()
            answer = re.sub(r"^(answer\s*:\s*)", "", answer, flags=re.I).strip().strip('"')
    return {
        "answer": answer,
        "model_tokens_after_skill": int(result.completion_tokens or 0),
        "prompt_tokens": int(result.prompt_tokens or 0),
        "raw": result.text,
    }


def run_interventional_skill_on_task(skill: Skill, env, task: Task, *, model: Any) -> Experience:
    env.reset(task)
    started = time.perf_counter()
    inter = run_interventions(skill, env, task)
    syn = synthesize_with_model(model, task.question, inter["opened"], inter["state"])
    predicted = str(syn["answer"] or "").strip()
    if syn["model_tokens_after_skill"] <= 0:
        raise RuntimeError("model_tokens_after_skill == 0; interventional contract violated")
    exp = Experience(
        task_id=task.task_id,
        question=task.question,
        gold_answer=task.gold_answer,
        capability=task.capability,
        stage=task.stage,
        split=task.split,
        predicted_answer=predicted,
        capability_context={
            "skill": skill.name,
            "executable": True,
            "interventional": True,
            "replaced_model": False,
            "model_tokens_after_skill": syn["model_tokens_after_skill"],
        },
    )
    for payload in inter.get("searches") or []:
        exp.tool_calls.append({"tool": "search", "arguments": {"query": payload.get("query")}, "ok": True})
        exp.observations.append({"tool": "search", "payload": payload})
        exp.actions.append({"tool": "search", "arguments": {"query": payload.get("query")}})
    for payload in inter.get("opened") or []:
        exp.tool_calls.append({"tool": "open", "arguments": {"document_id": payload.get("document_id")}, "ok": True})
        exp.observations.append({"tool": "open", "payload": payload})
        exp.actions.append({"tool": "open", "arguments": {"document_id": payload.get("document_id")}})
        exp.evidence.append(
            {
                "document_id": payload.get("document_id"),
                "title": payload.get("title"),
                "source_type": "wikipedia",
            }
        )
    exp.actions.append({"tool": "answer", "arguments": {"text": predicted}, "source": "policy_model"})
    gold = task.gold_answer or ""
    success = exact_match(predicted, gold) or any(exact_match(predicted, a) for a in (task.aliases or []))
    exp.success = success
    exp.reward = 1.0 if success else 0.0
    exp.outcome = {
        "predicted": predicted,
        "gold": gold,
        "success": success,
        "skill": skill.name,
        "executed": inter.get("executed"),
        "model_tokens_after_skill": syn["model_tokens_after_skill"],
        "replaced_model": False,
        "final_answer_source": "policy_model",
    }
    exp.cost = ExperienceCost(
        tokens=int(syn["prompt_tokens"]) + int(syn["model_tokens_after_skill"]),
        prompt_tokens=int(syn["prompt_tokens"]),
        completion_tokens=int(syn["model_tokens_after_skill"]),
        environment_steps=len(exp.tool_calls),
        tool_calls=len(exp.tool_calls),
        latency=time.perf_counter() - started,
    )
    if success:
        skill.success_count += 1
    else:
        skill.fail_count += 1
    return exp


def skill_has_leakage(skill: Skill) -> list[str]:
    from expolicy.r7.constants import LEAK_KEYS

    issues: list[str] = []
    dump = skill.model_dump(mode="json")

    def walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in LEAK_KEYS and key not in {"capability"}:
                    issues.append(f"forbidden_key:{path}.{key}" if path else f"forbidden_key:{key}")
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(value, f"{path}[{i}]")
        elif isinstance(obj, str):
            low = obj.lower()
            if "gold_answer" in low or "future_probe" in low:
                issues.append(f"leak_string:{path}")

    walk(dump)
    return issues
