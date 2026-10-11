"""Task pattern, bridge structure, and entity extraction for near-transfer probes."""

from __future__ import annotations

import re
from typing import Any

from expolicy.core.experience import Experience
from expolicy.environments.search.world import Task, World

# Capability buckets used by R1–R4. S0/S3 are not learning targets.
TARGET_CAPS = {
    "s1_multi_hop": ("multi_hop", "query_decomposition"),
    "s2_temporal": ("temporal_reasoning", "entity_disambiguation"),
    "s4_cross_doc": ("cross_document_synthesis",),
}

BUCKET_PRIMARY_CAP = {
    "s1_multi_hop": "multi_hop",
    "s2_temporal": "temporal_reasoning",
    "s4_cross_doc": "cross_document_synthesis",
}

PATTERN_FAMILY = {
    "mh_ceo": "person_employer_ceo",
    "mh_city": "person_employer_hq",
    "qd_adv": "person_advisor_employer_sector",
    "tm_old": "org_ceo_before_transition",
    "tm_new": "org_ceo_at_transition",
    "xd_evt": "event_org_and_lead",
    "ed_role": "shared_given_name_role",
    "fr_evt": "event_city_ignore_rumor",
    "sh_emp": "person_employer",
    "sh_birth": "person_birth_city",
    "sh_spec": "person_specialty",
    "sh_hq": "org_headquarters",
    "sh_found": "org_founded",
    "sh_prod": "org_product",
}

BRIDGE_BY_PATTERN = {
    "person_employer_ceo": "person→employer→ceo",
    "person_employer_hq": "person→employer→hq_city",
    "person_advisor_employer_sector": "person→advisor→employer→sector",
    "org_ceo_before_transition": "org→timeline→ceo_at_year",
    "org_ceo_at_transition": "org→timeline→incoming_ceo",
    "event_org_and_lead": "event→organizer + event→lead",
}


def task_code(task_id: str) -> str:
    parts = (task_id or "").split("_")
    if len(parts) >= 3 and parts[0] == "t":
        return f"{parts[1]}_{parts[2]}"
    return task_id or "unknown"


def normalized_pattern(task_id: str) -> str:
    return PATTERN_FAMILY.get(task_code(task_id), task_code(task_id))


def bridge_relation(task_id: str) -> str:
    return BRIDGE_BY_PATTERN.get(normalized_pattern(task_id), "unknown")


def bucket_for_capability(capability: str | None) -> str | None:
    for bucket, caps in TARGET_CAPS.items():
        if capability in caps:
            return bucket
    return None


def mentioned_entities(question: str, world: World | None) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {"people": [], "orgs": [], "events": [], "years": []}
    if not question:
        return found
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", question)
    found["years"] = years
    if world is None:
        return found
    for person in world.people.values():
        if person.name and person.name in question:
            found["people"].append(person.name)
    for org in world.orgs.values():
        if org.name and org.name in question:
            found["orgs"].append(org.name)
    for event in world.events.values():
        if event.name and event.name in question:
            found["events"].append(event.name)
    return found


def intermediate_entities(exp: Experience, world: World | None = None) -> list[str]:
    ents = mentioned_entities(exp.question, world)
    out: list[str] = []
    for key in ("people", "orgs", "events"):
        out.extend(ents[key])
    return out


def search_queries(exp: Experience) -> list[str]:
    queries: list[str] = []
    for call in exp.tool_calls:
        if call.get("tool") != "search":
            continue
        q = (call.get("arguments") or {}).get("query")
        if q:
            queries.append(str(q))
    return queries


def evidence_path(exp: Experience) -> list[str]:
    docs: list[str] = []
    for ev in exp.evidence:
        did = ev.get("document_id")
        if did:
            docs.append(str(did))
    if docs:
        return docs
    for call in exp.tool_calls:
        if call.get("tool") == "open":
            did = (call.get("arguments") or {}).get("document_id")
            if did:
                docs.append(str(did))
    return docs


def failed_action(exp: Experience) -> str:
    if exp.success:
        return ""
    tools = [str(c.get("tool")) for c in exp.tool_calls]
    if "search" not in tools:
        return "answered_without_search"
    if "open" not in tools:
        return "searched_but_never_opened"
    if not exp.predicted_answer:
        return "retrieved_but_no_answer"
    return f"wrong_span:{exp.failure_type.value if exp.failure_type else 'unknown'}"


def expected_failure_signature(capability: str | None) -> str:
    if capability in {"multi_hop", "query_decomposition"}:
        return "reasoning_failure"
    if capability in {"temporal_reasoning", "entity_disambiguation"}:
        return "reasoning_failure"
    if capability == "cross_document_synthesis":
        return "answer_synthesis_failure"
    return "unknown"


def gold_doc_key(task: Task) -> tuple[str, ...]:
    return tuple(sorted(task.gold_doc_ids))


def experience_record_fields(exp: Experience, world: World | None = None) -> dict[str, Any]:
    pattern = normalized_pattern(exp.task_id)
    return {
        "question": exp.question,
        "normalized_task_pattern": pattern,
        "capability": exp.capability,
        "failure_type": exp.failure_type.value if exp.failure_type else ("none" if exp.success else "unknown"),
        "intermediate_entities": intermediate_entities(exp, world),
        "bridge_relation": bridge_relation(exp.task_id),
        "search_queries": search_queries(exp),
        "successful_evidence_path": evidence_path(exp) if exp.success else [],
        "failed_action": failed_action(exp),
        "correction": _correction_text(exp, pattern),
        "final_outcome": "success" if exp.success else "failure",
    }


def _correction_text(exp: Experience, pattern: str) -> str:
    if exp.success:
        return f"Keep the {pattern} hop order; do not skip the bridge entity."
    if pattern in {"person_employer_ceo", "person_employer_hq"}:
        return "Search the named person, open their file, read the employer, then search that organization for the requested field."
    if pattern.startswith("org_ceo"):
        return "Open the leadership timeline, attach the asked year to the CEO interval, discard off-interval names."
    if pattern == "event_org_and_lead":
        return "Retrieve organizer and lead independently from the event file, bind both, then synthesize 'org; lead'."
    return "Re-decompose the question and open supporting documents before answering."
