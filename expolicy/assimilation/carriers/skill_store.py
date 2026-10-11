"""Executable skills that control the search runtime, not prompt-only hints."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from expolicy.core.experience import Experience, ExperienceCost
from expolicy.environments.search.environment import SearchEnvironment
from expolicy.environments.search.world import Task
from expolicy.experiments.patterns import mentioned_entities

_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")


class SkillStep(BaseModel):
    op: str
    query_template: str = ""
    pattern_template: str = ""
    rank: int = 0


SKILL_CAP_ALIASES = {
    "query_decomposition": "multi_hop",
}


def canonical_skill_capability(capability: str | None) -> str | None:
    if not capability:
        return None
    return SKILL_CAP_ALIASES.get(capability, capability)


class Skill(BaseModel):
    skill_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    capability: str | None = None
    preconditions: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    procedure: list[SkillStep] = Field(default_factory=list)
    query_template: str = ""
    decomposition_steps: list[str] = Field(default_factory=list)
    tool_sequence: list[str] = Field(default_factory=list)
    state_variables: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    stopping_condition: str = "opened_evidence"
    answer_synthesis_rule: str = ""
    tool_requirements: list[str] = Field(default_factory=lambda: ["search", "open"])
    termination_condition: str = "opened_evidence"
    verifier: str = "has_open"
    failure_recovery: str = ""
    provenance: list[str] = Field(default_factory=list)
    version: int = 1
    status: str = "active"
    success_count: int = 0
    fail_count: int = 0
    executable: bool = False
    validation_gain: float | None = None
    runtime_config: dict[str, Any] = Field(default_factory=dict)
    compiler_version: str = ""
    activation_reason: str = ""
    n_support: int = 0
    support_success_rate: float = 0.0
    support_failure_rate: float = 0.0
    source_experience_ids: list[str] = Field(default_factory=list)


def validate_skill_schema(skill: Skill) -> list[str]:
    errors: list[str] = []
    if not skill.name:
        errors.append("missing_name")
    if skill.executable:
        if not skill.capability:
            errors.append("missing_capability")
        if not skill.tool_sequence:
            errors.append("missing_tool_sequence")
        if not skill.decomposition_steps:
            errors.append("missing_decomposition_steps")
        if not skill.answer_synthesis_rule:
            errors.append("missing_answer_synthesis_rule")
        if not skill.verifier:
            errors.append("missing_verifier")
    return errors


class SkillStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "skills.json"
        self.skills: dict[str, Skill] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            for item in payload:
                skill = Skill.model_validate(item)
                self.skills[skill.skill_id] = skill

    def _persist(self) -> None:
        self.path.write_text(
            json.dumps([s.model_dump(mode="json") for s in self.skills.values()], indent=2),
            encoding="utf-8",
        )

    def create(self, skill: Skill) -> Skill:
        self.skills[skill.skill_id] = skill
        self._persist()
        return skill

    def summaries(self) -> list[str]:
        lines = []
        for s in self.skills.values():
            if s.status != "active":
                continue
            if s.executable:
                lines.append(f"{s.name}[{s.capability}]: {' → '.join(s.decomposition_steps[:6])}")
            else:
                lines.append(f"{s.name}: {[step.op for step in s.procedure]}")
        return lines

    def executable_for(self, capability: str | None) -> Skill | None:
        cap = canonical_skill_capability(capability)
        best: Skill | None = None
        for skill in self.skills.values():
            if skill.status != "active" or not skill.executable:
                continue
            if skill.capability != cap:
                continue
            if best is None or skill.version > best.version:
                best = skill
        return best

    def install(self, skill: Skill) -> Skill:
        """Activate ``skill`` and deprecate previous executable skills of the same capability."""
        cap = canonical_skill_capability(skill.capability)
        for existing in self.skills.values():
            if (
                existing.status == "active"
                and existing.executable
                and existing.capability == cap
                and existing.skill_id != skill.skill_id
            ):
                existing.status = "deprecated"
        return self.create(skill)

    def invoke(self, name_or_id: str, env: SearchEnvironment, **inputs: Any) -> dict[str, Any]:
        skill = self._resolve(name_or_id)
        if skill is None:
            return {"ok": False, "error": f"unknown skill {name_or_id}"}
        if skill.executable:
            task = env.current_task
            result = execute_skill(skill, env, task=task, query=str(inputs.get("query") or (task.question if task else "")))
            return result
        query = str(inputs.get("query") or "")
        opened: list[dict[str, Any]] = []
        searches: list[dict[str, Any]] = []
        fmt = {"query": query, **{k: v for k, v in inputs.items() if k != "query"}}
        for step in skill.procedure:
            if step.op == "search":
                q = (step.query_template or "{query}").format(**fmt)
                result = env.step("search", {"query": q})
                searches.append(result.payload)
            elif step.op == "open":
                hits = []
                if searches:
                    hits = (searches[-1].get("hits") or [])
                if hits:
                    rank = min(step.rank, len(hits) - 1)
                    doc_id = hits[rank]["document_id"]
                    opened.append(env.step("open", {"document_id": doc_id}).payload)
            elif step.op == "find" and opened:
                pattern = (step.pattern_template or query).format(**fmt)
                env.step("find", {"document_id": opened[-1]["document_id"], "pattern": pattern})
        ok = bool(opened) if skill.verifier == "has_open" else bool(searches)
        if ok:
            skill.success_count += 1
        else:
            skill.fail_count += 1
        self._persist()
        return {"ok": ok, "skill": skill.name, "opened": opened, "searches": searches}

    def refine(self, skill_id: str, procedure: list[SkillStep]) -> Skill:
        skill = self.skills[skill_id]
        skill.procedure = procedure
        skill.version += 1
        self._persist()
        return skill

    def deprecate(self, skill_id: str) -> None:
        if skill_id in self.skills:
            self.skills[skill_id].status = "deprecated"
            self._persist()

    def evaluate(self, skill_id: str) -> dict[str, Any]:
        skill = self.skills[skill_id]
        total = skill.success_count + skill.fail_count
        return {
            "skill_id": skill_id,
            "n": total,
            "success_rate": skill.success_count / total if total else 0.0,
        }

    def _resolve(self, name_or_id: str) -> Skill | None:
        if name_or_id in self.skills:
            return self.skills[name_or_id]
        for skill in self.skills.values():
            if skill.name == name_or_id and skill.status == "active":
                return skill
        return None

    @staticmethod
    def default_lookup_skill() -> Skill:
        return Skill(
            name="official_lookup",
            preconditions=["need_entity_fact"],
            procedure=[
                SkillStep(op="search", query_template="{query} official registry"),
                SkillStep(op="open", rank=0),
                SkillStep(op="find", pattern_template="{query}"),
            ],
            provenance=["builtin"],
            executable=False,
        )


def skill_a_multihop() -> Skill:
    return Skill(
        name="multi_hop_bridge_search",
        capability="multi_hop",
        executable=True,
        preconditions=["question_names_a_person", "asks_employer_field"],
        query_template="{person} ; {org} {target_field}",
        decomposition_steps=[
            "identify target relation",
            "search hop-1 entity",
            "verify bridge entity (employer)",
            "search hop-2 relation",
            "collect both supporting docs",
            "synthesize only after both hops verified",
        ],
        tool_sequence=["search", "open", "search", "open", "answer"],
        state_variables=["person", "org", "target_field", "hop1_doc", "hop2_doc"],
        evidence_requirements=["person_file", "org_file"],
        stopping_condition="both_hops_verified",
        answer_synthesis_rule="answer the requested field from the org file only after the employer bridge is confirmed",
        verifier="two_hops_open",
        failure_recovery="if org missing, search 'works at' snippet from hop-1; if still missing, open next hit",
        provenance=["r2_spec"],
    )


def skill_b_temporal() -> Skill:
    return Skill(
        name="temporal_resolution",
        capability="temporal_reasoning",
        executable=True,
        preconditions=["question_has_year_or_interval", "asks_ceo_or_role"],
        query_template="{org} leadership timeline {year}",
        decomposition_steps=[
            "extract temporal constraint",
            "identify candidate facts",
            "attach timestamps",
            "discard temporally invalid evidence",
            "rank valid evidence",
            "answer",
        ],
        tool_sequence=["search", "open", "find", "answer"],
        state_variables=["org", "year", "old_ceo", "new_ceo", "until", "from"],
        evidence_requirements=["leadership_timeline"],
        stopping_condition="year_bound_to_interval",
        answer_synthesis_rule="if year < transition: old CEO; if year >= transition: new CEO",
        verifier="timeline_open",
        failure_recovery="search '{org} chief executive until' and reopen encyclopedia timeline",
        provenance=["r2_spec"],
    )


def skill_c_crossdoc() -> Skill:
    return Skill(
        name="cross_document_synthesis",
        capability="cross_document_synthesis",
        executable=True,
        preconditions=["asks_two_fields", "event_named"],
        query_template="{event}",
        decomposition_steps=[
            "decompose requested fields",
            "retrieve evidence independently",
            "bind evidence to entity",
            "verify consistency",
            "synthesize structured answer",
        ],
        tool_sequence=["search", "open", "search", "open", "answer"],
        state_variables=["event", "org", "lead"],
        evidence_requirements=["event_file", "org_or_person_support"],
        stopping_condition="both_fields_bound",
        answer_synthesis_rule="return '{org}; {lead}' only if both bound from opened evidence",
        verifier="two_fields_bound",
        failure_recovery="if a field missing, search that field + event name and open the next official hit",
        provenance=["r2_spec"],
    )


def builtin_executable_skills() -> list[Skill]:
    return [skill_a_multihop(), skill_b_temporal(), skill_c_crossdoc()]


def execute_skill(
    skill: Skill,
    env: SearchEnvironment,
    *,
    task: Task | None = None,
    query: str = "",
    max_opens: int = 6,
) -> dict[str, Any]:
    question = query or (task.question if task else "")
    world = env.world
    ents = mentioned_entities(question, world)
    cfg = dict(skill.runtime_config or {})
    opened: list[dict[str, Any]] = []
    searches: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "person": (ents["people"] or [None])[0],
        "org": (ents["orgs"] or [None])[0],
        "event": (ents["events"] or [None])[0],
        "year": (ents["years"] or [None])[0],
        "target_field": _infer_target_field(question) or cfg.get("target_field_prior") or "requested_field",
    }
    compliance = {"planned": list(skill.tool_sequence), "executed": []}
    prefer_titles = [str(x) for x in (cfg.get("prefer_titles") or []) if x]
    max_opens = int(cfg.get("max_opens") or max_opens)
    recovery_suffix = str(cfg.get("recovery_suffix") or "official")
    hop2_prefer = str(cfg.get("hop2_prefer") or "")
    first_field = str(cfg.get("first_field") or "org")
    timeline_focus = str(cfg.get("timeline_focus") or "until")

    def do_search(q: str) -> dict[str, Any]:
        result = env.step("search", {"query": q})
        payload = result.payload if result.ok else {"error": result.error, "hits": []}
        searches.append(payload)
        compliance["executed"].append("search")
        return payload

    def do_open_from_hits(hits: list[dict[str, Any]], *, prefer: str = "") -> dict[str, Any] | None:
        if not hits:
            return None
        ranked = list(hits)
        def _title_rank(h: dict[str, Any]) -> tuple:
            title = (h.get("title") or "").lower()
            prefer_hit = 0 if prefer and prefer.lower() in title else 1
            cfg_hit = 0 if any(p.lower() in title for p in prefer_titles) else 1
            return (prefer_hit, cfg_hit, -float(h.get("reliability") or 0), int(h.get("rank") or 99))

        ranked = sorted(ranked, key=_title_rank)
        for hit in ranked[:max_opens]:
            did = hit.get("document_id")
            if not did:
                continue
            result = env.step("open", {"document_id": did})
            if not result.ok:
                continue
            opened.append(result.payload)
            compliance["executed"].append("open")
            return result.payload
        return None

    if skill.capability == "multi_hop":
        person = state["person"] or question
        hop1 = do_search(str(person))
        doc1 = do_open_from_hits(hop1.get("hits") or [], prefer=str(person).split()[0] if person else "")
        org = _extract_employer(opened) or state["org"]
        state["org"] = org
        hop2_q = f"{org} {state['target_field']}" if org else f"{person} {state['target_field']}"
        hop2 = do_search(hop2_q)
        prefer2 = hop2_prefer or ("registry" if state["target_field"] in {"ceo", "headquarters", "hq"} else "")
        do_open_from_hits(hop2.get("hits") or [], prefer=prefer2 or (org or ""))
        if len(opened) < 2:
            rec = do_search(f"{org or person} {recovery_suffix}".strip())
            do_open_from_hits(rec.get("hits") or [])
            compliance["executed"].append("recovery")
        answer = _synthesize_multihop(question, opened, state)
    elif skill.capability == "temporal_reasoning":
        org = state["org"] or question
        year = state["year"]
        tl_q = str(cfg.get("timeline_query") or f"{org} leadership timeline {year or ''}").strip()
        hop = do_search(tl_q)
        doc = do_open_from_hits(hop.get("hits") or [], prefer="Leadership timeline")
        if doc:
            env.step("find", {"document_id": doc["document_id"], "pattern": str(year or "chief executive")})
            compliance["executed"].append("find")
        if not opened:
            rec_q = f"{org} chief executive {timeline_focus}" if timeline_focus else f"{org} {recovery_suffix}"
            rec = do_search(rec_q.strip())
            do_open_from_hits(rec.get("hits") or [], prefer="Leadership")
            compliance["executed"].append("recovery")
        answer = _synthesize_temporal(question, opened, year)
    elif skill.capability == "cross_document_synthesis":
        event = state["event"] or question
        hop = do_search(str(event))
        do_open_from_hits(hop.get("hits") or [], prefer="Event file")
        org, lead = _extract_event_org_lead(opened)
        state["org"], state["lead"] = org, lead
        field_order = ["lead", "org"] if first_field == "lead" else ["org", "lead"]
        for field in field_order:
            if field == "org" and not org:
                hop_org = do_search(f"{event} organized")
                do_open_from_hits(hop_org.get("hits") or [])
            if field == "lead" and not lead:
                hop_lead = do_search(f"{event} led")
                do_open_from_hits(hop_lead.get("hits") or [])
            org, lead = _extract_event_org_lead(opened)
            state["org"], state["lead"] = org, lead
        answer = _synthesize_crossdoc(org, lead)
    else:
        hop = do_search(question)
        do_open_from_hits(hop.get("hits") or [])
        answer = _first_span(opened)

    both_hops = len(opened) >= 2
    evidence_complete = _evidence_complete(skill, opened, state)
    ok = bool(answer) and evidence_complete
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
        "state": {k: v for k, v in state.items() if v is not None},
        "compliance": compliance,
        "evidence_complete": evidence_complete,
        "both_hops": both_hops,
        "n_open": len(opened),
        "n_search": len(searches),
    }


def run_skill_on_task(skill: Skill, env: SearchEnvironment, task: Task) -> Experience:
    env.reset(task)
    started = time.perf_counter()
    result = execute_skill(skill, env, task=task, query=task.question)
    predicted = str(result.get("answer") or "").strip()
    exp = Experience(
        task_id=task.task_id,
        question=task.question,
        gold_answer=task.gold_answer,
        capability=task.capability,
        stage=task.stage,
        split=task.split,
        predicted_answer=predicted,
        capability_context={"skill": skill.name, "executable": True},
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
            {
                "document_id": payload.get("document_id"),
                "source_type": payload.get("source_type"),
                "reliability": payload.get("reliability"),
                "title": payload.get("title"),
            }
        )
    exp.actions.append({"tool": "answer", "arguments": {"text": predicted}})
    exp.observations.append({"tool": "answer", "text": predicted})
    from expolicy.agents.llm_agent import _normalize_answer, _soft_match, infer_failure

    success = _normalize_answer(predicted) == _normalize_answer(task.gold_answer) or _soft_match(
        predicted, task.gold_answer, task.aliases
    )
    exp.success = success
    exp.reward = 1.0 if success else 0.0
    exp.outcome = {
        "predicted": predicted,
        "gold": task.gold_answer,
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
    if not success:
        exp.failure_type = infer_failure(exp)
    return exp


def _infer_target_field(question: str) -> str:
    q = question.lower()
    if "chief executive" in q or "ceo" in q:
        return "ceo"
    if "headquarter" in q or "which city is the organization" in q:
        return "headquarters"
    if "sector" in q:
        return "sector"
    if "product" in q:
        return "product"
    return "requested_field"


def _extract_employer(opened: list[dict[str, Any]]) -> str | None:
    for doc in opened:
        text = doc.get("text") or ""
        m = re.search(r"works as a [^.]+ at ([^.]+)\.", text)
        if m and m.group(1):
            return m.group(1).strip()
        m = re.search(
            r"(?:researcher|engineer|archivist|director|correspondent|pilot) at ([^.]+)",
            text,
        )
        if m and m.group(1):
            return m.group(1).strip()
    return None


def _synthesize_multihop(question: str, opened: list[dict[str, Any]], state: dict[str, Any]) -> str:
    field = state.get("target_field")
    texts = [d.get("text") or "" for d in opened]
    blob = "\n".join(texts)
    if field == "ceo":
        m = re.search(r"current chief executive is ([^.]+)\.", blob)
        if m:
            return m.group(1).strip()
        m = re.search(r"chief executive of [^.]+ is ([^.]+)\.", blob)
        if m:
            return m.group(1).strip()
    if field == "headquarters":
        m = re.search(r"headquartered in ([^.]+)\.", blob)
        if m:
            return m.group(1).strip()
        # hop-1 city questions ask org HQ, not birth city
        m = re.search(r"organization that employs .+ headquartered\?", question)
        if m:
            m2 = re.search(r"headquartered in ([^.]+)\.", blob)
            if m2:
                return m2.group(1).strip()
    if field == "sector":
        m = re.search(r"is a ([a-z]+) organization", blob)
        if m:
            return m.group(1).strip()
    return ""


def _synthesize_temporal(question: str, opened: list[dict[str, Any]], year: str | None) -> str:
    blob = "\n".join(d.get("text") or "" for d in opened)
    m_old = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+) served as chief executive of (.+?) until (\d{4})", blob)
    m_new = re.search(r"In (\d{4}), ([A-Z][a-z]+ [A-Z][a-z]+) became chief executive", blob)
    asked = int(year) if year and str(year).isdigit() else None
    q = question.lower()
    if "became" in q and m_new:
        return m_new.group(2).strip()
    if asked is not None and m_old and m_new:
        until = int(m_old.group(3))
        if asked < until:
            return m_old.group(1).strip()
        return m_new.group(2).strip()
    if m_old and "until" in q:
        return m_old.group(1).strip()
    if m_new:
        return m_new.group(2).strip()
    if m_old:
        return m_old.group(1).strip()
    return ""


def _extract_event_org_lead(opened: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    blob = "\n".join(d.get("text") or "" for d in opened)
    org = None
    lead = None
    m = re.search(r"organized by ([^.]+?) and led by ([^.]+)\.", blob)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"organized by ([^.]+)\.", blob)
    if m:
        org = m.group(1).strip()
    m = re.search(r"led by ([^.]+)\.", blob)
    if m:
        lead = m.group(1).strip()
    m = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+) spoke at the closing session", blob)
    if m and not lead:
        lead = m.group(1).strip()
    return org, lead


def _synthesize_crossdoc(org: str | None, lead: str | None) -> str:
    if org and lead:
        return f"{org}; {lead}"
    return ""


def _first_span(opened: list[dict[str, Any]]) -> str:
    return ""


def _evidence_complete(skill: Skill, opened: list[dict[str, Any]], state: dict[str, Any]) -> bool:
    if skill.verifier == "two_hops_open":
        return len(opened) >= 2 and bool(state.get("org") or state.get("person"))
    if skill.verifier == "timeline_open":
        return any("timeline" in (d.get("title") or "").lower() or "chief executive" in (d.get("text") or "").lower() for d in opened)
    if skill.verifier == "two_fields_bound":
        return bool(state.get("org") and state.get("lead"))
    return bool(opened)
