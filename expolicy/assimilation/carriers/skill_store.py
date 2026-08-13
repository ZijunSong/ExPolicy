"""Executable skills, not prompt-only templates."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from expolicy.environments.search.environment import SearchEnvironment


class SkillStep(BaseModel):
    op: str
    query_template: str = ""
    pattern_template: str = ""
    rank: int = 0


class Skill(BaseModel):
    skill_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    preconditions: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    procedure: list[SkillStep] = Field(default_factory=list)
    tool_requirements: list[str] = Field(default_factory=lambda: ["search", "open"])
    termination_condition: str = "opened_evidence"
    verifier: str = "has_open"
    provenance: list[str] = Field(default_factory=list)
    version: int = 1
    status: str = "active"
    success_count: int = 0
    fail_count: int = 0


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
        return [f"{s.name}: {[step.op for step in s.procedure]}" for s in self.skills.values() if s.status == "active"]

    def invoke(self, name_or_id: str, env: SearchEnvironment, **inputs: Any) -> dict[str, Any]:
        skill = self._resolve(name_or_id)
        if skill is None:
            return {"ok": False, "error": f"unknown skill {name_or_id}"}
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
        )
