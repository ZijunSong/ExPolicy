"""Semantic memory: mergeable, deprecatable rules with conflict resolution."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from expolicy.core.experience import ExperienceAbstraction


class SemanticRule(BaseModel):
    rule_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    text: str
    scope: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    support: int = 1
    status: str = "active"
    superseded_by: str | None = None
    provenance: list[str] = Field(default_factory=list)
    updated_at: float = Field(default_factory=time.time)


class SemanticMemory:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "rules.jsonl"
        self.rules: list[SemanticRule] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.rules.append(SemanticRule.model_validate(json.loads(line)))

    def _persist(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            for rule in self.rules:
                handle.write(rule.model_dump_json() + "\n")

    def add_from_abstraction(self, abstraction: ExperienceAbstraction) -> SemanticRule:
        text = abstraction.reusable_principle.strip()
        if not text:
            text = abstraction.situation.strip() or abstraction.outcome
        existing = self._find_similar(text)
        if existing is not None:
            if existing.confidence < abstraction.confidence and text != existing.text:
                return self.supersede(existing, text, abstraction)
            existing.support += 1
            existing.confidence = min(1.0, existing.confidence + 0.05)
            existing.provenance.append(abstraction.experience_id)
            existing.updated_at = time.time()
            self._persist()
            return existing
        rule = SemanticRule(
            text=text,
            scope=list(abstraction.applicability_scope),
            confidence=abstraction.confidence,
            provenance=[abstraction.experience_id],
        )
        self.rules.append(rule)
        self._persist()
        return rule

    def _find_similar(self, text: str) -> SemanticRule | None:
        key = _norm(text)
        for rule in self.rules:
            if rule.status != "active":
                continue
            if _norm(rule.text) == key or _token_overlap(_norm(rule.text), key) >= 0.8:
                return rule
        return None

    def supersede(self, old: SemanticRule, new_text: str, abstraction: ExperienceAbstraction) -> SemanticRule:
        new = SemanticRule(
            text=new_text,
            scope=list(abstraction.applicability_scope) or old.scope,
            confidence=max(old.confidence, abstraction.confidence),
            support=old.support + 1,
            provenance=old.provenance + [abstraction.experience_id],
        )
        old.status = "superseded"
        old.superseded_by = new.rule_id
        self.rules.append(new)
        self._persist()
        return new

    def deprecate(self, rule_id: str, reason: str) -> None:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                rule.status = "deprecated"
                rule.updated_at = time.time()
        self._persist()

    def active_texts(self, k: int = 8) -> list[str]:
        active = [r for r in self.rules if r.status == "active"]
        active.sort(key=lambda r: (r.confidence * r.support, r.updated_at), reverse=True)
        return [r.text for r in active[:k]]

    def detect_conflicts(self) -> list[tuple[str, str]]:
        conflicts = []
        active = [r for r in self.rules if r.status == "active"]
        for i, a in enumerate(active):
            for b in active[i + 1 :]:
                if "prefer" in a.text.lower() and "prefer" in b.text.lower() and a.text != b.text:
                    if _token_overlap(_norm(a.text), _norm(b.text)) > 0.4:
                        conflicts.append((a.rule_id, b.rule_id))
        return conflicts


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _token_overlap(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
