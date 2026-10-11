"""Experience-conditioned activation of frozen executable skill primitives.

R5 causal contract: ``fit()`` may reuse R2 procedure primitives, but it must not
return a constant global skill for an arbitrary batch. Activation, version,
provenance, and at least one runtime configuration field are determined by the
acquired experiences of the requested capability.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable, Sequence

from expolicy.assimilation.carriers.skill_store import (
    Skill,
    canonical_skill_capability,
    skill_a_multihop,
    skill_b_temporal,
    skill_c_crossdoc,
    validate_skill_schema,
)
from expolicy.core.experience import Experience
from expolicy.experiments.patterns import normalized_pattern

COMPILER_VERSION = "r5.skill_assimilator.v1"
MIN_SUPPORT = 8
TARGET_SKILL_CAPS = ("multi_hop", "temporal_reasoning", "cross_document_synthesis")
CAP_ALIASES = {
    "multi_hop": {"multi_hop", "query_decomposition"},
    "temporal_reasoning": {"temporal_reasoning"},
    "cross_document_synthesis": {"cross_document_synthesis"},
}
PRIMITIVES = {
    "multi_hop": skill_a_multihop,
    "temporal_reasoning": skill_b_temporal,
    "cross_document_synthesis": skill_c_crossdoc,
}
LEAK_KEYS = {"gold_answer", "gold_doc_ids", "gold", "probe_task_ids", "future_probe"}


def maps_to_skill_capability(capability: str | None) -> str | None:
    cap = canonical_skill_capability(capability)
    if cap in PRIMITIVES:
        return cap
    return None


def experiences_for_capability(
    experiences: Sequence[Experience],
    capability: str,
    *,
    require_capability_match: bool = True,
) -> list[Experience]:
    if not require_capability_match:
        return list(experiences)
    allowed = CAP_ALIASES.get(capability, {capability})
    return [e for e in experiences if (e.capability or "") in allowed]


def _stable_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _search_queries(experiences: Sequence[Experience]) -> list[str]:
    queries: list[str] = []
    for exp in experiences:
        for call in exp.tool_calls:
            if call.get("tool") != "search":
                continue
            q = (call.get("arguments") or {}).get("query")
            if q:
                queries.append(str(q))
    return queries


def _evidence_titles(experiences: Sequence[Experience]) -> list[str]:
    titles: list[str] = []
    for exp in experiences:
        for ev in exp.evidence:
            title = ev.get("title")
            if title:
                titles.append(str(title))
    return titles


def _majority_pattern(experiences: Sequence[Experience]) -> str:
    pats = [normalized_pattern(e.task_id) for e in experiences if e.task_id]
    if not pats:
        return "unknown"
    return Counter(pats).most_common(1)[0][0]


def derive_runtime_config(experiences: Sequence[Experience], capability: str) -> dict[str, Any]:
    """Nontrivial config derived only from acquired trajectories (no gold / probe ids)."""
    n = len(experiences)
    n_success = sum(1 for e in experiences if e.success)
    titles = _evidence_titles([e for e in experiences if e.success] or list(experiences))
    prefer_titles = [t for t, _ in Counter(titles).most_common(3)]
    queries = _search_queries(experiences)
    qblob = " ".join(queries).lower()
    pattern = _majority_pattern(experiences)
    cfg: dict[str, Any] = {
        "primitive_library": "r2_frozen_v1",
        "majority_pattern": pattern,
        "n_support_used": n,
        "prefer_titles": prefer_titles,
        "max_opens": 6 if n_success >= max(1, n // 2) else 8,
    }
    if capability == "multi_hop":
        if pattern == "person_employer_hq":
            cfg["target_field_prior"] = "headquarters"
            cfg["hop2_prefer"] = "registry"
        elif pattern == "person_advisor_employer_sector":
            cfg["target_field_prior"] = "sector"
            cfg["hop2_prefer"] = ""
        else:
            cfg["target_field_prior"] = "ceo"
            cfg["hop2_prefer"] = "registry"
        if "works at" in qblob:
            cfg["recovery_suffix"] = "works at"
        elif "headquarter" in qblob:
            cfg["recovery_suffix"] = "headquarters official"
        else:
            cfg["recovery_suffix"] = "official"
    elif capability == "temporal_reasoning":
        cfg["timeline_focus"] = "became" if pattern == "org_ceo_at_transition" else "until"
        cfg["timeline_query"] = "{org} leadership timeline {year}"
        cfg["recovery_suffix"] = "chief executive until" if cfg["timeline_focus"] == "until" else "chief executive"
    elif capability == "cross_document_synthesis":
        cfg["first_field"] = "lead" if "led" in qblob and qblob.find("led") < (qblob.find("organized") if "organized" in qblob else 10**9) else "org"
        cfg["recovery_suffix"] = "organized"
    return cfg


def skill_has_leakage(skill: Skill, *, probe_ids: Iterable[str] = (), gold_doc_ids: Iterable[str] = ()) -> list[str]:
    issues: list[str] = []
    dump = skill.model_dump(mode="json")

    def walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in LEAK_KEYS:
                    issues.append(f"forbidden_key:{path}.{key}" if path else f"forbidden_key:{key}")
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(value, f"{path}[{i}]")

    walk(dump)
    blob = json.dumps(dump, default=str)
    for pid in probe_ids:
        if pid and pid in blob:
            issues.append(f"probe_task_id:{pid}")
    for did in gold_doc_ids:
        if did and did in blob:
            issues.append(f"gold_doc_id:{did}")
    return issues


class SkillAssimilator:
    def __init__(self, *, min_support: int = MIN_SUPPORT, compiler_version: str = COMPILER_VERSION) -> None:
        self.min_support = int(min_support)
        self.compiler_version = compiler_version

    def fit(
        self,
        experiences: Sequence[Experience],
        capability: str,
        *,
        frozen_primitives: dict[str, Any] | None = None,
        require_capability_match: bool = True,
        version: int = 1,
        probe_ids: Iterable[str] = (),
        gold_doc_ids: Iterable[str] = (),
    ) -> Skill | None:
        cap = maps_to_skill_capability(capability) or capability
        if cap not in PRIMITIVES:
            return None
        matched = experiences_for_capability(experiences, cap, require_capability_match=require_capability_match)
        n = len(matched)
        if n < self.min_support:
            return None
        primitive_fn = (frozen_primitives or PRIMITIVES)[cap]
        skill = primitive_fn()
        source_ids = sorted({e.experience_id for e in matched})
        n_success = sum(1 for e in matched if e.success)
        n_fail = n - n_success
        cfg = derive_runtime_config(matched, cap)
        skill.runtime_config = cfg
        skill.compiler_version = self.compiler_version
        skill.version = int(version)
        skill.status = "active"
        skill.executable = True
        skill.n_support = n
        skill.support_success_rate = n_success / n
        skill.support_failure_rate = n_fail / n
        skill.source_experience_ids = source_ids
        skill.activation_reason = f"n_support={n}>={self.min_support}; capability={cap}"
        skill.provenance = [
            f"compiler:{self.compiler_version}",
            f"primitive:{skill.name}",
            *[f"exp:{eid}" for eid in source_ids],
        ]
        skill.skill_id = _stable_id(
            {
                "compiler": self.compiler_version,
                "capability": cap,
                "version": skill.version,
                "source_experience_ids": source_ids,
                "runtime_config": cfg,
            }
        )
        errors = validate_skill_schema(skill)
        if errors:
            return None
        leaks = skill_has_leakage(skill, probe_ids=probe_ids, gold_doc_ids=gold_doc_ids)
        if leaks:
            raise RuntimeError(f"skill artifact leaked probe/gold fields: {leaks[:8]}")
        if skill.provenance == ["r2_spec"]:
            raise RuntimeError("fit() must not return the unmodified R2 builtin skill")
        return skill
