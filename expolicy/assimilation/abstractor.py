"""Typed experience abstraction. Not a free-text summary."""

from __future__ import annotations

from expolicy.core.experience import Experience, ExperienceAbstraction, ExperienceVerification
from expolicy.core.types import FailureType


class ExperienceAbstractor:
    def abstract(self, experience: Experience, verification: ExperienceVerification) -> ExperienceAbstraction:
        tools = [str(c.get("tool")) for c in experience.tool_calls]
        queries = [
            str((c.get("arguments") or {}).get("query"))
            for c in experience.tool_calls
            if c.get("tool") == "search"
        ]
        opened = [str(ev.get("document_id")) for ev in experience.evidence if ev.get("document_id")]
        failure = experience.failure_type.value if experience.failure_type else None
        principle = _principle(experience, queries, opened)
        procedural = tools.count("search") >= 2 or "find" in tools
        runtime_control = experience.failure_type in {
            FailureType.TOOL,
            FailureType.VERIFICATION,
            FailureType.PLANNING,
        }
        return ExperienceAbstraction(
            experience_id=experience.experience_id,
            preconditions=[f"capability={experience.capability}", f"hops={experience.capability_context.get('hops')}"],
            situation=experience.question,
            decision_point=_decision_point(tools),
            action_pattern=tools,
            evidence=opened,
            outcome=f"success={experience.success}; answer={experience.predicted_answer}",
            failure_mode=failure,
            correction=_correction(experience),
            reusable_principle=principle,
            applicability_scope=[experience.capability or "general"],
            not_applicable_when=["held-out future answers unknown", "different entity with shared surface form unless disambiguated"],
            confidence=verification.confidence,
            provenance=[experience.experience_id, *opened],
            procedural=procedural,
            declarative=not procedural,
            runtime_control=runtime_control,
            broad_transfer=bool(experience.success) and (experience.capability in {"multi_hop", "source_reliability", "query_decomposition"}),
        )


def _decision_point(tools: list[str]) -> str:
    if "search" in tools and "open" in tools:
        return "retrieve_then_read"
    if "search" in tools:
        return "retrieve_only"
    return "no_retrieval"


def _principle(experience: Experience, queries: list[str], opened: list[str]) -> str:
    cap = experience.capability or "general"
    if cap == "source_reliability":
        return "When sources disagree, prefer official/encyclopedia documents over blogs."
    if cap == "conflicting_evidence":
        return "Resolve conflicts by source reliability and explicit timestamps, not recency of a rumor."
    if cap == "multi_hop":
        return "Decompose multi-hop questions: find the bridging entity, then search that entity."
    if cap == "temporal_reasoning":
        return "Bind the asked year to the leadership/event interval before naming a person."
    if cap == "entity_disambiguation":
        return "Do not merge people who share a given name; bind role and organization."
    if experience.success:
        q = queries[0] if queries else experience.question
        return f"For {cap} questions like '{experience.question}', search '{q}' and read {opened[:2]}."
    return f"Failed {cap} episode: avoid answering before opening supporting documents."


def _correction(experience: Experience) -> str | None:
    if experience.success:
        return None
    if experience.failure_type == FailureType.RETRIEVAL:
        return "Issue a more specific query using the entity's full name and organization."
    if experience.failure_type == FailureType.ANSWER_SYNTHESIS:
        return "Copy the span from a high-reliability opened document instead of paraphrasing loosely."
    if experience.failure_type == FailureType.PLANNING:
        return "Call search before answer."
    return "Re-open gold-like documents and extract the short span."
