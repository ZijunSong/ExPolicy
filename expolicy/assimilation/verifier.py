"""Experience verifier: consistency, evidence, provenance. Low-trust → no weight update."""

from __future__ import annotations

from expolicy.core.experience import Experience, ExperienceVerification
from expolicy.environments.search.world import World


class ExperienceVerifier:
    def __init__(self, world: World | None = None) -> None:
        self.world = world

    def verify(self, experience: Experience) -> ExperienceVerification:
        reasons: list[str] = []
        consistency = 1.0
        evidence_score = 0.0
        provenance_score = 1.0

        if len(experience.actions) != len(experience.observations):
            consistency = 0.4
            reasons.append("action_observation_length_mismatch")

        tools = [c.get("tool") for c in experience.tool_calls]
        if experience.predicted_answer and "search" not in tools and "open" not in tools:
            consistency *= 0.5
            reasons.append("answered_without_retrieval")

        opened = [ev.get("document_id") for ev in experience.evidence if ev.get("document_id")]
        if opened:
            evidence_score = min(1.0, 0.3 * len(opened))
        else:
            reasons.append("no_opened_evidence")

        if self.world is not None:
            for doc_id in opened:
                if doc_id not in self.world.documents:
                    provenance_score = 0.0
                    reasons.append(f"unknown_document:{doc_id}")
            for call in experience.tool_calls:
                if call.get("tool") == "open" and call.get("ok") is False:
                    consistency *= 0.7
                    reasons.append("failed_open")

        if experience.branch_origin and not experience.parent_experience_id:
            provenance_score *= 0.5
            reasons.append("branch_missing_parent")

        if experience.success is False and experience.gold_answer and experience.predicted_answer == experience.gold_answer:
            consistency = 0.0
            reasons.append("label_inconsistency")

        confidence = (consistency + evidence_score + provenance_score) / 3.0
        valid = confidence >= 0.45 and "unknown_document" not in " ".join(reasons)
        if not valid and "low_confidence" not in reasons:
            reasons.append("low_confidence")
        return ExperienceVerification(
            experience_id=experience.experience_id,
            valid=valid,
            evidence_score=float(evidence_score),
            consistency_score=float(consistency),
            provenance_score=float(provenance_score),
            confidence=float(confidence),
            rejection_reasons=reasons,
        )
