"""Heuristic routing baseline. Learned policy lives in learned_policy.py."""

from __future__ import annotations

from expolicy.core.capability import CapabilityProfile
from expolicy.core.experience import AssimilationDecision, Experience, ExperienceAbstraction, ExperienceVerification
from expolicy.core.types import Carrier


class HeuristicRouter:
    name = "heuristic"

    def route(
        self,
        experience: Experience,
        verification: ExperienceVerification,
        abstraction: ExperienceAbstraction,
        profile: CapabilityProfile | None = None,
    ) -> AssimilationDecision:
        if not verification.valid or verification.confidence < 0.4:
            return _dec(experience, Carrier.DISCARD, 0.9, 0.0, 0.0, "invalid_or_low_confidence")
        if experience.success is False and abstraction.runtime_control:
            return _dec(experience, Carrier.HARNESS, 0.6, 0.2, 0.15, "runtime_control_failure")
        recurrence_hint = abstraction.recurrence
        if abstraction.procedural and recurrence_hint >= 1 and experience.success:
            return _dec(experience, Carrier.SKILL, 0.65, 0.35, 0.1, "procedural_reusable")
        if abstraction.declarative and recurrence_hint >= 2:
            return _dec(experience, Carrier.SEMANTIC, 0.7, 0.3, 0.08, "repeated_declarative")
        if abstraction.broad_transfer and verification.confidence >= 0.75 and experience.success:
            return _dec(experience, Carrier.WEIGHT, 0.55, 0.4, 0.25, "stable_broad_transfer")
        if experience.success is False:
            return _dec(experience, Carrier.EPISODIC, 0.6, 0.15, 0.05, "specific_failure_episode")
        return _dec(experience, Carrier.EPISODIC, 0.5, 0.12, 0.05, "specific_low_recurrence")


def _dec(
    experience: Experience,
    carrier: Carrier,
    confidence: float,
    utility: float,
    interference: float,
    rationale: str,
) -> AssimilationDecision:
    return AssimilationDecision(
        experience_id=experience.experience_id,
        carrier=carrier,
        confidence=confidence,
        expected_utility=utility,
        expected_interference=interference,
        update_spec={"rationale": rationale},
        rationale=rationale,
        policy_name="heuristic",
    )
