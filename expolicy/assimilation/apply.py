"""Apply an assimilation decision to persistent stores. Real writes only."""

from __future__ import annotations

from typing import Any

from expolicy.assimilation.carriers.episodic_memory import EpisodicMemory
from expolicy.assimilation.carriers.harness_store import HarnessStore
from expolicy.assimilation.carriers.semantic_memory import SemanticMemory
from expolicy.assimilation.carriers.skill_store import Skill, SkillStep, SkillStore
from expolicy.assimilation.carriers.weight_update import WeightUpdater
from expolicy.core.experience import AssimilationDecision, Experience, ExperienceAbstraction
from expolicy.core.state import PersistentAgentState
from expolicy.core.types import Carrier
from expolicy.logging.event_logger import EventLogger


class CarrierRuntime:
    def __init__(
        self,
        state: PersistentAgentState,
        *,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
        skills: SkillStore | None = None,
        harness: HarnessStore | None = None,
        weights: WeightUpdater | None = None,
        logger: EventLogger | None = None,
        enable_harness: bool = True,
        enable_weight: bool = True,
    ) -> None:
        self.state = state
        self.episodic = episodic
        self.semantic = semantic
        self.skills = skills
        self.harness = harness
        self.weights = weights
        self.logger = logger
        self.enable_harness = enable_harness
        self.enable_weight = enable_weight

    def apply(
        self,
        experience: Experience,
        abstraction: ExperienceAbstraction,
        decision: AssimilationDecision,
        *,
        verified_batch: list[Experience] | None = None,
    ) -> dict[str, Any]:
        carrier = decision.carrier
        info: dict[str, Any] = {"carrier": carrier.value}
        if carrier == Carrier.DISCARD:
            info["status"] = "discarded"
        elif carrier == Carrier.EPISODIC:
            if self.episodic is None:
                raise RuntimeError("episodic memory is not attached")
            self.episodic.add(experience, abstraction)
            if self.logger:
                self.logger.emit("MEMORY_UPDATED", {"kind": "episodic", "id": experience.experience_id})
            info["status"] = "episodic_added"
        elif carrier == Carrier.SEMANTIC:
            if self.semantic is None:
                raise RuntimeError("semantic memory is not attached")
            rule = self.semantic.add_from_abstraction(abstraction)
            if self.logger:
                self.logger.emit("MEMORY_UPDATED", {"kind": "semantic", "rule_id": rule.rule_id})
            info["status"] = "semantic_added"
            info["rule_id"] = rule.rule_id
        elif carrier == Carrier.SKILL:
            if self.skills is None:
                raise RuntimeError("skill store is not attached")
            skill = Skill(
                name=f"skill_{experience.capability or 'gen'}_{experience.task_id}",
                preconditions=abstraction.preconditions,
                procedure=[
                    SkillStep(op=op) if op in {"search", "open", "find"} else SkillStep(op="search")
                    for op in abstraction.action_pattern
                    if op in {"search", "open", "find"}
                ]
                or [SkillStep(op="search", query_template="{query}"), SkillStep(op="open")],
                provenance=[experience.experience_id],
            )
            self.skills.create(skill)
            if self.logger:
                self.logger.emit("SKILL_CREATED", {"skill_id": skill.skill_id})
            info["status"] = "skill_created"
            info["skill_id"] = skill.skill_id
        elif carrier == Carrier.HARNESS:
            if not self.enable_harness:
                info["status"] = "harness_disabled"
                return info
            if self.harness is None:
                raise RuntimeError("harness store is not attached")
            patch = _harness_patch_from_experience(experience)
            candidate = self.harness.propose(self.state.harness, patch)
            # Sandbox: static validation already done. Probe hook is the caller's responsibility
            # via evolve_loop (runs regression probes before commit). Here we commit config-only
            # patches that do not execute arbitrary code.
            self.harness.commit(self.state, candidate, {"source": experience.experience_id})
            if self.logger:
                self.logger.emit("HARNESS_PATCHED", {"version": candidate.version, "patch": patch})
            info["status"] = "harness_patched"
            info["patch"] = patch
        elif carrier == Carrier.WEIGHT:
            if not self.enable_weight:
                info["status"] = "weight_disabled"
                return info
            if self.weights is None:
                raise RuntimeError("weight updater is not attached")
            batch = verified_batch or [experience]
            summary = self.weights.train(batch, steps=min(20, max(5, len(batch) * 2)))
            self.state.adapter_path = summary["adapter_dir"]
            self.state.write_manifest()
            if self.logger:
                self.logger.emit("WEIGHT_UPDATED", summary)
            info["status"] = "weight_updated"
            info["train"] = summary
        else:
            raise RuntimeError(f"unhandled carrier {carrier}")
        return info


def _harness_patch_from_experience(experience: Experience) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    if experience.failure_type and experience.failure_type.value == "retrieval_failure":
        patch["search_top_k"] = 8
        patch["extra_search_on_conflict"] = True
    elif experience.failure_type and experience.failure_type.value == "verification_failure":
        patch["enable_verifier"] = True
        patch["min_evidence_before_answer"] = 2
    else:
        patch["evidence_checkpoint"] = True
        patch["min_evidence_before_answer"] = 1
    return patch
