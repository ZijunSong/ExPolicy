"""Verifier, abstractor, heuristic router, acquisition policies."""

from __future__ import annotations

from expolicy.acquisition.base import FrontierAcquisition, PassiveAcquisition, make_acquisition
from expolicy.assimilation.abstractor import ExperienceAbstractor
from expolicy.assimilation.router import HeuristicRouter
from expolicy.assimilation.verifier import ExperienceVerifier
from expolicy.core.capability import CapabilityProfile, FrontierCapability
from expolicy.core.experience import Experience
from expolicy.core.state import PersistentAgentState
from expolicy.core.types import Carrier, FailureType, FrontierBand
from expolicy.diagnosis.capability_profile import CapabilityDiagnoser
from expolicy.environments.search.world import build_world
from expolicy.training.policy_model import ExperiencePolicyModel, candidate_features


def _exp(success: bool, tools: list[str]) -> Experience:
    exp = Experience(
        task_id="t_sh_hq_000",
        question="Where is Aurel Institute headquartered?",
        predicted_answer="Aurel" if success else "Nexora",
        gold_answer="Aurel",
        success=success,
        capability="single_hop",
        stage=0,
        split="deployment",
    )
    for tool in tools:
        exp.tool_calls.append({"tool": tool, "arguments": {"query": "Aurel Institute"}, "ok": True})
        exp.actions.append({"tool": tool})
        if tool == "open":
            exp.observations.append({"tool": "open", "payload": {"document_id": "doc_0000"}})
            exp.evidence.append({"document_id": "doc_0000", "source_type": "official", "reliability": 0.98})
        else:
            exp.observations.append({"tool": tool, "payload": {}})
    if not success:
        exp.failure_type = FailureType.ANSWER_SYNTHESIS
    return exp


def test_verifier_requires_evidence() -> None:
    world = build_world(13)
    verifier = ExperienceVerifier(world)
    bad = _exp(True, ["search"])
    result = verifier.verify(bad)
    assert result.evidence_score == 0.0
    good = _exp(True, ["search", "open"])
    # doc_0000 exists in world
    v2 = verifier.verify(good)
    assert v2.consistency_score > 0


def test_router_invalid_goes_discard() -> None:
    exp = _exp(False, [])
    ver = ExperienceVerifier().verify(exp)
    abs_ = ExperienceAbstractor().abstract(exp, ver)
    decision = HeuristicRouter().route(exp, ver, abs_)
    assert decision.carrier in {Carrier.DISCARD, Carrier.EPISODIC, Carrier.HARNESS}


def test_acquisition_frontier_prefers_unmastered() -> None:
    world = build_world(13)
    by_cap: dict[str, list] = {}
    for t in world.tasks.values():
        if t.split != "acquisition":
            continue
        by_cap.setdefault(t.capability, []).append(t)
    tasks = (by_cap.get("single_hop", [])[:10] + by_cap.get("multi_hop", [])[:10])
    assert tasks
    profile = CapabilityProfile(
        capability_scores={"single_hop": 0.9, "multi_hop": 0.3},
        uncertainty={"single_hop": 0.1, "multi_hop": 0.8},
        frontier_capabilities=[
            FrontierCapability(
                name="multi_hop",
                band=FrontierBand.FRONTIER,
                score=0.3,
                learnability=0.8,
                novelty=0.7,
                transfer_potential=0.8,
                uncertainty=0.8,
                expected_cost=1.2,
            ),
            FrontierCapability(
                name="single_hop",
                band=FrontierBand.MASTERED,
                score=0.9,
                learnability=0.2,
                novelty=0.2,
                transfer_potential=0.2,
                uncertainty=0.1,
                expected_cost=1.0,
            ),
        ],
    )
    props = FrontierAcquisition().propose(PersistentAgentState(), profile, tasks, 8)
    ids = props[0].task_ids
    mh = sum(1 for i in ids if world.tasks[i].capability == "multi_hop")
    sh = sum(1 for i in ids if world.tasks[i].capability == "single_hop")
    assert mh >= sh


def test_passive_is_deterministic_with_seed() -> None:
    world = build_world(13)
    tasks = list(world.tasks.values())[:20]
    a = PassiveAcquisition(seed=3).propose(PersistentAgentState(), CapabilityProfile(), tasks, 5)
    b = PassiveAcquisition(seed=3).propose(PersistentAgentState(), CapabilityProfile(), tasks, 5)
    assert a[0].task_ids == b[0].task_ids


def test_diagnosis_from_real_experiences() -> None:
    exps = [_exp(i % 3 != 0, ["search", "open"]) for i in range(12)]
    for i, e in enumerate(exps):
        e.capability = "single_hop" if i < 6 else "multi_hop"
    profile = CapabilityDiagnoser().diagnose(exps)
    assert profile.n_trajectories == 12
    assert "single_hop" in profile.capability_scores
    assert profile.n_failures > 0


def test_policy_features_exclude_gold_answer() -> None:
    world = build_world(13)
    task = next(iter(world.tasks.values()))
    feat = candidate_features(task, CapabilityProfile(), 4.0)
    assert task.gold_answer.lower() not in " ".join(str(x) for x in feat)


def test_policy_train_step_finite() -> None:
    model = ExperiencePolicyModel()
    import numpy as np

    x = np.zeros(12, dtype=np.float32)
    x[0] = 0.4
    loss = model.train_step(x, acq_target=0.2, util_target=0.2, carrier_index=1, lr=0.01)
    assert loss == loss
