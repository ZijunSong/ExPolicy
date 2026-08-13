"""Online self-evolving loop. Real diagnose → acquire → execute → assimilate → measure → learn."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from expolicy.acquisition.base import make_acquisition
from expolicy.acquisition.generation import FailureSeekingGenerator, HardVariantGenerator
from expolicy.agents.persistent_agent import PersistentAgent
from expolicy.assimilation.abstractor import ExperienceAbstractor
from expolicy.assimilation.apply import CarrierRuntime
from expolicy.assimilation.carriers.harness_store import HarnessStore
from expolicy.assimilation.carriers.weight_update import WeightUpdater
from expolicy.assimilation.router import HeuristicRouter
from expolicy.assimilation.verifier import ExperienceVerifier
from expolicy.benchmarks.search_benchmark import ContinualSuite, SearchBenchmark
from expolicy.core.config import ExperimentConfig
from expolicy.core.experience import Experience, UtilityRecord
from expolicy.core.lifecycle import transition
from expolicy.core.types import Carrier, ExperienceStatus
from expolicy.diagnosis.capability_profile import CapabilityDiagnoser
from expolicy.evaluation.evaluator import Evaluator
from expolicy.evaluation.intervention import UtilityEvaluator
from expolicy.evaluation.metrics import aggregate, score_experience
from expolicy.logging.event_logger import EventLogger, append_jsonl
from expolicy.orchestration.checkpoint import bump_agent, snapshot_run
from expolicy.orchestration.scheduler import BudgetTracker
from expolicy.training.policy_model import LearnedAssimilation, candidate_features
from expolicy.training.policy_trainer import PolicyTrainer
from expolicy.training.replay_buffer import ReplayBuffer


class EvolveLoop:
    def __init__(
        self,
        cfg: ExperimentConfig,
        agent: PersistentAgent,
        benchmark: SearchBenchmark,
        output_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.agent = agent
        self.benchmark = benchmark
        self.suite = ContinualSuite(benchmark)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = agent.logger or EventLogger(self.output_dir / "events.jsonl")
        self.verifier = ExperienceVerifier(benchmark.world)
        self.abstractor = ExperienceAbstractor()
        self.router = HeuristicRouter()
        self.learned_asm = LearnedAssimilation()
        self.diagnoser = CapabilityDiagnoser(agent.embedder)
        self.budget = BudgetTracker(cfg.budget)
        self.buffer = ReplayBuffer(self.output_dir / "utility_records.jsonl", seed=cfg.seed)
        self.hard_gen = HardVariantGenerator(benchmark.world, seed=cfg.seed)
        self.fail_gen = FailureSeekingGenerator(benchmark.world, seed=cfg.seed)
        self.evaluator = Evaluator(agent.search_agent, self.output_dir)
        snapshot_run(self.output_dir, cfg.model_dump(mode="json"))

    def run(self) -> dict:
        history: list[Experience] = []
        stage = self.cfg.stage if self.cfg.stage is not None else 0
        while self.budget.remaining():
            capability = self.diagnoser.diagnose(history, previous=self.agent.state.capability_profile, agent_version=self.agent.state.version)
            self.agent.state.capability_profile = capability
            self.logger.emit("CAPABILITY_DIAGNOSED", capability.model_dump(mode="json"))
            pool = self.suite.acquisition_pool(stage) or self.suite.deployment_pool(stage)
            if not pool:
                break
            policy = make_acquisition(self.cfg.acquisition, seed=self.cfg.seed)
            proposals = policy.propose(self.agent.state, capability, pool, self.cfg.budget.acquisition_batch, recent=history)
            task_ids = proposals[0].task_ids if proposals else [t.task_id for t in pool[: self.cfg.budget.acquisition_batch]]
            self.logger.emit("ACQUISITION_DECISION", {"task_ids": task_ids, "mode": self.cfg.acquisition})
            for task_id in task_ids:
                if not self.budget.remaining():
                    break
                task = self.benchmark.world.tasks[task_id]
                before = self.agent.state
                experience = self.agent.run_task(task)
                history.append(experience)
                self.budget.consume(experience.cost)
                append_jsonl(self.output_dir / "experiences.jsonl", json.loads(experience.model_dump_json()))
                verification = self.verifier.verify(experience)
                experience.mark(transition(experience.status, ExperienceStatus.VERIFIED if verification.valid else ExperienceStatus.REJECTED))
                self.logger.emit("EXPERIENCE_VERIFIED", verification.model_dump(mode="json"))
                if not verification.valid:
                    continue
                abstraction = self.abstractor.abstract(experience, verification)
                experience.mark(transition(ExperienceStatus.VERIFIED, ExperienceStatus.ABSTRACTED))
                self.logger.emit("EXPERIENCE_ABSTRACTED", json.loads(abstraction.model_dump_json()))
                if self.cfg.assimilation == "learned":
                    decision = self.learned_asm.route(experience, verification, abstraction, capability)
                else:
                    decision = self.router.route(experience, verification, abstraction, capability)
                if self.cfg.variant == "B5" and decision.carrier in {Carrier.WEIGHT, Carrier.HARNESS, Carrier.SKILL}:
                    decision.carrier = Carrier.EPISODIC
                if self.cfg.variant == "B6" and decision.carrier != Carrier.WEIGHT:
                    decision.carrier = Carrier.WEIGHT
                self.logger.emit("ASSIMILATION_DECISION", json.loads(decision.model_dump(mode="json")))
                dest_root = self.output_dir / "agents"
                dest_root.mkdir(exist_ok=True)
                child = bump_agent(self.agent.state, dest_root, f"assimilate:{decision.carrier.value}", [experience.experience_id])
                runtime = self._runtime(child)
                runtime.apply(experience, abstraction, decision, verified_batch=[experience])
                self.agent.state = child
                self.agent.search_agent.harness = child.harness
                self.logger.emit("AGENT_VERSION_CREATED", {"version": child.version})
                probe = self.suite.validation_pool(stage)[: self.cfg.probe_limit]
                retention = self.suite.retention_pool(stage)[: self.cfg.probe_limit]
                util_eval = UtilityEvaluator(self._eval_state, logger=self.logger)
                realized = util_eval.intervene(before, child, probe, retention, experience, decision.carrier)
                feat = candidate_features(task, capability, budget_left=float(self.budget.config.acquisition_batch))
                record = UtilityRecord(
                    agent_version_before=before.version,
                    agent_version_after=child.version,
                    experience_id=experience.experience_id,
                    carrier=decision.carrier,
                    candidate_features=feat.tolist(),
                    realized_utility=realized.utility,
                    future_gain=realized.future_gain,
                    retention_delta=realized.retention_gain,
                    cost=experience.cost,
                )
                self.buffer.add(record)
                if len(self.buffer) >= 4:
                    trainer = PolicyTrainer(self.learned_asm.model, self.buffer)
                    trainer.fit(steps=20, batch_size=4)
                    self.learned_asm.model.save(self.output_dir / "experience_policy.npz")
        metrics = aggregate([score_experience(e) for e in history])
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return {"metrics": metrics, "n_experiences": len(history), "budget": self.budget.snapshot()}

    def _runtime(self, state) -> CarrierRuntime:
        self.agent.state = state
        return CarrierRuntime(
            state,
            episodic=self.agent.episodic,
            semantic=self.agent.semantic,
            skills=self.agent.skills,
            harness=HarnessStore(state.harness_dir()),
            weights=WeightUpdater(self.cfg.model.path, Path(state.adapter_dir())) if self.cfg.enable_weight_update else None,
            logger=self.logger,
            enable_harness=self.cfg.enable_harness_evolution,
            enable_weight=self.cfg.enable_weight_update,
        )

    def _eval_state(self, state, tasks: Sequence) -> list[Experience]:
        prev = self.agent.state
        self.agent.state = state
        self.agent.search_agent.harness = state.harness
        out = [self.agent.run_task(t) for t in tasks]
        self.agent.state = prev
        self.agent.search_agent.harness = prev.harness
        return out
