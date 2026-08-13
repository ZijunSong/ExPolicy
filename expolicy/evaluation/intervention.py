"""A/B intervention utility: clone agent, apply one carrier, evaluate probes."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from expolicy.assimilation.apply import CarrierRuntime
from expolicy.core.experience import (
    AssimilationDecision,
    Experience,
    ExperienceAbstraction,
    ExperienceUtility,
    UtilityRecord,
)
from expolicy.core.state import PersistentAgentState
from expolicy.core.types import Carrier
from expolicy.core.utility import compute_utility, to_experience_utility
from expolicy.environments.search.world import Task
from expolicy.evaluation.metrics import aggregate, score_experience
from expolicy.logging.event_logger import EventLogger


class UtilityEvaluator:
    def __init__(
        self,
        run_fn: Callable[[PersistentAgentState, Sequence[Task]], list[Experience]],
        logger: EventLogger | None = None,
    ) -> None:
        self.run_fn = run_fn
        self.logger = logger

    def intervene(
        self,
        before_state: PersistentAgentState,
        after_state: PersistentAgentState,
        future_probe: Sequence[Task],
        retention_probe: Sequence[Task],
        experience: Experience,
        carrier: Carrier,
    ) -> ExperienceUtility:
        before_future = self.run_fn(before_state, future_probe)
        after_future = self.run_fn(after_state, future_probe)
        before_ret = self.run_fn(before_state, retention_probe) if retention_probe else []
        after_ret = self.run_fn(after_state, retention_probe) if retention_probe else []
        bf = aggregate([score_experience(e) for e in before_future])
        af = aggregate([score_experience(e) for e in after_future])
        br = aggregate([score_experience(e) for e in before_ret]) if before_ret else {"accuracy": 0.0}
        ar = aggregate([score_experience(e) for e in after_ret]) if after_ret else {"accuracy": 0.0}
        future_gain = float(af["accuracy"] - bf["accuracy"])
        retention_gain = float(ar.get("accuracy", 0.0) - br.get("accuracy", 0.0))
        regression = min(0.0, retention_gain)
        learning = compute_utility(
            future_gain=future_gain,
            transfer_gain=0.0,
            retention_gain=retention_gain,
            cost=experience.cost,
            regression=regression,
            interference=max(0.0, -retention_gain),
        )
        util = to_experience_utility(
            experience.experience_id,
            learning,
            before_metrics={"future": bf, "retention": br},
            after_metrics={"future": af, "retention": ar},
            carrier=carrier,
        )
        if self.logger:
            self.logger.emit(
                "UTILITY_MEASURED",
                {"experience_id": experience.experience_id, "utility": util.utility, "carrier": carrier.value},
            )
        return util


def carrier_matrix(
    runtime_factory: Callable[[PersistentAgentState], CarrierRuntime],
    eval_fn: Callable[[PersistentAgentState, Sequence[Task]], list[Experience]],
    base_state: PersistentAgentState,
    dest_root: Path,
    experience: Experience,
    abstraction: ExperienceAbstraction,
    probe: Sequence[Task],
    carriers: Sequence[Carrier] | None = None,
) -> dict[str, float]:
    carriers = list(carriers or list(Carrier))
    matrix: dict[str, float] = {}
    base_rows = [score_experience(e) for e in eval_fn(base_state, probe)]
    base_acc = aggregate(base_rows)["accuracy"]
    for carrier in carriers:
        child_dir = dest_root / f"interv_{experience.experience_id[:8]}_{carrier.value}"
        child = base_state.snapshot_into(child_dir, copy_stores=True)
        runtime = runtime_factory(child)
        decision = AssimilationDecision(
            experience_id=experience.experience_id,
            carrier=carrier,
            confidence=1.0,
            expected_utility=0.0,
            expected_interference=0.0,
            policy_name="oracle_sweep",
        )
        runtime.apply(experience, abstraction, decision, verified_batch=[experience])
        after_rows = [score_experience(e) for e in eval_fn(child, probe)]
        matrix[carrier.value] = float(aggregate(after_rows)["accuracy"] - base_acc)
    return matrix
