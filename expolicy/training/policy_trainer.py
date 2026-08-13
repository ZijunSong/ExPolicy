"""Train Experience Policy from realized utility records (no imitation of heuristics)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from expolicy.core.errors import TrainingDivergenceError
from expolicy.core.types import Carrier
from expolicy.training.policy_model import CARRIERS, ExperiencePolicyModel
from expolicy.training.replay_buffer import ReplayBuffer


class PolicyTrainer:
    def __init__(self, model: ExperiencePolicyModel, buffer: ReplayBuffer) -> None:
        self.model = model
        self.buffer = buffer

    def fit(self, steps: int = 200, batch_size: int = 8, lr: float = 1e-2) -> dict:
        if len(self.buffer) == 0:
            raise TrainingDivergenceError("empty utility replay buffer")
        losses = []
        for step in range(steps):
            batch = self.buffer.sample(min(batch_size, len(self.buffer)))
            step_loss = 0.0
            for rec in batch:
                x = np.array(rec.candidate_features or rec.state_features, dtype=np.float32)
                if x.size != self.model.in_dim:
                    if x.size > self.model.in_dim:
                        x = x[: self.model.in_dim]
                    else:
                        x = np.pad(x, (0, self.model.in_dim - x.size))
                carrier_index = None
                if rec.carrier in CARRIERS:
                    carrier_index = CARRIERS.index(rec.carrier)
                loss = self.model.train_step(
                    x,
                    acq_target=float(rec.realized_utility),
                    util_target=float(rec.realized_utility),
                    carrier_index=carrier_index,
                    lr=lr,
                )
                if not np.isfinite(loss):
                    raise TrainingDivergenceError(f"NaN policy loss at step {step}")
                step_loss += loss
            losses.append(step_loss / max(len(batch), 1))
        return {"steps": steps, "mean_loss": float(sum(losses) / len(losses)), "final_loss": float(losses[-1])}


class OPDTrainer:
    """Distillation-style follow-up on verified abstractions (tiny LoRA via WeightUpdater)."""

    def __init__(self, weight_updater: object) -> None:
        self.weight_updater = weight_updater

    def fit(self, experiences: list, **kwargs: object) -> dict:
        return self.weight_updater.train(experiences, **kwargs)  # type: ignore[no-any-return]


def dump_utility_dataset(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(rec.model_dump_json() + "\n")
