"""Train Experience Policy from a utility JSONL buffer."""

from __future__ import annotations

from pathlib import Path

from expolicy.training.policy_model import ExperiencePolicyModel
from expolicy.training.policy_trainer import PolicyTrainer
from expolicy.training.replay_buffer import ReplayBuffer


def train_policy(buffer_path: Path, out_path: Path, *, steps: int = 200) -> dict:
    buffer = ReplayBuffer(buffer_path)
    model = ExperiencePolicyModel()
    trainer = PolicyTrainer(model, buffer)
    summary = trainer.fit(steps=steps)
    model.save(out_path)
    summary["out"] = str(out_path)
    summary["n_records"] = len(buffer)
    return summary


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/experience_policy.yaml")
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--out", default="outputs/experience_policy.npz")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(train_policy(Path(args.buffer), Path(args.out), steps=args.steps), indent=2))
