"""Replay buffer for experience-policy updates."""

from __future__ import annotations

import json
import random
from pathlib import Path

from expolicy.core.experience import UtilityRecord


class ReplayBuffer:
    def __init__(self, path: Path | None = None, seed: int = 0) -> None:
        self.path = Path(path) if path else None
        self.rng = random.Random(seed)
        self.records: list[UtilityRecord] = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.records.append(UtilityRecord.model_validate(json.loads(line)))

    def add(self, record: UtilityRecord) -> None:
        self.records.append(record)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")

    def sample(self, n: int) -> list[UtilityRecord]:
        if not self.records:
            return []
        n = min(n, len(self.records))
        return self.rng.sample(self.records, n)

    def __len__(self) -> int:
        return len(self.records)
