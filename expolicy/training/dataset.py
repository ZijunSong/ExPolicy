"""ExperienceUtilityDataset built from intervention records."""

from __future__ import annotations

from pathlib import Path

from expolicy.core.experience import UtilityRecord
from expolicy.logging.event_logger import iter_jsonl


class ExperienceUtilityDataset:
    def __init__(self, records: list[UtilityRecord]) -> None:
        self.records = records

    @classmethod
    def from_jsonl(cls, path: Path) -> ExperienceUtilityDataset:
        rows = []
        for raw in iter_jsonl(path):
            rows.append(UtilityRecord.model_validate(raw))
        return cls(rows)

    def split_guard(self) -> None:
        for rec in self.records:
            if rec.split_guard in {"future", "retention"}:
                raise ValueError(f"utility dataset contains eval split {rec.split_guard}")

    def __len__(self) -> int:
        return len(self.records)
