"""Episodic memory with vector retrieval, recency, capability tags, provenance."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from expolicy.core.experience import Experience, ExperienceAbstraction
from expolicy.models.embedder import Embedder, VectorIndex


class EpisodicMemory:
    def __init__(self, root: Path, embedder: Embedder) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.records_path = self.root / "records.jsonl"
        self.index = VectorIndex.load(self.root, embedder.dim)
        self.records: list[dict[str, Any]] = []
        if self.records_path.exists():
            for line in self.records_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.records.append(json.loads(line))

    def add(self, experience: Experience, abstraction: ExperienceAbstraction | None = None) -> None:
        text = (
            f"{experience.question}\n{experience.predicted_answer}\n"
            f"{abstraction.reusable_principle if abstraction else ''}"
        )
        record = {
            "experience_id": experience.experience_id,
            "task_id": experience.task_id,
            "question": experience.question,
            "answer": experience.predicted_answer,
            "success": experience.success,
            "capability": experience.capability,
            "timestamp": time.time(),
            "principle": abstraction.reusable_principle if abstraction else "",
            "provenance": experience.provenance,
        }
        self.records.append(record)
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        vec = self.embedder.encode([text])
        self.index.add([experience.experience_id], vec)
        self.index.save(self.root)

    def retrieve(self, query: str, k: int = 5, *, capability: str | None = None) -> list[str]:
        if not self.records:
            return []
        vec = self.embedder.encode([query])[0]
        hits = self.index.search(vec, k=max(k * 3, k))
        by_id = {row["experience_id"]: row for row in self.records}
        now = time.time()
        scored: list[tuple[float, str]] = []
        for eid, sim in hits:
            row = by_id.get(eid)
            if row is None:
                continue
            if capability and row.get("capability") not in {None, capability}:
                recency = 1.0
            else:
                recency = 1.0
            age_days = max((now - float(row.get("timestamp") or now)) / 86400.0, 0.0)
            recency = 1.0 / (1.0 + 0.05 * age_days)
            cap_bonus = 0.15 if capability and row.get("capability") == capability else 0.0
            scored.append((sim + recency * 0.1 + cap_bonus, eid))
        scored.sort(reverse=True)
        lines = []
        for _, eid in scored[:k]:
            row = by_id[eid]
            lines.append(
                f"[{row.get('capability')}] Q={row.get('question')} -> {row.get('answer')} "
                f"(success={row.get('success')}) {row.get('principle') or ''}"
            )
        return lines

    def __len__(self) -> int:
        return len(self.records)
