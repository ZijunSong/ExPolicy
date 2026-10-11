"""Episodic memory with capability-aware hybrid retrieval and structured cards."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from expolicy.core.experience import Experience, ExperienceAbstraction
from expolicy.environments.search.world import World
from expolicy.experiments.patterns import (
    expected_failure_signature,
    experience_record_fields,
    normalized_pattern,
)
from expolicy.models.embedder import Embedder, VectorIndex

MEMORY_TOKEN_BUDGET = 90  # ~words per card; keep prompt injection bounded

HYBRID_WEIGHTS = {
    "semantic": 0.35,
    "capability": 0.25,
    "failure": 0.15,
    "pattern": 0.20,
    "recency_success": 0.05,
}


def _clip_card(text: str, budget: int = MEMORY_TOKEN_BUDGET) -> str:
    words = text.split()
    if len(words) <= budget:
        return text
    return " ".join(words[:budget]) + " …"


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

    def add(
        self,
        experience: Experience,
        abstraction: ExperienceAbstraction | None = None,
        *,
        world: World | None = None,
    ) -> None:
        fields = experience_record_fields(experience, world)
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
            **fields,
            "situation": (abstraction.situation if abstraction else experience.question)[:400],
            "correction_abs": abstraction.correction if abstraction else fields["correction"],
            "applicability": list(abstraction.applicability_scope) if abstraction else [experience.capability or "general"],
            "not_applicable_when": list(abstraction.not_applicable_when) if abstraction else ["different capability", "same answer shortcut"],
        }
        self.records.append(record)
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        # Index on question + pattern + capability, not the answer (avoids answer copying).
        text = (
            f"{experience.question}\npattern={fields['normalized_task_pattern']}\n"
            f"capability={experience.capability}\nbridge={fields['bridge_relation']}\n"
            f"{abstraction.reusable_principle if abstraction else fields['correction']}"
        )
        vec = self.embedder.encode([text])
        self.index.add([experience.experience_id], vec)
        self.index.save(self.root)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        capability: str | None = None,
        failure_signature: str | None = None,
        task_pattern: str | None = None,
        mode: str = "hybrid",
        return_debug: bool = False,
    ) -> list[str] | tuple[list[str], list[dict[str, Any]]]:
        if not self.records:
            return ([], []) if return_debug else []
        if mode == "legacy":
            lines, debug = self._retrieve_legacy(query, k=k, capability=capability)
            return (lines, debug) if return_debug else lines
        vec = self.embedder.encode([query])[0]
        hits = self.index.search(vec, k=max(len(self.records), k))
        sim_by_id = {eid: float(sim) for eid, sim in hits}
        now = time.time()
        fail_q = failure_signature or expected_failure_signature(capability)
        pattern_q = task_pattern or ""
        scored: list[tuple[float, dict[str, Any], dict[str, float]]] = []
        w = HYBRID_WEIGHTS
        for row in self.records:
            eid = row["experience_id"]
            semantic = sim_by_id.get(eid, 0.0)
            cap_match = 1.0 if capability and row.get("capability") == capability else 0.0
            row_fail = str(row.get("failure_type") or "")
            if fail_q and row_fail == fail_q:
                fail_match = 1.0
            elif fail_q and row_fail not in {"", "none"}:
                fail_match = 0.35
            else:
                fail_match = 0.0
            row_pat = str(row.get("normalized_task_pattern") or normalized_pattern(str(row.get("task_id") or "")))
            if pattern_q and row_pat == pattern_q:
                pat_match = 1.0
            elif pattern_q and row_pat.split("_")[:1] == pattern_q.split("_")[:1]:
                pat_match = 0.4
            else:
                pat_match = 0.0
            age_days = max((now - float(row.get("timestamp") or now)) / 86400.0, 0.0)
            recency = 1.0 / (1.0 + 0.05 * age_days)
            success_w = 0.55 if row.get("success") else 0.45
            rec_s = 0.5 * recency + 0.5 * success_w
            score = (
                w["semantic"] * semantic
                + w["capability"] * cap_match
                + w["failure"] * fail_match
                + w["pattern"] * pat_match
                + w["recency_success"] * rec_s
            )
            parts = {
                "semantic": semantic,
                "capability": cap_match,
                "failure": fail_match,
                "pattern": pat_match,
                "recency_success": rec_s,
                "score": score,
            }
            scored.append((score, row, parts))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]
        lines = [_format_card(row) for _, row, _ in top]
        debug = [
            {
                "experience_id": row["experience_id"],
                "task_id": row.get("task_id"),
                "capability": row.get("capability"),
                "pattern": row.get("normalized_task_pattern"),
                "success": row.get("success"),
                "score_parts": parts,
                "score": parts["score"],
            }
            for _, row, parts in top
        ]
        return (lines, debug) if return_debug else lines

    def _retrieve_legacy(
        self, query: str, k: int, capability: str | None
    ) -> tuple[list[str], list[dict[str, Any]]]:
        vec = self.embedder.encode([query])[0]
        hits = self.index.search(vec, k=max(k * 3, k))
        by_id = {row["experience_id"]: row for row in self.records}
        now = time.time()
        scored: list[tuple[float, str]] = []
        for eid, sim in hits:
            row = by_id.get(eid)
            if row is None:
                continue
            age_days = max((now - float(row.get("timestamp") or now)) / 86400.0, 0.0)
            recency = 1.0 / (1.0 + 0.05 * age_days)
            cap_bonus = 0.15 if capability and row.get("capability") == capability else 0.0
            scored.append((sim + recency * 0.1 + cap_bonus, eid))
        scored.sort(reverse=True)
        lines = []
        debug = []
        for sim, eid in scored[:k]:
            row = by_id[eid]
            lines.append(
                f"[{row.get('capability')}] Q={row.get('question')} -> {row.get('answer')} "
                f"(success={row.get('success')}) {row.get('principle') or ''}"
            )
            debug.append({"experience_id": eid, "score": sim, "mode": "legacy"})
        return lines, debug

    def last_retrieval_hit_rate(self, debug: list[dict[str, Any]], *, capability: str | None, pattern: str | None) -> dict[str, float]:
        if not debug:
            return {"hit_rate": 0.0, "n": 0.0}
        cap_hits = sum(1 for d in debug if capability and d.get("capability") == capability)
        pat_hits = sum(1 for d in debug if pattern and d.get("pattern") == pattern)
        return {
            "n": float(len(debug)),
            "capability_hit_rate": cap_hits / len(debug),
            "pattern_hit_rate": pat_hits / len(debug),
            "hit_rate": (cap_hits + pat_hits) / (2 * len(debug)),
        }

    def __len__(self) -> int:
        return len(self.records)


def _format_card(row: dict[str, Any]) -> str:
    queries = row.get("search_queries") or []
    qshow = "; ".join(str(q) for q in queries[:3]) or "entity then bridge relation"
    fail = row.get("failure_type") or "none"
    correction = row.get("correction") or row.get("correction_abs") or ""
    warning = "; ".join(row.get("not_applicable_when") or ["do not copy prior answers"])
    apply = ", ".join(row.get("applicability") or [str(row.get("capability") or "")])
    card = (
        f"Situation: {row.get('normalized_task_pattern') or ''} / {row.get('capability')} "
        f"(bridge {row.get('bridge_relation') or 'n/a'}).\n"
        f"Relevant failure: {fail}; action={row.get('failed_action') or 'n/a'}.\n"
        f"Useful evidence pattern: search [{qshow}]; then open hop-1 and hop-2 support.\n"
        f"Corrected procedure: {correction}\n"
        f"Applicability: {apply}.\n"
        f"Warning / when not to use: {warning}; never reuse a stored answer or document id."
    )
    return _clip_card(card)
