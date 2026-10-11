"""JSON-serializable persistent capability state."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from expolicy.r9_lab.io_util import atomic_write_json, read_json, read_jsonl, sha256_json, write_json


EPISODIC_NAME = "memory_state.jsonl"
SEMANTIC_NAME = "semantic_entries.json"
SKILL_NAME = "best_skill.md"
META_NAME = "STATE.json"
HASH_NAME = "hashes.json"
RETRIEVAL_TRACE = "retrieval_trace.jsonl"


@dataclass
class EpisodicRecord:
    task_id: str
    env: str
    instruction: str
    trajectory: list[dict[str, Any]]
    success: bool
    verifier_outcome: str = "CORRECT"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "EpisodicRecord":
        return cls(
            task_id=str(row["task_id"]),
            env=str(row.get("env") or ""),
            instruction=str(row.get("instruction") or ""),
            trajectory=list(row.get("trajectory") or []),
            success=bool(row.get("success")),
            verifier_outcome=str(row.get("verifier_outcome") or "CORRECT"),
        )


@dataclass
class SemanticEntry:
    scope: str
    trigger: str
    principle: str
    failure_pattern: str = ""
    recommended_action_pattern: str = ""
    evidence_task_ids: list[str] = field(default_factory=list)

    def dedup_key(self) -> tuple[str, str, str]:
        return (self.scope.strip(), self.trigger.strip(), self.principle.strip())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SemanticEntry":
        return cls(
            scope=str(row.get("scope") or ""),
            trigger=str(row.get("trigger") or ""),
            principle=str(row.get("principle") or ""),
            failure_pattern=str(row.get("failure_pattern") or ""),
            recommended_action_pattern=str(row.get("recommended_action_pattern") or ""),
            evidence_task_ids=[str(x) for x in (row.get("evidence_task_ids") or [])],
        )


@dataclass
class PersistentState:
    """Directory-backed persistent state. JSON-serializable except Path root."""

    root: str
    episodic: list[EpisodicRecord] = field(default_factory=list)
    semantic: list[SemanticEntry] = field(default_factory=list)
    skill_markdown: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    committed_task_ids: list[str] = field(default_factory=list)

    def root_path(self) -> Path:
        return Path(self.root)

    def content_payload(self) -> dict[str, Any]:
        return {
            "episodic": [r.to_dict() for r in self.episodic],
            "semantic": [s.to_dict() for s in self.semantic],
            "skill_markdown": self.skill_markdown,
            "committed_task_ids": list(self.committed_task_ids),
            "extra": dict(self.extra),
        }

    def content_hash(self) -> str:
        return sha256_json(self.content_payload())

    def to_dict(self) -> dict[str, Any]:
        body = self.content_payload()
        body["root"] = self.root
        body["content_hash"] = self.content_hash()
        return body

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, root: str | None = None) -> "PersistentState":
        return cls(
            root=str(root or payload.get("root") or ""),
            episodic=[EpisodicRecord.from_dict(r) for r in payload.get("episodic") or []],
            semantic=[SemanticEntry.from_dict(r) for r in payload.get("semantic") or []],
            skill_markdown=str(payload.get("skill_markdown") or ""),
            extra=dict(payload.get("extra") or {}),
            committed_task_ids=[str(x) for x in payload.get("committed_task_ids") or []],
        )

    def save(self) -> str:
        root = self.root_path()
        root.mkdir(parents=True, exist_ok=True)
        jsonl = root / EPISODIC_NAME
        jsonl.write_text(
            "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in self.episodic),
            encoding="utf-8",
        )
        atomic_write_json(root / SEMANTIC_NAME, [s.to_dict() for s in self.semantic])
        (root / SKILL_NAME).write_text(self.skill_markdown, encoding="utf-8")
        digest = self.content_hash()
        atomic_write_json(root / META_NAME, self.to_dict())
        atomic_write_json(root / HASH_NAME, {"content_hash": digest})
        return digest

    @classmethod
    def load(cls, root: Path | str) -> "PersistentState":
        root_p = Path(root)
        meta = root_p / META_NAME
        if meta.exists():
            state = cls.from_dict(read_json(meta), root=str(root_p))
            return state
        episodic = [EpisodicRecord.from_dict(r) for r in read_jsonl(root_p / EPISODIC_NAME)]
        sem_path = root_p / SEMANTIC_NAME
        semantic = [SemanticEntry.from_dict(r) for r in (read_json(sem_path) if sem_path.exists() else [])]
        skill_path = root_p / SKILL_NAME
        skill = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""
        return cls(root=str(root_p), episodic=episodic, semantic=semantic, skill_markdown=skill)

    @classmethod
    def create(cls, root: Path | str) -> "PersistentState":
        state = cls(root=str(root))
        state.save()
        return state

    def append_retrieval_trace(self, row: dict[str, Any]) -> None:
        from expolicy.r9_lab.io_util import append_jsonl

        append_jsonl(self.root_path() / RETRIEVAL_TRACE, row)


def token_overlap(query: str, text: str) -> float:
    q = set((query or "").lower().split())
    t = set((text or "").lower().split())
    if not q:
        return 0.0
    return len(q & t) / max(1, len(q))
