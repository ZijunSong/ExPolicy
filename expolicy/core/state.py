"""Persistent agent state: weights, memories, skills, harness, lineage."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from expolicy.core.capability import CapabilityProfile
from expolicy.core.errors import CheckpointError
from expolicy.core.types import JSONDict


class HarnessConfig(BaseModel):
    max_steps: int = 10
    search_top_k: int = 5
    enable_verifier: bool = True
    evidence_checkpoint: bool = True
    rollback_on_conflict: bool = False
    require_citation: bool = True
    context_packing: str = "recent_evidence"
    max_context_chars: int = 6000
    tool_order: list[str] = Field(default_factory=lambda: ["search", "open", "find", "answer"])
    extra_search_on_conflict: bool = True
    min_evidence_before_answer: int = 1
    temperature: float = 0.0
    n_consistency: int = 1
    version: int = 0
    notes: str = ""

    def allowed_patch_keys(self) -> set[str]:
        return {
            "max_steps",
            "search_top_k",
            "enable_verifier",
            "evidence_checkpoint",
            "rollback_on_conflict",
            "require_citation",
            "context_packing",
            "max_context_chars",
            "tool_order",
            "extra_search_on_conflict",
            "min_evidence_before_answer",
            "temperature",
            "n_consistency",
            "notes",
        }


class LineageRecord(BaseModel):
    parent: str | None
    child: str
    reason: str
    timestamp: float = Field(default_factory=time.time)
    experience_ids: list[str] = Field(default_factory=list)
    carriers_used: list[str] = Field(default_factory=list)
    metrics_delta: JSONDict = Field(default_factory=dict)


class PersistentAgentState(BaseModel):
    """In-memory handle. Heavy stores live on disk under ``root``."""

    version: str = "agent_v000"
    root: str = ""
    model_name: str = ""
    adapter_path: str | None = None
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    capability_profile: CapabilityProfile = Field(default_factory=CapabilityProfile)
    lineage: list[LineageRecord] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    metrics: JSONDict = Field(default_factory=dict)
    extra: JSONDict = Field(default_factory=dict)

    def version_index(self) -> int:
        suffix = self.version.rsplit("_v", 1)[-1]
        try:
            return int(suffix)
        except ValueError as exc:
            raise CheckpointError(f"unrecognized agent version {self.version}") from exc

    def next_version(self) -> str:
        return f"agent_v{self.version_index() + 1:03d}"

    def root_path(self) -> Path:
        if not self.root:
            raise CheckpointError("PersistentAgentState.root is empty")
        return Path(self.root)

    def episodic_dir(self) -> Path:
        return self.root_path() / "episodic_memory"

    def semantic_dir(self) -> Path:
        return self.root_path() / "semantic_memory"

    def skills_dir(self) -> Path:
        return self.root_path() / "skills"

    def harness_dir(self) -> Path:
        return self.root_path() / "harness"

    def adapter_dir(self) -> Path:
        return self.root_path() / "model_adapter"

    def write_manifest(self) -> None:
        root = self.root_path()
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(
            json.dumps(self.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        (root / "capability_profile.json").write_text(
            self.capability_profile.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (root / "metrics.json").write_text(
            json.dumps(self.metrics, indent=2),
            encoding="utf-8",
        )
        (root / "lineage.json").write_text(
            json.dumps([item.model_dump(mode="json") for item in self.lineage], indent=2),
            encoding="utf-8",
        )
        self.harness_dir().mkdir(parents=True, exist_ok=True)
        (self.harness_dir() / "harness.json").write_text(
            self.harness.model_dump_json(indent=2),
            encoding="utf-8",
        )
        self.episodic_dir().mkdir(parents=True, exist_ok=True)
        self.semantic_dir().mkdir(parents=True, exist_ok=True)
        self.skills_dir().mkdir(parents=True, exist_ok=True)
        self.adapter_dir().mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, root: str | Path) -> PersistentAgentState:
        root_p = Path(root)
        manifest = root_p / "manifest.json"
        if not manifest.exists():
            raise CheckpointError(f"missing manifest.json under {root_p}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["root"] = str(root_p)
        return cls.model_validate(payload)

    def snapshot_into(self, dest: str | Path, *, copy_stores: bool = True) -> PersistentAgentState:
        dest_p = Path(dest)
        if dest_p.exists():
            raise CheckpointError(f"refusing to overwrite existing agent dir {dest_p}")
        dest_p.mkdir(parents=True, exist_ok=True)
        cloned = self.model_copy(deep=True)
        cloned.root = str(dest_p)
        cloned.write_manifest()
        if copy_stores and self.root and Path(self.root).exists():
            for name in ("episodic_memory", "semantic_memory", "skills", "model_adapter", "harness"):
                src = Path(self.root) / name
                dst = dest_p / name
                if src.exists():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
        return cloned


def empty_agent(root: str | Path, *, model_name: str = "", version: str = "agent_v000") -> PersistentAgentState:
    state = PersistentAgentState(version=version, root=str(root), model_name=model_name)
    state.write_manifest()
    return state
