"""Constrained harness evolution: proposal → validate → sandbox → probes → commit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from expolicy.core.errors import HarnessPatchRejected
from expolicy.core.state import HarnessConfig, PersistentAgentState


class HarnessStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.history_path = self.root / "history.jsonl"

    def propose(self, current: HarnessConfig, patch: dict[str, Any]) -> HarnessConfig:
        allowed = current.allowed_patch_keys()
        illegal = set(patch) - allowed
        if illegal:
            raise HarnessPatchRejected(f"harness patch contains illegal keys {sorted(illegal)}")
        if "tool_order" in patch:
            tools = patch["tool_order"]
            if not isinstance(tools, list) or any(t not in {"search", "open", "find", "answer", "invoke_skill"} for t in tools):
                raise HarnessPatchRejected("tool_order is not a permitted tool list")
        data = current.model_dump()
        data.update(patch)
        data["version"] = current.version + 1
        return HarnessConfig.model_validate(data)

    def regression_test(
        self,
        candidate: HarnessConfig,
        probe_fn: Callable[[HarnessConfig], dict[str, Any]],
        *,
        min_accuracy: float,
    ) -> dict[str, Any]:
        metrics = probe_fn(candidate)
        acc = float(metrics.get("accuracy") or 0.0)
        if acc + 1e-9 < min_accuracy:
            raise HarnessPatchRejected(
                f"harness regression: accuracy {acc:.3f} < required {min_accuracy:.3f}"
            )
        return metrics

    def commit(self, state: PersistentAgentState, candidate: HarnessConfig, metrics: dict[str, Any]) -> None:
        prev = state.harness
        state.harness = candidate
        record = {
            "from_version": prev.version,
            "to_version": candidate.version,
            "harness": candidate.model_dump(mode="json"),
            "metrics": metrics,
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        state.write_manifest()

    def rollback(self, state: PersistentAgentState) -> HarnessConfig:
        if not self.history_path.exists():
            raise HarnessPatchRejected("no harness history to rollback")
        lines = [json.loads(line) for line in self.history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) < 2:
            raise HarnessPatchRejected("cannot rollback the initial harness")
        prev = HarnessConfig.model_validate(lines[-2]["harness"])
        state.harness = prev
        state.write_manifest()
        return prev
