"""Frozen assimilators. Acquisition code does not edit prompts, capacity, or hyperparameters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from expolicy.paper_b.constants import INDUCER_VERSION
from expolicy.paper_c.features import LeakageError
from expolicy.r9_lab.assimilators.episodic import apply_episodic
from expolicy.r9_lab.constants import EPISODIC_TOP_K
from expolicy.r9_lab.io_util import sha256_file, sha256_json
from expolicy.r9_lab.state.persistent_state import PersistentState

SYNTH_NAME = "synthetic_content"
SYNTH_VERSION = "c-synth-content-1"
SYNTH_PROMPT = (
    "Commit non-empty family-tagged experience into the persistent family index. "
    "Empty payloads increment a count and do not write a family. No gold and no probe tasks."
)
_GOLD_KEYS = {"gold", "gold_answer", "gold_sql", "correct_answer"}
_BLOCKED_BUCKETS = {"future_test", "shadow_reward_probe", "retention_probe", "shadow_select", "shadow_eval"}


class AssimilatorDrift(RuntimeError):
    pass


class FrozenAssimilator(Protocol):
    name: str
    version: str
    prompt: str

    def algorithm_hash(self) -> str: ...

    def apply(self, state: PersistentState, experiences: Sequence[Mapping[str, Any]]) -> dict[str, Any]: ...


def guard_experiences(experiences: Sequence[Mapping[str, Any]]) -> None:
    for item in experiences:
        leaked = sorted(set(item) & _GOLD_KEYS)
        if leaked:
            raise LeakageError(f"assimilator refused gold fields: {leaked}")
        bucket = str(item.get("bucket") or item.get("split") or "")
        if bucket in _BLOCKED_BUCKETS:
            raise LeakageError(f"probe or future experience cannot be assimilated: {bucket}")


def _result(
    *,
    name: str,
    algorithm_hash: str,
    accepted: bool,
    changed: bool,
    n: int,
    teacher_tokens: int = 0,
    actor_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "action": name,
        "accepted": accepted,
        "changed": changed,
        "n": n,
        "algorithm_hash": algorithm_hash,
        "teacher_tokens": teacher_tokens,
        "actor_tokens": actor_tokens,
        "validation_calls": 0,
        "online_algorithm_edit": False,
    }


class SyntheticContentAssimilator:
    """Harness F. A family skill is written only when the experience still has content.

    This is the executable stand-in used to test the acquisition protocol. It is not a
    result on a public environment, and it does not certify paper B.
    """

    name = SYNTH_NAME
    version = SYNTH_VERSION
    prompt = SYNTH_PROMPT

    def __init__(self, *, max_entries: int = 256) -> None:
        self.max_entries = int(max_entries)
        self._hash = sha256_json(
            {
                "name": self.name,
                "version": self.version,
                "prompt": self.prompt,
                "max_entries": self.max_entries,
                "source": sha256_file(Path(__file__).resolve()),
            }
        )

    def algorithm_hash(self) -> str:
        return self._hash

    def apply(self, state: PersistentState, experiences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        guard_experiences(experiences)
        counts = dict(state.extra.get("family_counts") or {})
        empty_writes = int(state.extra.get("empty_writes") or 0)
        written = 0
        for item in experiences:
            if len(state.committed_task_ids) + written >= self.max_entries:
                break
            instruction = str(item.get("instruction") or "").strip()
            family = str(item.get("family") or "").strip()
            if item.get("content_control") == "empty" or not instruction or not family:
                empty_writes += 1
                continue
            counts[family] = int(counts.get(family) or 0) + 1
            task_id = str(item.get("task_id") or f"anon:{family}:{written}")
            if task_id not in state.committed_task_ids:
                state.committed_task_ids.append(task_id)
            written += 1
        state.extra["family_counts"] = counts
        state.extra["empty_writes"] = empty_writes
        state.save()
        return _result(
            name=self.name,
            algorithm_hash=self.algorithm_hash(),
            accepted=True,
            changed=written > 0 or empty_writes > 0,
            n=written,
        )


class EpisodicFrozenAssimilator:
    """A1 episodic update with the repository's frozen top-k. C0 is not claimed."""

    name = "episodic_replay"
    version = "a1-frozen"
    prompt = "Store verified successful trajectories. Retrieval top_k stays at the A1 constant."

    def __init__(self) -> None:
        source = Path(apply_episodic.__code__.co_filename)
        self._hash = sha256_json(
            {
                "name": self.name,
                "version": self.version,
                "prompt": self.prompt,
                "top_k": EPISODIC_TOP_K,
                "source": sha256_file(source),
            }
        )

    def algorithm_hash(self) -> str:
        return self._hash

    def apply(self, state: PersistentState, experiences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        guard_experiences(experiences)
        if EPISODIC_TOP_K != 4:
            raise AssimilatorDrift("episodic top_k left the frozen A1 value")
        result = apply_episodic(state, [dict(item) for item in experiences])
        return _result(
            name=self.name,
            algorithm_hash=self.algorithm_hash(),
            accepted=True,
            changed=bool(result.get("changed")),
            n=int(result.get("n") or 0),
        )


def paper_b_candidate() -> dict[str, Any]:
    """B's inducer is visible to the manifest and is not an approved C assimilator."""
    return {
        "name": "paper_b_inducer",
        "version": INDUCER_VERSION,
        "c0_status": "NOT_RUN",
        "usable_as_c_assimilator": False,
        "reason": "B has no content-gain pass on two distinct public environments",
    }


def assert_same_algorithm(expected: str, result: Mapping[str, Any]) -> None:
    if result.get("online_algorithm_edit"):
        raise AssimilatorDrift("acquisition is not allowed to edit the assimilator")
    if result.get("algorithm_hash") != expected:
        raise AssimilatorDrift("assimilator hash changed during the run")
