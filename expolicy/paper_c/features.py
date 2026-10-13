"""Features visible before a candidate is executed. Probe labels are not features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from expolicy.paper_c.constants import (
    FORBIDDEN_FEATURE_KEYS,
    HISTORICAL_FEATURES,
    PUBLIC_CANDIDATE_KEYS,
)
from expolicy.r9_lab.state.persistent_state import PersistentState


class LeakageError(RuntimeError):
    pass


@dataclass
class FamilyStats:
    outcomes: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.outcomes)

    def success_rate(self) -> float | None:
        if not self.outcomes:
            return None
        return sum(self.outcomes) / len(self.outcomes)

    def fail_rate(self) -> float | None:
        rate = self.success_rate()
        if rate is None:
            return None
        return 1.0 - rate

    def progress(self) -> float | None:
        if len(self.outcomes) < 2:
            return None
        previous = sum(self.outcomes[:-1]) / (len(self.outcomes) - 1)
        full = sum(self.outcomes) / len(self.outcomes)
        return full - previous

    def variance(self) -> float | None:
        if len(self.outcomes) < 2:
            return None
        mean = sum(self.outcomes) / len(self.outcomes)
        return sum((item - mean) ** 2 for item in self.outcomes) / (len(self.outcomes) - 1)


@dataclass
class AcquisitionHistory:
    """Outcomes of rounds that already finished. The current candidate is not in here yet."""

    by_family: dict[str, FamilyStats] = field(default_factory=dict)

    def stats(self, family: str) -> FamilyStats:
        return self.by_family.setdefault(family, FamilyStats())

    def observe(self, family: str, success: float) -> None:
        self.stats(family).outcomes.append(float(success))


def model_feature_names(families: Sequence[str], envs: Sequence[str]) -> tuple[str, ...]:
    names = ["n_committed", "public_difficulty", "public_estimated_tools"]
    names += [f"env_{env}" for env in envs]
    names += [f"family__{family}" for family in families]
    names += [f"covered__{family}" for family in families]
    return tuple(names)


def state_free_feature_names(families: Sequence[str], envs: Sequence[str]) -> tuple[str, ...]:
    blocked = {"n_committed"} | {f"covered__{family}" for family in families}
    return tuple(name for name in model_feature_names(families, envs) if name not in blocked)


def env_feature_names(envs: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"env_{env}" for env in envs)


def assert_feature_keys(keys: Sequence[str]) -> None:
    banned = [key for key in keys if key in FORBIDDEN_FEATURE_KEYS]
    if banned:
        raise LeakageError(f"post-acquisition fields used as features: {banned}")
    for key in keys:
        allowed = (
            key in {"n_committed", "public_difficulty", "public_estimated_tools"}
            or key in HISTORICAL_FEATURES
            or key.startswith("env_")
            or key.startswith("family__")
            or key.startswith("covered__")
        )
        if not allowed:
            raise LeakageError(f"feature is outside the pre-acquisition whitelist: {key}")


def pre_acquisition_features(
    state: PersistentState,
    *,
    family: str,
    env: str,
    families: Sequence[str],
    public: Mapping[str, Any],
    history: AcquisitionHistory | None,
    declared_envs: Sequence[str],
) -> dict[str, float | None]:
    """State summary and public candidate metadata. Current-rollout outcomes are refused."""
    unknown = [key for key in public if key not in PUBLIC_CANDIDATE_KEYS]
    if unknown:
        raise LeakageError(f"candidate metadata is not on the pre-acquisition whitelist: {unknown}")
    if env not in declared_envs:
        raise LeakageError(f"undeclared environment: {env}")
    if family not in families:
        raise LeakageError(f"candidate family is outside the declared pool: {family}")
    counts = dict(state.extra.get("family_counts") or {})
    features: dict[str, float | None] = {
        "n_committed": float(len(state.committed_task_ids)),
        "public_difficulty": float(public["public_difficulty"]),
        "public_estimated_tools": float(public["public_estimated_tools"]),
    }
    for name in declared_envs:
        features[f"env_{name}"] = 1.0 if name == env else 0.0
    for name in families:
        features[f"family__{name}"] = 1.0 if name == family else 0.0
        covered = name == family and int(counts.get(name) or 0) >= 1
        features[f"covered__{name}"] = 1.0 if covered else 0.0
    stats = None if history is None else history.by_family.get(family)
    features["hist_fail"] = None if stats is None or stats.n == 0 else stats.fail_rate()
    features["hist_progress"] = None if stats is None else stats.progress()
    features["hist_n"] = None if stats is None or stats.n == 0 else float(stats.n)
    assert_feature_keys(features.keys())
    return features


def vectorize(features: Mapping[str, Any], names: Sequence[str]) -> list[float] | None:
    values: list[float] = []
    for name in names:
        value = features.get(name)
        if value is None:
            return None
        values.append(float(value))
    return values


def visibility_report(families: Sequence[str], envs: Sequence[str]) -> dict[str, Any]:
    model_names = model_feature_names(families, envs)
    overlap = sorted(set(model_names) & FORBIDDEN_FEATURE_KEYS)
    if overlap:
        raise LeakageError(f"model features collide with forbidden keys: {overlap}")
    return {
        "model_features": list(model_names),
        "historical_features_next_round_only": list(HISTORICAL_FEATURES),
        "forbidden_current_rollout": sorted(FORBIDDEN_FEATURE_KEYS),
        "public_candidate_keys": sorted(PUBLIC_CANDIDATE_KEYS),
        "state_features_removed_in_m_no_state": ["n_committed", *[f"covered__{family}" for family in families]],
        "rule": "trajectory length, success, failure, and tool errors update the next round only",
    }
