"""Features that exist before an assimilation action. Probe labels are not features."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from expolicy.paper_a.constants import (
    CONTEXT_FEATURES,
    EXPERIENCE_FEATURES,
    FORBIDDEN_FEATURE_KEYS,
    FORBIDDEN_FEATURE_SPLITS,
    STATE_FEATURES,
)
from expolicy.r9_lab.state.persistent_state import PersistentState, token_overlap


class LeakageError(RuntimeError):
    pass


_NUMBER = re.compile(r"\d+")
_SPACE = re.compile(r"\s+")


def template_key(item: Mapping[str, Any]) -> str:
    if item.get("source_group"):
        return str(item["source_group"])
    if item.get("template_id"):
        return f"{item.get('env')}:tmpl:{item['template_id']}"
    text = _SPACE.sub(" ", _NUMBER.sub("#", str(item.get("instruction") or "").lower())).strip()
    return f"{item.get('env')}:{text}"


def assert_feature_keys(keys: Sequence[str]) -> None:
    banned = [key for key in keys if key in FORBIDDEN_FEATURE_KEYS]
    if banned:
        raise LeakageError(f"post-action or probe fields used as features: {banned}")


def keys_for_variant(variant: str) -> tuple[str, ...]:
    if variant == "A1":
        return CONTEXT_FEATURES + EXPERIENCE_FEATURES
    if variant == "A2":
        return CONTEXT_FEATURES
    if variant in {"M", "A3", "A4"}:
        return CONTEXT_FEATURES + EXPERIENCE_FEATURES + STATE_FEATURES
    raise KeyError(variant)


def _rate(flag: bool, rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row.get(key)) is flag) / len(rows)


def predecision_features(
    state: PersistentState,
    experiences: Sequence[Mapping[str, Any]],
    *,
    env: str,
    capability: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Build the allowlisted feature vector. Current probe scores are ignored if present."""
    rows = [dict(exp) for exp in experiences]
    for row in rows:
        assert_feature_keys([key for key in row.keys() if key in FORBIDDEN_FEATURE_KEYS and key in {
            "shadow_after",
            "utility",
            "U",
            "delta_shadow",
            "delta_retention",
            "y_after",
            "future_success",
        }])
    n = len(rows) or 1
    successes = [bool(row.get("success")) for row in rows]
    templates = [template_key(row) for row in rows]
    unique = len(set(templates))
    memory_texts = [rec.instruction for rec in state.episodic]
    memory_texts.extend(f"{entry.trigger} {entry.principle}" for entry in state.semantic)
    similarities: list[float] = []
    conflicts = 0
    compared = 0
    for row in rows:
        instruction = str(row.get("instruction") or "")
        if not memory_texts:
            similarities.append(0.0)
            continue
        scored = [(token_overlap(instruction, text), text) for text in memory_texts]
        best, _ = max(scored, key=lambda item: item[0])
        similarities.append(best)
        compared += 1
        if best >= 0.5:
            # A close stored trajectory with the opposite outcome is a conflict, not a match.
            nearest = max(state.episodic, key=lambda rec: token_overlap(instruction, rec.instruction), default=None)
            if nearest is not None and bool(nearest.success) != bool(row.get("success")) and token_overlap(
                instruction, nearest.instruction
            ) >= 0.5:
                conflicts += 1
    _check_capability(capability)
    cap_shadow = None if not capability else capability.get("shadow")
    cap_retention = None if not capability else capability.get("retention")
    features = {
        "env_db": 1.0 if env == "db" else 0.0,
        "env_os": 1.0 if env == "os" else 0.0,
        "env_kg": 1.0 if env == "kg" else 0.0,
        "mean_steps": float(sum(float(row.get("steps") or 0.0) for row in rows) / n),
        "batch_success_rate": float(sum(successes) / n) if rows else 0.0,
        "error_rate": float(1.0 - (sum(successes) / n)) if rows else 0.0,
        "parser_invalid_rate": (
            sum(1 for row in rows if row.get("parser_valid") is False) / n if rows else 0.0
        ),
        "infra_fail_rate": _rate(True, rows, "infra_failure") if rows else 0.0,
        "repetition_rate": float(1.0 - unique / n) if rows else 0.0,
        "memory_similarity": float(sum(similarities) / len(similarities)) if similarities else 0.0,
        "conflict_rate": float(conflicts / compared) if compared else 0.0,
        "memory_size": float(len(state.episodic) + len(state.semantic)),
        "n_episodic": float(len(state.episodic)),
        "n_semantic": float(len(state.semantic)),
        "skill_size": float(len(state.skill_markdown or "")),
        "capability_shadow_hist": None if cap_shadow is None else float(cap_shadow),
        "capability_retention_hist": None if cap_retention is None else float(cap_retention),
    }
    assert_feature_keys(features.keys())
    return features


def _check_capability(capability: Mapping[str, Any] | None) -> None:
    if not capability:
        return
    source = str(capability.get("source") or "")
    if source in FORBIDDEN_FEATURE_SPLITS or "shadow_reward" in source or "future" in source:
        raise LeakageError(f"capability summary cannot come from a probe or future split: {source}")
    allowed = {"calibration", "assim_validation", "historical_state", "train_fold", "foundation"}
    if source not in allowed:
        raise LeakageError(f"unknown capability source: {source}")


def vectorize(features: Mapping[str, Any], names: Sequence[str]) -> list[float] | None:
    values: list[float] = []
    for name in names:
        value = features.get(name)
        if value is None:
            return None
        values.append(float(value))
    return values


def apply_simple_rule(features: Mapping[str, float], rules: Sequence[Mapping[str, Any]]) -> str:
    """Frozen rule list from the config. This is a baseline, not a fitted model."""
    default = "DISCARD"
    for rule in rules:
        if "default" in rule and "when" not in rule:
            default = str(rule["default"])
            continue
        when = dict(rule.get("when") or {})
        if _when_matches(features, when):
            return str(rule["action"])
    return default


def _when_matches(features: Mapping[str, float], when: Mapping[str, Any]) -> bool:
    for key, expected in when.items():
        if key.endswith("_lt"):
            if not (float(features[key[: -len("_lt")]]) < float(expected)):
                return False
        elif key.endswith("_lte"):
            if not (float(features[key[: -len("_lte")]]) <= float(expected)):
                return False
        elif key.endswith("_gte"):
            if not (float(features[key[: -len("_gte")]]) >= float(expected)):
                return False
        elif key.endswith("_gt"):
            if not (float(features[key[: -len("_gt")]]) > float(expected)):
                return False
        else:
            raise KeyError(key)
    return True
