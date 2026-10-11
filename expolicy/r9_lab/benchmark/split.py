"""Frozen R9 split construction for LifelongAgentBench.

If official entry_dict.json files exist, load them. Production default: fail if
missing. `--synthetic` is for unit tests only and writes a flagged split that
production scripts refuse unless EXPLICITLY requested.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from expolicy.r9_lab.constants import (
    BUCKET_MIN,
    BUCKET_TARGETS,
    BUCKETS,
    ENVS,
    FORMAL_BUCKETS,
    LAB_DATA_REL,
    LAB_ROOT,
    OUTPUT_ROOT,
    SPLIT_SEED,
)
from expolicy.r9_lab.io_util import atomic_write_json, sha256_file, sha256_json, write_json


@dataclass
class TaskRecord:
    task_id: str
    env: str
    sample_index: str
    instruction: str
    skill_list: list[str] = field(default_factory=list)
    difficulty: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def official_data_path(env: str) -> Path:
    return LAB_ROOT / LAB_DATA_REL[env]


def official_data_available() -> dict[str, bool]:
    return {env: official_data_path(env).is_file() for env in ENVS}


def _instruction_of(item: Mapping[str, Any]) -> str:
    for key in ("instruction", "question"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _skills_of(item: Mapping[str, Any]) -> list[str]:
    raw = item.get("skill_list") or item.get("skills") or []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def _difficulty_of(item: Mapping[str, Any], skills: Sequence[str]) -> int:
    for key in ("difficulty", "difficulty_level"):
        if key in item and item[key] is not None:
            try:
                return int(item[key])
            except (TypeError, ValueError):
                pass
    return len(skills)


def load_official_entries(env: str) -> list[TaskRecord]:
    path = official_data_path(env)
    if not path.is_file():
        raise FileNotFoundError(
            f"Official LAB data missing for env={env}: {path}. "
            "Production default is to fail. Use synthetic mode only for unit tests."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"entry_dict.json for {env} is not a dict of sample_index -> item")
    records: list[TaskRecord] = []
    for key, item in raw.items():
        if not isinstance(item, dict):
            continue
        instruction = _instruction_of(item)
        skills = _skills_of(item)
        records.append(
            TaskRecord(
                task_id=f"{env}:{key}",
                env=env,
                sample_index=str(key),
                instruction=instruction,
                skill_list=skills,
                difficulty=_difficulty_of(item, skills),
            )
        )
    records.sort(key=lambda r: (r.env, r.sample_index))
    return records


def build_synthetic_catalog(*, n_per_env: int = 600) -> dict[str, list[TaskRecord]]:
    """Unit-test catalog. Clearly distinct instructions; never used as official LAB data."""
    catalog: dict[str, list[TaskRecord]] = {}
    for env in ENVS:
        rows: list[TaskRecord] = []
        for i in range(n_per_env):
            skills = [f"{env}_skill_{i % 8}"]
            difficulty = i % 5
            rows.append(
                TaskRecord(
                    task_id=f"{env}:syn{i}",
                    env=env,
                    sample_index=str(i),
                    instruction=(
                        f"[SYNTHETIC {env}] instruction index={i} "
                        f"skill={skills[0]} difficulty={difficulty}"
                    ),
                    skill_list=skills,
                    difficulty=difficulty,
                )
            )
        catalog[env] = rows
    return catalog


def _stratum_key(record: TaskRecord) -> tuple[str, int]:
    skill_sig = "|".join(sorted(record.skill_list)) if record.skill_list else "none"
    return skill_sig, int(record.difficulty)


def _scaled_sizes(n_available: int) -> dict[str, int]:
    """Allocate bucket sizes. model_screen is reserved first and isolated.

    If the catalog is smaller than the target sum, scale formal buckets but keep
    future_test >= 60 and shadow_reward_probe >= 40 when possible.
    """
    screen_n = min(BUCKET_TARGETS["model_screen"], n_available)
    remaining = n_available - screen_n
    formal_targets = {k: v for k, v in BUCKET_TARGETS.items() if k != "model_screen"}
    target_sum = sum(formal_targets.values())
    sizes: dict[str, int] = {"model_screen": screen_n}
    if remaining <= 0:
        for k in formal_targets:
            sizes[k] = 0
        return sizes
    if remaining >= target_sum:
        sizes.update(formal_targets)
        extra = remaining - target_sum
        sizes["acquisition_pool"] += extra
        return sizes
    # Scale, then enforce mins.
    raw = {k: max(0, int(round(v * remaining / target_sum))) for k, v in formal_targets.items()}
    # Fix rounding so sum == remaining.
    drift = remaining - sum(raw.values())
    raw["acquisition_pool"] = max(0, raw["acquisition_pool"] + drift)
    for k, mn in BUCKET_MIN.items():
        if raw.get(k, 0) < mn and remaining >= mn:
            deficit = mn - raw[k]
            donor = "acquisition_pool" if raw.get("acquisition_pool", 0) >= deficit else "calibration"
            take = min(deficit, raw.get(donor, 0))
            raw[donor] = raw.get(donor, 0) - take
            raw[k] += take
    sizes.update(raw)
    return sizes


def _stratified_draw(
    records: Sequence[TaskRecord],
    rng: random.Random,
) -> list[TaskRecord]:
    groups: dict[tuple[str, int], list[TaskRecord]] = defaultdict(list)
    for rec in records:
        groups[_stratum_key(rec)].append(rec)
    keys = sorted(groups.keys())
    for key in keys:
        rng.shuffle(groups[key])
    ordered: list[TaskRecord] = []
    while any(groups[k] for k in keys):
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop())
    return ordered


def construct_env_split(
    records: Sequence[TaskRecord],
    *,
    seed: int = SPLIT_SEED,
) -> dict[str, list[TaskRecord]]:
    rng = random.Random(seed)
    unique: list[TaskRecord] = []
    seen_id: set[str] = set()
    seen_instr: set[str] = set()
    for rec in records:
        instr = rec.instruction.strip()
        if rec.task_id in seen_id or instr in seen_instr:
            continue
        seen_id.add(rec.task_id)
        seen_instr.add(instr)
        unique.append(rec)
    ordered = _stratified_draw(unique, rng)
    sizes = _scaled_sizes(len(ordered))
    out: dict[str, list[TaskRecord]] = {b: [] for b in BUCKETS}
    cursor = 0
    for bucket in BUCKETS:
        n = sizes.get(bucket, 0)
        out[bucket] = ordered[cursor : cursor + n]
        cursor += n
    return out


def _assert_disjoint(split: Mapping[str, Mapping[str, Sequence[TaskRecord]]]) -> None:
    # Within-env bucket disjointness (task_id and exact instruction).
    for env, buckets in split.items():
        ids: dict[str, set[str]] = {}
        qs: dict[str, set[str]] = {}
        for name, rows in buckets.items():
            ids[name] = {r.task_id for r in rows}
            qs[name] = {r.instruction.strip() for r in rows}
        names = list(ids)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if ids[a] & ids[b]:
                    raise ValueError(f"{env} task_id overlap {a} vs {b}: {ids[a] & ids[b]}")
                if qs[a] & qs[b]:
                    raise ValueError(f"{env} instruction overlap {a} vs {b}")
    # model_screen of every env vs formal buckets of every env.
    screen_ids: set[str] = set()
    screen_qs: set[str] = set()
    formal_ids: set[str] = set()
    formal_qs: set[str] = set()
    for env, buckets in split.items():
        for rec in buckets.get("model_screen") or []:
            screen_ids.add(rec.task_id)
            screen_qs.add(rec.instruction.strip())
        for name in FORMAL_BUCKETS:
            for rec in buckets.get(name) or []:
                formal_ids.add(rec.task_id)
                formal_qs.add(rec.instruction.strip())
    if screen_ids & formal_ids:
        raise ValueError(f"model_screen vs formal task_id overlap: {screen_ids & formal_ids}")
    if screen_qs & formal_qs:
        raise ValueError("model_screen vs formal exact-instruction overlap")


def construct_split(
    catalog: Mapping[str, Sequence[TaskRecord]],
    *,
    seed: int = SPLIT_SEED,
    synthetic: bool = False,
) -> dict[str, Any]:
    per_env: dict[str, dict[str, list[TaskRecord]]] = {}
    for env in ENVS:
        per_env[env] = construct_env_split(catalog[env], seed=seed + ENVS.index(env))
    _assert_disjoint(per_env)
    payload: dict[str, Any] = {
        "seed": seed,
        "synthetic": bool(synthetic),
        "synthetic_flag": "SYNTHETIC_SPLIT_FOR_UNIT_TESTS_ONLY" if synthetic else None,
        "envs": {
            env: {bucket: [r.to_dict() for r in rows] for bucket, rows in buckets.items()}
            for env, buckets in per_env.items()
        },
        "notes": {
            "model_screen_isolated": True,
            "official_sample_order": "LAB sample_order=default is sequential Task.get_sample_index_list(); this split is R9's disjoint experiment split, not a claim that LAB has an ML train/valid/test split.",
        },
    }
    payload["sha256"] = sha256_json({k: v for k, v in payload.items() if k != "sha256"})
    return payload


def write_split(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(digest + "\n", encoding="utf-8")
    return digest


def load_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def records_in_bucket(payload: Mapping[str, Any], env: str, bucket: str) -> list[TaskRecord]:
    rows = payload["envs"][env][bucket]
    return [TaskRecord(**{k: r[k] for k in TaskRecord.__dataclass_fields__}) for r in rows]


def all_bucket_ids(payload: Mapping[str, Any], bucket: str) -> set[str]:
    ids: set[str] = set()
    for env in payload["envs"]:
        for row in payload["envs"][env].get(bucket) or []:
            ids.add(row["task_id"])
    return ids


def assert_production_split(payload: Mapping[str, Any], *, allow_synthetic: bool) -> None:
    if payload.get("synthetic") and not allow_synthetic:
        raise RuntimeError(
            "Refusing synthetic split.json in production. "
            "Pass --allow-synthetic only for unit tests / dry-run."
        )


def build_and_write(
    *,
    out_path: Path,
    seed: int = SPLIT_SEED,
    synthetic: bool = False,
    n_per_env: int = 600,
) -> tuple[dict[str, Any], str]:
    if synthetic:
        catalog = build_synthetic_catalog(n_per_env=n_per_env)
    else:
        missing = {env: str(official_data_path(env)) for env, ok in official_data_available().items() if not ok}
        if missing:
            raise FileNotFoundError(
                "Official LAB entry_dict.json missing; production default is fail. "
                f"missing={missing}. Use --synthetic only for unit tests."
            )
        catalog = {env: load_official_entries(env) for env in ENVS}
    payload = construct_split(catalog, seed=seed, synthetic=synthetic)
    digest = write_split(out_path, payload)
    payload["file_sha256"] = digest
    return payload, digest


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build frozen R9 LAB split")
    p.add_argument("--synthetic", action="store_true", help="Unit-test synthetic split only")
    p.add_argument("--out", type=Path, default=OUTPUT_ROOT / "split.json")
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--n-per-env", type=int, default=600)
    args = p.parse_args(argv)
    payload, digest = build_and_write(
        out_path=args.out,
        seed=args.seed,
        synthetic=args.synthetic,
        n_per_env=args.n_per_env,
    )
    print(json.dumps({"path": str(args.out), "sha256": digest, "synthetic": payload["synthetic"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
