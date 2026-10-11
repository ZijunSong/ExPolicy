#!/usr/bin/env python3
"""Enumerate playable ALFWorld games and freeze the R8 split (fail-closed).

Usage:
  python scripts/r8/build_alfworld_splits.py --seed 20260821 \\
      --out outputs/r8_backend_validation/split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from expolicy.r8.constants import (
    ALFWORLD_DATA,
    BUCKETS,
    EASIER_FAMILIES,
    FAMILIES,
    PER_FAMILY_QUOTA,
    RETENTION_TOTAL,
    SPLIT_SEED,
)

SKIP_PREFIXES = ("pick_and_place_with_movable_recep",)


def _family_of(task_dir: str) -> str | None:
    for prefix in SKIP_PREFIXES:
        if task_dir.startswith(prefix):
            return None
    for fam in FAMILIES:
        if task_dir.startswith(fam):
            return fam
    return None


def _is_playable(trial_dir: Path) -> bool:
    game = trial_dir / "game.tw-pddl"
    traj = trial_dir / "traj_data.json"
    if not game.is_file() or not traj.is_file():
        return False
    try:
        raw = game.read_text(encoding="utf-8")[:400]
    except OSError:
        return False
    if '"solvable": false' in raw.lower() or '"solvable":false' in raw.lower():
        return False
    return True


def _load_meta(trial_dir: Path, family: str) -> dict:
    traj_path = trial_dir / "traj_data.json"
    task_type = family
    task_id = trial_dir.name
    instructions: list[str] = []
    try:
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        task_type = str(traj.get("task_type") or family)
        task_id = str(traj.get("task_id") or trial_dir.name)
        anns = ((traj.get("turk_annotations") or {}).get("anns") or [])
        for ann in anns:
            desc = str(ann.get("task_desc") or "").strip()
            if desc:
                instructions.append(desc)
    except Exception:
        pass
    rel = trial_dir.relative_to(ALFWORLD_DATA).as_posix()
    gamefile_rel = f"{rel}/game.tw-pddl"
    gamefile_abs = str(trial_dir / "game.tw-pddl")
    return {
        "id": f"{family}/{task_id}",
        "trial_id": task_id,
        "task_type": task_type,
        "family": family,
        "gamefile": gamefile_rel,
        "gamefile_abs": gamefile_abs,
        "trial_dir": str(trial_dir),
        "instructions": instructions,
        "instruction_key": " || ".join(sorted({x.lower().strip() for x in instructions})),
    }


def enumerate_playable(split_name: str = "train") -> dict[str, list[dict]]:
    root = ALFWORLD_DATA / "json_2.1.1" / split_name
    if not root.is_dir():
        raise FileNotFoundError(f"ALFWorld split missing: {root}")
    by_fam: dict[str, list[dict]] = {fam: [] for fam in FAMILIES}
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        family = _family_of(task_dir.name)
        if family is None:
            continue
        for trial_dir in sorted(p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")):
            if not _is_playable(trial_dir):
                continue
            by_fam[family].append(_load_meta(trial_dir, family))
    return by_fam


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _symlink_trial(dst_root: Path, item: dict) -> None:
    src = Path(item["trial_dir"])
    family_dir = src.parent.name
    dest = dst_root / family_dir / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.symlink(src, dest)


def _skillopt_items(items: list[dict], prefix: str) -> list[dict]:
    out = []
    for i, item in enumerate(items):
        out.append(
            {
                "id": f"{prefix}:{i:04d}:{item['trial_id']}",
                "gamefile": item["gamefile"],
                "task_type": item["task_type"],
            }
        )
    return out


def _fail(msg: str) -> None:
    raise SystemExit(f"FAIL-CLOSED: {msg}")


def build_split(seed: int, out_dir: Path) -> dict:
    if not (ALFWORLD_DATA / "json_2.1.1" / "train").is_dir():
        _fail(f"ALFWORLD data missing at {ALFWORLD_DATA}")
    by_fam = enumerate_playable("train")
    rng = random.Random(seed)

    needed = sum(PER_FAMILY_QUOTA.values())
    for fam in FAMILIES:
        n = len(by_fam[fam])
        if n < needed:
            _fail(f"family {fam} has {n} playable train games; need {needed}")

    buckets: dict[str, list[dict]] = {name: [] for name in BUCKETS}
    leftovers: dict[str, list[dict]] = {}
    for fam in FAMILIES:
        items = list(by_fam[fam])
        items.sort(key=lambda x: x["id"])
        rng.shuffle(items)
        cursor = 0
        for bucket, k in PER_FAMILY_QUOTA.items():
            buckets[bucket].extend(items[cursor : cursor + k])
            cursor += k
        leftovers[fam] = items[cursor:]

    easier_pool: list[dict] = []
    for fam in EASIER_FAMILIES:
        easier_pool.extend(leftovers[fam])
    easier_pool.sort(key=lambda x: x["id"])
    rng.shuffle(easier_pool)
    if len(easier_pool) < RETENTION_TOTAL:
        _fail(f"easier leftover pool has {len(easier_pool)}; need retention {RETENTION_TOTAL}")
    buckets["retention_probe"] = easier_pool[:RETENTION_TOTAL]

    # Overlap checks.
    id_sets = {name: {x["id"] for x in rows} for name, rows in buckets.items()}
    path_sets = {name: {x["gamefile"] for x in rows} for name, rows in buckets.items()}
    inst_sets = {name: {x["instruction_key"] for x in rows if x["instruction_key"]} for name, rows in buckets.items()}
    names = list(buckets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if id_sets[a] & id_sets[b]:
                _fail(f"id overlap {a} ∩ {b} = {len(id_sets[a] & id_sets[b])}")
            if path_sets[a] & path_sets[b]:
                _fail(f"gamefile overlap {a} ∩ {b}")
            if inst_sets[a] & inst_sets[b]:
                _fail(f"instruction overlap {a} ∩ {b} = {len(inst_sets[a] & inst_sets[b])}")

    for fam in FAMILIES:
        for bucket, k in PER_FAMILY_QUOTA.items():
            got = sum(1 for x in buckets[bucket] if x["family"] == fam)
            if got != k:
                _fail(f"{bucket}/{fam} count {got} != {k}")
    if len(buckets["retention_probe"]) != RETENTION_TOTAL:
        _fail("retention_probe size mismatch")

    counts = {
        name: {
            "total": len(rows),
            "by_family": {fam: sum(1 for x in rows if x["family"] == fam) for fam in FAMILIES},
        }
        for name, rows in buckets.items()
    }
    inventory = {fam: len(by_fam[fam]) for fam in FAMILIES}

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "source_split": "json_2.1.1/train",
        "data_root": str(ALFWORLD_DATA),
        "families": list(FAMILIES),
        "quotas": dict(PER_FAMILY_QUOTA),
        "retention_total": RETENTION_TOTAL,
        "easier_families": list(EASIER_FAMILIES),
        "playable_train_inventory": inventory,
        "counts": counts,
        "notes": [
            "Official valid_seen / valid_unseen are NOT used as future_test: per-family playable counts fail the >=30 quota.",
            "future_test is carved from train playable games and must never enter MemRL Q update, SkillOpt validation, or controller fitting.",
            "SkillOpt official test split for backend training is shadow_reward_probe, not future_test.",
        ],
        "buckets": buckets,
    }

    if out_dir.exists():
        for child in ("views", "skillopt", "skillopt_smoke"):
            p = out_dir / child
            if p.is_dir():
                shutil.rmtree(p)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_json = out_dir / "split.json"
    _write_json(split_json, payload)
    digest = _sha256_file(split_json)
    (out_dir / "split.sha256").write_text(digest + "\n", encoding="utf-8")

    # MemRL directory views (symlinks to original trial dirs).
    for name, rows in buckets.items():
        view = out_dir / "views" / name
        view.mkdir(parents=True, exist_ok=True)
        for item in rows:
            _symlink_trial(view, item)

    # SkillOpt path manifests. Backend training uses acquisition/val/shadow.
    skillopt_root = out_dir / "skillopt"
    mapping = {
        "train": "acquisition_pool",
        "val": "assim_validation",
        "test": "shadow_reward_probe",
    }
    for split_name, bucket in mapping.items():
        items = _skillopt_items(buckets[bucket], split_name)
        _write_json(skillopt_root / split_name / "items.json", items)
    _write_json(
        skillopt_root / "split_manifest.json",
        {
            "benchmark": "ALFWorld",
            "manifest_type": "path_split",
            "source": "r8_frozen_split",
            "seed": seed,
            "split_sha256": digest,
            "counts": {k: len(buckets[v]) for k, v in mapping.items()},
            "path_root": str(ALFWORLD_DATA),
            "notes": ["test is shadow_reward_probe, not future_test"],
        },
    )

    # Tiny MemRL/SkillOpt smoke view: 2 games / family from calibration.
    smoke_view = out_dir / "views" / "smoke"
    smoke_view.mkdir(parents=True, exist_ok=True)
    smoke_items = []
    for fam in FAMILIES:
        smoke_items.extend([x for x in buckets["calibration"] if x["family"] == fam][:2])
    for item in smoke_items:
        _symlink_trial(smoke_view, item)
    smoke = out_dir / "skillopt_smoke"
    smoke_train, smoke_val, smoke_test = [], [], []
    for fam in FAMILIES:
        cal = [x for x in buckets["calibration"] if x["family"] == fam]
        av = [x for x in buckets["assim_validation"] if x["family"] == fam]
        sh = [x for x in buckets["shadow_reward_probe"] if x["family"] == fam]
        smoke_train.extend(cal[:2])
        smoke_val.extend(av[:1])
        smoke_test.extend(sh[:1])
    _write_json(smoke / "train" / "items.json", _skillopt_items(smoke_train, "smoke_train"))
    _write_json(smoke / "val" / "items.json", _skillopt_items(smoke_val, "smoke_val"))
    _write_json(smoke / "test" / "items.json", _skillopt_items(smoke_test, "smoke_test"))
    _write_json(
        smoke / "split_manifest.json",
        {
            "benchmark": "ALFWorld",
            "purpose": "official_smoke_only",
            "seed": seed,
            "split_sha256": digest,
            "leaks_future_test": False,
        },
    )

    # Future-test SkillOpt eval manifest (eval-only; never passed to train.py as val).
    fut = out_dir / "skillopt_future_eval"
    _write_json(fut / "test" / "items.json", _skillopt_items(buckets["future_test"], "future"))
    _write_json(fut / "train" / "items.json", [])
    _write_json(fut / "val" / "items.json", [])
    _write_json(
        fut / "split_manifest.json",
        {
            "benchmark": "ALFWorld",
            "purpose": "future_test_eval_only",
            "seed": seed,
            "split_sha256": digest,
        },
    )

    audit_md = out_dir.parent.parent / "r8_hetero" / "audit" / "SPLIT_AUDIT.md"
    # Also write beside the split and to the protocol path.
    audit_paths = [
        out_dir / "SPLIT_AUDIT.md",
        ROOT / "outputs" / "r8_hetero" / "audit" / "SPLIT_AUDIT.md",
    ]
    lines = [
        "# R8 ALFWorld Split Audit",
        "",
        f"- Created: {payload['created_at']}",
        f"- Seed: `{seed}`",
        f"- Source: `{ALFWORLD_DATA}/json_2.1.1/train` playable `game.tw-pddl` only",
        f"- split.json sha256: `{digest}`",
        "- Official valid_seen / valid_unseen **not** used for future_test (quota fail-closed).",
        "",
        "## Playable train inventory",
        "",
        "| family | n |",
        "|---|---:|",
    ]
    for fam, n in inventory.items():
        lines.append(f"| {fam} | {n} |")
    lines += [
        "",
        "## Frozen buckets",
        "",
        "| bucket | total | " + " | ".join(FAMILIES) + " |",
        "|---|---:|" + "|".join(["---:" ] * len(FAMILIES)) + "|",
    ]
    for name in BUCKETS:
        byf = counts[name]["by_family"]
        cells = " | ".join(str(byf[fam]) for fam in FAMILIES)
        lines.append(f"| {name} | {counts[name]['total']} | {cells} |")
    lines += [
        "",
        "## Isolation",
        "",
        "- gamefile / trial id overlap across buckets: **0**",
        "- turk instruction_key overlap across buckets: **0**",
        "- `future_test` is absent from SkillOpt train/val and from MemRL data_path views used for updates",
        "- SkillOpt backend `test` split = `shadow_reward_probe` (held-out gate / diagnostic), not future_test",
        "",
        "## FAIL-CLOSED result: PASS",
        "",
    ]
    text = "\n".join(lines)
    for path in audit_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if path.parent.name == "audit":
            shutil.copy2(split_json, path.parent / "split.json")
            (path.parent / "split.sha256").write_text(digest + "\n", encoding="utf-8")

    return {"sha256": digest, "counts": counts, "inventory": inventory, "out": str(out_dir)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "r8_backend_validation" / "split")
    args = p.parse_args()
    result = build_split(args.seed, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
