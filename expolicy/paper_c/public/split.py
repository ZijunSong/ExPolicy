"""Role assignment for the KG pilot. Future-test stays isolated."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from expolicy.paper_c.public.families import family_rank, kg_family
from expolicy.r9_lab.io_util import sha256_json


def load_bucket(split_path: Path, env: str, bucket: str) -> list[dict[str, Any]]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    rows = payload["envs"][env][bucket]
    return [dict(row) for row in rows]


def assign_c0_screen(split_path: Path, *, env: str = "kg", seed: int = 20261013) -> dict[str, Any]:
    """Carve acquisition and screen probes from the historical acquisition pool.

    Previously scored shadow and retention probes stay development-only.
    future_test is listed and then left unused.
    """
    pool = load_bucket(split_path, env, "acquisition_pool")
    for row in pool:
        row["family"] = kg_family(str(row.get("instruction") or ""))
        row["role_source"] = "acquisition_pool"
    ordered = list(pool)
    acquisition = ordered[:48]
    shadow = ordered[48:96]
    retention = ordered[96:120]
    unused = ordered[120:]
    families = sorted({str(row["family"]) for row in ordered})
    sequences = [acquisition[index * 16 : (index + 1) * 16] for index in range(3)]
    return {
        "env": env,
        "seed": seed,
        "family_rank": family_rank(families, seed),
        "families": families,
        "sequences": sequences,
        "shadow_screen": shadow,
        "retention_screen": retention,
        "unused_acquisition_pool": unused,
        "development_only": ["shadow_reward_probe", "retention_probe", "calibration", "model_screen", "assim_validation"],
        "isolated": ["future_test"],
        "reduced": len(shadow) < 48 or len(retention) < 24 or len(sequences) < 3,
        "content_hash": sha256_json(
            {
                "acquisition": [row["task_id"] for row in acquisition],
                "shadow": [row["task_id"] for row in shadow],
                "retention": [row["task_id"] for row in retention],
            }
        ),
    }


def render_split_audit(assignment: dict[str, Any]) -> str:
    lines = [
        "# C 划分审计",
        "",
        f"环境：`{assignment['env']}`。future_test 保持隔离，历史 probe 只作为开发数据。",
        "",
        f"任务族：{', '.join(assignment['families'])}",
        f"平局顺序：{assignment['family_rank']}",
        f"采集序列：{len(assignment['sequences'])} × {len(assignment['sequences'][0]) if assignment['sequences'] else 0}",
        f"screen shadow：{len(assignment['shadow_screen'])}",
        f"screen retention：{len(assignment['retention_screen'])}",
        f"划分哈希：`{assignment['content_hash']}`",
        "",
        "这批 acquisition_pool 任务来自既有 R9 划分。重新执行可以得到新轨迹，但不能把本 screen 写成独立确认集。",
        "",
    ]
    return "\n".join(lines)
