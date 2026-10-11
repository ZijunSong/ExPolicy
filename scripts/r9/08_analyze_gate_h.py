#!/usr/bin/env python3
"""Compute Gate H (H1–H6). Writes R9_GATE_H.json and R9_ASSIM_MATRIX.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from expolicy.r9_lab.constants import OUTPUT_ROOT
from expolicy.r9_lab.evaluation.gate_h import analyze_gate_h
from expolicy.r9_lab.evaluation.h4_features import analyze_h4
from expolicy.r9_lab.io_util import atomic_write_json, read_json
from expolicy.r9_lab.resume import run_cell_once


MATRIX_FIELDS = [
    "batch_id",
    "env",
    "difficulty",
    "skill_count",
    "base_success_rate",
    "batch_success_rate",
    "failure_type_stats",
    "U_discard",
    "U_episodic",
    "U_semantic",
    "U_skillopt",
    "best_action",
    "second_best_action",
    "best_margin",
]


def collect_rows(matrix_root: Path) -> list[dict]:
    rows = []
    for batch_dir in sorted(matrix_root.glob("*")):
        if not batch_dir.is_dir() or batch_dir.name.startswith("_"):
            continue
        u = {}
        meta = {}
        for action_dir in batch_dir.iterdir():
            if not action_dir.is_dir() or action_dir.name.startswith("_"):
                continue
            done = action_dir / "DONE.json"
            if not done.exists():
                continue
            payload = read_json(done)
            body = payload.get("payload") or payload
            meta = body
            u[str(body.get("action") or action_dir.name)] = float(util_of(body))
        if len(u) < 4:
            continue
        ranked = sorted(u.items(), key=lambda kv: kv[1], reverse=True)
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else ("", best[1])
        rows.append(
            {
                "batch_id": batch_dir.name,
                "env": str(meta.get("env") or batch_dir.name.split("-")[0]),
                "difficulty": meta.get("difficulty") or 0,
                "skill_count": meta.get("skill_count") or 0,
                "base_success_rate": meta.get("base_success_rate") or 0,
                "batch_success_rate": meta.get("batch_success_rate") or 0,
                "mean_steps": meta.get("mean_steps") or 0,
                "invalid_action_rate": meta.get("invalid_action_rate") or 0,
                "tokens": meta.get("tokens") or 0,
                "memory_size": meta.get("memory_size") or 0,
                "skill_size": meta.get("skill_size") or 0,
                "shadow_n": meta.get("shadow_n") or 0,
                "failure_type_stats": json.dumps(meta.get("failure_type_stats") or {}),
                "U_discard": u.get("DISCARD", 0.0),
                "U_episodic": u.get("EPISODIC_REPLAY", 0.0),
                "U_semantic": u.get("SEMANTIC_MEMORY", 0.0),
                "U_skillopt": u.get("SKILLOPT_EDIT", 0.0),
                "best_action": best[0],
                "second_best_action": second[0],
                "best_margin": float(best[1] - second[1]),
            }
        )
    return rows


def util_of(body: dict) -> float:
    util = body.get("utility") or {}
    return float(util.get("U") or body.get("U") or 0.0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = p.parse_args(argv)
    out = args.output_root

    def _job() -> dict:
        rows = collect_rows(out / "matrix")
        csv_path = out / "R9_ASSIM_MATRIX.csv"
        csv_rows = [{k: r.get(k, "") for k in MATRIX_FIELDS} for r in rows]
        if csv_rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=MATRIX_FIELDS)
                w.writeheader()
                w.writerows(csv_rows)
        feat = analyze_h4(rows) if rows else {}
        report = analyze_gate_h(rows, feature_report=feat, leakage_ok=True)
        atomic_write_json(out / "R9_GATE_H.json", report)
        return report

    print(json.dumps(run_cell_once(out / "gate_h", _job, resume=args.resume), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
