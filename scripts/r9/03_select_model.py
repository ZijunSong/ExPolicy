#!/usr/bin/env python3
"""Apply pre-registered model selection rule. Freeze model_manifest.json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from expolicy.r9_lab.constants import MODEL_LADDER, OUTPUT_ROOT, model_by_id
from expolicy.r9_lab.io_util import sha256_file
from expolicy.r9_lab.manifest import freeze_manifest
from expolicy.r9_lab.resume import run_cell_once
from expolicy.r9_lab.selection import EnvScreenStats, evaluate_model, select_model


def load_screen_csv(path: Path) -> list:
    with path.open() as f:
        return list(csv.DictReader(f))


def results_from_csv(rows: list[dict]) -> list:
    by_model: dict[str, dict[str, EnvScreenStats]] = {}
    for row in rows:
        mid = row["model_id"]
        env = row["env"]
        by_model.setdefault(mid, {})
        by_model[mid][env] = EnvScreenStats(
            success=float(row["success"]),
            parser_valid_rate=float(row["parser_valid_rate"]),
            infra_failure_rate=float(row["infra_failure_rate"]),
            n=int(float(row.get("n") or 0)),
        )
    return [evaluate_model(mid, envs) for mid, envs in by_model.items()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args(argv)
    out = args.output_root
    csv_path = args.csv or (out / "R9_MODEL_SCREEN.csv")

    def _job() -> dict:
        rows = load_screen_csv(csv_path)
        outcome = select_model(results_from_csv(rows))
        md = out / "audit" / "MODEL_SELECTION.md"
        md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# R9 model selection",
            "",
            f"status: {outcome.status}",
            "rule: lowest Full-PASS Score; tie-break Q0>Q1>Q2>Q3; band 0.10-0.70 not post-hoc changed",
            "",
        ]
        for r in outcome.ranked:
            lines.append(f"- {r.model_id} full_pass={r.full_pass} score={r.score} reason={r.reason}")
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        split_path = out / "split.json"
        split_sha = sha256_file(split_path) if split_path.exists() else ""
        if outcome.selected is None:
            manifest = {
                "selected_model_id": None,
                "status": outcome.status,
                "hf_repo": None,
                "local_path": None,
                "revision_or_snapshot": None,
                "dtype": "bfloat16",
                "thinking_mode": None,
                "selection_score": None,
                "screen_split_sha256": split_sha,
            }
        else:
            spec = model_by_id(outcome.selected.model_id)
            manifest = {
                "selected_model_id": spec.model_id,
                "status": "SELECTED",
                "hf_repo": spec.hf_repo,
                "local_path": str(spec.local_path),
                "revision_or_snapshot": None,
                "dtype": "bfloat16",
                "thinking_mode": spec.thinking,
                "served_name": spec.served_name,
                "selection_score": outcome.selected.score,
                "screen_split_sha256": split_sha,
            }
        freeze_manifest(out / "model_manifest.json", manifest)
        return {"status": outcome.status, "manifest": str(out / "model_manifest.json")}

    print(json.dumps(run_cell_once(out / "select_model", _job, resume=args.resume), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
