#!/usr/bin/env python3
"""Audit LAB split semantics and write frozen split.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expolicy.r9_lab.audit.leakage import assert_model_screen_isolated
from expolicy.r9_lab.benchmark.lab_data import ensure_lab_assets
from expolicy.r9_lab.benchmark.split import (
    assert_production_split,
    build_and_write,
    official_data_available,
)
from expolicy.r9_lab.constants import BUCKETS, ENVS, OUTPUT_ROOT, SPLIT_SEED
from expolicy.r9_lab.resume import run_cell_once


def write_split_audit(path: Path, payload: dict) -> None:
    lines = [
        "# R9 split audit",
        "",
        f"seed: {payload.get('seed')}",
        f"synthetic: {payload.get('synthetic')}",
        f"sha256: {payload.get('sha256')}",
        "",
    ]
    for env in ENVS:
        lines.append(f"## {env}")
        for b in BUCKETS:
            n = len(payload["envs"][env].get(b) or [])
            lines.append(f"- {b}: {n}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = p.parse_args(argv)
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)

    def _job() -> dict:
        allow = bool(args.synthetic or args.allow_synthetic)
        if not args.synthetic:
            missing = {e: ok for e, ok in official_data_available().items() if not ok}
            if missing and not allow:
                try:
                    ensure_lab_assets(timeout=45.0)
                except Exception as exc:  # noqa: BLE001
                    raise FileNotFoundError(
                        f"Official LAB data missing: {missing}; materialize failed: {exc}"
                    ) from exc
                missing = {e: ok for e, ok in official_data_available().items() if not ok}
                if missing:
                    raise FileNotFoundError(f"Official LAB data missing after materialize: {missing}")
        use_syn = bool(args.synthetic or (not all(official_data_available().values()) and allow))
        payload, digest = build_and_write(
            out_path=out / "split.json",
            seed=args.seed,
            synthetic=use_syn,
        )
        assert_production_split(payload, allow_synthetic=allow)
        assert_model_screen_isolated(payload)
        write_split_audit(out / "audit" / "SPLIT_AUDIT.md", payload)
        (out / "audit").mkdir(parents=True, exist_ok=True)
        (out / "split.json.sha256").write_text(digest + "\n", encoding="utf-8")
        return {"sha256": digest, "synthetic": payload["synthetic"]}

    print(json.dumps(run_cell_once(out / "audit_split", _job, resume=args.resume), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
