#!/usr/bin/env python3
"""Materialize official LifelongAgentBench entry_dict.json from HuggingFace.

Standalone CLI; also imported by 00_preflight / 01_audit_split.
Does not invent task gold. Does not patch third_party source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expolicy.r9_lab.benchmark.lab_data import (
    LabDataError,
    ensure_lab_assets,
    generate_chat_history_items,
    materialize_all,
)
from expolicy.r9_lab.constants import LAB_ROOT, OUTPUT_ROOT


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=LAB_ROOT / ".hf_cache" / "LifelongAgentBench")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--chat-history-only", action="store_true")
    p.add_argument("--no-chat-history", action="store_true")
    p.add_argument("--timeout", type=float, default=45.0)
    args = p.parse_args(argv)
    try:
        if args.chat_history_only:
            report = {"chat_history": generate_chat_history_items()}
        elif args.no_chat_history:
            report = materialize_all(
                cache_dir=args.cache_dir, skip_download=args.skip_download, timeout=args.timeout
            )
        else:
            report = ensure_lab_assets(
                cache_dir=args.cache_dir, skip_download=args.skip_download, timeout=args.timeout
            )
    except LabDataError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    audit = OUTPUT_ROOT / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "LAB_DATA_MATERIALIZE.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, default=str))
    errors = (report.get("data") or report).get("errors") if isinstance(report, dict) else None
    if isinstance(report, dict) and report.get("data", {}).get("errors"):
        return 1
    if isinstance(errors, dict) and errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
