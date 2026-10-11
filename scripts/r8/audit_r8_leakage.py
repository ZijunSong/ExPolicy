#!/usr/bin/env python3
"""Fail-closed leakage audit for the frozen R8 ALFWorld split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from expolicy.r8.constants import BUCKETS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=Path, default=ROOT / "outputs" / "r8_backend_validation" / "split" / "split.json")
    args = p.parse_args()
    payload = json.loads(args.split.read_text(encoding="utf-8"))
    sha_path = args.split.with_name("split.sha256")
    digest = hashlib.sha256(args.split.read_bytes()).hexdigest()
    if sha_path.exists() and sha_path.read_text().strip() != digest:
        raise SystemExit("split.json sha256 mismatch")

    buckets = payload["buckets"]
    ids = {name: {x["id"] for x in buckets[name]} for name in BUCKETS}
    paths = {name: {x["gamefile"] for x in buckets[name]} for name in BUCKETS}
    inst = {name: {x.get("instruction_key") for x in buckets[name] if x.get("instruction_key")} for name in BUCKETS}
    names = list(BUCKETS)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if ids[a] & ids[b]:
                raise SystemExit(f"id leak {a}∩{b}")
            if paths[a] & paths[b]:
                raise SystemExit(f"path leak {a}∩{b}")
            if inst[a] & inst[b]:
                raise SystemExit(f"instruction leak {a}∩{b}")

    future_paths = paths["future_test"]
    skillopt = args.split.parent / "skillopt"
    for split_name in ("train", "val", "test"):
        items = json.loads((skillopt / split_name / "items.json").read_text())
        leaked = {it["gamefile"] for it in items} & future_paths
        if leaked:
            raise SystemExit(f"future_test leaked into SkillOpt {split_name}: {len(leaked)}")
    print("PASS leakage audit")


if __name__ == "__main__":
    main()
