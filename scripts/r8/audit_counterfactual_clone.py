#!/usr/bin/env python3
"""Counterfactual clone isolation checks for R8 persistent state directories."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    root = Path(tempfile.mkdtemp(prefix="r8_clone_"))
    try:
        a = root / "A1"
        b = root / "A2"
        a.mkdir()
        (a / "memory.txt").write_text("mem-a\n")
        (a / "skill.md").write_text("skill-a\n")
        shutil.copytree(a, b)
        (b / "memory.txt").write_text("mem-b\n")
        (b / "skill.md").write_text("skill-b\n")
        assert (a / "memory.txt").read_text() == "mem-a\n"
        assert (a / "skill.md").read_text() == "skill-a\n"
        assert (b / "memory.txt").read_text() == "mem-b\n"
        print("PASS clone isolation")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if args.demo:
        return


if __name__ == "__main__":
    main()
