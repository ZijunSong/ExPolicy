#!/usr/bin/env python3
"""Materialize the Helion Archive corpus and print split statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expolicy.environments.search.world import build_world, world_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=Path("data/world/helion.json"))
    args = parser.parse_args()
    world = build_world(args.seed)
    world.save(args.out)
    print(json.dumps(world_stats(world), indent=2))


if __name__ == "__main__":
    main()
