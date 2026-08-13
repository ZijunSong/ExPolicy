"""On-disk corpus helpers."""

from __future__ import annotations

from pathlib import Path

from expolicy.environments.search.world import World, build_world


def load_or_build(path: Path, seed: int = 13) -> World:
    if path.exists():
        return World.load(path)
    world = build_world(seed)
    world.save(path)
    return world
