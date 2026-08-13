"""Optional adapter that loads an on-disk Helion corpus (same as SearchEnvironment)."""

from pathlib import Path

from expolicy.environments.search.environment import SearchEnvironment
from expolicy.environments.search.world import World


def load_local_search(world_path: Path) -> SearchEnvironment:
    return SearchEnvironment(World.load(world_path))
