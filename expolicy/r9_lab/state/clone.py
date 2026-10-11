"""Deep-copy persistent state to a new directory. Clone A must not affect clone B."""

from __future__ import annotations

import shutil
from pathlib import Path

from expolicy.r9_lab.state.persistent_state import PersistentState


def clone_state(src: PersistentState | Path | str, dest: Path | str) -> PersistentState:
    if isinstance(src, PersistentState):
        src.save()
        src_path = src.root_path()
    else:
        src_path = Path(src)
        if not (src_path / "STATE.json").exists():
            PersistentState.load(src_path).save()
    dest_path = Path(dest)
    if dest_path.exists():
        shutil.rmtree(dest_path)
    # Several workers may clone one S0 while a save is replacing *.tmp files.
    shutil.copytree(src_path, dest_path, ignore=lambda _dir, names: [name for name in names if name.endswith(".tmp")])
    cloned = PersistentState.load(dest_path)
    cloned.root = str(dest_path)
    cloned.save()
    return cloned
