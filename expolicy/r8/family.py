"""Map ALFWorld gamefile paths onto the six frozen R8 families."""

from __future__ import annotations

from expolicy.r8.constants import FAMILIES


def family_from_gamefile(path: str) -> str | None:
    text = path.replace("\\", "/")
    for fam in FAMILIES:
        if f"/{fam}" in f"/{text}" or text.startswith(fam):
            return fam
    return None
