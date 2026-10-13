"""Task families from the public question text. Gold programs are not labels."""

from __future__ import annotations

import re

COMPARE = re.compile(
    r"\b(more than|less than|fewer|most|least|compare|larger|smaller|older|younger|highest|lowest)\b",
    re.I,
)
FILTER = re.compile(r"\b(which|whose|where|that)\b", re.I)
ATTRIBUTE = re.compile(r"^(what|who|when)\b", re.I)


def kg_family(question: str) -> str:
    text = " ".join(str(question or "").split())
    if COMPARE.search(text):
        return "compare"
    if FILTER.search(text):
        return "filter"
    if ATTRIBUTE.search(text):
        return "attribute"
    return "relation"


def family_rank(families: list[str], seed: int) -> dict[str, int]:
    """Frozen tie order. Alphabetical order is not the tie break."""
    import random

    ordered = list(dict.fromkeys(families))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return {name: index for index, name in enumerate(ordered)}
