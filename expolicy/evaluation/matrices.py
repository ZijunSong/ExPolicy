"""Experience-cluster × future-capability transfer matrix from realized utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from expolicy.core.experience import Experience, ExperienceUtility


def transfer_matrix(
    experiences: Iterable[Experience],
    utilities: Iterable[ExperienceUtility],
) -> dict[str, dict[str, float]]:
    util_by_id = {u.experience_id: u for u in utilities}
    sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for exp in experiences:
        util = util_by_id.get(exp.experience_id)
        if util is None:
            continue
        src = exp.capability or "unknown"
        # future gain attributed to the experience's own cluster; callers can pass tagged extras
        dst = str((util.after_metrics or {}).get("target_capability") or src)
        sums[src][dst].append(util.future_gain)
    return {
        src: {dst: sum(vals) / len(vals) for dst, vals in dests.items()}
        for src, dests in sums.items()
    }
