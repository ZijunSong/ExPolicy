"""Sequence-level comparisons. Expanded tasks are not independent samples."""

from __future__ import annotations

import math
from collections.abc import Sequence

from expolicy.paper_a.stats import interval_claim, normal_mean_ci


def paired_summary(diffs: Sequence[float], *, wide: float) -> dict[str, object]:
    values = [float(item) for item in diffs]
    if not values:
        return {"status": "missing", "n": 0, "diff": None, "ci": None, "population_noise_claim": False}
    claim = interval_claim(values)
    center, lo, hi = normal_mean_ci(values)
    width = hi - lo
    return {
        "status": "ok",
        "n": len(values),
        "diff": center,
        "ci": [lo, hi],
        "width": width,
        "wide": bool(claim["population_noise_claim"]) and width > float(wide),
        "zero_width": bool(claim["zero_width"]),
        "population_noise_claim": bool(claim["population_noise_claim"]) and width <= float(wide),
        "crosses_zero": lo < 0.0 < hi,
    }


def recovery_point(c_train: float, c_save: float) -> int | None:
    """Sequences required to recover offline training cost when deployment saves cost.

    ``c_save <= 0`` has no recovery point. The figure does not transfer to a new environment.
    """
    if float(c_save) <= 0.0:
        return None
    return int(math.ceil(float(c_train) / float(c_save)))
