"""Learning and cost efficiency from realized metrics."""

from __future__ import annotations


def gain_per_unit(delta_capability: float, amount: float) -> float:
    if amount <= 0:
        return 0.0
    return delta_capability / amount


def efficiency_report(before: dict, after: dict, *, tokens: int, steps: int, gpu_hours: float) -> dict:
    delta = float(after.get("accuracy") or 0.0) - float(before.get("accuracy") or 0.0)
    return {
        "delta_accuracy": delta,
        "gain_per_1k_steps": gain_per_unit(delta, steps / 1000.0) if steps else 0.0,
        "gain_per_1m_tokens": gain_per_unit(delta, tokens / 1_000_000.0) if tokens else 0.0,
        "gain_per_gpu_hour": gain_per_unit(delta, gpu_hours) if gpu_hours else 0.0,
        "tokens": tokens,
        "steps": steps,
        "gpu_hours": gpu_hours,
    }
