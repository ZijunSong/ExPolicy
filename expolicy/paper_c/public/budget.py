"""Episode budget. A reservation is spent even when the episode fails or rolls back."""

from __future__ import annotations

from typing import Any


class BudgetExhausted(RuntimeError):
    pass


class EpisodeBudget:
    def __init__(self, cap: int, *, stage: str):
        self.cap = int(cap)
        self.stage = stage
        self.used = 0
        self.events: list[dict[str, Any]] = []

    def reserve(self, n: int = 1, *, kind: str, detail: str = "") -> None:
        count = int(n)
        if count < 1:
            raise ValueError("reservation must be positive")
        if self.used + count > self.cap:
            self.events.append(
                {"kind": "refused", "requested": count, "used": self.used, "cap": self.cap, "detail": detail}
            )
            raise BudgetExhausted(f"{self.stage} cap {self.cap} would be exceeded by {count} {kind}")
        self.used += count
        self.events.append({"kind": kind, "reserved": count, "used": self.used, "detail": detail})

    @property
    def remaining(self) -> int:
        return self.cap - self.used
