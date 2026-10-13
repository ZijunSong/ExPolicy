"""Two cost books. Offline future evaluation is overhead and does not buy a method a better budget."""

from __future__ import annotations

from typing import Any

from expolicy.paper_c.constants import ACCOUNT_COMPUTE, ACCOUNT_INTERACTION, ACCOUNT_OVERHEAD


class BudgetLedger:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        *,
        account: str,
        kind: str,
        interaction_units: float = 0.0,
        actor_tokens: int = 0,
        teacher_tokens: int = 0,
        tool_calls: int = 0,
        validation_calls: int = 0,
        gpu_hours: float = 0.0,
        wall_s: float = 0.0,
        detail: str = "",
    ) -> dict[str, Any]:
        if account not in {ACCOUNT_INTERACTION, ACCOUNT_COMPUTE, ACCOUNT_OVERHEAD}:
            raise ValueError(f"unknown cost account: {account}")
        row = {
            "account": account,
            "kind": kind,
            "interaction_units": float(interaction_units),
            "actor_tokens": int(actor_tokens),
            "teacher_tokens": int(teacher_tokens),
            "tool_calls": int(tool_calls),
            "validation_calls": int(validation_calls),
            "gpu_hours": float(gpu_hours),
            "wall_s": float(wall_s),
            "detail": detail,
        }
        self.rows.append(row)
        return row

    def interaction_total(self) -> float:
        return sum(row["interaction_units"] for row in self.rows if row["account"] == ACCOUNT_INTERACTION)

    def compute_tokens(self) -> int:
        return sum(row["actor_tokens"] + row["teacher_tokens"] for row in self.rows if row["account"] != ACCOUNT_OVERHEAD)

    def overhead_interactions(self) -> float:
        return sum(row["interaction_units"] for row in self.rows if row["account"] == ACCOUNT_OVERHEAD)

    def totals(self) -> dict[str, float]:
        return {
            "interaction": self.interaction_total(),
            "compute_tokens": float(self.compute_tokens()),
            "evaluation_overhead": self.overhead_interactions(),
        }
