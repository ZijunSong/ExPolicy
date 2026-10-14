"""Episode cap. A reservation is spent once, and a second reference is not a new run."""

from __future__ import annotations

from expolicy.paper_d.constants import ProtocolError


class CostLedger:
    def __init__(self, cap: int) -> None:
        if cap < 0:
            raise ProtocolError("episode cap must be non-negative")
        self.cap = int(cap)
        self._reserved: dict[str, int] = {}
        self._committed: set[str] = set()

    def reserve(self, execution_id: str, episodes: int) -> bool:
        if execution_id in self._reserved:
            return True
        if episodes < 0:
            raise ProtocolError("cannot reserve a negative episode count")
        if self.spent + episodes > self.cap:
            return False
        self._reserved[execution_id] = int(episodes)
        return True

    def commit(self, execution_id: str) -> None:
        if execution_id not in self._reserved:
            raise ProtocolError("cannot commit an execution that was not reserved")
        self._committed.add(execution_id)

    @property
    def spent(self) -> int:
        return sum(self._reserved.values())

    @property
    def remaining(self) -> int:
        return self.cap - self.spent
