"""SQLite episode ledger. Reservations survive process restarts."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from expolicy.paper_d.constants import ProtocolError


class PersistentLedger:
    def __init__(self, path: Path, *, cap: int) -> None:
        if cap < 0:
            raise ProtocolError("episode cap must be non-negative")
        self.path = Path(path)
        self.cap = int(cap)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                logical_execution_id TEXT NOT NULL,
                job_kind TEXT NOT NULL,
                planned_episodes INTEGER NOT NULL,
                state TEXT NOT NULL,
                reserved_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL,
                tokens INTEGER,
                tool_calls INTEGER,
                result_hash TEXT
            )
            """
        )

    def close(self) -> None:
        self._db.close()

    def spent(self) -> int:
        row = self._db.execute(
            "SELECT COALESCE(SUM(planned_episodes), 0) FROM attempts WHERE state != 'REVOKED'"
        ).fetchone()
        return int(row[0])

    def reserve_attempt(self, attempt_id: str, logical_execution_id: str, *, job_kind: str, episodes: int = 1) -> bool:
        if episodes < 0:
            raise ProtocolError("cannot reserve a negative episode count")
        now = time.time()
        try:
            self._db.execute("BEGIN IMMEDIATE")
            existing = self._db.execute(
                "SELECT state FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if existing is not None:
                self._db.execute("COMMIT")
                return True
            spent = self.spent()
            if spent + episodes > self.cap:
                self._db.execute("COMMIT")
                return False
            self._db.execute(
                """
                INSERT INTO attempts (
                    attempt_id, logical_execution_id, job_kind, planned_episodes, state, reserved_at
                ) VALUES (?, ?, ?, ?, 'RESERVED', ?)
                """,
                (attempt_id, logical_execution_id, job_kind, episodes, now),
            )
            self._db.execute("COMMIT")
        except sqlite3.Error:
            self._db.execute("ROLLBACK")
            raise
        return True

    def mark_started(self, attempt_id: str) -> None:
        self._db.execute(
            "UPDATE attempts SET state = 'STARTED', started_at = ? WHERE attempt_id = ?",
            (time.time(), attempt_id),
        )

    def finish_attempt(self, attempt_id: str, *, tokens: int, tool_calls: int, result_hash: str) -> None:
        self._db.execute(
            """
            UPDATE attempts
            SET state = 'COMPLETED', finished_at = ?, tokens = ?, tool_calls = ?, result_hash = ?
            WHERE attempt_id = ?
            """,
            (time.time(), tokens, tool_calls, result_hash, attempt_id),
        )

    def fail_attempt(self, attempt_id: str) -> None:
        self._db.execute(
            "UPDATE attempts SET state = 'FAILED', finished_at = ? WHERE attempt_id = ?",
            (time.time(), attempt_id),
        )

    def completed_hash(self, logical_execution_id: str) -> str | None:
        row = self._db.execute(
            """
            SELECT result_hash FROM attempts
            WHERE logical_execution_id = ? AND state = 'COMPLETED'
            ORDER BY finished_at DESC LIMIT 1
            """,
            (logical_execution_id,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return str(row[0])
