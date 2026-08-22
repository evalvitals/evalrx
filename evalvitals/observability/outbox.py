"""Durable, idempotent delivery queue for EvalVitals observability events.

The outbox is deliberately a delivery mechanism, not another long-term run
format.  A JSONL run log remains available during the migration, while this
SQLite database makes an interrupted or temporarily disconnected Langfuse
session resumable without dropping the events that happened in between.
"""

from __future__ import annotations

import json
import random
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class ObservabilityOutbox:
    """A small SQLite outbox with deterministic event ids and retry state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS outbox_events (
                    event_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    event_seq INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS outbox_ready "
                "ON outbox_events(state, next_attempt_at, created_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def enqueue(self, envelope: dict[str, Any]) -> None:
        """Insert an event once; duplicate calls are intentionally harmless."""
        event_id = str(envelope["event_id"])
        encoded = json.dumps(envelope, ensure_ascii=False, default=str, sort_keys=True)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO outbox_events
                    (event_id, trace_id, event_seq, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_id, str(envelope["trace_id"]), int(envelope["event_seq"]), encoded, time.time()),
            )

    def drain(self, publish: Callable[[dict[str, Any]], None], *, limit: int = 100) -> tuple[int, int]:
        """Try ready events once, returning ``(published, failed)``.

        The caller controls scheduling.  A failed publish stays durable and is
        retried with bounded exponential backoff plus jitter on a later flush.
        """
        now = time.time()
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, payload, attempts FROM outbox_events
                WHERE state = 'pending' AND next_attempt_at <= ?
                ORDER BY trace_id, event_seq
                LIMIT ?
                """,
                (now, limit),
            ).fetchall()

        published = failed = 0
        for row in rows:
            event_id = str(row["event_id"])
            try:
                publish(json.loads(str(row["payload"])))
            except Exception as exc:  # caller owns the transport and may be offline
                attempts = int(row["attempts"]) + 1
                delay = min(300.0, (2 ** min(attempts, 8)) + random.random())
                with self._lock, self._connect() as connection:
                    connection.execute(
                        """UPDATE outbox_events
                           SET attempts = ?, next_attempt_at = ?, last_error = ?
                           WHERE event_id = ?""",
                        (attempts, time.time() + delay, str(exc)[:1000], event_id),
                    )
                failed += 1
            else:
                with self._lock, self._connect() as connection:
                    connection.execute("DELETE FROM outbox_events WHERE event_id = ?", (event_id,))
                published += 1
        return published, failed

    def pending_count(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM outbox_events").fetchone()
        return int(row["n"])
