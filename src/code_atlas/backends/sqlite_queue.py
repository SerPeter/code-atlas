"""SQLite-backed event bus — in-process fallback for the Valkey/Redis event queue.

Matches :class:`code_atlas.events.EventBus`'s public interface (same method
signatures and return shapes) so :mod:`code_atlas.indexing.consumers` and
:mod:`code_atlas.indexing.daemon` need no changes beyond what they're handed
at construction time.

Two tables model a real redelivery/PEL (pending-entries-list) queue, not a
plain FIFO: ``messages`` holds published payloads; ``deliveries`` records,
per ``(topic, group, consumer)``, which messages have been claimed and
whether they've been acked. A third ``groups`` table tracks which consumer
groups have been created, so :meth:`SqliteEventBus.stream_group_info`
matches ``EventBus``'s "unknown group reports zero backlog" behavior.

Message ids are ``b"<rowid>-0"`` — the ``"<int>-<int>"`` byte-string shape
:func:`code_atlas.indexing.consumers._stream_id_key` parses for newest-wins
dedup.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import aiosqlite

from code_atlas.events import Event, Topic, encode_event

if TYPE_CHECKING:
    from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    payload BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_topic ON messages(topic);

CREATE TABLE IF NOT EXISTS deliveries (
    message_id INTEGER NOT NULL,
    topic TEXT NOT NULL,
    grp TEXT NOT NULL,
    consumer TEXT NOT NULL,
    delivered_at REAL NOT NULL,
    acked_at REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_deliveries_message_group ON deliveries(message_id, topic, grp);
CREATE INDEX IF NOT EXISTS ix_deliveries_consumer_pending ON deliveries(topic, grp, consumer, acked_at);

CREATE TABLE IF NOT EXISTS groups (
    topic TEXT NOT NULL,
    grp TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (topic, grp)
);
"""

_POLL_INTERVAL_S = 0.1


def _parse_msg_id(msg_id: bytes) -> int:
    """Extract the integer rowid from a ``b"<rowid>-0"`` message id."""
    rowid, _, _seq = msg_id.partition(b"-")
    return int(rowid)


class SqliteEventBus:
    """Async SQLite-backed event queue — drop-in fallback for :class:`~code_atlas.events.EventBus`.

    Backed by ``aiosqlite`` in WAL mode. One connection is opened lazily on
    first use and reused for the bus's lifetime.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._connect_lock = asyncio.Lock()
        # Guards the claim critical section in read_batch/_claim_new_messages:
        # SELECT-undelivered then INSERT-delivery must be atomic per (topic,
        # group) or two concurrent claimers could both grab the same message.
        self._claim_lock = asyncio.Lock()

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            async with self._connect_lock:
                if self._conn is None:
                    self._db_path.parent.mkdir(parents=True, exist_ok=True)
                    conn = await aiosqlite.connect(self._db_path)
                    await conn.execute("PRAGMA journal_mode=WAL")
                    await conn.execute("PRAGMA synchronous=NORMAL")
                    await conn.executescript(_SCHEMA)
                    await conn.commit()
                    self._conn = conn
        assert self._conn is not None
        return self._conn

    async def ping(self) -> bool:
        """Health check — returns True if the local database is reachable."""
        conn = await self._get_conn()
        await conn.execute("SELECT 1")
        return True

    async def ensure_group(self, topic: Topic, group: str) -> None:
        """Idempotently register a consumer group."""
        conn = await self._get_conn()
        await conn.execute(
            "INSERT OR IGNORE INTO groups (topic, grp, created_at) VALUES (?, ?, ?)",
            (topic.value, group, time.time()),
        )
        await conn.commit()

    async def publish(self, topic: Topic, event: Event) -> bytes:
        """Publish an event. Returns the message ID (``b"<rowid>-0"``)."""
        conn = await self._get_conn()
        payload = encode_event(event)[b"data"]
        cur = await conn.execute(
            "INSERT INTO messages (topic, payload, created_at) VALUES (?, ?, ?)",
            (topic.value, payload, time.time()),
        )
        await conn.commit()
        rowid = cur.lastrowid
        await cur.close()
        return f"{rowid}-0".encode()

    async def publish_many(self, topic: Topic, events: list[Event]) -> list[bytes]:
        """Publish multiple events in a single transaction."""
        if not events:
            return []
        conn = await self._get_conn()
        now = time.time()
        ids: list[bytes] = []
        for event in events:
            payload = encode_event(event)[b"data"]
            cur = await conn.execute(
                "INSERT INTO messages (topic, payload, created_at) VALUES (?, ?, ?)",
                (topic.value, payload, now),
            )
            ids.append(f"{cur.lastrowid}-0".encode())
            await cur.close()
        await conn.commit()
        return ids

    async def _claim_new_messages(
        self, topic: Topic, group: str, consumer: str, count: int
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        """Atomically select undelivered messages for (topic, group) and record delivery."""
        conn = await self._get_conn()
        async with self._claim_lock:
            cur = await conn.execute(
                """
                SELECT m.id, m.payload FROM messages m
                WHERE m.topic = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM deliveries d WHERE d.message_id = m.id AND d.topic = ? AND d.grp = ?
                  )
                ORDER BY m.id
                LIMIT ?
                """,
                (topic.value, topic.value, group, count),
            )
            rows = await cur.fetchall()
            await cur.close()
            if not rows:
                return []
            now = time.time()
            await conn.executemany(
                "INSERT INTO deliveries (message_id, topic, grp, consumer, delivered_at, acked_at) "
                "VALUES (?, ?, ?, ?, ?, NULL)",
                [(row[0], topic.value, group, consumer, now) for row in rows],
            )
            await conn.commit()
        return [(f"{row[0]}-0".encode(), {b"data": row[1]}) for row in rows]

    async def read_batch(
        self,
        topic: Topic,
        group: str,
        consumer: str,
        *,
        count: int = 10,
        block_ms: int = 2000,
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        """Claim a batch of undelivered messages, polling until *block_ms* elapses.

        SQLite has no server-side blocking read, so this polls every
        ``_POLL_INTERVAL_S`` up to *block_ms*, returning an empty list if
        nothing arrives in time. Returns the same shape as ``EventBus.read_batch``.
        """
        deadline = time.monotonic() + block_ms / 1000
        while True:
            rows = await self._claim_new_messages(topic, group, consumer, count)
            if rows:
                return rows
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return []
            await asyncio.sleep(min(_POLL_INTERVAL_S, remaining))

    async def read_pending(
        self,
        topic: Topic,
        group: str,
        consumer: str,
        *,
        count: int = 10,
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        """Replay this consumer's un-acked deliveries (the PEL) — survives a process restart."""
        conn = await self._get_conn()
        cur = await conn.execute(
            """
            SELECT m.id, m.payload FROM deliveries d
            JOIN messages m ON m.id = d.message_id
            WHERE d.topic = ? AND d.grp = ? AND d.consumer = ? AND d.acked_at IS NULL
            ORDER BY m.id
            LIMIT ?
            """,
            (topic.value, group, consumer, count),
        )
        rows = await cur.fetchall()
        await cur.close()
        return [(f"{row[0]}-0".encode(), {b"data": row[1]}) for row in rows]

    async def ack(self, topic: Topic, group: str, *msg_ids: bytes) -> int:
        """Acknowledge messages after successful processing. Returns the count acked."""
        if not msg_ids:
            return 0
        conn = await self._get_conn()
        rowids = [_parse_msg_id(mid) for mid in msg_ids]
        placeholders = ",".join("?" * len(rowids))
        cur = await conn.execute(
            f"UPDATE deliveries SET acked_at = ? WHERE topic = ? AND grp = ? "
            f"AND acked_at IS NULL AND message_id IN ({placeholders})",
            (time.time(), topic.value, group, *rowids),
        )
        await conn.commit()
        count = cur.rowcount
        await cur.close()
        return count

    async def stream_group_info(self, topic: Topic, group: str) -> dict[str, int | None]:
        """Return pending + lag counts for a consumer group.

        Returns ``{"pending": 0, "lag": 0}`` if the group was never
        registered via ``ensure_group`` (a missing group genuinely has no
        backlog).
        """
        conn = await self._get_conn()
        cur = await conn.execute("SELECT 1 FROM groups WHERE topic = ? AND grp = ?", (topic.value, group))
        exists = (await cur.fetchone()) is not None
        await cur.close()
        if not exists:
            return {"pending": 0, "lag": 0}

        cur = await conn.execute(
            "SELECT COUNT(*) FROM deliveries WHERE topic = ? AND grp = ? AND acked_at IS NULL",
            (topic.value, group),
        )
        pending = await self._scalar(cur)
        await cur.close()

        cur = await conn.execute(
            """
            SELECT COUNT(*) FROM messages m
            WHERE m.topic = ?
              AND NOT EXISTS (
                  SELECT 1 FROM deliveries d WHERE d.message_id = m.id AND d.topic = ? AND d.grp = ?
              )
            """,
            (topic.value, topic.value, group),
        )
        lag = await self._scalar(cur)
        await cur.close()
        return {"pending": pending, "lag": lag}

    async def stream_group_info_multi(self, queries: list[tuple[Topic, str]]) -> list[dict[str, int | None]]:
        """Return pending + lag counts for multiple consumer groups."""
        return [await self.stream_group_info(topic, group) for topic, group in queries]

    async def flush(self) -> None:
        """Clear all messages and deliveries for a full reindex. Consumer group registrations survive."""
        conn = await self._get_conn()
        await conn.execute("DELETE FROM deliveries")
        await conn.execute("DELETE FROM messages")
        await conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def _debug_pending_count(self, topic: Topic, group: str) -> int:
        """Test helper — count un-acked deliveries for (topic, group) across all consumers."""
        conn = await self._get_conn()
        cur = await conn.execute(
            "SELECT COUNT(*) FROM deliveries WHERE topic = ? AND grp = ? AND acked_at IS NULL",
            (topic.value, group),
        )
        count = await self._scalar(cur)
        await cur.close()
        return count

    @staticmethod
    async def _scalar(cur: aiosqlite.Cursor) -> int:
        """Read a single ``COUNT(*)``-style integer from a cursor's first row."""
        row = await cur.fetchone()
        assert row is not None
        return int(row[0])
