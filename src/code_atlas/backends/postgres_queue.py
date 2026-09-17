"""Postgres-backed event bus — a third queue backend beside Valkey and SQLite (ADR-0056).

Matches :class:`code_atlas.events.EventBus`'s public interface method for method, and
its *semantics* where the two embedded mirrors disagree: a delivery outlives the message
it points at (a trimmed or flushed message replays with empty fields, which consumers
ack), and a consumer registration is a real row that ``drop_consumer`` removes along with
whatever it still held.

**One database, many projects.** Valkey prefixes keys with the project and SQLite keeps a
file per project; here every row is keyed by ``(project, topic)``, so one database serves
a whole fleet.

**Tables, not a queue extension.** Plain tables in the ``atlas_queue`` schema, with row
locks:

- ``messages``   — ``(project, topic, id)`` payloads; ``id`` is an identity column.
- ``groups``     — per-group cursor ``last_id``: the newest id ever handed to the group.
- ``deliveries`` — the pending-entries list, one row per delivered-but-unacked message.
  Ack deletes the row. No foreign key to ``messages``, so trimming never cascades.
- ``consumers``  — registrations with ``seen_at``, for ``consumer_registrations``' idle.
- ``leases``     — the indexer lease, one row per project.
- ``rate_bucket`` / ``rate_scale`` — the embedding rate limiter's shared state
  (``search/ratelimit.py``).
- ``meta``       — ``schema_version``, so a process refuses a schema newer than its code
  and migrates an older one (:func:`ensure_queue_schema`).

**Ids are assigned in commit order.** A cursor is only correct if no message can commit
*behind* it. With a bare identity column a publisher can take id 10, stall, and commit
after another publisher's id 11 was already read and the cursor moved to 11 — and id 10
is then skipped forever. Every publish therefore takes ``pg_advisory_xact_lock`` on
``(project, topic)`` *before* the insert draws its ids, and holds it to commit, so ids of
one stream become visible strictly in order.

**Blocking reads wake on LISTEN/NOTIFY.** A publish fires ``pg_notify`` inside its
transaction, which Postgres delivers at commit. One dedicated connection per process,
outside the pool, listens; a waiting reader holds no pool connection and falls back to a
bounded poll if the listener is gone.

**The bus owns no connection.** :class:`PostgresConnections` is the process's one pool and
one listener, opened by the composition root (``code_atlas.backends.use_backends``) and
shared with the embedding rate limiter (ADR-0038).

**Every time comes from the server clock** (``clock_timestamp()``): idle times, reclaim
cutoffs and lease expiry. The Valkey Lua does the same on purpose — a container clock was
measured 16s away from its host.

Message ids are ``b"<id>-0"``, the ``"<int>-<int>"`` shape
:func:`code_atlas.indexing.consumers._stream_id_key` parses for newest-wins dedup.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import TYPE_CHECKING, Any, Self

import asyncpg
from loguru import logger

from code_atlas.events import ConsumerGroupMissingError, Event, StreamGroupInfo, Topic, encode_event
from code_atlas.telemetry import get_tracer

if TYPE_CHECKING:
    from collections.abc import Callable

    from code_atlas.settings import PostgresSettings

# Same span names as the Valkey ``EventBus``: a trace should not change shape because a
# deployment chose another queue.
_tracer = get_tracer(__name__)

SCHEMA = "atlas_queue"

# One statement per entry: asyncpg runs a multi-statement string through the simple
# query protocol, but keeping them apart makes a failing one name itself.
_DDL = (
    f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.messages (
        project text NOT NULL,
        topic   text NOT NULL,
        id      bigint GENERATED ALWAYS AS IDENTITY,
        payload bytea NOT NULL,
        PRIMARY KEY (project, topic, id)
    )""",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.groups (
        project text NOT NULL,
        topic   text NOT NULL,
        grp     text NOT NULL,
        last_id bigint NOT NULL DEFAULT 0,
        PRIMARY KEY (project, topic, grp)
    )""",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.deliveries (
        project        text NOT NULL,
        topic          text NOT NULL,
        grp            text NOT NULL,
        message_id     bigint NOT NULL,
        consumer       text NOT NULL,
        delivered_at   timestamptz NOT NULL,
        delivery_count integer NOT NULL DEFAULT 1,
        PRIMARY KEY (project, topic, grp, message_id)
    )""",
    f"CREATE INDEX IF NOT EXISTS deliveries_by_consumer ON {SCHEMA}.deliveries (project, topic, grp, consumer)",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.consumers (
        project  text NOT NULL,
        topic    text NOT NULL,
        grp      text NOT NULL,
        consumer text NOT NULL,
        seen_at  timestamptz NOT NULL,
        PRIMARY KEY (project, topic, grp, consumer)
    )""",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.leases (
        project    text PRIMARY KEY,
        owner      text NOT NULL,
        expires_at timestamptz NOT NULL
    )""",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.rate_bucket (
        key   text PRIMARY KEY,
        level double precision,
        ts    bigint NOT NULL
    )""",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.rate_scale (
        key     text PRIMARY KEY,
        v       double precision NOT NULL,
        next_at bigint NOT NULL
    )""",
)

# NOTIFY channel shared by every project in the database; the payload names the stream.
_CHANNEL = "atlas_queue_wake"

# Upper bound on one wait between claim attempts. The notification is the fast path; this
# only bounds how long a lost notification (a dropped listener connection) can delay a read.
_POLL_FALLBACK_S = 1.0

_DEFAULT_STREAM_MAXLEN = 100_000
"""Mirrors ``RedisSettings.stream_maxlen``'s default — the same retained-backlog budget."""

_POOL_MAX_SIZE = 16
"""Connections in the process's one pool, shared by the event bus and the rate limiter.

Sized for what one indexing process runs at once. The limiter holds a connection for each
concurrent acquire, and acquires are bounded by the embedding client's concurrency gate:
at most ``max_concurrency``, whose largest provider default is 8 (litellm). The bus adds
the AST and embed consumers' claim loops, the embed workers' acks, the orchestrator's
publishes, the lease renewal and the drain poll -- statements a few milliseconds long,
about 8 at a time at the embed stage's peak. 8 + 8 = 16.

Undersizing is latency, not failure: no path holds a pool connection while acquiring a
second one, so a caller past the limit waits in ``pool.acquire`` and cannot deadlock. The
two pools this replaced (10 for the bus, 4 per embedding client's limiter) were sized
independently and summed past this once a process held two embedding clients.
"""


QUEUE_SCHEMA_VERSION = 1
"""The shape of the ``atlas_queue`` tables this code reads and writes.

Bump it with every change to a table's shape, and add the statements that take the
previous version there to ``_MIGRATIONS``. ``_DDL`` always describes the current shape.
"""

_MIGRATIONS: dict[int, tuple[str, ...]] = {}
"""Statements that upgrade a database *to* each version, keyed by that version.

Applied oldest first, in the same transaction as the version stamp. Empty while the schema
is at version 1: version 1 is the shape ``_DDL`` has always created.
"""

_UNVERSIONED = 1
"""The version of a database whose tables predate the version row.

Only the version-1 code ever created ``atlas_queue`` without stamping it, so an
unstamped database with tables in it has the version-1 shape.
"""


class QueueSchemaVersionError(RuntimeError):
    """The queue database was stamped by newer code than this."""

    def __init__(self, stored: int, code: int) -> None:
        super().__init__(
            f"Postgres queue schema v{stored} is newer than code v{code}. "
            "Downgrade is not supported — update your Code Atlas installation."
        )
        self.stored = stored
        self.code = code


async def ensure_queue_schema(conn: asyncpg.Connection) -> None:
    """Bring the ``atlas_queue`` schema to ``QUEUE_SCHEMA_VERSION``, or refuse.

    ``CREATE ... IF NOT EXISTS`` alone only adds what is missing, so a changed table would
    silently run against its old shape. The ``meta`` table's ``schema_version`` row says
    which shape is there:

    - newer than this code: raise :class:`QueueSchemaVersionError` before touching anything;
    - older: apply ``_MIGRATIONS`` in order, then stamp the current version;
    - absent: a fresh database gets the current version, and one whose tables already
      exist was made by the version-1 code that predates the row (``_UNVERSIONED``).

    Serialised with a transaction-scoped advisory lock: ``CREATE ... IF NOT EXISTS`` is not
    race-free between two transactions (both can pass the existence check and one then
    fails on the catalog's unique index), and two processes starting together is the
    normal case — an agent client spawns several MCP sessions at once. The same lock makes
    a migration run once.
    """
    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended('atlas_queue:schema', 0))")
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        await conn.execute(f"CREATE TABLE IF NOT EXISTS {SCHEMA}.meta (key text PRIMARY KEY, value text NOT NULL)")
        stored = await conn.fetchval(f"SELECT value FROM {SCHEMA}.meta WHERE key = 'schema_version'")
        if stored is not None:
            version = int(stored)
        elif await conn.fetchval(f"SELECT to_regclass('{SCHEMA}.messages') IS NOT NULL"):
            version = _UNVERSIONED
        else:
            version = QUEUE_SCHEMA_VERSION

        if version > QUEUE_SCHEMA_VERSION:
            raise QueueSchemaVersionError(version, QUEUE_SCHEMA_VERSION)
        for target in range(version + 1, QUEUE_SCHEMA_VERSION + 1):
            logger.info("Migrating Postgres queue schema v{} → v{}", target - 1, target)
            for statement in _MIGRATIONS[target]:
                await conn.execute(statement)

        for statement in _DDL:
            await conn.execute(statement)
        await conn.execute(
            f"INSERT INTO {SCHEMA}.meta (key, value) VALUES ('schema_version', $1) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            str(QUEUE_SCHEMA_VERSION),
        )


def connect_kwargs(settings: PostgresSettings) -> dict[str, Any]:
    """asyncpg connection arguments for *settings*. The one place the secret is unwrapped."""
    return {
        "host": settings.host,
        "port": settings.port,
        "user": settings.user,
        "password": settings.password.get_secret_value() or None,
        "database": settings.database,
        # asyncpg defaults to 60s, which turns an unreachable server into a minute-long hang.
        "timeout": 10,
    }


class PostgresConnections:
    """The process's connections to the queue database: one pool and one LISTEN connection.

    Owned by the composition root, handed to :class:`PostgresEventBus` and
    ``PostgresRateLimiter``, and closed only by whoever constructed it (ADR-0038). Both are
    opened on first use, so selecting the backend connects to nothing.

    Every connection carries ``application_name`` ``code-atlas:<pid>``, so
    ``pg_stat_activity`` attributes each one to the process holding it.
    """

    def __init__(self, settings: PostgresSettings) -> None:
        self._settings = settings
        self.application_name = f"code-atlas:{os.getpid()}"
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()
        self._listener: asyncpg.Connection | None = None
        self._listener_lock = asyncio.Lock()
        self._subscriptions: set[tuple[str, Callable[..., None]]] = set()

    @property
    def address(self) -> str:
        """``host:port/database`` — never the password."""
        return f"{self._settings.host}:{self._settings.port}/{self._settings.database}"

    def _connect_kwargs(self) -> dict[str, Any]:
        return {**connect_kwargs(self._settings), "server_settings": {"application_name": self.application_name}}

    async def pool(self) -> asyncpg.Pool:
        """The shared pool, created (with the queue schema) on first use."""
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    pool = await asyncpg.create_pool(min_size=1, max_size=_POOL_MAX_SIZE, **self._connect_kwargs())
                    try:
                        async with pool.acquire() as conn:
                            await ensure_queue_schema(conn)
                    except BaseException:
                        await pool.close()
                        raise
                    self._pool = pool
        return self._pool

    async def listen(self, channel: str, callback: Callable[..., None]) -> bool:
        """Subscribe *callback* to *channel* on the listener connection. False if it cannot be had.

        Outside the pool deliberately: a listening connection is held for the process's whole
        life, and parking it in the pool would starve the queries it exists to wake. A
        listener that dropped is reopened with every subscription it carried.
        """
        key = (channel, callback)
        listener = self._listener
        if key in self._subscriptions and listener is not None and not listener.is_closed():
            return True
        async with self._listener_lock:
            conn = self._listener
            try:
                if conn is None or conn.is_closed():
                    conn = await asyncpg.connect(**self._connect_kwargs())
                    self._listener = conn
                    for ch, cb in self._subscriptions:
                        await conn.add_listener(ch, cb)
                if key not in self._subscriptions:
                    await conn.add_listener(channel, callback)
                    self._subscriptions.add(key)
            except Exception:
                logger.opt(exception=True).debug("Postgres queue listener unavailable; reads will poll")
                return False
            return True

    async def close(self) -> None:
        """Close the listener and the pool. Idempotent."""
        listener, self._listener = self._listener, None
        if listener is not None and not listener.is_closed():
            with contextlib.suppress(Exception):
                await listener.close()
        pool, self._pool = self._pool, None
        if pool is not None:
            await pool.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close on the way out, including on an exception (ADR-0038)."""
        await self.close()


def _msg_id(raw: int) -> bytes:
    return f"{raw}-0".encode()


def _parse_msg_id(msg_id: bytes) -> int:
    head, _, _seq = msg_id.partition(b"-")
    return int(head)


def _fields(payload: bytes | None) -> dict[bytes, bytes]:
    """Valkey's shape: ``{b"data": ...}``, or empty for a message that was trimmed away."""
    return {} if payload is None else {b"data": bytes(payload)}


def _rowcount(status: str) -> int:
    """Row count from a command tag such as ``"DELETE 3"``."""
    return int(status.rsplit(" ", 1)[-1])


class PostgresEventBus:
    """Async Postgres-backed event queue — see the module docstring for the design."""

    # Private switch for the commit-order test only: that test proves the skip it prevents
    # by turning it off. Never False in production.
    _commit_order_lock: bool = True

    def __init__(
        self,
        connections: PostgresConnections,
        *,
        project_name: str = "",
        stream_maxlen: int = _DEFAULT_STREAM_MAXLEN,
    ) -> None:
        self._connections = connections
        self._project = project_name
        self._maxlen = stream_maxlen if stream_maxlen > 0 else None
        self._wake: dict[str, asyncio.Event] = {}
        # Publishes since the last trim, per topic. Trimming is approximate on purpose:
        # an exact trim on every publish would scan maxlen index entries each time.
        self._since_trim: dict[Topic, int] = {}

    @property
    def address(self) -> str:
        """``host:port/database`` — never the password."""
        return self._connections.address

    # -- connections -----------------------------------------------------------

    async def _get_pool(self) -> asyncpg.Pool:
        return await self._connections.pool()

    def _wake_key(self, topic: Topic) -> str:
        return f"{self._project}:{topic.value}"

    def _on_notify(self, _conn: object, _pid: int, _channel: str, payload: str) -> None:
        event = self._wake.get(payload)
        if event is not None:
            event.set()

    async def ping(self) -> bool:
        """Health check — True if Postgres answers."""
        pool = await self._get_pool()
        return await pool.fetchval("SELECT 1") == 1

    # -- groups and publishing -------------------------------------------------

    async def ensure_group(self, topic: Topic, group: str) -> None:
        """Idempotently create a consumer group reading from the start of the stream (id 0)."""
        pool = await self._get_pool()
        await pool.execute(
            f"INSERT INTO {SCHEMA}.groups (project, topic, grp, last_id) VALUES ($1, $2, $3, 0) ON CONFLICT DO NOTHING",
            self._project,
            topic.value,
            group,
        )

    async def _insert(self, conn: asyncpg.Connection, topic: Topic, payloads: list[bytes]) -> list[int]:
        """Insert *payloads* in order and fire the wake-up; returns their ids.

        One statement. The advisory lock CTE is joined into the insert's source, so it is
        taken before the first id is drawn, and it is transaction-scoped, so it is held
        until the caller's commit — which is what puts ids in commit order. ``pg_notify``
        is queued in the same transaction and delivered only when it commits.
        """
        lock = (
            "SELECT pg_advisory_xact_lock(hashtextextended('atlas_queue:' || $1 || ':' || $2, 0))"
            if self._commit_order_lock
            else "SELECT 1"
        )
        rows = await conn.fetch(
            f"""
            WITH l AS ({lock}),
            ins AS (
                INSERT INTO {SCHEMA}.messages (project, topic, payload)
                SELECT $1, $2, u.p FROM l, unnest($3::bytea[]) WITH ORDINALITY AS u(p, n)
                ORDER BY u.n
                RETURNING id
            ),
            wake AS (SELECT pg_notify('{_CHANNEL}', $1 || ':' || $2) FROM l)
            SELECT ins.id FROM ins, wake ORDER BY ins.id
            """,
            self._project,
            topic.value,
            payloads,
        )
        return [int(r["id"]) for r in rows]

    async def _maybe_trim(self, conn: asyncpg.Connection, topic: Topic, published: int) -> None:
        """Keep roughly the newest ``stream_maxlen`` messages, like ``XADD MAXLEN ~``.

        Runs once per tenth of the budget published rather than on every publish; the
        deliveries of trimmed messages survive and replay with empty fields.
        """
        if self._maxlen is None:
            return
        count = self._since_trim.get(topic, 0) + published
        if count < max(1, self._maxlen // 10):
            self._since_trim[topic] = count
            return
        self._since_trim[topic] = 0
        await conn.execute(
            f"""
            DELETE FROM {SCHEMA}.messages
            WHERE project = $1 AND topic = $2 AND id < (
                SELECT id FROM {SCHEMA}.messages WHERE project = $1 AND topic = $2
                ORDER BY id DESC OFFSET $3 LIMIT 1
            )
            """,
            self._project,
            topic.value,
            self._maxlen - 1,
        )

    async def publish(self, topic: Topic, event: Event) -> bytes:
        """Publish an event. Returns the message ID (``b"<id>-0"``)."""
        with _tracer.start_as_current_span("eventbus.publish", attributes={"topic": topic.value}):
            return (await self._publish(topic, [event]))[0]

    async def publish_many(self, topic: Topic, events: list[Event]) -> list[bytes]:
        """Publish multiple events in one statement (one round-trip)."""
        if not events:
            return []
        with _tracer.start_as_current_span(
            "eventbus.publish_many", attributes={"topic": topic.value, "count": len(events)}
        ):
            return await self._publish(topic, events)

    async def _publish(self, topic: Topic, events: list[Event]) -> list[bytes]:
        pool = await self._get_pool()
        payloads = [encode_event(e)[b"data"] for e in events]
        async with pool.acquire() as conn:
            ids = await self._insert(conn, topic, payloads)
            await self._maybe_trim(conn, topic, len(ids))
        return [_msg_id(i) for i in ids]

    # -- reading ---------------------------------------------------------------

    async def _claim(
        self, topic: Topic, group: str, consumer: str, count: int
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        """Deliver up to *count* never-delivered messages and record the deliveries atomically.

        The group row lock serialises claimers of one group, so two consumers can never be
        handed the same message. The claim statement runs *after* the lock is granted and
        so reads a snapshot that includes whatever the previous holder committed.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            last_id = await conn.fetchval(
                f"SELECT last_id FROM {SCHEMA}.groups WHERE project = $1 AND topic = $2 AND grp = $3 FOR UPDATE",
                self._project,
                topic.value,
                group,
            )
            if last_id is None:
                raise ConsumerGroupMissingError(topic, group)
            rows = await conn.fetch(
                f"""
                WITH m AS (
                    SELECT id, payload FROM {SCHEMA}.messages
                    WHERE project = $1 AND topic = $2 AND id > $4
                    ORDER BY id LIMIT $5
                ),
                d AS (
                    INSERT INTO {SCHEMA}.deliveries (project, topic, grp, message_id, consumer, delivered_at)
                    SELECT $1, $2, $3, id, $6, clock_timestamp() FROM m
                ),
                c AS (
                    INSERT INTO {SCHEMA}.consumers (project, topic, grp, consumer, seen_at)
                    VALUES ($1, $2, $3, $6, clock_timestamp())
                    ON CONFLICT (project, topic, grp, consumer) DO UPDATE SET seen_at = excluded.seen_at
                ),
                u AS (
                    UPDATE {SCHEMA}.groups SET last_id = (SELECT max(id) FROM m)
                    WHERE project = $1 AND topic = $2 AND grp = $3 AND EXISTS (SELECT 1 FROM m)
                )
                SELECT id, payload FROM m ORDER BY id
                """,
                self._project,
                topic.value,
                group,
                last_id,
                count,
                consumer,
            )
        return [(_msg_id(r["id"]), _fields(r["payload"])) for r in rows]

    async def read_batch(
        self,
        topic: Topic,
        group: str,
        consumer: str,
        *,
        count: int = 10,
        block_ms: int = 2000,
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        """Claim a batch of new messages, waiting up to *block_ms* for one to arrive.

        The wake event is cleared *before* each claim, so a publish that commits after the
        claim's snapshot always leaves it set and the next wait returns at once. No pool
        connection is held while waiting.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + block_ms / 1000
        with _tracer.start_as_current_span(
            "eventbus.read_batch", attributes={"topic": topic.value, "group": group, "consumer": consumer}
        ):
            listening = block_ms > 0 and await self._connections.listen(_CHANNEL, self._on_notify)
            wake = self._wake.setdefault(self._wake_key(topic), asyncio.Event())
            while True:
                wake.clear()
                rows = await self._claim(topic, group, consumer, count)
                if rows:
                    return rows
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return []
                timeout = min(remaining, _POLL_FALLBACK_S if listening else 0.1)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(wake.wait(), timeout)

    async def read_pending(
        self,
        topic: Topic,
        group: str,
        consumer: str,
        *,
        count: int = 10,
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        """Replay this consumer's unacked deliveries, oldest first.

        Like ``XREADGROUP ... 0`` this counts as a redelivery: it refreshes the delivery
        time, so a live consumer replaying its own work is not reclaimed as idle.
        """
        with _tracer.start_as_current_span(
            "eventbus.read_pending", attributes={"topic": topic.value, "group": group, "consumer": consumer}
        ):
            pool = await self._get_pool()
            rows = await pool.fetch(
                f"""
                WITH r AS (
                    SELECT message_id FROM {SCHEMA}.deliveries
                    WHERE project = $1 AND topic = $2 AND grp = $3 AND consumer = $4
                    ORDER BY message_id LIMIT $5
                ),
                u AS (
                    UPDATE {SCHEMA}.deliveries d
                    SET delivered_at = clock_timestamp(), delivery_count = d.delivery_count + 1
                    FROM r
                    WHERE d.project = $1 AND d.topic = $2 AND d.grp = $3 AND d.message_id = r.message_id
                      AND d.consumer = $4
                    RETURNING d.message_id
                ),
                c AS (
                    INSERT INTO {SCHEMA}.consumers (project, topic, grp, consumer, seen_at)
                    VALUES ($1, $2, $3, $4, clock_timestamp())
                    ON CONFLICT (project, topic, grp, consumer) DO UPDATE SET seen_at = excluded.seen_at
                )
                SELECT u.message_id, m.payload FROM u
                LEFT JOIN {SCHEMA}.messages m ON m.project = $1 AND m.topic = $2 AND m.id = u.message_id
                ORDER BY u.message_id
                """,
                self._project,
                topic.value,
                group,
                consumer,
                count,
            )
            if not rows:
                await self._require_group(topic, group)
            return [(_msg_id(r["message_id"]), _fields(r["payload"])) for r in rows]

    async def _require_group(self, topic: Topic, group: str) -> None:
        """Raise :class:`ConsumerGroupMissingError` unless *group* exists. Only asked on an empty
        replay: a delivery row cannot outlive its group, so rows prove the group is there."""
        pool = await self._get_pool()
        exists = await pool.fetchval(
            f"SELECT 1 FROM {SCHEMA}.groups WHERE project = $1 AND topic = $2 AND grp = $3",
            self._project,
            topic.value,
            group,
        )
        if exists is None:
            raise ConsumerGroupMissingError(topic, group)

    async def reclaim_abandoned(
        self,
        topic: Topic,
        group: str,
        consumer: str,
        *,
        min_idle_ms: int,
        count: int = 10,
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        """Take over deliveries idle for at least *min_idle_ms*, like ``XAUTOCLAIM``.

        ``SKIP LOCKED`` lets two reclaimers sweep at once without handing one entry to both.
        Idle is measured on the server clock.
        """
        with _tracer.start_as_current_span(
            "eventbus.reclaim_abandoned", attributes={"topic": topic.value, "group": group, "consumer": consumer}
        ):
            pool = await self._get_pool()
            rows = await pool.fetch(
                f"""
                WITH r AS (
                    SELECT message_id FROM {SCHEMA}.deliveries
                    WHERE project = $1 AND topic = $2 AND grp = $3
                      AND delivered_at <= clock_timestamp() - make_interval(secs => $5::double precision / 1000)
                    ORDER BY message_id LIMIT $6
                    FOR UPDATE SKIP LOCKED
                ),
                u AS (
                    UPDATE {SCHEMA}.deliveries d
                    SET consumer = $4, delivered_at = clock_timestamp(), delivery_count = d.delivery_count + 1
                    FROM r
                    WHERE d.project = $1 AND d.topic = $2 AND d.grp = $3 AND d.message_id = r.message_id
                    RETURNING d.message_id
                ),
                c AS (
                    INSERT INTO {SCHEMA}.consumers (project, topic, grp, consumer, seen_at)
                    VALUES ($1, $2, $3, $4, clock_timestamp())
                    ON CONFLICT (project, topic, grp, consumer) DO UPDATE SET seen_at = excluded.seen_at
                )
                SELECT u.message_id, m.payload FROM u
                LEFT JOIN {SCHEMA}.messages m ON m.project = $1 AND m.topic = $2 AND m.id = u.message_id
                ORDER BY u.message_id
                """,
                self._project,
                topic.value,
                group,
                consumer,
                float(min_idle_ms),
                count,
            )
            return [(_msg_id(r["message_id"]), _fields(r["payload"])) for r in rows]

    async def ack(self, topic: Topic, group: str, *msg_ids: bytes) -> int:
        """Acknowledge messages; returns how many were pending."""
        if not msg_ids:
            return 0
        pool = await self._get_pool()
        status = await pool.execute(
            f"DELETE FROM {SCHEMA}.deliveries WHERE project = $1 AND topic = $2 AND grp = $3 "
            "AND message_id = ANY($4::bigint[])",
            self._project,
            topic.value,
            group,
            [_parse_msg_id(m) for m in msg_ids],
        )
        return _rowcount(status)

    # -- registrations ---------------------------------------------------------

    async def consumer_registrations(self, topic: Topic, group: str) -> list[tuple[str, int, int]]:
        """``(name, pending, idle_ms)`` for every consumer registered in *group*.

        *idle_ms* is time since the consumer last read, reclaimed or replayed — on the
        server clock — so a live consumer blocking on an empty stream still reads as busy.
        """
        pool = await self._get_pool()
        rows = await pool.fetch(
            f"""
            SELECT c.consumer,
                   (SELECT count(*) FROM {SCHEMA}.deliveries d
                    WHERE d.project = c.project AND d.topic = c.topic AND d.grp = c.grp
                      AND d.consumer = c.consumer) AS pending,
                   GREATEST(0, (extract(epoch FROM clock_timestamp() - c.seen_at) * 1000))::bigint AS idle_ms
            FROM {SCHEMA}.consumers c
            WHERE c.project = $1 AND c.topic = $2 AND c.grp = $3
            ORDER BY c.consumer
            """,
            self._project,
            topic.value,
            group,
        )
        return [(str(r["consumer"]), int(r["pending"]), int(r["idle_ms"])) for r in rows]

    async def drop_consumer(self, topic: Topic, group: str, consumer: str) -> int:
        """Deregister *consumer*, returning how many pending entries went with it.

        Destroys those entries rather than reassigning them, exactly like
        ``XGROUP DELCONSUMER`` — the return value is a leak detector.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            status = await conn.execute(
                f"DELETE FROM {SCHEMA}.deliveries WHERE project = $1 AND topic = $2 AND grp = $3 AND consumer = $4",
                self._project,
                topic.value,
                group,
                consumer,
            )
            await conn.execute(
                f"DELETE FROM {SCHEMA}.consumers WHERE project = $1 AND topic = $2 AND grp = $3 AND consumer = $4",
                self._project,
                topic.value,
                group,
                consumer,
            )
        return _rowcount(status)

    # -- indexer lease ---------------------------------------------------------

    async def acquire_indexer_lease(self, owner: str, ttl_ms: int) -> bool:
        """Take the indexer lease, or return False if a live one is held (``SET NX PX``)."""
        pool = await self._get_pool()
        got = await pool.fetchval(
            f"""
            INSERT INTO {SCHEMA}.leases AS l (project, owner, expires_at)
            VALUES ($1, $2, clock_timestamp() + make_interval(secs => $3::double precision / 1000))
            ON CONFLICT (project) DO UPDATE SET owner = excluded.owner, expires_at = excluded.expires_at
            WHERE l.expires_at <= clock_timestamp()
            RETURNING true
            """,
            self._project,
            owner,
            float(ttl_ms),
        )
        return bool(got)

    async def force_acquire_indexer_lease(self, owner: str, ttl_ms: int) -> bool:
        """Take the lease out from under whoever holds it. See ``EventBus`` for why."""
        pool = await self._get_pool()
        await pool.execute(
            f"""
            INSERT INTO {SCHEMA}.leases (project, owner, expires_at)
            VALUES ($1, $2, clock_timestamp() + make_interval(secs => $3::double precision / 1000))
            ON CONFLICT (project) DO UPDATE SET owner = excluded.owner, expires_at = excluded.expires_at
            """,
            self._project,
            owner,
            float(ttl_ms),
        )
        return True

    async def renew_indexer_lease(self, owner: str, ttl_ms: int) -> bool:
        """Extend the lease only while *owner* still holds a live one (compare-and-set)."""
        pool = await self._get_pool()
        status = await pool.execute(
            f"""
            UPDATE {SCHEMA}.leases
            SET expires_at = clock_timestamp() + make_interval(secs => $3::double precision / 1000)
            WHERE project = $1 AND owner = $2 AND expires_at > clock_timestamp()
            """,
            self._project,
            owner,
            float(ttl_ms),
        )
        return _rowcount(status) > 0

    async def release_indexer_lease(self, owner: str) -> bool:
        """Release the lease if *owner* holds it (compare-and-delete)."""
        pool = await self._get_pool()
        live = await pool.fetchval(
            f"DELETE FROM {SCHEMA}.leases WHERE project = $1 AND owner = $2 RETURNING expires_at > clock_timestamp()",
            self._project,
            owner,
        )
        return bool(live)

    async def read_indexer_lease(self) -> str | None:
        """Current lease holder, for diagnostics. ``None`` when the lease is free."""
        pool = await self._get_pool()
        return await pool.fetchval(
            f"SELECT owner FROM {SCHEMA}.leases WHERE project = $1 AND expires_at > clock_timestamp()",
            self._project,
        )

    # -- backlog ---------------------------------------------------------------

    async def stream_group_info(self, topic: Topic, group: str) -> StreamGroupInfo:
        """Pending + lag for one group; ``{"pending": 0, "lag": 0}`` for an unknown group."""
        return (await self.stream_group_info_multi([(topic, group)]))[0]

    async def stream_group_info_multi(self, queries: list[tuple[Topic, str]]) -> list[StreamGroupInfo]:
        """Pending + lag for several groups in one statement.

        ``lag`` is exact — the number of messages past the group's cursor that still exist —
        so, unlike Valkey after a trim, it is never ``None``. It is never 0 while a readable
        message remains, which is the only property callers rely on.
        """
        if not queries:
            return []
        pool = await self._get_pool()
        rows = await pool.fetch(
            f"""
            SELECT q.ord,
                   (SELECT count(*) FROM {SCHEMA}.deliveries d
                    WHERE d.project = $1 AND d.topic = q.topic AND d.grp = q.grp) AS pending,
                   (SELECT count(*) FROM {SCHEMA}.messages m
                    WHERE m.project = $1 AND m.topic = q.topic AND m.id > g.last_id) AS lag,
                   g.last_id IS NOT NULL AS known
            FROM unnest($2::text[], $3::text[]) WITH ORDINALITY AS q(topic, grp, ord)
            LEFT JOIN {SCHEMA}.groups g ON g.project = $1 AND g.topic = q.topic AND g.grp = q.grp
            ORDER BY q.ord
            """,
            self._project,
            [t.value for t, _ in queries],
            [g for _, g in queries],
        )
        return [
            {"pending": int(r["pending"]), "lag": int(r["lag"])} if r["known"] else {"pending": 0, "lag": 0}
            for r in rows
        ]

    async def flush(self) -> None:
        """Delete this project's messages for a full reindex.

        Groups, registrations and deliveries survive — deliveries of flushed messages replay
        with empty fields and consumers ack them, as with ``XTRIM`` on Valkey.
        """
        pool = await self._get_pool()
        await pool.execute(f"DELETE FROM {SCHEMA}.messages WHERE project = $1", self._project)

    async def delete_project_queue(self) -> int | None:
        """Delete this project's messages, groups, deliveries, registrations and expired lease.

        Returns the number of rows removed. One transaction, keyed by project, so no other
        project's rows and none of the rate limiter's shared rows are reachable from it. A
        *live* lease row stays: whoever removes a project holds it and releases it by
        compare-and-delete, and deleting it under them would let a second indexer in.
        """
        pool = await self._get_pool()
        removed = 0
        async with pool.acquire() as conn, conn.transaction():
            for table in ("messages", "groups", "deliveries", "consumers"):
                status = await conn.execute(f"DELETE FROM {SCHEMA}.{table} WHERE project = $1", self._project)
                removed += _rowcount(status)
            status = await conn.execute(
                f"DELETE FROM {SCHEMA}.leases WHERE project = $1 AND expires_at <= clock_timestamp()", self._project
            )
            removed += _rowcount(status)
        return removed
