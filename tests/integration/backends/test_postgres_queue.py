"""`PostgresEventBus` behaviour the other buses do not share, or cannot show (ADR-0056).

The shared scenarios live in `test_queue_conformance.py`. These pin what is specific to
building a stream on tables: ids that must become visible in commit order, trimming that
leaves pending entries behind, real consumer registrations, wake-ups that hold no pool
connection, and expiry measured on the server clock rather than this process's.
"""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta

import asyncpg
import pytest
import time_machine

from code_atlas.backends import postgres_queue, use_backends
from code_atlas.backends.postgres_queue import (
    SCHEMA,
    PostgresConnections,
    PostgresEventBus,
    QueueSchemaVersionError,
    connect_kwargs,
    ensure_queue_schema,
)
from code_atlas.events import FileChanged, Topic, decode_event, encode_event
from code_atlas.indexing.consumers import BatchPolicy, TierConsumer
from code_atlas.indexing.orchestrator import index_project
from code_atlas.search.ratelimit import PostgresRateLimiter, RateBudget
from code_atlas.settings import AtlasSettings, BackendSettings, EmbeddingSettings, derive_project_name

pytestmark = pytest.mark.integration

TOPIC = Topic.FILE_CHANGED


def _event(path: str) -> FileChanged:
    return FileChanged(path=path, change_type="modified", project_name="p")


def _paths(batch: list[tuple[bytes, dict[bytes, bytes]]]) -> list[str]:
    out = []
    for _mid, fields in batch:
        event = decode_event(TOPIC, fields)
        assert isinstance(event, FileChanged)
        out.append(event.path)
    return out


# ---------------------------------------------------------------------------
# Commit order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lock", [True, False], ids=["with-lock", "lock-disabled"])
async def test_a_message_committed_behind_the_cursor_is_not_skipped(pg_settings, monkeypatch, lock) -> None:
    """Publisher A draws id N and stalls; publisher B draws N+1 and commits; a reader moves
    the group's cursor to N+1; then A commits. Without the commit-order lock, N sits behind
    the cursor and is never delivered to anybody.

    Run both ways on purpose. The lock-disabled case asserts the loss *happens*, which is
    what proves this test can see the failure at all -- a test that passes with the fix
    removed guards nothing.
    """
    monkeypatch.setattr(PostgresEventBus, "_commit_order_lock", lock)
    async with PostgresConnections(pg_settings) as connections:
        bus = PostgresEventBus(connections, project_name="test-commit-order")
        await bus.ensure_group(TOPIC, "g")
        stalled = await asyncpg.connect(**connect_kwargs(pg_settings))
        try:
            tx = stalled.transaction()
            await tx.start()
            (stalled_id,) = await bus._insert(stalled, TOPIC, [encode_event(_event("stalled"))[b"data"]])

            late = asyncio.create_task(bus.publish(TOPIC, _event("late")))
            # With the lock the publish blocks behind the stalled one, so this waits out its
            # full second. Without it the publish must be *finished* before the reader moves the
            # cursor -- a fixed second was a bet on machine load, so wait for the fact instead.
            await asyncio.wait({late}, timeout=1.0 if lock else 30.0)
            if not lock:
                assert late.done(), "the unlocked publish did not commit within 30s"

            first = await bus.read_batch(TOPIC, "g", "reader", count=10, block_ms=0)
            await tx.commit()
            await late
            rest = await bus.read_batch(TOPIC, "g", "reader", count=10, block_ms=1000)
        finally:
            await stalled.close()

    delivered = _paths(first) + _paths(rest)
    if lock:
        assert first == [], "the second publisher committed past a stalled one -- ids are not in commit order"
        assert delivered == ["stalled", "late"]
    else:
        assert delivered == ["late"], (
            f"expected the lock-disabled run to lose message {stalled_id}; it did not, so this "
            f"test no longer reproduces the skip it exists to catch (delivered={delivered})"
        )


# ---------------------------------------------------------------------------
# Trimming and flush keep pending entries, with empty fields
# ---------------------------------------------------------------------------


async def _message_count(pg_settings, project: str) -> int:
    conn = await asyncpg.connect(**connect_kwargs(pg_settings))
    try:
        return int(await conn.fetchval(f"SELECT count(*) FROM {SCHEMA}.messages WHERE project = $1", project))
    finally:
        await conn.close()


async def test_trimming_keeps_the_newest_and_leaves_pending_as_empty_fields(pg_settings) -> None:
    project = "test-trim"
    async with PostgresConnections(pg_settings) as connections:
        bus = PostgresEventBus(connections, project_name=project, stream_maxlen=10)
        await bus.ensure_group(TOPIC, "g")
        await bus.publish(TOPIC, _event("early"))
        (held,) = await bus.read_batch(TOPIC, "g", "c1", count=1, block_ms=500)

        for i in range(24):
            await bus.publish(TOPIC, _event(f"m{i}"))

        assert await _message_count(pg_settings, project) == 10

        replay = await bus.read_pending(TOPIC, "g", "c1", count=10)
        assert replay == [(held[0], {})], "a trimmed delivery must replay with empty fields, like Valkey"
        assert await bus.ack(TOPIC, "g", held[0]) == 1

        # Lag counts what can still be read, not what was ever published.
        assert await bus.stream_group_info(TOPIC, "g") == {"pending": 0, "lag": 10}
        assert _paths(await bus.read_batch(TOPIC, "g", "c1", count=100, block_ms=500)) == [
            f"m{i}" for i in range(14, 24)
        ]


async def test_flush_keeps_pending_entries_for_consumers_to_ack(pg_settings) -> None:
    async with PostgresConnections(pg_settings) as connections:
        bus = PostgresEventBus(connections, project_name="test-flush")
        await bus.ensure_group(TOPIC, "g")
        await bus.publish_many(TOPIC, [_event("a"), _event("b")])
        held = await bus.read_batch(TOPIC, "g", "c1", count=1, block_ms=500)

        await bus.flush()

        assert await bus.stream_group_info(TOPIC, "g") == {"pending": 1, "lag": 0}
        assert await bus.read_pending(TOPIC, "g", "c1", count=10) == [(held[0][0], {})]
        assert await bus.read_batch(TOPIC, "g", "c1", count=10, block_ms=50) == []


# ---------------------------------------------------------------------------
# Registrations
# ---------------------------------------------------------------------------


class _NullConsumer(TierConsumer):
    async def process_batch(self, events, batch_id):
        return None


def _consumer(bus, group, name, **kw):
    return _NullConsumer(bus, TOPIC, group, name, BatchPolicy(time_window_s=0.1, max_batch_size=5), **kw)


async def test_registrations_prune_like_valkey(pg_settings) -> None:
    """The consumer's prune/deregister policy, which the SQLite bus cannot exercise."""
    async with PostgresConnections(pg_settings) as connections:
        bus = PostgresEventBus(connections, project_name="test-registrations")
        await bus.ensure_group(TOPIC, "g")
        await bus.publish(TOPIC, _event("held"))
        assert await bus.read_batch(TOPIC, "g", "ast-dead-busy", count=1, block_ms=500)
        await bus.read_batch(TOPIC, "g", "ast-dead-idle", count=1, block_ms=10)

        live = _consumer(bus, "g", "ast-live", stale_consumer_idle_ms=0)
        await bus.read_batch(TOPIC, "g", live.consumer_name, count=1, block_ms=10)
        regs = {name: pending for name, pending, _idle in await bus.consumer_registrations(TOPIC, "g")}
        assert regs == {"ast-dead-busy": 1, "ast-dead-idle": 0, "ast-live": 0}

        await live._prune_consumer_registrations()
        names = [name for name, _p, _i in await bus.consumer_registrations(TOPIC, "g")]
        assert names == ["ast-dead-busy", "ast-live"], "a registration still holding work was pruned"

        assert await bus.drop_consumer(TOPIC, "g", "ast-dead-busy") == 1, "drop must report the work it destroyed"
        await live._deregister_self()
        assert await bus.consumer_registrations(TOPIC, "g") == []


async def test_idle_is_measured_on_the_server_clock(pg_settings) -> None:
    async with PostgresConnections(pg_settings) as connections:
        bus = PostgresEventBus(connections, project_name="test-idle")
        await bus.ensure_group(TOPIC, "g")
        await bus.read_batch(TOPIC, "g", "c1", count=1, block_ms=10)
        await asyncio.sleep(0.3)
        with time_machine.travel(timedelta(hours=5), tick=True):
            ((_name, _pending, idle_ms),) = await bus.consumer_registrations(TOPIC, "g")
        assert 250 <= idle_ms < 60_000


async def test_lease_expiry_ignores_the_client_clock(pg_settings) -> None:
    """A client clock far in the future must not expire a live lease — the server decides."""
    async with PostgresConnections(pg_settings) as a, PostgresConnections(pg_settings) as b:
        first = PostgresEventBus(a, project_name="test-lease-clock")
        second = PostgresEventBus(b, project_name="test-lease-clock")
        assert await first.acquire_indexer_lease("owner-a", 60_000) is True
        with time_machine.travel(timedelta(hours=2), tick=True):
            assert await second.acquire_indexer_lease("owner-b", 60_000) is False
            assert await second.read_indexer_lease() == "owner-a"


# ---------------------------------------------------------------------------
# Blocking reads
# ---------------------------------------------------------------------------


async def test_a_blocked_reader_wakes_on_notify_and_holds_no_pool_connection(pg_settings, monkeypatch) -> None:
    """The wake-up must come from NOTIFY, not the poll fallback -- so the fallback is pushed
    out past the test's own deadline -- and the wait must not pin a pool connection."""
    monkeypatch.setattr(postgres_queue, "_POLL_FALLBACK_S", 30.0)
    async with PostgresConnections(pg_settings) as reading, PostgresConnections(pg_settings) as writing:
        reader = PostgresEventBus(reading, project_name="test-wake")
        writer = PostgresEventBus(writing, project_name="test-wake")
        await reader.ensure_group(TOPIC, "g")
        started = time.monotonic()
        task = asyncio.create_task(reader.read_batch(TOPIC, "g", "c1", count=10, block_ms=20_000))
        # Until the reader has listened and finished its empty claim, a publish could land
        # before the LISTEN (and wake nobody) or the claim could still hold its connection. A
        # fixed half-second assumed both happen that fast on any machine; wait for them instead.
        pool = reading._pool
        assert pool is not None
        deadline = time.monotonic() + 10
        while reading._listener is None or pool.get_idle_size() != pool.get_size():
            assert time.monotonic() < deadline, "the reader never settled into its wait"
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.2)  # well into the wait: a held connection would still be held

        assert not task.done()
        assert pool.get_idle_size() == pool.get_size(), "a waiting reader is holding a pool connection"

        await writer.publish(TOPIC, _event("wake"))
        batch = await asyncio.wait_for(task, timeout=10)

        assert _paths(batch) == ["wake"]
        assert time.monotonic() - started < 10, "woke by polling, not by NOTIFY"


async def test_schema_creation_survives_concurrent_first_use(pg_settings) -> None:
    """Several processes start together; unserialised CREATE IF NOT EXISTS races on the catalog."""
    conn = await asyncpg.connect(**connect_kwargs(pg_settings))
    try:
        await conn.execute(f"DROP SCHEMA {SCHEMA} CASCADE")
    finally:
        await conn.close()

    processes = [PostgresConnections(pg_settings) for _ in range(6)]
    buses = [PostgresEventBus(c, project_name=f"test-ddl-{i}") for i, c in enumerate(processes)]
    try:
        assert await asyncio.gather(*(b.ping() for b in buses)) == [True] * 6
    finally:
        for c in processes:
            await c.close()


# ---------------------------------------------------------------------------
# Removing a project
# ---------------------------------------------------------------------------


async def _rows(pg_settings, project: str) -> dict[str, int]:
    conn = await asyncpg.connect(**connect_kwargs(pg_settings))
    try:
        return {
            table: int(await conn.fetchval(f"SELECT count(*) FROM {SCHEMA}.{table} WHERE project = $1", project))
            for table in ("messages", "groups", "deliveries", "consumers", "leases")
        }
    finally:
        await conn.close()


async def test_removing_a_project_takes_its_rows_and_only_its_rows(pg_settings) -> None:
    async with PostgresConnections(pg_settings) as connections:
        gone = PostgresEventBus(connections, project_name="test-rm-gone")
        kept = PostgresEventBus(connections, project_name="test-rm-kept")
        for b in (gone, kept):
            await b.ensure_group(TOPIC, "g")
            await b.publish_many(TOPIC, [_event("a"), _event("b")])
            await b.read_batch(TOPIC, "g", "c1", count=1, block_ms=500)
        assert await gone.acquire_indexer_lease("dead-holder", 1) is True  # expires at once
        assert await kept.acquire_indexer_lease("live-holder", 60_000) is True
        await PostgresRateLimiter(connections).penalize(RateBudget("m", 0, 0))
        await asyncio.sleep(0.05)
        before = await _rows(pg_settings, "test-rm-kept")

        removed = await gone.delete_project_queue()

        assert await _rows(pg_settings, "test-rm-gone") == dict.fromkeys(before, 0)
        assert removed == 2 + 1 + 1 + 1 + 1, "messages, group, delivery, registration and the expired lease"
        assert await _rows(pg_settings, "test-rm-kept") == before
        conn = await asyncpg.connect(**connect_kwargs(pg_settings))
        try:
            assert await conn.fetchval(f"SELECT count(*) FROM {SCHEMA}.rate_scale WHERE key = 'rl:m:scale'") == 1
        finally:
            await conn.close()


async def test_removing_a_project_leaves_a_live_lease_to_its_holder(pg_settings) -> None:
    """The remover holds this lease itself; deleting it mid-removal would let an indexer in."""
    async with PostgresConnections(pg_settings) as connections:
        bus = PostgresEventBus(connections, project_name="test-rm-leased")
        assert await bus.acquire_indexer_lease("remover", 60_000) is True

        await bus.delete_project_queue()

        assert await bus.read_indexer_lease() == "remover"
        assert await bus.release_indexer_lease("remover") is True


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

_REAL_VERSION = postgres_queue.QUEUE_SCHEMA_VERSION


async def _stored_version(conn: asyncpg.Connection) -> str | None:
    return await conn.fetchval(f"SELECT value FROM {SCHEMA}.meta WHERE key = 'schema_version'")


async def _stamp(conn: asyncpg.Connection, version: int) -> None:
    await conn.execute(
        f"INSERT INTO {SCHEMA}.meta (key, value) VALUES ('schema_version', $1) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        str(version),
    )


async def test_a_fresh_database_is_stamped_with_the_current_version(pg_settings) -> None:
    conn = await asyncpg.connect(**connect_kwargs(pg_settings))
    try:
        await conn.execute(f"DROP SCHEMA {SCHEMA} CASCADE")
        await ensure_queue_schema(conn)
        assert await _stored_version(conn) == str(_REAL_VERSION)
    finally:
        await conn.close()


async def test_a_database_from_before_the_version_row_is_read_as_version_one(pg_settings, monkeypatch) -> None:
    """The version-1 code created every table and no `meta`. Those databases are version 1,
    and taking them for fresh would skip every migration a later version adds -- so this
    runs as a version-2 codebase with one migration, which only a v1 reading applies."""
    monkeypatch.setattr(postgres_queue, "QUEUE_SCHEMA_VERSION", 2)
    monkeypatch.setattr(postgres_queue, "_MIGRATIONS", {2: (f"CREATE TABLE {SCHEMA}.test_migrated (step int)",)})
    conn = await asyncpg.connect(**connect_kwargs(pg_settings))
    try:
        await conn.execute(f"INSERT INTO {SCHEMA}.messages (project, topic, payload) VALUES ('test-old', 't', 'x')")
        await conn.execute(f"DROP TABLE {SCHEMA}.meta")

        await ensure_queue_schema(conn)

        assert await conn.fetchval(f"SELECT to_regclass('{SCHEMA}.test_migrated') IS NOT NULL"), (
            "an unstamped database with tables was taken for a fresh one and skipped its migration"
        )
        assert await _stored_version(conn) == "2"
        assert await conn.fetchval(f"SELECT count(*) FROM {SCHEMA}.messages WHERE project = 'test-old'") == 1, (
            "stamping an existing database must not rebuild it"
        )
    finally:
        await conn.execute(f"DROP TABLE IF EXISTS {SCHEMA}.test_migrated")
        await _stamp(conn, _REAL_VERSION)
        await conn.close()


async def test_a_newer_schema_is_refused_before_anything_is_touched(pg_settings) -> None:
    conn = await asyncpg.connect(**connect_kwargs(pg_settings))
    try:
        await _stamp(conn, _REAL_VERSION + 1)
        try:
            with pytest.raises(QueueSchemaVersionError) as excinfo:
                await ensure_queue_schema(conn)
            message = str(excinfo.value)
            assert f"v{_REAL_VERSION + 1}" in message
            assert f"v{_REAL_VERSION}" in message
            assert await _stored_version(conn) == str(_REAL_VERSION + 1), "a refused schema must not be restamped"

            # And through the pool a process actually uses: nothing opens against it.
            async with PostgresConnections(pg_settings) as connections:
                with pytest.raises(QueueSchemaVersionError):
                    await PostgresEventBus(connections, project_name="test-newer").ping()
        finally:
            await _stamp(conn, _REAL_VERSION)
    finally:
        await conn.close()


async def test_an_older_schema_is_migrated_in_order(pg_settings, monkeypatch) -> None:
    """The mechanism, exercised with migrations that exist only in this test: the real list
    is empty while the schema is at version 1.

    Migration 3 writes into the table migration 2 creates, so running them out of order, or
    skipping one, fails outright rather than passing."""
    monkeypatch.setattr(postgres_queue, "QUEUE_SCHEMA_VERSION", _REAL_VERSION + 2)
    monkeypatch.setattr(
        postgres_queue,
        "_MIGRATIONS",
        {
            _REAL_VERSION + 1: (f"CREATE TABLE {SCHEMA}.test_migrated (step int NOT NULL)",),
            _REAL_VERSION + 2: (f"INSERT INTO {SCHEMA}.test_migrated (step) VALUES ({_REAL_VERSION + 2})",),
        },
    )
    conn = await asyncpg.connect(**connect_kwargs(pg_settings))
    try:
        await _stamp(conn, _REAL_VERSION)
        await ensure_queue_schema(conn)

        assert await _stored_version(conn) == str(_REAL_VERSION + 2)
        assert await conn.fetchval(f"SELECT array_agg(step) FROM {SCHEMA}.test_migrated") == [_REAL_VERSION + 2]

        # Already current: a second run applies nothing.
        await ensure_queue_schema(conn)
        assert await conn.fetchval(f"SELECT count(*) FROM {SCHEMA}.test_migrated") == 1
    finally:
        await conn.execute(f"DROP TABLE IF EXISTS {SCHEMA}.test_migrated")
        await _stamp(conn, _REAL_VERSION)
        await conn.close()


# ---------------------------------------------------------------------------
# The real thing
# ---------------------------------------------------------------------------


async def test_an_index_runs_end_to_end_on_a_postgres_queue(pg_settings, tmp_path) -> None:
    """`index_project` over the embedded graph with the queue in Postgres: entities land, the
    stream drains, and nothing is left pending -- the path `atlas index` takes."""
    root = tmp_path / "test_pgproj"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "models.py").write_text(
        'class User:\n    """A user."""\n\n    def save(self):\n        return helper()\n\n\n'
        "def helper():\n    return 1\n",
        encoding="utf-8",
    )
    settings = AtlasSettings(
        project_root=root,
        backend=BackendSettings(graph={"sqlite": {}}, queue={"postgres": pg_settings}),
        embeddings=EmbeddingSettings(enabled=False),
    )
    async with use_backends(settings) as backends:
        graph, bus = backends.graph, backends.bus
        assert isinstance(bus, PostgresEventBus)
        await graph.ensure_schema()
        result = await index_project(settings, graph, bus, drain_timeout_s=60.0, limiter=backends.limiter)  # ty: ignore[invalid-argument-type]

        assert result.drained, "the pipeline did not drain through the Postgres queue"
        assert result.files_published >= 2
        assert await graph.count_entities(derive_project_name(root)) >= 3
        assert await bus.stream_group_info(TOPIC, "ast") == {"pending": 0, "lag": 0}


async def test_an_index_holds_one_pool_and_one_listener(pg_settings, tmp_path, monkeypatch) -> None:
    """A whole index -- bus, lease, consumers and the embedding rate limiter -- holds at most one
    pool plus one LISTEN connection to the queue database.

    Every `EmbedClient` used to build its own limiter, and every limiter its own pool beside
    the bus's, so the count followed the number of places that built an embedding client.
    Embeddings are on (stubbed at the provider boundary) so the limiter takes its row locks
    while the bus is busy.

    The pool is shrunk for the run so the bound does not depend on how much load a small
    corpus can generate: at 4, two independent pools and a listener can reach 9, one shared
    pool and a listener cannot pass 5. Measured before the change, with the bus and limiter
    each holding a pool, the same run peaked at 10 connections.
    """
    from code_atlas.bench import stub_provider

    monkeypatch.setattr(postgres_queue, "_POOL_MAX_SIZE", 4)
    root = tmp_path / "test_pgconn"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    for i in range(30):
        (root / "pkg" / f"m{i}.py").write_text(
            f'class K{i}:\n    """Class {i}."""\n\n    def run(self):\n        return f{i}()\n\n\n'
            f'def f{i}():\n    """Function {i}."""\n    return {i}\n',
            encoding="utf-8",
        )
    settings = AtlasSettings(
        project_root=root,
        backend=BackendSettings(graph={"sqlite": {}}, queue={"postgres": pg_settings}),
        embeddings=EmbeddingSettings(dimension=16, batch_size=2, max_concurrency=8),
    )
    # Named per process on the pool and the listener, so this counts exactly our connections.
    application_name = PostgresConnections(pg_settings).application_name

    probe = await asyncpg.connect(**connect_kwargs(pg_settings))
    peak = 0
    stop = asyncio.Event()

    async def sample() -> None:
        nonlocal peak
        while not stop.is_set():
            count = await probe.fetchval(
                "SELECT count(*) FROM pg_stat_activity WHERE application_name = $1", application_name
            )
            peak = max(peak, int(count))
            await asyncio.sleep(0.01)

    sampler = asyncio.create_task(sample())
    try:
        with stub_provider(16):
            async with use_backends(settings) as backends:
                graph = backends.graph
                await graph.ensure_schema()
                result = await index_project(
                    settings,
                    graph,  # ty: ignore[invalid-argument-type]
                    backends.bus,  # ty: ignore[invalid-argument-type]
                    drain_timeout_s=120.0,
                    limiter=backends.limiter,
                )
                assert result.drained
                assert (await graph.count_embeddings_by_project()).get(derive_project_name(root), 0) > 0, (
                    "nothing was embedded, so the limiter was never exercised"
                )
    finally:
        stop.set()
        await sampler
        await probe.close()

    assert peak >= 2, f"saw {peak} connection(s): the pool and the listener were never both open"
    assert peak <= postgres_queue._POOL_MAX_SIZE + 1, (
        f"{peak} connections for one process: more than one pool ({postgres_queue._POOL_MAX_SIZE}) plus one listener"
    )
