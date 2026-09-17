"""The three event buses must behave alike wherever the pipeline depends on it.

`EventBus` (Valkey), `SqliteEventBus` and `PostgresEventBus` are hand-written mirrors of
one interface, and the consumers in `indexing/consumers.py` are written against the
Valkey semantics. Each scenario here runs against all three, so a mirror that drifts
fails by name rather than as a pipeline that quietly stops draining.

Only behaviour the three genuinely share is asserted here. Where they differ on purpose
-- SQLite's `flush` deletes deliveries and its `drop_consumer` is a no-op, while Valkey
and Postgres keep pending entries with empty fields and have real registrations -- the
Postgres side is pinned in `test_postgres_queue.py` instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from code_atlas.backends.postgres_queue import PostgresConnections, PostgresEventBus
from code_atlas.backends.sqlite_queue import SqliteEventBus
from code_atlas.events import ConsumerGroupMissingError, EventBus, FileChanged, Topic, decode_event, redis_client
from code_atlas.indexing.consumers import BatchPolicy, TierConsumer
from code_atlas.search.ratelimit import RateBudget, RateLimiter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

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


def _id_key(msg_id: bytes) -> tuple[int, int]:
    """The parse `consumers._stream_id_key` performs: an id that fails it breaks dedup."""
    head, _, seq = msg_id.decode().partition("-")
    return int(head), int(seq)


@pytest.fixture(params=["valkey", "sqlite", "postgres"])
async def make_bus(request, tmp_path) -> AsyncIterator[Callable[[str], Any]]:
    """A factory of buses on one backend: ``make_bus(project)``.

    Calling it twice with the same project gives two clients of the same queue -- two
    processes, as far as the backend can tell. SQLite keeps a file per project, which is
    how that backend isolates projects, so the project picks the file.

    Each network bus gets connections of its own, because two processes would: what gets
    closed at the end is those connections, which the buses only borrow.
    """
    opened: list[Any] = []
    kind = request.param

    if kind == "valkey":
        redis_settings = request.getfixturevalue("settings").redis

        def factory(project: str) -> Any:
            client = redis_client(redis_settings)
            opened.append(client)
            return EventBus(client, redis_settings, project_name=project)

    elif kind == "sqlite":

        def factory(project: str) -> Any:
            bus = SqliteEventBus(tmp_path / f"{project}.sqlite3")
            opened.append(bus)
            return bus

    else:
        pg = request.getfixturevalue("pg_settings")

        def factory(project: str) -> Any:
            connections = PostgresConnections(pg)
            opened.append(connections)
            return PostgresEventBus(connections, project_name=project)

    yield factory
    for resource in opened:
        await (resource.aclose() if hasattr(resource, "aclose") else resource.close())


@pytest.fixture
async def bus(make_bus) -> Any:
    b = make_bus("test-main")
    try:
        await b.ping()
    except Exception as exc:
        pytest.skip(f"queue backend not available: {exc}")
    return b


async def test_publish_read_ack(bus) -> None:
    await bus.ensure_group(TOPIC, "g")
    ids = await bus.publish_many(TOPIC, [_event("a"), _event("b")])
    ids.append(await bus.publish(TOPIC, _event("c")))

    batch = await bus.read_batch(TOPIC, "g", "c1", count=10, block_ms=500)

    assert _paths(batch) == ["a", "b", "c"]
    assert [mid for mid, _ in batch] == ids
    assert [_id_key(i) for i in ids] == sorted(_id_key(i) for i in ids), "ids must grow in publish order"
    assert await bus.ack(TOPIC, "g", *ids) == 3
    assert await bus.read_pending(TOPIC, "g", "c1", count=10) == []
    assert await bus.read_batch(TOPIC, "g", "c1", count=10, block_ms=50) == []


async def test_pending_replays_only_to_its_own_consumer(bus) -> None:
    await bus.ensure_group(TOPIC, "g")
    await bus.publish_many(TOPIC, [_event(f"m{i}") for i in range(4)])

    mine = await bus.read_batch(TOPIC, "g", "c1", count=2, block_ms=500)
    theirs = await bus.read_batch(TOPIC, "g", "c2", count=2, block_ms=500)

    replay = await bus.read_pending(TOPIC, "g", "c1", count=10)
    assert [mid for mid, _ in replay] == [mid for mid, _ in mine]
    assert _paths(replay) == _paths(mine)
    assert {mid for mid, _ in replay}.isdisjoint({mid for mid, _ in theirs})

    await bus.ack(TOPIC, "g", *(mid for mid, _ in mine))
    assert await bus.read_pending(TOPIC, "g", "c1", count=10) == []


async def test_reclaim_takes_only_idle_entries(bus) -> None:
    await bus.ensure_group(TOPIC, "g")
    await bus.publish(TOPIC, _event("orphan"))
    dead = await bus.read_batch(TOPIC, "g", "dead", count=1, block_ms=500)
    assert len(dead) == 1

    assert await bus.reclaim_abandoned(TOPIC, "g", "live", min_idle_ms=60_000, count=10) == []

    await asyncio.sleep(0.3)
    adopted = await bus.reclaim_abandoned(TOPIC, "g", "live", min_idle_ms=150, count=10)
    assert [mid for mid, _ in adopted] == [mid for mid, _ in dead]
    assert _paths(adopted) == ["orphan"]
    assert [mid for mid, _ in await bus.read_pending(TOPIC, "g", "live", count=10)] == [mid for mid, _ in dead]


async def test_concurrent_consumers_never_share_a_message(bus) -> None:
    await bus.ensure_group(TOPIC, "g")
    await bus.publish_many(TOPIC, [_event(f"m{i}") for i in range(40)])

    batches = await asyncio.gather(*(bus.read_batch(TOPIC, "g", f"c{i}", count=4, block_ms=200) for i in range(12)))

    seen = [mid for batch in batches for mid, _ in batch]
    assert len(seen) == len(set(seen)), "one message was handed to two consumers"
    assert len(seen) == 40


async def test_group_info_counts_pending_and_lag(bus) -> None:
    assert await bus.stream_group_info(TOPIC, "never-created") == {"pending": 0, "lag": 0}

    await bus.ensure_group(TOPIC, "g")
    await bus.publish_many(TOPIC, [_event("a"), _event("b"), _event("c")])
    assert await bus.stream_group_info(TOPIC, "g") == {"pending": 0, "lag": 3}

    batch = await bus.read_batch(TOPIC, "g", "c1", count=2, block_ms=500)
    assert await bus.stream_group_info(TOPIC, "g") == {"pending": 2, "lag": 1}

    await bus.ack(TOPIC, "g", batch[0][0])
    multi = await bus.stream_group_info_multi([(TOPIC, "g"), (Topic.EMBED_DIRTY, "embed")])
    assert multi == [{"pending": 1, "lag": 1}, {"pending": 0, "lag": 0}]


async def test_a_never_published_stream_reads_as_an_empty_group(bus) -> None:
    """Single and multi answer alike for a stream nothing has ever touched.

    The drain wait asks about every stage at once, before a quiet stage has a stream. Valkey's
    pipelined `XINFO GROUPS` used to raise "no such key" there while the single query
    reported zero backlog, so a caller's answer depended on which of the two it used.
    """
    queries = [(TOPIC, "never"), (Topic.EMBED_DIRTY, "never-either")]

    assert await bus.stream_group_info(TOPIC, "never") == {"pending": 0, "lag": 0}
    assert await bus.stream_group_info_multi(queries) == [{"pending": 0, "lag": 0}, {"pending": 0, "lag": 0}]

    # A missing stream beside a live one must not poison the live answer.
    await bus.ensure_group(TOPIC, "g")
    await bus.publish(TOPIC, _event("a"))
    assert await bus.stream_group_info_multi([(TOPIC, "g"), (Topic.EMBED_DIRTY, "never")]) == [
        {"pending": 0, "lag": 1},
        {"pending": 0, "lag": 0},
    ]


async def test_flush_empties_the_stream_and_keeps_the_group(bus) -> None:
    await bus.ensure_group(TOPIC, "g")
    await bus.publish_many(TOPIC, [_event("old1"), _event("old2")])

    await bus.flush()

    assert await bus.read_batch(TOPIC, "g", "c1", count=10, block_ms=50) == []
    await bus.publish(TOPIC, _event("new"))
    assert _paths(await bus.read_batch(TOPIC, "g", "c1", count=10, block_ms=500)) == ["new"]


async def test_lease_is_exclusive_across_two_clients(make_bus) -> None:
    first, second = make_bus("test-main"), make_bus("test-main")
    try:
        await first.ping()
    except Exception as exc:
        pytest.skip(f"queue backend not available: {exc}")

    assert await first.acquire_indexer_lease("owner-a", 60_000) is True
    assert await second.acquire_indexer_lease("owner-b", 60_000) is False
    assert await second.read_indexer_lease() == "owner-a"

    assert await second.renew_indexer_lease("owner-b", 60_000) is False
    assert await second.release_indexer_lease("owner-b") is False
    assert await first.renew_indexer_lease("owner-a", 60_000) is True

    assert await second.force_acquire_indexer_lease("owner-b", 60_000) is True
    assert await first.renew_indexer_lease("owner-a", 60_000) is False, "a displaced holder must fail to renew"
    assert await first.read_indexer_lease() == "owner-b"

    assert await second.release_indexer_lease("owner-b") is True
    assert await first.read_indexer_lease() is None


async def test_an_expired_lease_passes_on(make_bus) -> None:
    first, second = make_bus("test-main"), make_bus("test-main")
    try:
        await first.ping()
    except Exception as exc:
        pytest.skip(f"queue backend not available: {exc}")

    assert await first.acquire_indexer_lease("owner-a", 300) is True
    assert await second.acquire_indexer_lease("owner-b", 60_000) is False
    await asyncio.sleep(0.6)
    assert await second.read_indexer_lease() is None
    assert await second.acquire_indexer_lease("owner-b", 60_000) is True


async def test_projects_are_isolated(make_bus) -> None:
    suffix = uuid.uuid4().hex[:6]
    alpha, beta = make_bus(f"test-alpha-{suffix}"), make_bus(f"test-beta-{suffix}")
    try:
        await alpha.ping()
    except Exception as exc:
        pytest.skip(f"queue backend not available: {exc}")

    for b in (alpha, beta):
        await b.ensure_group(TOPIC, "g")
    await alpha.publish(TOPIC, _event("alpha-only"))

    assert await beta.read_batch(TOPIC, "g", "c1", count=10, block_ms=50) == []
    assert await beta.stream_group_info(TOPIC, "g") == {"pending": 0, "lag": 0}

    assert await alpha.acquire_indexer_lease("a", 60_000) is True
    assert await beta.acquire_indexer_lease("b", 60_000) is True, "one project's lease blocked another's"

    await beta.flush()
    assert _paths(await alpha.read_batch(TOPIC, "g", "c1", count=10, block_ms=500)) == ["alpha-only"]


async def test_removing_a_projects_queue_data_spares_every_other_project(make_bus) -> None:
    """`atlas project rm` empties one project's queue and nothing else.

    SQLite answers ``None`` and keeps everything: its file is shared by every project of a
    checkout and has no project column, so there is no subset it could remove.
    """
    suffix = uuid.uuid4().hex[:6]
    gone, kept = make_bus(f"test-gone-{suffix}"), make_bus(f"test-kept-{suffix}")
    try:
        await gone.ping()
    except Exception as exc:
        pytest.skip(f"queue backend not available: {exc}")

    for b in (gone, kept):
        await b.ensure_group(TOPIC, "g")
        await b.publish_many(TOPIC, [_event("a"), _event("b")])
        assert len(await b.read_batch(TOPIC, "g", "c1", count=1, block_ms=500)) == 1
    assert await kept.acquire_indexer_lease("kept-owner", 60_000) is True

    removed = await gone.delete_project_queue()

    assert await kept.stream_group_info(TOPIC, "g") == {"pending": 1, "lag": 1}
    assert len(await kept.read_pending(TOPIC, "g", "c1", count=10)) == 1
    assert await kept.read_indexer_lease() == "kept-owner"
    if removed is None:
        assert isinstance(gone, SqliteEventBus), "only the embedded queue may decline to remove anything"
        assert await gone.stream_group_info(TOPIC, "g") == {"pending": 1, "lag": 1}
    else:
        assert removed > 0
        assert await gone.stream_group_info(TOPIC, "g") == {"pending": 0, "lag": 0}


async def test_removing_a_project_named_rl_leaves_the_rate_limiter_keys(settings) -> None:
    """The limiter's shared keys are `{prefix}:rl:{model}:*` -- exactly what a
    `{prefix}:{project}:*` pattern would match for a project called `rl`."""
    async with redis_client(settings.redis) as client:
        bus = EventBus(client, settings.redis, project_name="rl")
        try:
            await bus.ping()
        except Exception as exc:
            pytest.skip(f"Valkey not available: {exc}")
        await RateLimiter(client, stream_prefix=settings.redis.stream_prefix).penalize(RateBudget("m", 0, 0))
        scale_key = f"{settings.redis.stream_prefix}:rl:m:scale"
        assert await client.exists(scale_key) == 1

        await bus.ensure_group(TOPIC, "g")
        await bus.publish(TOPIC, _event("a"))
        assert await bus.delete_project_queue() == 1

        assert await client.exists(scale_key) == 1, "removing project 'rl' deleted the shared rate-limit state"
        assert await bus.stream_group_info(TOPIC, "g") == {"pending": 0, "lag": 0}


async def test_reading_a_missing_group_raises_on_every_bus(bus) -> None:
    """Never created: Valkey answers NOGROUP, and the SQL buses must say the same rather than
    hand back an empty batch that looks exactly like an idle stream."""
    await bus.publish(TOPIC, _event("a"))

    with pytest.raises(ConsumerGroupMissingError):
        await bus.read_batch(TOPIC, "never", "c1", count=10, block_ms=50)
    with pytest.raises(ConsumerGroupMissingError):
        await bus.read_pending(TOPIC, "never", "c1", count=10)

    await bus.ensure_group(TOPIC, "never")
    assert _paths(await bus.read_batch(TOPIC, "never", "c1", count=10, block_ms=500)) == ["a"]


async def test_a_consumer_whose_group_is_deleted_fails_loudly_and_recovers(bus) -> None:
    """`atlas project rm` under an idle daemon: its consumer must crash (the supervisor logs and
    restarts it) instead of polling a group that no longer exists until the process restarts."""
    consumer = _NullConsumer(bus, TOPIC, "g", "c1", BatchPolicy(time_window_s=0.05, max_batch_size=5))
    run = asyncio.create_task(consumer.run())
    try:
        deadline = asyncio.get_running_loop().time() + 10
        while not await _group_exists(bus):
            assert not run.done(), "the consumer stopped before it created its group"
            assert asyncio.get_running_loop().time() < deadline, "the consumer never created its group"
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.3)  # into its read loop

        if await bus.delete_project_queue() is None:
            assert isinstance(bus, SqliteEventBus)
            pytest.skip("the embedded queue has no per-project data to delete; its missing-group read is covered above")

        with pytest.raises(ConsumerGroupMissingError):
            await asyncio.wait_for(run, timeout=10)
    finally:
        consumer.stop()
        if not run.done():
            run.cancel()
            with contextlib.suppress(asyncio.CancelledError, ConsumerGroupMissingError):
                await run

    # The supervisor's restart: run() re-creates the group and consumes again.
    consumer = _NullConsumer(bus, TOPIC, "g", "c1", BatchPolicy(time_window_s=0.05, max_batch_size=5))
    run = asyncio.create_task(consumer.run())
    try:
        await bus.publish(TOPIC, _event("after"))
        deadline = asyncio.get_running_loop().time() + 10
        while not consumer.seen:
            assert not run.done(), f"the restarted consumer stopped: {run.exception() if run.done() else ''}"
            assert asyncio.get_running_loop().time() < deadline, "the restarted consumer never read"
            await asyncio.sleep(0.05)
    finally:
        consumer.stop()
        await asyncio.wait_for(run, timeout=10)


class _NullConsumer(TierConsumer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.seen: list[str] = []

    async def process_batch(self, events, batch_id):
        self.seen.extend(e.path for e in events if isinstance(e, FileChanged))


async def _group_exists(bus: Any) -> bool:
    try:
        await bus.read_pending(TOPIC, "g", "probe", count=1)
    except ConsumerGroupMissingError:
        return False
    return True
