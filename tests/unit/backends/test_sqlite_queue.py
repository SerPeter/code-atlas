"""Unit tests for SqliteEventBus — the in-process fallback event queue (no infrastructure needed)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
import time_machine

from code_atlas.backends.sqlite_queue import SqliteEventBus
from code_atlas.events import FileChanged, Topic, decode_event

if TYPE_CHECKING:
    from pathlib import Path


def _event(path: str = "a.py") -> FileChanged:
    return FileChanged(path=path, change_type="modified")


def _decode_file_changed(fields: dict[bytes, bytes]) -> FileChanged:
    """Decode a FILE_CHANGED event, narrowing ``decode_event``'s ``FileChanged | EmbedDirty``
    union — every event in this module is published as FILE_CHANGED, so it's always a
    ``FileChanged`` at runtime; the isinstance assert makes that explicit for ty too.
    """
    event = decode_event(Topic.FILE_CHANGED, fields)
    assert isinstance(event, FileChanged)
    return event


# ---------------------------------------------------------------------------
# publish + read_batch
# ---------------------------------------------------------------------------


class TestPublishAndReadBatch:
    async def test_read_batch_delivers_published_event(self, tmp_path: Path) -> None:
        bus = SqliteEventBus(tmp_path / "queue.sqlite3")
        await bus.ensure_group(Topic.FILE_CHANGED, "ast")
        await bus.publish(Topic.FILE_CHANGED, _event("a.py"))

        batch = await bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", block_ms=100)

        assert len(batch) == 1
        msg_id, fields = batch[0]
        event = _decode_file_changed(fields)
        assert event.path == "a.py"
        assert event.change_type == "modified"
        assert msg_id  # non-empty
        await bus.close()

    async def test_publish_many_and_read_batch_delivers_all(self, tmp_path: Path) -> None:
        bus = SqliteEventBus(tmp_path / "queue.sqlite3")
        await bus.ensure_group(Topic.FILE_CHANGED, "ast")
        ids = await bus.publish_many(Topic.FILE_CHANGED, [_event("a.py"), _event("b.py"), _event("c.py")])
        assert len(ids) == 3

        batch = await bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", count=10, block_ms=100)
        assert len(batch) == 3
        paths = {_decode_file_changed(fields).path for _mid, fields in batch}
        assert paths == {"a.py", "b.py", "c.py"}
        await bus.close()

    async def test_read_batch_does_not_redeliver_already_claimed_messages(self, tmp_path: Path) -> None:
        bus = SqliteEventBus(tmp_path / "queue.sqlite3")
        await bus.ensure_group(Topic.FILE_CHANGED, "ast")
        await bus.publish(Topic.FILE_CHANGED, _event("a.py"))
        first = await bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", block_ms=100)
        assert len(first) == 1

        second = await bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", block_ms=50)
        assert second == []
        await bus.close()

    async def test_read_batch_returns_empty_when_nothing_published(self, tmp_path: Path) -> None:
        bus = SqliteEventBus(tmp_path / "queue.sqlite3")
        await bus.ensure_group(Topic.FILE_CHANGED, "ast")
        batch = await bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", block_ms=50)
        assert batch == []
        await bus.close()


# ---------------------------------------------------------------------------
# Message id format — "<int>-<int>" bytes (consumers.py._stream_id_key contract)
# ---------------------------------------------------------------------------


class TestMessageIdFormat:
    async def test_publish_returns_int_dash_int_bytes(self, tmp_path: Path) -> None:
        bus = SqliteEventBus(tmp_path / "queue.sqlite3")
        msg_id = await bus.publish(Topic.FILE_CHANGED, _event())
        assert isinstance(msg_id, bytes)
        ms, _, seq = msg_id.partition(b"-")
        assert ms.isdigit()
        assert seq.isdigit()
        await bus.close()

    async def test_read_batch_ids_are_int_dash_int_bytes(self, tmp_path: Path) -> None:
        bus = SqliteEventBus(tmp_path / "queue.sqlite3")
        await bus.ensure_group(Topic.FILE_CHANGED, "ast")
        await bus.publish_many(Topic.FILE_CHANGED, [_event("a.py"), _event("b.py")])
        batch = await bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", block_ms=100)
        for msg_id, _fields in batch:
            ms, _, seq = msg_id.partition(b"-")
            assert ms.isdigit()
            assert seq.isdigit()
        await bus.close()


# ---------------------------------------------------------------------------
# read_pending — PEL replay across a simulated process restart
# ---------------------------------------------------------------------------


class TestReadPendingCrashRecovery:
    async def test_read_pending_replays_after_restart_without_ack(self, tmp_path: Path) -> None:
        db_path = tmp_path / "queue.sqlite3"

        bus1 = SqliteEventBus(db_path)
        await bus1.ensure_group(Topic.FILE_CHANGED, "ast")
        await bus1.publish(Topic.FILE_CHANGED, _event("a.py"))
        delivered = await bus1.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", block_ms=100)
        assert len(delivered) == 1
        # Simulate a crash: never ack, drop the connection without closing cleanly.
        await bus1.close()

        # New process, same db file, same (topic, group, consumer) identity.
        bus2 = SqliteEventBus(db_path)
        pending = await bus2.read_pending(Topic.FILE_CHANGED, "ast", "ast-0")
        assert len(pending) == 1
        assert pending[0][0] == delivered[0][0]
        event = _decode_file_changed(pending[0][1])
        assert event.path == "a.py"
        await bus2.close()

    async def test_read_pending_empty_for_different_consumer(self, tmp_path: Path) -> None:
        db_path = tmp_path / "queue.sqlite3"
        bus = SqliteEventBus(db_path)
        await bus.ensure_group(Topic.FILE_CHANGED, "ast")
        await bus.publish(Topic.FILE_CHANGED, _event("a.py"))
        await bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", block_ms=100)

        pending = await bus.read_pending(Topic.FILE_CHANGED, "ast", "ast-1")
        assert pending == []
        await bus.close()


# ---------------------------------------------------------------------------
# ack
# ---------------------------------------------------------------------------


class TestAck:
    async def test_ack_removes_from_pending(self, tmp_path: Path) -> None:
        bus = SqliteEventBus(tmp_path / "queue.sqlite3")
        await bus.ensure_group(Topic.FILE_CHANGED, "ast")
        await bus.publish(Topic.FILE_CHANGED, _event("a.py"))
        delivered = await bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", block_ms=100)
        msg_id = delivered[0][0]

        assert await bus._debug_pending_count(Topic.FILE_CHANGED, "ast") == 1
        acked = await bus.ack(Topic.FILE_CHANGED, "ast", msg_id)
        assert acked == 1
        assert await bus._debug_pending_count(Topic.FILE_CHANGED, "ast") == 0

        pending = await bus.read_pending(Topic.FILE_CHANGED, "ast", "ast-0")
        assert pending == []
        await bus.close()

    async def test_ack_of_unknown_id_returns_zero(self, tmp_path: Path) -> None:
        bus = SqliteEventBus(tmp_path / "queue.sqlite3")
        acked = await bus.ack(Topic.FILE_CHANGED, "ast", b"9999-0")
        assert acked == 0
        await bus.close()


# ---------------------------------------------------------------------------
# flush
# ---------------------------------------------------------------------------


class TestFlush:
    async def test_flush_clears_messages_but_group_still_usable(self, tmp_path: Path) -> None:
        bus = SqliteEventBus(tmp_path / "queue.sqlite3")
        await bus.ensure_group(Topic.FILE_CHANGED, "ast")
        await bus.publish(Topic.FILE_CHANGED, _event("a.py"))
        delivered = await bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", block_ms=100)
        assert len(delivered) == 1

        await bus.flush()

        assert await bus._debug_pending_count(Topic.FILE_CHANGED, "ast") == 0
        assert await bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", block_ms=50) == []

        # Group registration survives flush — a fresh ensure_group + publish/read still works.
        await bus.ensure_group(Topic.FILE_CHANGED, "ast")
        await bus.publish(Topic.FILE_CHANGED, _event("b.py"))
        batch = await bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", block_ms=100)
        assert len(batch) == 1
        assert _decode_file_changed(batch[0][1]).path == "b.py"
        await bus.close()


# ---------------------------------------------------------------------------
# stream_group_info / stream_group_info_multi
# ---------------------------------------------------------------------------


class TestStreamGroupInfo:
    async def test_unknown_group_reports_zero(self, tmp_path: Path) -> None:
        bus = SqliteEventBus(tmp_path / "queue.sqlite3")
        info = await bus.stream_group_info(Topic.FILE_CHANGED, "ast")
        assert info == {"pending": 0, "lag": 0}
        await bus.close()

    async def test_pending_and_lag_counts(self, tmp_path: Path) -> None:
        bus = SqliteEventBus(tmp_path / "queue.sqlite3")
        await bus.ensure_group(Topic.FILE_CHANGED, "ast")
        await bus.publish_many(Topic.FILE_CHANGED, [_event("a.py"), _event("b.py"), _event("c.py")])

        # Claim 2 of 3 — 1 remains undelivered (lag), the 2 claimed are pending (unacked).
        delivered = await bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", count=2, block_ms=100)
        assert len(delivered) == 2

        info = await bus.stream_group_info(Topic.FILE_CHANGED, "ast")
        assert info == {"pending": 2, "lag": 1}

        await bus.ack(Topic.FILE_CHANGED, "ast", delivered[0][0])
        info = await bus.stream_group_info(Topic.FILE_CHANGED, "ast")
        assert info == {"pending": 1, "lag": 1}
        await bus.close()

    async def test_stream_group_info_multi_matches_individual_calls(self, tmp_path: Path) -> None:
        bus = SqliteEventBus(tmp_path / "queue.sqlite3")
        await bus.ensure_group(Topic.FILE_CHANGED, "ast")
        await bus.ensure_group(Topic.EMBED_DIRTY, "embed")
        await bus.publish(Topic.FILE_CHANGED, _event("a.py"))

        results = await bus.stream_group_info_multi([(Topic.FILE_CHANGED, "ast"), (Topic.EMBED_DIRTY, "embed")])
        assert results == [
            await bus.stream_group_info(Topic.FILE_CHANGED, "ast"),
            await bus.stream_group_info(Topic.EMBED_DIRTY, "embed"),
        ]
        await bus.close()


class TestIndexerLeaseExpiry:
    """A lease outliving its holder is the whole reason the TTL exists.

    Every recovery path in the system rests on this: the daemon's catch-up waits 90s
    because a dead holder's lease expires inside 60, `atlas index` waits rather than
    skipping, and --force exists only for the window before expiry. None of it was
    covered, because covering it meant a test that sleeps for a minute.

    time-machine moves the clock instead. The lease is real and the SQLite row is real;
    only `time.time()` is a fiction.
    """

    @staticmethod
    def _bus(tmp_path: Path) -> SqliteEventBus:
        return SqliteEventBus(tmp_path / "queue.sqlite3")

    async def test_a_live_lease_blocks_a_second_holder(self, tmp_path: Path) -> None:
        bus = self._bus(tmp_path)
        try:
            assert await bus.acquire_indexer_lease("first", 60_000) is True
            assert await bus.acquire_indexer_lease("second", 60_000) is False
            assert await bus.read_indexer_lease() == "first"
        finally:
            await bus.close()

    async def test_an_expired_lease_passes_to_the_next_taker(self, tmp_path: Path) -> None:
        """The dead-holder case: without it the 60s TTL was an untested assumption that
        three separate waiting strategies had been built on top of."""
        bus = self._bus(tmp_path)
        try:
            assert await bus.acquire_indexer_lease("dead-holder", 60_000) is True

            with time_machine.travel(datetime.now(UTC) + timedelta(seconds=61), tick=False):
                assert await bus.read_indexer_lease() is None, "an expired lease must read as free"
                assert await bus.acquire_indexer_lease("next", 60_000) is True
                assert await bus.read_indexer_lease() == "next"
        finally:
            await bus.close()

    async def test_a_lease_just_short_of_expiry_still_holds(self, tmp_path: Path) -> None:
        """The other side of the boundary, so the test above cannot pass by the TTL being
        ignored altogether."""
        bus = self._bus(tmp_path)
        try:
            assert await bus.acquire_indexer_lease("holder", 60_000) is True

            with time_machine.travel(datetime.now(UTC) + timedelta(seconds=59), tick=False):
                assert await bus.acquire_indexer_lease("interloper", 60_000) is False
                assert await bus.read_indexer_lease() == "holder"
        finally:
            await bus.close()

    async def test_renewal_pushes_the_expiry_out(self, tmp_path: Path) -> None:
        bus = self._bus(tmp_path)
        try:
            await bus.acquire_indexer_lease("holder", 60_000)

            with time_machine.travel(datetime.now(UTC) + timedelta(seconds=30), tick=False):
                assert await bus.renew_indexer_lease("holder", 60_000) is True

            # 61s after the original take, but only 31s after the renewal.
            with time_machine.travel(datetime.now(UTC) + timedelta(seconds=61), tick=False):
                assert await bus.acquire_indexer_lease("interloper", 60_000) is False
                assert await bus.read_indexer_lease() == "holder"
        finally:
            await bus.close()

    async def test_renewal_by_a_stranger_is_refused(self, tmp_path: Path) -> None:
        """Compare-and-set: a process that stalled past its TTL must not renew a lease
        that has since passed to someone else."""
        bus = self._bus(tmp_path)
        try:
            await bus.acquire_indexer_lease("holder", 60_000)
            assert await bus.renew_indexer_lease("stranger", 60_000) is False
            assert await bus.read_indexer_lease() == "holder"
        finally:
            await bus.close()


class TestAsyncContextManager:
    """Closing has to survive the exit paths nobody remembers.

    The bug that prompted this: all four infra fixtures called `pytest.skip()` between
    constructing a client and closing it. `Skipped` is a BaseException, so it is not even
    caught by `except Exception`, and every skipped run abandoned a live connection --
    surfacing later as a ResourceWarning blamed on whichever unrelated test the GC
    happened to interrupt.
    """

    async def test_exit_closes_on_the_happy_path(self, tmp_path: Path) -> None:
        async with SqliteEventBus(tmp_path / "queue.sqlite3") as bus:
            await bus.ping()
            assert bus._conn is not None
        assert bus._conn is None

    async def test_exit_closes_when_the_body_raises(self, tmp_path: Path) -> None:
        bus = SqliteEventBus(tmp_path / "queue.sqlite3")

        async def use_and_raise() -> None:
            async with bus:
                await bus.ping()
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await use_and_raise()

        assert bus._conn is None, "an exception must not leak the connection"

    async def test_exit_closes_on_a_base_exception(self, tmp_path: Path) -> None:
        """pytest.skip raises BaseException, which is exactly the path that leaked.

        A bare `try/except Exception` around the body would not have caught this; the
        context manager does.
        """

        class _SkipLike(BaseException):
            pass

        bus = SqliteEventBus(tmp_path / "queue.sqlite3")

        async def use_and_skip() -> None:
            async with bus:
                await bus.ping()
                raise _SkipLike

        with pytest.raises(_SkipLike):
            await use_and_skip()

        assert bus._conn is None, "a BaseException must not leak the connection either"

    def test_every_client_supports_the_protocol(self) -> None:
        """Derived from the classes, not a hand-written list -- a fifth backend that
        forgets the protocol should fail here rather than leak in production."""
        from code_atlas.backends.sqlite_graph import SqliteGraphClient
        from code_atlas.events import EventBus
        from code_atlas.graph.client import GraphClient

        for cls in (GraphClient, SqliteGraphClient, EventBus, SqliteEventBus):
            assert hasattr(cls, "__aenter__"), f"{cls.__name__} cannot be used with async with"
            assert hasattr(cls, "__aexit__"), f"{cls.__name__} cannot be used with async with"
