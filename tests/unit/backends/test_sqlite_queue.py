"""Unit tests for SqliteEventBus — the in-process fallback event queue (no infrastructure needed)."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
