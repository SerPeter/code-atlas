"""Integration tests for the event-driven pipeline.

Requires Memgraph + Valkey (provided by conftest fixtures).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from code_atlas.events import (
    EmbedDirty,
    EntityRef,
    EventBus,
    FileChanged,
    Topic,
    decode_event,
)
from code_atlas.indexing.consumers import _MAX_BATCH_FAILURES, ASTConsumer, BatchPolicy, EmbedConsumer
from code_atlas.indexing.orchestrator import _wait_for_drain
from code_atlas.search.embeddings import EmbedCache, build_embed_text

if TYPE_CHECKING:
    from code_atlas.events import Event
    from code_atlas.graph.client import GraphClient
    from code_atlas.settings import AtlasSettings

# All tests in this module require a live Redis/Valkey
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def _clean_streams(event_bus: EventBus):
    """Delete test streams before and after each test to avoid state leakage."""
    for topic in Topic:
        key = f"{event_bus._prefix}:{topic.value}"
        await event_bus._redis.delete(key)
    yield
    for topic in Topic:
        key = f"{event_bus._prefix}:{topic.value}"
        await event_bus._redis.delete(key)


def _write_python_file(root, rel_path: str, content: str) -> None:
    """Write a Python file under *root* at the given relative path."""
    full = root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")


def _file_changed(settings: AtlasSettings, rel_path: str, change_type: str = "modified") -> FileChanged:
    """Build a FileChanged event with full project identity for the test project."""
    return FileChanged(
        path=rel_path,
        change_type=change_type,
        timestamp=time.time(),
        project_name=settings.project_root.resolve().name,
        project_root=str(settings.project_root),
    )


async def _pel_count(event_bus: EventBus, topic: Topic, group: str) -> int:
    """Return the number of un-ACKed (pending) messages for *group*."""
    info = await event_bus._redis.xpending(f"{event_bus._prefix}:{topic.value}", group)
    return int(info["pending"])


async def _wait_until(predicate, *, timeout_s: float = 10.0, interval_s: float = 0.1) -> None:
    """Poll *predicate* until it returns True or *timeout_s* elapses."""
    async with asyncio.timeout(timeout_s):
        while not predicate():
            await asyncio.sleep(interval_s)


# ---------------------------------------------------------------------------
# EventBus tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_clean_streams")
async def test_publish_and_consume(event_bus: EventBus) -> None:
    """Publish FileChanged events, read back via XREADGROUP, verify decode."""
    group = "test-group"
    consumer = "test-consumer"

    await event_bus.ensure_group(Topic.FILE_CHANGED, group)

    # Publish two events
    ev1 = FileChanged(path="src/main.py", change_type="modified", timestamp=1000.0)
    ev2 = FileChanged(path="src/utils.py", change_type="created", timestamp=1001.0)
    await event_bus.publish(Topic.FILE_CHANGED, ev1)
    await event_bus.publish(Topic.FILE_CHANGED, ev2)

    # Read them back
    messages = await event_bus.read_batch(Topic.FILE_CHANGED, group, consumer, count=10, block_ms=500)
    assert len(messages) == 2

    decoded = [decode_event(Topic.FILE_CHANGED, fields) for _, fields in messages]
    assert decoded[0] == ev1
    assert decoded[1] == ev2

    # ACK
    msg_ids = [mid for mid, _ in messages]
    acked = await event_bus.ack(Topic.FILE_CHANGED, group, *msg_ids)
    assert acked == 2


@pytest.mark.usefixtures("_clean_streams")
async def test_dedup_within_batch(event_bus: EventBus) -> None:
    """Same file path published multiple times — consumer should dedup to 1."""
    group = "test-dedup"
    consumer = "test-dedup-0"

    await event_bus.ensure_group(Topic.FILE_CHANGED, group)

    # Publish same path 5 times
    for i in range(5):
        await event_bus.publish(
            Topic.FILE_CHANGED,
            FileChanged(path="src/main.py", change_type="modified", timestamp=1000.0 + i),
        )

    # Read all messages
    messages = await event_bus.read_batch(Topic.FILE_CHANGED, group, consumer, count=10, block_ms=500)
    assert len(messages) == 5

    # Apply dedup logic (same as TierConsumer): latest event wins per dedup key
    pending: dict[str, FileChanged] = {}
    for _, fields in messages:
        event = decode_event(Topic.FILE_CHANGED, fields)
        assert isinstance(event, FileChanged)
        pending[event.path] = event

    assert len(pending) == 1
    assert pending["src/main.py"].timestamp == 1004.0


# ---------------------------------------------------------------------------
# AST consumer tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_clean_streams")
async def test_ast_consumes_file_changed(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """AST consumer processes FileChanged from the file-changed topic and writes entities to graph."""
    await graph_client.ensure_schema()

    # Write a Python file for the AST consumer to parse
    _write_python_file(settings.project_root, "hello.py", "def greet(name: str) -> str:\n    return f'Hello {name}'\n")

    consumer = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )

    # Publish a FileChanged and let the consumer process it
    project_name = settings.project_root.resolve().name
    await event_bus.publish(
        Topic.FILE_CHANGED,
        FileChanged(
            path="hello.py",
            change_type="created",
            timestamp=time.time(),
            project_name=project_name,
            project_root=str(settings.project_root),
        ),
    )

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(1.0)
    consumer.stop()
    await asyncio.wait_for(task, timeout=5.0)

    assert consumer.stats.files_processed >= 1
    assert consumer.stats.entities_added >= 1


@pytest.mark.usefixtures("_clean_streams")
async def test_file_hash_gate_skips_unchanged(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """Hash gate skips a file when content hasn't changed between runs."""
    await graph_client.ensure_schema()

    _write_python_file(settings.project_root, "stable.py", "X = 42\n")

    project_name = settings.project_root.resolve().name
    ev = FileChanged(
        path="stable.py",
        change_type="modified",
        timestamp=time.time(),
        project_name=project_name,
        project_root=str(settings.project_root),
    )

    # First run: processes the file and stores its hash
    c1 = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    await event_bus.publish(Topic.FILE_CHANGED, ev)
    task = asyncio.create_task(c1.run())
    await asyncio.sleep(1.0)
    c1.stop()
    await asyncio.wait_for(task, timeout=5.0)
    assert c1.stats.files_processed >= 1

    # Second run: same file, same content — should be skipped
    c2 = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    await event_bus.publish(Topic.FILE_CHANGED, ev)
    task = asyncio.create_task(c2.run())
    await asyncio.sleep(1.0)
    c2.stop()
    await asyncio.wait_for(task, timeout=5.0)

    assert c2.stats.files_skipped >= 1
    assert c2.stats.files_processed == 0


@pytest.mark.usefixtures("_clean_streams")
async def test_file_hash_gate_processes_modified(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """Hash gate allows a file through when content changes between runs."""
    await graph_client.ensure_schema()

    _write_python_file(settings.project_root, "changing.py", "X = 1\n")

    project_name = settings.project_root.resolve().name
    ev = FileChanged(
        path="changing.py",
        change_type="modified",
        timestamp=time.time(),
        project_name=project_name,
        project_root=str(settings.project_root),
    )

    # First run
    c1 = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    await event_bus.publish(Topic.FILE_CHANGED, ev)
    task = asyncio.create_task(c1.run())
    await asyncio.sleep(1.0)
    c1.stop()
    await asyncio.wait_for(task, timeout=5.0)
    assert c1.stats.files_processed >= 1

    # Modify the file
    _write_python_file(settings.project_root, "changing.py", "X = 2\nY = 3\n")

    # Second run: changed content — should process again
    c2 = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    await event_bus.publish(Topic.FILE_CHANGED, ev)
    task = asyncio.create_task(c2.run())
    await asyncio.sleep(1.0)
    c2.stop()
    await asyncio.wait_for(task, timeout=5.0)

    assert c2.stats.files_processed >= 1


@pytest.mark.usefixtures("_clean_streams")
async def test_cooldown_defers_rapid_edits(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """Per-file cooldown defers rapid re-edits so only the first is processed immediately."""
    await graph_client.ensure_schema()

    _write_python_file(settings.project_root, "rapid.py", "A = 1\n")

    project_name = settings.project_root.resolve().name

    consumer = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
        cooldown_s=60.0,  # Long cooldown — second event should be deferred
    )

    # Publish first event
    await event_bus.publish(
        Topic.FILE_CHANGED,
        FileChanged(
            path="rapid.py",
            change_type="modified",
            timestamp=time.time(),
            project_name=project_name,
            project_root=str(settings.project_root),
        ),
    )

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(1.0)

    # First event should be processed
    assert consumer.stats.files_processed >= 1
    first_processed = consumer.stats.files_processed

    # Publish a second event for the same file — should be deferred
    await event_bus.publish(
        Topic.FILE_CHANGED,
        FileChanged(
            path="rapid.py",
            change_type="modified",
            timestamp=time.time(),
            project_name=project_name,
            project_root=str(settings.project_root),
        ),
    )
    await asyncio.sleep(1.0)

    consumer.stop()
    await asyncio.wait_for(task, timeout=5.0)

    # Second event deferred — files_processed should not have increased
    assert consumer.stats.files_processed == first_processed
    assert consumer.stats.files_deferred >= 1


@pytest.mark.usefixtures("_clean_streams")
async def test_cooldown_disabled_processes_all(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """With cooldown_s=0, all events are processed immediately (reindex mode)."""
    await graph_client.ensure_schema()

    _write_python_file(settings.project_root, "nodelay.py", "Z = 1\n")

    project_name = settings.project_root.resolve().name

    consumer = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
        cooldown_s=0.0,  # No cooldown
    )

    # Publish two events for the same file
    for i in range(2):
        await event_bus.publish(
            Topic.FILE_CHANGED,
            FileChanged(
                path="nodelay.py",
                change_type="modified",
                timestamp=time.time() + i,
                project_name=project_name,
                project_root=str(settings.project_root),
            ),
        )
    # Small gap so they arrive in separate batches
    await asyncio.sleep(0.1)

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(2.0)
    consumer.stop()
    await asyncio.wait_for(task, timeout=5.0)

    # No deferral when cooldown is disabled
    assert consumer.stats.files_deferred == 0


# ---------------------------------------------------------------------------
# Pipeline durability (S7): PEL retention, cooldown deferral, poison parking
# ---------------------------------------------------------------------------


class _FlakyASTConsumer(ASTConsumer):
    """Raises on the first process_batch call, delegates afterwards."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0

    async def process_batch(self, events: list[Event], batch_id: str) -> set[str]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("first flush fails")
        return await super().process_batch(events, batch_id)


class _PoisonASTConsumer(ASTConsumer):
    """Raises whenever the batch contains poison.py — poisons every co-batched event."""

    async def process_batch(self, events: list[Event], batch_id: str) -> set[str]:
        if any(isinstance(e, FileChanged) and e.path == "poison.py" for e in events):
            raise RuntimeError("poison batch")
        return await super().process_batch(events, batch_id)


@pytest.mark.usefixtures("_clean_streams")
async def test_pel_retained_when_first_flush_fails(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """Crash-recovery messages reclaimed from the PEL survive a failed first flush.

    Before the fix the startup reclaim re-read self-ACKed the messages via
    _dedup_put; the failed flush then lost them forever (empty PEL on retry).
    """
    await graph_client.ensure_schema()

    _write_python_file(settings.project_root, "crash_a.py", "A = 1\n")
    _write_python_file(settings.project_root, "crash_b.py", "B = 2\n")

    await event_bus.ensure_group(Topic.FILE_CHANGED, "ast")
    await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "crash_a.py"))
    await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "crash_b.py"))

    # Simulated crash: deliver to ('ast', 'ast-0') without ACKing
    delivered = await event_bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", count=10, block_ms=500)
    assert len(delivered) == 2
    assert await _pel_count(event_bus, Topic.FILE_CHANGED, "ast") == 2

    consumer = _FlakyASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0.5, max_batch_size=10, block_ms=50),
    )
    task = asyncio.create_task(consumer.run())
    try:
        await _wait_until(lambda: consumer.stats.files_processed >= 2, timeout_s=15.0)
    finally:
        consumer.stop()
        await asyncio.wait_for(task, timeout=10.0)

    assert consumer.stats.files_processed >= 2
    assert await _pel_count(event_bus, Topic.FILE_CHANGED, "ast") == 0


@pytest.mark.usefixtures("_clean_streams")
async def test_cooldown_deferred_event_survives_shutdown(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """A cooldown-deferred event stays un-ACKed in the PEL across shutdown.

    Before the fix the deferred event was ACKed and held only in memory —
    stopping the consumer dropped the change forever.
    """
    await graph_client.ensure_schema()

    _write_python_file(settings.project_root, "held.py", "V = 1\n")

    c1 = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
        cooldown_s=60.0,
    )
    await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "held.py"))
    task = asyncio.create_task(c1.run())
    try:
        await _wait_until(lambda: c1.stats.files_processed >= 1)

        # Second change during the cooldown window — must be deferred, not ACKed
        _write_python_file(settings.project_root, "held.py", "V = 2\nW = 3\n")
        await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "held.py"))
        await _wait_until(lambda: c1.stats.files_deferred >= 1)
    finally:
        c1.stop()
        await asyncio.wait_for(task, timeout=10.0)

    # The deferred change survived shutdown in the PEL
    assert await _pel_count(event_bus, Topic.FILE_CHANGED, "ast") >= 1

    # A fresh consumer (same group/consumer name) reclaims and processes it
    c2 = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
        cooldown_s=0.0,
    )
    task = asyncio.create_task(c2.run())
    try:
        await _wait_until(lambda: c2.stats.files_processed >= 1)
    finally:
        c2.stop()
        await asyncio.wait_for(task, timeout=10.0)

    assert c2.stats.files_processed >= 1
    assert await _pel_count(event_bus, Topic.FILE_CHANGED, "ast") == 0


@pytest.mark.usefixtures("_clean_streams")
async def test_cooldown_deferred_event_processed_after_expiry(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """A deferred event is redelivered from the PEL and processed once the cooldown expires."""
    await graph_client.ensure_schema()

    _write_python_file(settings.project_root, "expire.py", "E = 1\n")

    consumer = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
        cooldown_s=2.0,
    )
    await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "expire.py"))
    task = asyncio.create_task(consumer.run())
    try:
        await _wait_until(lambda: consumer.stats.files_processed >= 1)

        _write_python_file(settings.project_root, "expire.py", "E = 2\nF = 3\n")
        await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "expire.py"))

        # Deferred while cooling down, then processed after the 2s cooldown expires
        await _wait_until(lambda: consumer.stats.files_processed >= 2, timeout_s=15.0)
    finally:
        consumer.stop()
        await asyncio.wait_for(task, timeout=10.0)

    assert consumer.stats.files_deferred >= 1
    assert await _pel_count(event_bus, Topic.FILE_CHANGED, "ast") == 0


@pytest.mark.usefixtures("_clean_streams")
async def test_poison_batch_parked_after_retry_cap(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """A deterministically-failing message is parked after _MAX_BATCH_FAILURES batches.

    Before the fix every merged batch failed forever and the good event was
    never processed; after, the poison message is ACKed (parked) and the good
    event processes once it lands in a poison-free batch.
    """
    await graph_client.ensure_schema()

    _write_python_file(settings.project_root, "good.py", "G = 1\n")

    consumer = _PoisonASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    task = asyncio.create_task(consumer.run())
    try:
        # Poison first so its failure count stays ahead of the good event's
        await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "poison.py"))
        await asyncio.sleep(0.3)
        await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "good.py"))

        await _wait_until(lambda: consumer.stats.files_processed >= 1, timeout_s=20.0)
    finally:
        consumer.stop()
        await asyncio.wait_for(task, timeout=10.0)

    assert consumer.stats.files_processed >= 1
    # Poison message was parked (ACKed) — nothing left pending
    assert await _pel_count(event_bus, Topic.FILE_CHANGED, "ast") == 0
    assert _MAX_BATCH_FAILURES == 5  # parking threshold pinned by the durability contract


# ---------------------------------------------------------------------------
# Stream trim / drain semantics (S7 d+e)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_clean_streams")
async def test_publish_many_does_not_trim_backlog(event_bus: EventBus) -> None:
    """publish_many must not silently trim an unconsumed backlog (old hard 10k cap)."""
    events = [FileChanged(path=f"f_{i}.py", change_type="modified", timestamp=float(i)) for i in range(20_000)]
    await event_bus.publish_many(Topic.FILE_CHANGED, events)

    key = f"{event_bus._prefix}:{Topic.FILE_CHANGED.value}"
    assert await event_bus._redis.xlen(key) == 20_000


@pytest.mark.usefixtures("_clean_streams")
async def test_null_lag_reported_unknown_not_drained(event_bus: EventBus) -> None:
    """NULL stream lag means 'unknown', not 'drained' — drain must not report success."""
    key = f"{event_bus._prefix}:{Topic.FILE_CHANGED.value}"
    await event_bus.ensure_group(Topic.FILE_CHANGED, "ast")
    msg_ids = [
        await event_bus.publish(Topic.FILE_CHANGED, FileChanged(path=f"lag_{i}.py", change_type="modified"))
        for i in range(10)
    ]

    # Force server-reported NULL lag: SETID to a mid-stream ID without
    # ENTRIESREAD invalidates the group's entries-read counter, so the server
    # cannot compute lag. (Trimming past the read position no longer forces
    # NULL on Valkey 8.1+ — it recovers an exact lag when last-delivered-id
    # precedes the first remaining entry.)
    await event_bus._redis.xgroup_setid(key, "ast", msg_ids[4])

    info = await event_bus.stream_group_info(Topic.FILE_CHANGED, "ast")
    assert info["lag"] is None

    drained = await _wait_for_drain(event_bus, timeout_s=1.5, embed_enabled=False)
    assert drained is False


@pytest.mark.usefixtures("_clean_streams")
async def test_flush_preserves_consumer_groups(event_bus: EventBus) -> None:
    """EventBus.flush() trims streams but must NOT destroy live consumer groups."""
    key = f"{event_bus._prefix}:{Topic.FILE_CHANGED.value}"
    await event_bus.ensure_group(Topic.FILE_CHANGED, "ast")
    for i in range(3):
        await event_bus.publish(Topic.FILE_CHANGED, FileChanged(path=f"pre_{i}.py", change_type="modified"))
    # Read some without ACK so the group has a live PEL
    await event_bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", count=2, block_ms=500)

    await event_bus.flush()

    groups = await event_bus._redis.xinfo_groups(key)
    names = set()
    for g in groups:
        name = g.get(b"name", g.get("name", b""))
        names.add(name.decode() if isinstance(name, bytes) else name)
    assert "ast" in names

    # A consumer keeps receiving new events without NOGROUP
    await event_bus.publish(Topic.FILE_CHANGED, FileChanged(path="post.py", change_type="modified"))
    messages = await event_bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", count=10, block_ms=1000)
    paths = {decode_event(Topic.FILE_CHANGED, fields).path for _, fields in messages}  # type: ignore[union-attr]
    assert "post.py" in paths


# ---------------------------------------------------------------------------
# Body-only edits (S3 e2e)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_clean_streams")
async def test_body_only_edit_publishes_embed_dirty(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """A body-only edit (same signature/docstring) is classified modified and re-published.

    Before the fix the edit passed the file hash gate, classified 'unchanged'
    (source excluded from content_hash), published no EmbedDirty, and wrote
    back the new file_hash — permanently sealing stale source and embeddings.
    """
    await graph_client.ensure_schema()

    embed_key = f"{event_bus._prefix}:{Topic.EMBED_DIRTY.value}"
    _write_python_file(settings.project_root, "body.py", "def f():\n    return 1\n")

    c1 = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "body.py"))
    task = asyncio.create_task(c1.run())
    await asyncio.sleep(1.0)
    c1.stop()
    await asyncio.wait_for(task, timeout=5.0)
    assert c1.stats.files_processed >= 1
    xlen_before = await event_bus._redis.xlen(embed_key)
    assert xlen_before >= 1

    # Body-only edit: same signature, no docstring change
    _write_python_file(settings.project_root, "body.py", "def f():\n    return 2\n")

    c2 = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "body.py"))
    task = asyncio.create_task(c2.run())
    await asyncio.sleep(1.0)
    c2.stop()
    await asyncio.wait_for(task, timeout=5.0)

    assert c2.stats.entities_modified >= 1
    assert await event_bus._redis.xlen(embed_key) > xlen_before

    project_name = settings.project_root.resolve().name
    rows = await graph_client.execute(
        "MATCH (c:Callable {project_name: $p, name: 'f'}) RETURN c.source AS src",
        {"p": project_name},
    )
    assert rows
    assert "return 2" in (rows[0]["src"] or "")


# ---------------------------------------------------------------------------
# Explicit anchors + staleness (Phase 3 — e2e through the real pipeline)
# ---------------------------------------------------------------------------


async def test_anchor_stale_after_edit_and_broken_after_delete(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """Exit criterion: edit an anchored function -> the note is flagged stale within
    seconds; delete its file entirely -> the note is flagged has_broken_anchors."""
    await graph_client.ensure_schema()
    project_name = settings.project_root.resolve().name

    _write_python_file(settings.project_root, "anchor_target.py", "def bar():\n    return 1\n")

    c1 = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "anchor_target.py", "created"))
    task = asyncio.create_task(c1.run())
    await asyncio.sleep(1.0)
    c1.stop()
    await asyncio.wait_for(task, timeout=5.0)

    target_records = await graph_client.execute(
        "MATCH (c:Callable {project_name: $p, name: 'bar'}) RETURN c.uid AS uid", {"p": project_name}
    )
    assert target_records
    target_uid = target_records[0]["uid"]

    _write_python_file(
        settings.project_root,
        "docs/notes/anchor-note.md",
        f"---\nid: anchor-note\nkind: note\nanchors: [{target_uid}]\n---\n\nDocs the bar function.\n",
    )

    c2 = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "docs/notes/anchor-note.md", "created"))
    task = asyncio.create_task(c2.run())
    await asyncio.sleep(1.0)
    c2.stop()
    await asyncio.wait_for(task, timeout=5.0)

    note_uid = f"{project_name}:note:anchor-note"
    records = await graph_client.execute(
        "MATCH (n:Note {uid: $uid})-[r:DOCUMENTS {link_type: 'anchor'}]->(b) RETURN r.stale AS stale, b.uid AS buid",
        {"uid": note_uid},
    )
    assert len(records) == 1
    assert records[0]["buid"] == target_uid
    assert records[0]["stale"] is False

    # Edit the anchored function — content_hash drifts, marked stale within this same batch.
    _write_python_file(settings.project_root, "anchor_target.py", "def bar():\n    return 2\n")
    c3 = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "anchor_target.py"))
    task = asyncio.create_task(c3.run())
    await asyncio.sleep(1.0)
    c3.stop()
    await asyncio.wait_for(task, timeout=5.0)

    records = await graph_client.execute(
        "MATCH (n:Note {uid: $uid})-[r:DOCUMENTS {link_type: 'anchor'}]->() RETURN r.stale AS stale",
        {"uid": note_uid},
    )
    assert records[0]["stale"] is True

    # Delete the anchored function's file entirely — note flagged has_broken_anchors.
    (settings.project_root / "anchor_target.py").unlink()
    c4 = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "anchor_target.py", "deleted"))
    task = asyncio.create_task(c4.run())
    await asyncio.sleep(1.0)
    c4.stop()
    await asyncio.wait_for(task, timeout=5.0)

    records = await graph_client.execute(
        "MATCH (n:Note {uid: $uid}) RETURN n.has_broken_anchors AS broken", {"uid": note_uid}
    )
    assert records[0]["broken"] is True


# ---------------------------------------------------------------------------
# Cross-file member DEFINES (S5 e2e)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_clean_streams")
async def test_go_cross_file_method_attaches_to_receiver_type(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """A Go method whose receiver type lives in another file of the same package
    gets a DEFINES edge from that TypeDef via post-batch resolution.

    Before the fix the edge was emitted from the nonexistent fabricated uid
    '<p>:internal.server.routes.Server' and silently dropped.
    """
    pytest.importorskip("tree_sitter_go")
    await graph_client.ensure_schema()

    server_go = settings.project_root / "internal" / "server" / "server.go"
    server_go.parent.mkdir(parents=True, exist_ok=True)
    server_go.write_text("package server\n\ntype Server struct{}\n", encoding="utf-8")
    routes_go = settings.project_root / "internal" / "server" / "routes.go"
    routes_go.write_text("package server\n\nfunc (s *Server) Routes() {}\n", encoding="utf-8")

    consumer = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "internal/server/server.go", "created"))
    await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "internal/server/routes.go", "created"))

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(1.5)
    consumer.stop()
    # run()'s finally triggers _flush_deferred_resolution for the member rels
    await asyncio.wait_for(task, timeout=10.0)

    project_name = settings.project_root.resolve().name
    rows = await graph_client.execute(
        "MATCH (t:TypeDef {project_name: $p, name: 'Server'})-[:DEFINES]->(c:Callable {name: 'Routes'}) "
        "RETURN count(*) AS n",
        {"p": project_name},
    )
    assert rows[0]["n"] == 1


# ---------------------------------------------------------------------------
# Deferred-resolution / hash-write ordering (finding: consumers.py:~694)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_clean_streams")
async def test_deferred_calls_lost_if_process_dies_before_flush(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """A crash between the file-hash write and the deferred resolution flush
    must not permanently drop the pending relationship.

    Reindex-mode policy (time_window_s=0) sets resolve_batch_interval=5, so
    one process_batch call does NOT flush deferred CALLS rels — they sit only
    in process memory. Before the fix, the file's hash was written
    unconditionally in that same call; a crash before the (later) flush left
    the hash gate believing the file unchanged, so a fresh consumer's re-parse
    of the same event skipped it and the rel was gone forever.
    """
    await graph_client.ensure_schema()

    _write_python_file(
        settings.project_root,
        "mod.py",
        "def helper():\n    return 1\n\n\ndef main():\n    return helper()\n",
    )

    project_name = settings.project_root.resolve().name
    ev_mod = _file_changed(settings, "mod.py", "created")

    # Reindex-mode policy (time_window_s=0) -> resolve_batch_interval=5: a
    # batch's deferred rels are NOT flushed within that same call.
    consumer = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )

    # Warm-up batch: _last_resolve_time inits to 0.0, and the monotonic clock
    # (asyncio loop.time()) is NOT epoch-relative, so the very first call's
    # time-based flush condition is spuriously true regardless of the batch
    # interval. An empty batch harmlessly "uses up" that spurious flush (there
    # is nothing pending yet) and resets _last_resolve_time to now.
    await consumer.process_batch([], "batch-0")

    # Process ONE batch directly — bypasses run()'s guaranteed final-flush
    # (in its `finally`), which would otherwise mask this exact race.
    await consumer.process_batch([ev_mod], "batch-1")
    assert consumer._pending_call_rels  # sanity: the rel is still pending in-memory

    # Simulate a hard crash: drop the instance WITHOUT ever flushing.
    del consumer

    # "Restart": a fresh consumer instance re-processes the SAME event.
    c2 = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    await c2.process_batch([ev_mod], "batch-2")
    await c2._flush_deferred_resolution()

    rows = await graph_client.execute(
        "MATCH (a:Callable {project_name: $p, name: 'main'})-[:CALLS]->(b:Callable {name: 'helper'}) "
        "RETURN count(*) AS n",
        {"p": project_name},
    )
    assert rows[0]["n"] == 1


@pytest.mark.usefixtures("_clean_streams")
async def test_anchor_only_file_hash_withheld_until_flush(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """A markdown file whose ONLY deferred rel is an anchor must follow the
    same withhold path as import/call/type/member rels — its file_hash must
    NOT be written immediately, only staged in _pending_file_hashes until
    _flush_deferred_resolution actually resolves the anchor.

    Regression test for a copy/paste gap: the withhold condition listed
    import_rels/call_rels/type_rels/member_rels but omitted anchor_rels, so
    a note's hash was written immediately even though its anchor rel was
    still sitting unresolved in memory — a crash before the next flush would
    silently drop the anchor forever while the hash gate believed the file
    unchanged.
    """
    await graph_client.ensure_schema()
    project_name = settings.project_root.resolve().name

    _write_python_file(settings.project_root, "anchor_target2.py", "def baz():\n    return 1\n")

    c1 = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    await c1.process_batch([_file_changed(settings, "anchor_target2.py", "created")], "batch-0")

    target_records = await graph_client.execute(
        "MATCH (c:Callable {project_name: $p, name: 'baz'}) RETURN c.uid AS uid", {"p": project_name}
    )
    assert target_records
    target_uid = target_records[0]["uid"]

    _write_python_file(
        settings.project_root,
        "docs/notes/anchor-only-note.md",
        f"---\nid: anchor-only-note\nkind: note\nanchors: [{target_uid}]\n---\n\nAnchors baz.\n",
    )
    note_rel_path = "docs/notes/anchor-only-note.md"
    ev_note = _file_changed(settings, note_rel_path, "created")

    # Reindex-mode policy (time_window_s=0) -> resolve_batch_interval=5: this
    # batch's deferred anchor rel is NOT flushed within this same call, so the
    # note's hash must be withheld rather than written immediately.
    await c1.process_batch([ev_note], "batch-1")

    assert c1._pending_anchor_rels  # sanity: the anchor rel is pending in-memory
    assert note_rel_path in c1._pending_file_hashes.get(project_name, {})


# ---------------------------------------------------------------------------
# Detector timing (finding: consumers.py:~485)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_clean_streams")
async def test_same_batch_test_mapping_detector_resolves(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """A TESTS edge resolves even when the subject and its test file land in
    the SAME batch on a fresh (empty-graph) index.

    Before the fix, TestMappingDetector ran during the parse phase, before
    this batch's own entities were upserted — its graph lookup for the
    subject function found nothing (not written yet), so no TESTS edge was
    ever created for same-batch pairs.
    """
    await graph_client.ensure_schema()

    _write_python_file(settings.project_root, "sub.py", "def subject():\n    return 1\n")
    _write_python_file(settings.project_root, "test_sub.py", "def test_subject():\n    pass\n")

    # Publish both BEFORE starting the consumer so a single read_batch picks
    # up both messages together — i.e. they land in ONE process_batch call.
    await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "sub.py", "created"))
    await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "test_sub.py", "created"))

    consumer = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    task = asyncio.create_task(consumer.run())
    try:
        await _wait_until(lambda: consumer.stats.files_processed >= 2, timeout_s=10.0)
    finally:
        consumer.stop()
        await asyncio.wait_for(task, timeout=10.0)

    project_name = settings.project_root.resolve().name
    rows = await graph_client.execute(
        "MATCH (t:Callable {project_name: $p, name: 'test_subject'})-[:TESTS]->(s:Callable {name: 'subject'}) "
        "RETURN count(*) AS n",
        {"p": project_name},
    )
    assert rows[0]["n"] == 1


# ---------------------------------------------------------------------------
# Unreadable file retry (finding: consumers.py:~560)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_clean_streams")
async def test_unreadable_file_is_deferred_not_dropped(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient OSError reading a file (Windows lock/AV) defers the event
    for retry instead of silently ACKing it with only a warning.

    Before the fix the FileChanged message was ACKed after process_batch
    returned normally, and the change was lost until another edit fired a
    fresh event for the same file.
    """
    await graph_client.ensure_schema()

    _write_python_file(settings.project_root, "locked.py", "L = 1\n")
    locked_path = (settings.project_root / "locked.py").resolve()

    real_read_bytes = Path.read_bytes
    fail_count = 0

    def _flaky_read_bytes(self: Path) -> bytes:
        nonlocal fail_count
        if self.resolve() == locked_path and fail_count < 1:
            fail_count += 1
            raise PermissionError("simulated Windows file lock")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _flaky_read_bytes)

    consumer = ASTConsumer(
        event_bus,
        graph_client,
        settings,
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )
    await event_bus.publish(Topic.FILE_CHANGED, _file_changed(settings, "locked.py", "created"))

    task = asyncio.create_task(consumer.run())
    try:
        await _wait_until(lambda: consumer.stats.files_processed >= 1, timeout_s=10.0)
    finally:
        consumer.stop()
        await asyncio.wait_for(task, timeout=10.0)

    assert fail_count == 1  # the flaky read actually fired once
    assert consumer.stats.files_processed >= 1
    assert await _pel_count(event_bus, Topic.FILE_CHANGED, "ast") == 0

    project_name = settings.project_root.resolve().name
    rows = await graph_client.execute(
        "MATCH (v:Value {project_name: $p, name: 'L'}) RETURN count(*) AS n",
        {"p": project_name},
    )
    assert rows[0]["n"] == 1


# ---------------------------------------------------------------------------
# Embed-stage lost update (finding: consumers.py:~944)
# ---------------------------------------------------------------------------


class _StallingEmbedClient:
    """Fake embed client: stalls on texts containing *stall_marker* until released."""

    def __init__(self, stall_marker: str, dimension: int = 8) -> None:
        self.max_concurrency = 2
        self.batch_size = 10
        self._dimension = dimension
        self._stall_marker = stall_marker
        self._release = asyncio.Event()
        self.calls = 0

    def release(self) -> None:
        self._release.set()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if any(self._stall_marker in t for t in texts):
            await self._release.wait()
        return [[0.1] * self._dimension for _ in texts]


@pytest.mark.usefixtures("_clean_streams")
async def test_embed_concurrent_workers_no_lost_update(
    event_bus: EventBus,
    graph_client: GraphClient,
    settings: AtlasSettings,
) -> None:
    """A slow worker holding a stale read must not clobber a fresher vector
    written by a faster, later-dispatched worker for the same entity.

    Before the fix, EmbedDirty dedup only happened within a single batch — a
    second concurrent worker for the same uid could write a fresher vector
    while the first (slow) worker was still embedding; that first worker's
    unconditional write then landed LAST and overwrote the fresher data with
    stale data (lost update).
    """
    await graph_client.ensure_schema()
    project_name = settings.project_root.resolve().name
    uid = f"{project_name}:mod.f"

    # Seed a Callable entity directly — this test targets the embed stage's
    # own concurrency, not the AST/parsing stage.
    await graph_client.execute_write(
        "CREATE (c:Callable {uid: $uid, project_name: $p, qualified_name: 'mod.f', name: 'f', "
        "kind: 'function', file_path: 'mod.py', signature: 'def f()', docstring: 'v1', source: ''})",
        {"uid": uid, "p": project_name},
    )

    embed = _StallingEmbedClient(stall_marker="v1", dimension=graph_client._dimension)
    consumer = EmbedConsumer(
        event_bus,
        graph_client,
        embed,  # type: ignore[arg-type]
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
        max_concurrency=2,
    )
    ref = EntityRef(qualified_name=uid, node_type="Callable", file_path="mod.py")

    task = asyncio.create_task(consumer.run())
    try:
        # Batch 1: triggers the slow ("v1") embed — stalls until released.
        await event_bus.publish(Topic.EMBED_DIRTY, EmbedDirty(entity=ref, significance="HIGH"))
        await _wait_until(lambda: embed.calls >= 1, timeout_s=5.0)

        # While worker A is stalled mid-embed, the entity changes to v2 and a
        # second EmbedDirty fires — simulating the AST consumer's real update.
        await graph_client.execute_write("MATCH (c:Callable {uid: $uid}) SET c.docstring = 'v2'", {"uid": uid})
        await event_bus.publish(Topic.EMBED_DIRTY, EmbedDirty(entity=ref, significance="HIGH"))

        # Give a second (un-deferred, pre-fix) worker a chance to race ahead
        # and write v2 while worker A is still stalled.
        await asyncio.sleep(1.0)

        embed.release()  # unstall worker A — its write (if unguarded) lands now
        # Let everything settle: worker A's write, plus the deferred retry's
        # PEL reclaim + re-read (now sees v2) + its own write.
        await asyncio.sleep(2.0)
    finally:
        consumer.stop()
        await asyncio.wait_for(task, timeout=10.0)

    rows = await graph_client.execute(
        "MATCH (c:Callable {uid: $uid}) RETURN c.docstring AS d, c.embed_hash AS h",
        {"uid": uid},
    )
    assert rows
    expected_hash_v2 = EmbedCache.hash_text(
        build_embed_text(
            {
                "_label": "Callable",
                "qualified_name": "mod.f",
                "kind": "function",
                "signature": "def f()",
                "docstring": "v2",
                "source": "",
            }
        )
    )
    assert rows[0]["h"] == expected_hash_v2


# ---------------------------------------------------------------------------
# Embed-stage retry cap (finding: consumers.py:~845 — "infinite PEL retries")
# ---------------------------------------------------------------------------


class _AlwaysFailEmbedConsumer(EmbedConsumer):
    """Simulates TEI being permanently down — every batch fails."""

    async def process_batch(self, events: list[Event], batch_id: str) -> set[str] | None:
        raise RuntimeError("TEI down")


@pytest.mark.usefixtures("_clean_streams")
async def test_embed_poison_capped_not_infinite(event_bus: EventBus, graph_client: GraphClient) -> None:
    """Embed-stage retries are bounded by the shared poison cap — a
    permanently failing embed batch (TEI down) does not retry forever.

    This is already fixed as a side effect of wave 1's poison cap, which is
    shared across AST/Embed stages via TierConsumer.run()'s PEL reclaim loop
    (no consumers.py change needed for this sub-claim) — asserted here so a
    future regression on the shared reclaim path is caught.
    """
    consumer = _AlwaysFailEmbedConsumer(
        event_bus,
        graph_client,
        None,  # type: ignore[arg-type]  # never touched — process_batch raises before any embed call
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
        max_concurrency=1,
    )
    ref = EntityRef(qualified_name="p:mod.f", node_type="Callable", file_path="mod.py")
    await event_bus.publish(Topic.EMBED_DIRTY, EmbedDirty(entity=ref, significance="HIGH"))

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(3.0)
    consumer.stop()
    await asyncio.wait_for(task, timeout=10.0)

    assert await _pel_count(event_bus, Topic.EMBED_DIRTY, "embed") == 0
