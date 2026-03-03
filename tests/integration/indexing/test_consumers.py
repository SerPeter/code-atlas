"""Integration tests for the event-driven pipeline.

Requires Memgraph + Valkey (provided by conftest fixtures).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest

from code_atlas.events import (
    EventBus,
    FileChanged,
    Topic,
    decode_event,
)
from code_atlas.indexing.consumers import ASTConsumer, BatchPolicy

if TYPE_CHECKING:
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
    assert "rapid.py" in consumer._deferred


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

    # Both events processed — no deferral
    assert not consumer._deferred
