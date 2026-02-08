"""Integration tests for the event-driven pipeline.

Requires a running Redis/Valkey instance (``docker compose up -d valkey``).
"""

from __future__ import annotations

import asyncio

import pytest
import redis.asyncio as aioredis

from code_atlas.events import (
    ASTDirty,
    EventBus,
    FileChanged,
    Topic,
    decode_event,
)
from code_atlas.pipeline import Tier1GraphConsumer
from code_atlas.settings import AtlasSettings, RedisSettings

# All tests in this module require a live Redis/Valkey
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def redis_settings() -> RedisSettings:
    """Default Redis settings (localhost:6379)."""
    return RedisSettings()


@pytest.fixture
async def bus(redis_settings: RedisSettings):
    """EventBus connected to Redis, skip if unavailable."""
    b = EventBus(redis_settings)
    try:
        await b.ping()
    except aioredis.ConnectionError, OSError:
        pytest.skip("Redis/Valkey not available")
    yield b
    await b.close()


@pytest.fixture
async def _clean_streams(bus: EventBus):
    """Delete test streams before and after each test to avoid state leakage."""
    for topic in Topic:
        key = f"{bus._prefix}:{topic.value}"
        await bus._redis.delete(key)
    yield
    for topic in Topic:
        key = f"{bus._prefix}:{topic.value}"
        await bus._redis.delete(key)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_clean_streams")
async def test_publish_and_consume(bus: EventBus) -> None:
    """Publish FileChanged events, read back via XREADGROUP, verify decode."""
    group = "test-group"
    consumer = "test-consumer"

    await bus.ensure_group(Topic.FILE_CHANGED, group)

    # Publish two events
    ev1 = FileChanged(path="src/main.py", change_type="modified", timestamp=1000.0)
    ev2 = FileChanged(path="src/utils.py", change_type="created", timestamp=1001.0)
    await bus.publish(Topic.FILE_CHANGED, ev1)
    await bus.publish(Topic.FILE_CHANGED, ev2)

    # Read them back
    messages = await bus.read_batch(Topic.FILE_CHANGED, group, consumer, count=10, block_ms=500)
    assert len(messages) == 2

    decoded = [decode_event(Topic.FILE_CHANGED, fields) for _, fields in messages]
    assert decoded[0] == ev1
    assert decoded[1] == ev2

    # ACK
    msg_ids = [mid for mid, _ in messages]
    acked = await bus.ack(Topic.FILE_CHANGED, group, *msg_ids)
    assert acked == 2


@pytest.mark.usefixtures("_clean_streams")
async def test_dedup_within_batch(bus: EventBus) -> None:
    """Same file path published multiple times — consumer should dedup to 1."""
    group = "test-dedup"
    consumer = "test-dedup-0"

    await bus.ensure_group(Topic.FILE_CHANGED, group)

    # Publish same path 5 times
    for i in range(5):
        await bus.publish(
            Topic.FILE_CHANGED,
            FileChanged(path="src/main.py", change_type="modified", timestamp=1000.0 + i),
        )

    # Read all messages
    messages = await bus.read_batch(Topic.FILE_CHANGED, group, consumer, count=10, block_ms=500)
    assert len(messages) == 5

    # Apply dedup logic (same as TierConsumer): latest event wins per dedup key
    pending: dict[str, FileChanged] = {}
    for _, fields in messages:
        event = decode_event(Topic.FILE_CHANGED, fields)
        assert isinstance(event, FileChanged)
        pending[event.path] = event

    assert len(pending) == 1
    assert pending["src/main.py"].timestamp == 1004.0


@pytest.mark.usefixtures("_clean_streams")
async def test_tier1_publishes_downstream(bus: EventBus) -> None:
    """Run Tier1 briefly, verify ASTDirty appears on the ast-dirty stream."""
    # Set up consumer group for downstream
    await bus.ensure_group(Topic.AST_DIRTY, "test-downstream")

    # Publish a FileChanged event
    await bus.publish(
        Topic.FILE_CHANGED,
        FileChanged(path="src/app.py", change_type="modified", timestamp=2000.0),
    )

    # Run Tier1 for a short period then stop
    # Tier1 needs graph + settings but we're only testing event flow here;
    # it doesn't call graph in its current implementation.
    from unittest.mock import AsyncMock  # noqa: PLC0415

    mock_graph = AsyncMock()
    test_settings = AtlasSettings()
    tier1 = Tier1GraphConsumer(bus, mock_graph, test_settings)

    async def stop_after_delay() -> None:
        await asyncio.sleep(1.5)
        tier1.stop()

    await asyncio.gather(tier1.run(), stop_after_delay())

    # Read from the downstream ast-dirty stream
    messages = await bus.read_batch(Topic.AST_DIRTY, "test-downstream", "test-ds-0", count=10, block_ms=500)
    assert len(messages) >= 1

    event = decode_event(Topic.AST_DIRTY, messages[0][1])
    assert isinstance(event, ASTDirty)
    assert "src/app.py" in event.paths
