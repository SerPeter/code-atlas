"""Event types and Redis Streams event bus for the indexing pipeline."""

from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import orjson
import redis.asyncio as aioredis

from code_atlas.telemetry import get_tracer

if TYPE_CHECKING:
    from code_atlas.settings import RedisSettings

_tracer = get_tracer(__name__)


# ---------------------------------------------------------------------------
# Event types (frozen dataclasses — lightweight, stdlib-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileChanged:
    """A file was created, modified, or deleted."""

    path: str
    change_type: str  # "created" | "modified" | "deleted"
    project_name: str = ""  # monorepo sub-project (empty = derive from settings)
    project_root: str = ""  # absolute path to project root (monorepo sub-project roots differ)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class EntityRef:
    """Reference to a code entity within EmbedDirty."""

    qualified_name: str
    node_type: str
    file_path: str


@dataclass(frozen=True)
class ASTDirty:
    """A single file needs AST re-parsing (published by Tier 1, consumed by Tier 2)."""

    path: str
    project_name: str = ""  # monorepo sub-project (forwarded from FileChanged)
    project_root: str = ""  # absolute path to project root (forwarded from FileChanged)


@dataclass(frozen=True)
class EmbedDirty:
    """A single entity needs re-embedding (published by Tier 2, consumed by Tier 3)."""

    entity: EntityRef
    significance: str  # "MODERATE" | "HIGH"


# Type alias for any pipeline event
Event = FileChanged | ASTDirty | EmbedDirty


class Significance(StrEnum):
    """How significant a change is for downstream re-embedding."""

    NONE = "NONE"
    TRIVIAL = "TRIVIAL"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------


class Topic(StrEnum):
    """Redis Stream keys for the pipeline."""

    FILE_CHANGED = "file-changed"
    AST_DIRTY = "ast-dirty"
    EMBED_DIRTY = "embed-dirty"


# Map topic → event class for deserialization
_TOPIC_EVENT_MAP: dict[Topic, type[Event]] = {
    Topic.FILE_CHANGED: FileChanged,
    Topic.AST_DIRTY: ASTDirty,
    Topic.EMBED_DIRTY: EmbedDirty,
}


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def encode_event(event: Event) -> dict[bytes, bytes]:
    """Serialize an event for XADD. Returns ``{b"data": <json_bytes>}``."""
    return {b"data": orjson.dumps(asdict(event))}


def decode_event(topic: Topic, data: dict[bytes, bytes]) -> Event:
    """Deserialize a Redis Stream message back into a typed event."""
    raw = orjson.loads(data[b"data"])
    cls = _TOPIC_EVENT_MAP[topic]

    # Reconstruct nested dataclasses that json.loads flattens to dicts
    if cls is EmbedDirty:
        raw["entity"] = EntityRef(**raw["entity"])

    return cls(**raw)


# ---------------------------------------------------------------------------
# EventBus — thin wrapper over redis.asyncio
# ---------------------------------------------------------------------------


class EventBus:
    """Thin async wrapper over Redis Streams for pipeline events.

    Implements "dumb pipes, smart endpoints": the bus only routes messages,
    consumers implement their own batching and dedup.
    """

    def __init__(self, settings: RedisSettings, *, project_name: str = "") -> None:
        url = f"redis://{settings.host}:{settings.port}/{settings.db}"
        if settings.password:
            url = f"redis://:{settings.password}@{settings.host}:{settings.port}/{settings.db}"
        self._redis = aioredis.from_url(url, decode_responses=False)
        self._prefix = settings.stream_prefix
        self._project = project_name

    def _stream_key(self, topic: Topic) -> str:
        if self._project:
            return f"{self._prefix}:{self._project}:{topic.value}"
        return f"{self._prefix}:{topic.value}"

    async def ping(self) -> bool:
        """Health check — returns True if Redis is reachable."""
        return await self._redis.ping()

    async def ensure_group(self, topic: Topic, group: str) -> None:
        """Idempotently create a consumer group (starts reading new messages)."""
        try:
            await self._redis.xgroup_create(self._stream_key(topic), group, id="0", mkstream=True)
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def publish(self, topic: Topic, event: Event, *, maxlen: int = 10_000) -> bytes:
        """Publish an event to a stream. Returns the message ID."""
        with _tracer.start_as_current_span("eventbus.publish", attributes={"topic": topic.value}):
            return await self._redis.xadd(self._stream_key(topic), encode_event(event), maxlen=maxlen, approximate=True)

    async def publish_many(self, topic: Topic, events: list[Event], *, maxlen: int = 10_000) -> list[bytes]:
        """Publish multiple events in a single pipeline round-trip."""
        if not events:
            return []
        with _tracer.start_as_current_span(
            "eventbus.publish_many", attributes={"topic": topic.value, "count": len(events)}
        ):
            key = self._stream_key(topic)
            async with self._redis.pipeline(transaction=False) as pipe:
                for event in events:
                    pipe.xadd(key, encode_event(event), maxlen=maxlen, approximate=True)
                return await pipe.execute()

    async def read_batch(
        self,
        topic: Topic,
        group: str,
        consumer: str,
        *,
        count: int = 10,
        block_ms: int = 2000,
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        """Pull a batch of messages via XREADGROUP.

        Returns list of ``(message_id, field_dict)`` tuples, or empty list
        if the block timeout expires with no messages.
        """
        with _tracer.start_as_current_span(
            "eventbus.read_batch", attributes={"topic": topic.value, "group": group, "consumer": consumer}
        ):
            result: Any = await self._redis.xreadgroup(
                group,
                consumer,
                {self._stream_key(topic): ">"},
                count=count,
                block=block_ms,
            )
            if not result:
                return []
            # result shape: [[stream_key, [(msg_id, fields), ...]]]
            return result[0][1]

    async def read_pending(
        self,
        topic: Topic,
        group: str,
        consumer: str,
        *,
        count: int = 10,
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        """Re-read unacknowledged (pending) messages from the PEL.

        Uses ``XREADGROUP`` with ID ``"0"`` to fetch messages that were
        delivered but never ACKed (e.g. after a failed batch).  Returns
        the same shape as :meth:`read_batch`.  Returns an empty list when
        no pending messages remain.
        """
        with _tracer.start_as_current_span(
            "eventbus.read_pending", attributes={"topic": topic.value, "group": group, "consumer": consumer}
        ):
            result: Any = await self._redis.xreadgroup(
                group,
                consumer,
                {self._stream_key(topic): "0"},
                count=count,
            )
            if not result:
                return []
            return result[0][1]

    async def ack(self, topic: Topic, group: str, *msg_ids: bytes) -> int:
        """Acknowledge messages after successful processing."""
        return await self._redis.xack(self._stream_key(topic), group, *msg_ids)

    async def stream_group_info(self, topic: Topic, group: str) -> dict[str, int]:
        """Return pending + lag counts for a consumer group via XINFO GROUPS.

        Returns ``{"pending": N, "lag": N}`` or ``{"pending": 0, "lag": 0}``
        if the group does not exist yet.
        """
        try:
            groups = await self._redis.xinfo_groups(self._stream_key(topic))
        except aioredis.ResponseError:
            return {"pending": 0, "lag": 0}

        for g in groups:
            # Redis returns dicts with byte or str keys depending on decode_responses
            name = g.get(b"name", g.get("name", b""))
            if isinstance(name, bytes):
                name = name.decode()
            if name == group:
                pending = g.get(b"pending", g.get("pending", 0))
                lag = g.get(b"lag", g.get("lag", 0))
                return {"pending": int(pending), "lag": int(lag or 0)}

        return {"pending": 0, "lag": 0}

    async def stream_group_info_multi(self, queries: list[tuple[Topic, str]]) -> list[dict[str, int]]:
        """Return pending + lag counts for multiple consumer groups in one pipelined RTT.

        Each entry in *queries* is ``(topic, group_name)``.  Returns a list of
        ``{"pending": N, "lag": N}`` dicts in the same order.
        """
        if not queries:
            return []

        pipe = self._redis.pipeline(transaction=False)
        for topic, _group in queries:
            pipe.xinfo_groups(self._stream_key(topic))
        results = await pipe.execute()

        out: list[dict[str, int]] = []
        for (_topic, group), raw in zip(queries, results, strict=True):
            if isinstance(raw, Exception):
                out.append({"pending": 0, "lag": 0})
                continue
            found = False
            for g in raw:
                name = g.get(b"name", g.get("name", b""))
                if isinstance(name, bytes):
                    name = name.decode()
                if name == group:
                    pending = g.get(b"pending", g.get("pending", 0))
                    lag = g.get(b"lag", g.get("lag", 0))
                    out.append({"pending": int(pending), "lag": int(lag or 0)})
                    found = True
                    break
            if not found:
                out.append({"pending": 0, "lag": 0})
        return out

    async def flush(self) -> None:
        """Clear all pipeline streams for a full reindex.

        Trims all stream entries and destroys consumer groups to clear stale
        pending entry lists (PEL). Groups are recreated by ``ensure_group()``
        when consumers start.
        """
        pipe = self._redis.pipeline(transaction=False)
        for topic in Topic:
            pipe.xtrim(self._stream_key(topic), 0, approximate=False)
        await pipe.execute()

        # Destroy consumer groups to purge stale PEL entries from previous runs.
        for topic in Topic:
            key = self._stream_key(topic)
            try:
                groups = await self._redis.xinfo_groups(key)
            except Exception:
                continue
            for g in groups:
                name = g.get(b"name", g.get("name", b""))
                if isinstance(name, bytes):
                    name = name.decode()
                with contextlib.suppress(Exception):
                    await self._redis.xgroup_destroy(key, name)

    async def close(self) -> None:
        """Close the connection pool."""
        await self._redis.aclose()
