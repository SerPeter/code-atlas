"""Event types and Redis Streams event bus for the indexing pipeline."""

from __future__ import annotations

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
class EmbedDirty:
    """A single entity needs re-embedding (published by AST stage, consumed by Embed stage)."""

    entity: EntityRef
    significance: str  # "MODERATE" | "HIGH"


# Type alias for any pipeline event
Event = FileChanged | EmbedDirty


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
    EMBED_DIRTY = "embed-dirty"


# Map topic → event class for deserialization
_TOPIC_EVENT_MAP: dict[Topic, type[Event]] = {
    Topic.FILE_CHANGED: FileChanged,
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
        self._maxlen: int | None = settings.stream_maxlen if settings.stream_maxlen > 0 else None

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

    async def publish(self, topic: Topic, event: Event) -> bytes:
        """Publish an event to a stream. Returns the message ID.

        Streams are trimmed to ``RedisSettings.stream_maxlen`` (approximate);
        0 disables trimming. Callers cannot pass their own ceiling.
        """
        with _tracer.start_as_current_span("eventbus.publish", attributes={"topic": topic.value}):
            return await self._redis.xadd(
                self._stream_key(topic), encode_event(event), maxlen=self._maxlen, approximate=True
            )

    async def publish_many(self, topic: Topic, events: list[Event]) -> list[bytes]:
        """Publish multiple events in a single pipeline round-trip."""
        if not events:
            return []
        with _tracer.start_as_current_span(
            "eventbus.publish_many", attributes={"topic": topic.value, "count": len(events)}
        ):
            key = self._stream_key(topic)
            async with self._redis.pipeline(transaction=False) as pipe:
                for event in events:
                    pipe.xadd(key, encode_event(event), maxlen=self._maxlen, approximate=True)
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

    async def stream_group_info(self, topic: Topic, group: str) -> dict[str, int | None]:
        """Return pending + lag counts for a consumer group via XINFO GROUPS.

        Returns ``{"pending": N, "lag": N}``. ``lag`` is ``None`` when Redis
        reports it as unknown (the stream was trimmed past the group's read
        position) — callers must treat that as NOT drained, never as 0.
        Returns ``{"pending": 0, "lag": 0}`` if the group does not exist yet
        (a missing group genuinely has no backlog).
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
                return {"pending": int(pending), "lag": int(lag) if lag is not None else None}

        return {"pending": 0, "lag": 0}

    async def stream_group_info_multi(self, queries: list[tuple[Topic, str]]) -> list[dict[str, int | None]]:
        """Return pending + lag counts for multiple consumer groups in one pipelined RTT.

        Each entry in *queries* is ``(topic, group_name)``.  Returns a list of
        ``{"pending": N, "lag": N}`` dicts in the same order.  ``lag`` is
        ``None`` when unknown (stream trimmed past the group's read position);
        callers must treat that as NOT drained.
        """
        if not queries:
            return []

        pipe = self._redis.pipeline(transaction=False)
        for topic, _group in queries:
            pipe.xinfo_groups(self._stream_key(topic))
        results = await pipe.execute()

        out: list[dict[str, int | None]] = []
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
                    out.append({"pending": int(pending), "lag": int(lag) if lag is not None else None})
                    found = True
                    break
            if not found:
                out.append({"pending": 0, "lag": 0})
        return out

    async def flush(self) -> None:
        """Trim all pipeline streams for a full reindex.

        Consumer groups are preserved — live consumers keep running; PEL
        entries whose data was trimmed are ACKed by consumers when redelivered
        with empty fields.
        """
        pipe = self._redis.pipeline(transaction=False)
        for topic in Topic:
            pipe.xtrim(self._stream_key(topic), 0, approximate=False)
        await pipe.execute()

    async def close(self) -> None:
        """Close the connection pool."""
        await self._redis.aclose()
