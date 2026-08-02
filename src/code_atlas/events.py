"""Event types and Redis Streams event bus for the indexing pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import orjson
import redis.asyncio as aioredis
from loguru import logger

from code_atlas.telemetry import get_tracer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

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
# Indexer lease
# ---------------------------------------------------------------------------

# Long enough to survive a slow batch, short enough that a killed indexer does not block
# the next one for a whole session. Renewed at a third of the TTL.
INDEXER_LEASE_TTL_MS = 60_000
_LEASE_RENEW_S = INDEXER_LEASE_TTL_MS / 3000


class IndexerBusyError(RuntimeError):
    """Another process holds the indexer lease for this project."""

    def __init__(self, holder: str) -> None:
        super().__init__(f"another indexer holds the lease for this project: {holder}")
        self.holder = holder


def new_lease_owner() -> str:
    """Identity for a lease holder — host, pid and a nonce.

    The nonce matters on top of the pid: a recycled pid must not be able to renew or
    release a lease that a previous process of the same number took out.
    """
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


@asynccontextmanager
async def hold_indexer_lease(bus: Any, *, ttl_ms: int = INDEXER_LEASE_TTL_MS) -> AsyncIterator[str]:
    """Hold the project's indexer lease for the duration of the block.

    Raises :class:`IndexerBusyError` if someone else holds it, rather than indexing anyway —
    two processes writing the same nodes is how a single index run got split across two
    code versions, and how Memgraph's MVCC conflicts turned into dropped files.

    Renewal runs in the background so a long index cannot lose the lease mid-run, and the
    release is a compare-and-delete, so a process that stalled past its TTL cannot free a
    lease that has since passed to someone else.
    """
    owner = new_lease_owner()
    if not await bus.acquire_indexer_lease(owner, ttl_ms):
        holder = await bus.read_indexer_lease()
        raise IndexerBusyError(holder or "unknown")

    async def _renew() -> None:
        while True:
            await asyncio.sleep(_LEASE_RENEW_S)
            if not await bus.renew_indexer_lease(owner, ttl_ms):
                logger.warning("Indexer lease lost while still indexing (owner={})", owner)
                return

    renewer = asyncio.create_task(_renew())
    try:
        yield owner
    finally:
        renewer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await renewer
        await bus.release_indexer_lease(owner)


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
        return await self._redis.ping()  # type: ignore[invalid-await]  # stub widened to Awaitable[bool] | bool

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
                self._stream_key(topic),
                encode_event(event),  # type: ignore[invalid-argument-type]  # dict[bytes,bytes] is invariant-incompatible with the stub's broader byte-like union
                maxlen=self._maxlen,
                approximate=True,
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
                    pipe.xadd(
                        key,
                        encode_event(event),  # type: ignore[invalid-argument-type]  # see publish()
                        maxlen=self._maxlen,
                        approximate=True,
                    )
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

    async def reclaim_abandoned(
        self,
        topic: Topic,
        group: str,
        consumer: str,
        *,
        min_idle_ms: int,
        count: int = 10,
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        """Claim messages left pending by a consumer that is no longer running.

        Consumer names now carry a process identity, so a killed process's PEL entries
        belong to a name nothing will ever use again. ``XREADGROUP ... 0`` only reads the
        caller's own history, so without this they are orphaned permanently.

        *min_idle_ms* is what keeps this from stealing live work: an entry is only taken
        once no one has touched it for that long, which a running consumer's own batch
        never satisfies.
        """
        with _tracer.start_as_current_span(
            "eventbus.reclaim_abandoned", attributes={"topic": topic.value, "group": group, "consumer": consumer}
        ):
            try:
                result: Any = await self._redis.xautoclaim(
                    self._stream_key(topic), group, consumer, min_idle_time=min_idle_ms, count=count
                )
            except aioredis.ResponseError:
                return []  # group or stream does not exist yet
            # XAUTOCLAIM returns [next_cursor, [(msg_id, fields), ...], deleted_ids]
            return list(result[1]) if len(result) > 1 else []

    async def ack(self, topic: Topic, group: str, *msg_ids: bytes) -> int:
        """Acknowledge messages after successful processing."""
        return await self._redis.xack(self._stream_key(topic), group, *msg_ids)

    async def consumer_registrations(self, topic: Topic, group: str) -> list[tuple[str, int, int]]:
        """``(name, pending, idle_ms)`` for every consumer registered in *group*.

        A registration outlives the process that made it: Redis creates one on first read
        and never removes it. Now that names carry a process identity, every index run
        leaves one behind — so something has to enumerate them to clean up.

        *idle_ms* is time since the consumer last issued any command, not since it last
        got a message, so a live consumer blocking on an empty stream still reads as busy.
        """
        try:
            infos = await self._redis.xinfo_consumers(self._stream_key(topic), group)
        except aioredis.ResponseError:
            return []  # group or stream does not exist yet

        out: list[tuple[str, int, int]] = []
        for info in infos:
            name = info.get(b"name", info.get("name", b""))
            if isinstance(name, bytes):
                name = name.decode()
            pending = info.get(b"pending", info.get("pending", 0))
            idle_ms = info.get(b"idle", info.get("idle", 0))
            out.append((name, int(pending), int(idle_ms)))
        return out

    async def drop_consumer(self, topic: Topic, group: str, consumer: str) -> int:
        """Deregister *consumer*, returning how many pending entries went with it.

        Destroys those entries rather than reassigning them, so callers must confirm the
        PEL is empty first — the return value is a leak detector, not a status code.
        """
        try:
            return int(await self._redis.xgroup_delconsumer(self._stream_key(topic), group, consumer))
        except aioredis.ResponseError:
            return 0  # group or stream does not exist yet

    # -- Indexer lease ---------------------------------------------------------
    #
    # Unique consumer names stop two processes corrupting each other's PEL, but they do
    # not stop two processes indexing the same project into the same graph at once. The
    # lease is that invariant, and it lives in the store both indexers must already reach
    # in order to index at all — unlike a PID file, which cannot see a peer in Docker,
    # across a WSL boundary, or under another user.

    def _lease_key(self) -> str:
        return f"{self._prefix}:{self._project}:indexer-lease" if self._project else f"{self._prefix}:indexer-lease"

    async def acquire_indexer_lease(self, owner: str, ttl_ms: int) -> bool:
        """Take the indexer lease, or return False if someone else holds it."""
        return bool(await self._redis.set(self._lease_key(), owner.encode(), nx=True, px=ttl_ms))

    async def renew_indexer_lease(self, owner: str, ttl_ms: int) -> bool:
        """Extend the lease, but only while *owner* still holds it.

        Compare-and-set: a holder that stalled past its TTL and lost the lease must not
        silently take it back from whoever legitimately acquired it.
        """
        script = (
            "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('pexpire', KEYS[1], ARGV[2]) else "
            "return 0 end"
        )
        return bool(
            await self._redis.eval(  # type: ignore[invalid-await]  # stub widened to Awaitable[str] | str
                script, 1, self._lease_key(), owner.encode(), str(ttl_ms)
            )
        )

    async def release_indexer_lease(self, owner: str) -> bool:
        """Release the lease if *owner* holds it. Compare-and-delete, so a process can
        never free a lease that has already passed to someone else."""
        script = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"
        return bool(
            await self._redis.eval(  # type: ignore[invalid-await]  # stub widened to Awaitable[str] | str
                script, 1, self._lease_key(), owner.encode()
            )
        )

    async def read_indexer_lease(self) -> str | None:
        """Current lease holder, for diagnostics. ``None`` when the lease is free."""
        raw = await self._redis.get(self._lease_key())
        return raw.decode() if raw else None

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
