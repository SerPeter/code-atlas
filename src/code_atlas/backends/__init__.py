"""In-process fallback backends (SQLite queue, SQLite graph) selected via config.

Factory functions here decide, per :class:`~code_atlas.settings.BackendSettings`,
whether to construct the network-backed implementation (Valkey ``EventBus``,
Memgraph ``GraphClient``) or its embedded SQLite counterpart.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from code_atlas.backends.postgres_queue import PostgresConnections, PostgresEventBus
from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.backends.sqlite_queue import SqliteEventBus
from code_atlas.events import EventBus, redis_client
from code_atlas.graph.client import GraphClient
from code_atlas.search.ratelimit import PostgresRateLimiter, RateLimiter, SqliteRateLimiter
from code_atlas.settings import derive_project_name, ensure_sqlite_data_dir

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from pathlib import Path

    import redis.asyncio as aioredis

    from code_atlas.search.ratelimit import Limiter
    from code_atlas.settings import AtlasSettings

__all__ = [
    "Backends",
    "QueueConnections",
    "connected",
    "create_event_bus",
    "create_graph_client",
    "create_rate_limiter",
    "graph_backend_label",
    "queue_backend_label",
    "use_backends",
    "use_queue",
]

type QueueBus = EventBus | SqliteEventBus | PostgresEventBus


@dataclass(frozen=True)
class Backends:
    """The connections one process owns, handed down rather than reconstructed.

    `bus` is None only when the caller asked for a graph alone -- `atlas search` and
    `atlas ui` never publish, and opening a queue connection they will not use is a
    connection to leak.

    `limiter` is always present and draws on the same connection as `bus` (ADR-0044):
    pacing an embedding call does not require a bus, and must not open a second pool.
    It connects on first use, so a command that never embeds never reaches the store.
    """

    graph: GraphClient | SqliteGraphClient
    limiter: Limiter
    bus: QueueBus | None = None


class QueueConnections:
    """The network connections a process holds to its queue service -- at most one of each.

    The event bus and the rate limiter are handed these and never create or close their
    own. Before this, every `EmbedClient` built a limiter and every limiter opened its own
    Valkey client or asyncpg pool beside the bus's, so the connection count followed the
    number of places that constructed an embedding client rather than the work.

    Lazy: a client is built the first time a bus or limiter asks for it, and nothing
    connects until one issues a command. `close` closes only what was built.
    """

    def __init__(self, settings: AtlasSettings) -> None:
        self._settings = settings
        self._redis: aioredis.Redis | None = None
        self._postgres: PostgresConnections | None = None

    def redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = redis_client(self._settings.redis)
        return self._redis

    def postgres(self) -> PostgresConnections:
        pg = self._settings.backend.queue.postgres
        if pg is None:
            msg = "no [backend.queue.postgres] is declared"
            raise RuntimeError(msg)
        if self._postgres is None:
            self._postgres = PostgresConnections(pg)
        return self._postgres

    async def close(self) -> None:
        redis, self._redis = self._redis, None
        postgres, self._postgres = self._postgres, None
        if redis is not None:
            await redis.aclose()
        if postgres is not None:
            await postgres.close()


@asynccontextmanager
async def connected(
    settings: AtlasSettings,
    *,
    with_bus: bool = True,
    project_name: str | None = None,
    on_unreachable: Callable[[str], Exception] | None = None,
) -> AsyncGenerator[Backends]:
    """`use_backends`, plus the reachability check every command was writing by hand.

    Eight commands opened a graph, pinged it, logged the same message and exited 1 --
    the same five lines each time, and in most of them the `raise` happened *before* the
    try/finally that closed the client, so a failed ping leaked the connection it had
    just opened. Doing it once fixes that everywhere, since the scope closes on the way
    out however the block ends.

    *on_unreachable* builds the exception to raise, because the CLI wants
    `typer.Exit(1)` and nothing in this package should import typer. Omitting it
    re-raises the original error.
    """
    async with use_backends(settings, with_bus=with_bus, project_name=project_name) as backends:
        label = graph_backend_label(backends.graph, settings)
        try:
            await backends.graph.ping()
        except Exception as exc:
            logger.error("Cannot reach {} — {}", label, exc)
            if on_unreachable is not None:
                raise on_unreachable(label) from exc
            raise
        logger.info("Connected to {}", label)
        yield backends


@asynccontextmanager
async def use_queue(
    settings: AtlasSettings,
    *,
    bus: QueueBus | None = None,
    limiter: Limiter | None = None,
    with_bus: bool = True,
    project_name: str | None = None,
) -> AsyncGenerator[tuple[QueueBus | None, Limiter]]:
    """The queue half of `use_backends`: ``(bus, limiter)`` over one set of connections.

    Separate because `atlas index` needs the queue before the graph -- it probes the
    embedding dimension, paced by the limiter, before a graph can be sized -- and the
    bus and limiter must still share their connection there.

    The same ownership rule as `use_backends`: a *bus* or *limiter* passed in is reused
    and never closed; what this builds, it closes.

    *project_name* binds the bus to a project other than this checkout's -- `atlas project
    rm` guards and clears the queue of the project it removes.
    """
    async with AsyncExitStack() as stack:
        connections = QueueConnections(settings)
        stack.push_async_callback(connections.close)
        if bus is None and with_bus:
            bus = await create_event_bus(settings, connections, project_name=project_name)
            if isinstance(bus, SqliteEventBus):
                # The embedded bus owns a file handle; the network buses own nothing.
                await stack.enter_async_context(bus)
        if limiter is None:
            limiter = create_rate_limiter(settings, connections)
            if isinstance(limiter, SqliteRateLimiter):
                await stack.enter_async_context(limiter)
        yield bus, limiter


@asynccontextmanager
async def use_backends(
    settings: AtlasSettings,
    *,
    graph: GraphClient | SqliteGraphClient | None = None,
    bus: QueueBus | None = None,
    limiter: Limiter | None = None,
    with_bus: bool = True,
    project_name: str | None = None,
) -> AsyncGenerator[Backends]:
    """Use the connections this process needs, opening only the ones it does not have.

    Named for borrowing rather than for opening or owning, because it may do neither:
    hand it a live client and it reuses that one untouched. `open_*` promised an open
    that often does not happen, and "scope" described the mechanism rather than what a
    caller wants from it.

    The single place a command acquires connections. Before this, eleven CLI entry
    points each ran their own construct/ping/report/close sequence, and every one was a
    chance to miss an exit path -- which is how four fixtures came to call pytest.skip()
    between constructing a client and closing it.

    Pass *graph*, *bus* or *limiter* to reuse one the caller already holds. A reused client
    is deliberately not entered into the stack, so **ownership follows creation**: this
    closes what it opened and never what it was handed. That distinction is not
    hypothetical -- the MCP server's health check already holds a live graph and, usually,
    the daemon's bus, and opening a second connection to the same Memgraph to ask it
    whether it is reachable would be absurd. It is the same `own_graph`/`own_bus`
    bookkeeping health.py did by hand, expressed once and structurally.

    The bus and the limiter share one connection per service (`QueueConnections`): one
    Valkey client, or one asyncpg pool plus one LISTEN connection.

    AsyncExitStack rather than nested `async with` because the connections are
    conditional: if the queue fails to open, the stack still unwinds a graph it created,
    which a hand-written try/finally over optional objects gets wrong more often
    than not.

    Reachability stays the caller's business -- `atlas index` exits 1 on an unreachable
    graph while the MCP server degrades to query-only, and folding that choice in here
    would force one of them to fight it.
    """
    async with AsyncExitStack() as stack:
        # Enter for the lifecycle, but keep the object we constructed rather than what
        # __aenter__ hands back. Our clients return self, so the two are the same thing
        # in production -- but binding the return value means a test double whose
        # __aenter__ yields a fresh auto-mock silently swaps the client underneath the
        # caller, and the body then operates on an object the test never sees. The
        # reference we close and the reference we hand out should be the same one.
        if graph is None:
            graph = await create_graph_client(settings)
            await stack.enter_async_context(graph)
        bus, limiter = await stack.enter_async_context(
            use_queue(settings, bus=bus, limiter=limiter, with_bus=with_bus, project_name=project_name)
        )
        yield Backends(graph=graph, limiter=limiter, bus=bus)


def _sqlite_queue_path(settings: AtlasSettings) -> Path:
    return ensure_sqlite_data_dir(settings) / "queue.sqlite3"


def _sqlite_graph_path(settings: AtlasSettings) -> Path:
    return ensure_sqlite_data_dir(settings) / "graph.sqlite3"


async def create_event_bus(
    settings: AtlasSettings, connections: QueueConnections, *, project_name: str | None = None
) -> QueueBus:
    """Build the event bus selected by ``settings.backend.queue``, over *connections*.

    Bound to *project_name*, or to this checkout's project when it is omitted.

    - ``[backend.queue.sqlite]`` declared: always build a :class:`SqliteEventBus` under
      ``project_root / backend.sqlite_data_dir``.
    - ``[backend.queue.valkey]`` declared: always build a real :class:`EventBus`;
      unreachable Valkey fails loudly. Declaring a backend is choosing it.
    - ``[backend.queue.postgres]`` declared: always build a :class:`PostgresEventBus`
      (ADR-0056). Never probed for: ``auto`` does not consider Postgres.
    - **neither** declared: probe a real :class:`EventBus` via ``ping()`` and fall back to
      :class:`SqliteEventBus` with a logged warning if unreachable.
    """
    choice = settings.backend.queue_choice
    project_name = project_name or derive_project_name(settings.project_root)

    if choice == "sqlite":
        return SqliteEventBus(_sqlite_queue_path(settings))

    if choice == "valkey":
        return EventBus(connections.redis(), settings.redis, project_name=project_name)

    if settings.backend.queue.postgres is not None:
        return PostgresEventBus(connections.postgres(), project_name=project_name)

    bus = EventBus(connections.redis(), settings.redis, project_name=project_name)
    try:
        await bus.ping()
    except Exception:
        logger.warning("Valkey unreachable — falling back to in-process SQLite event queue")
        return SqliteEventBus(_sqlite_queue_path(settings))
    return bus


def create_rate_limiter(settings: AtlasSettings, connections: QueueConnections) -> Limiter:
    """The embedding rate limiter for whichever queue backend this project is configured for.

    Keyed on the configured queue backend rather than on a probe of its own. The bus factory
    already probes once at startup, and a second probe here would reintroduce exactly
    the connect timeout this selection exists to remove.

    An undeclared queue backend therefore resolves to Valkey, not to a guess: the Valkey limiter degrades
    on its own when the store is absent, and since it now holds a retry cooldown that
    costs a couple of timeouts per index rather than one per batch. Choosing SQLite for
    ``"auto"`` would be worse -- it would pace against a private file while the rest of
    the fleet paced against Valkey, and the shared budget the buckets exist to enforce
    would quietly stop being shared.

    The network limiters draw on *connections*, the same ones the bus uses. The SQLite
    limiter keeps its own file, ``ratelimit.sqlite3``, apart from the queue's (ADR-0044).
    """
    if settings.backend.queue_choice == "sqlite":
        return SqliteRateLimiter(ensure_sqlite_data_dir(settings) / "ratelimit.sqlite3")
    if settings.backend.queue.postgres is not None:
        return PostgresRateLimiter(connections.postgres())
    return RateLimiter(connections.redis(), stream_prefix=settings.redis.stream_prefix)


def graph_backend_label(client: GraphClient | SqliteGraphClient, settings: AtlasSettings) -> str:
    """Human-readable ``"<backend> at <address>"`` for whichever graph backend *client* actually is.

    Used in place of hardcoded "Memgraph at ..." log/error messages at construction
    call sites, so they stay honest once the backend can also be a SqliteGraphClient.
    """
    if isinstance(client, SqliteGraphClient):
        return f"SQLite (embedded) at {_sqlite_graph_path(settings)}"
    return f"Memgraph at {settings.memgraph.host}:{settings.memgraph.port}"


def queue_backend_label(client: QueueBus, settings: AtlasSettings) -> str:
    """Human-readable ``"<backend> at <address>"`` for whichever queue backend *client* actually is.

    Used in place of hardcoded "Valkey at ..." log/error messages at construction
    call sites, so they stay honest once the backend can also be a SqliteEventBus.
    """
    if isinstance(client, SqliteEventBus):
        return f"SQLite (embedded) at {_sqlite_queue_path(settings)}"
    if isinstance(client, PostgresEventBus):
        return f"Postgres at {client.address}"
    return f"Valkey at {settings.redis.host}:{settings.redis.port}"


async def create_graph_client(settings: AtlasSettings) -> GraphClient | SqliteGraphClient:
    """Build the graph client selected by ``settings.backend.graph``.

    - ``[backend.graph.sqlite]`` declared: always build a :class:`SqliteGraphClient` under
      ``project_root / backend.sqlite_data_dir``.
    - ``[backend.graph.memgraph]`` declared: always build a real :class:`GraphClient`;
      unreachable Memgraph fails loudly. Declaring a backend is choosing it -- the old
      shape let a fully configured Memgraph be read and then not used.
    - **neither** declared: probe a real :class:`GraphClient` via ``ping()`` and fall back
      to :class:`SqliteGraphClient` with a logged warning if unreachable. The fallback
      graph is EMPTY, so the next index rebuilds everything and re-buys every vector --
      which is why it is reserved for the case where nothing was asked for.
    """
    choice = settings.backend.graph_choice
    dimension = settings.embeddings.dimension or 768
    embeddings_enabled = settings.embeddings.enabled

    if choice == "sqlite":
        return SqliteGraphClient(
            _sqlite_graph_path(settings), dimension=dimension, embeddings_enabled=embeddings_enabled
        )

    if choice == "memgraph":
        return GraphClient(settings)

    graph = GraphClient(settings)
    try:
        await graph.ping()
    except Exception:
        await graph.close()
        logger.warning("Memgraph unreachable — falling back to in-process SQLite graph backend")
        return SqliteGraphClient(
            _sqlite_graph_path(settings), dimension=dimension, embeddings_enabled=embeddings_enabled
        )
    return graph
