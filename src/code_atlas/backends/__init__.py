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

from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.backends.sqlite_queue import SqliteEventBus
from code_atlas.events import EventBus
from code_atlas.graph.client import GraphClient
from code_atlas.settings import derive_project_name

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from code_atlas.settings import AtlasSettings

__all__ = [
    "Backends",
    "create_event_bus",
    "create_graph_client",
    "graph_backend_label",
    "queue_backend_label",
    "use_backends",
]


@dataclass(frozen=True)
class Backends:
    """The connections one process owns, handed down rather than reconstructed.

    `bus` is None only when the caller asked for a graph alone -- `atlas search` and
    `atlas ui` never publish, and opening a queue connection they will not use is a
    connection to leak.
    """

    graph: GraphClient | SqliteGraphClient
    bus: EventBus | SqliteEventBus | None = None


@asynccontextmanager
async def use_backends(
    settings: AtlasSettings,
    *,
    graph: GraphClient | SqliteGraphClient | None = None,
    bus: EventBus | SqliteEventBus | None = None,
    with_bus: bool = True,
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

    Pass *graph* or *bus* to reuse a connection the caller already holds. A reused client
    is deliberately not entered into the stack, so **ownership follows creation**: this
    closes what it opened and never what it was handed. That distinction is not
    hypothetical -- the MCP server's health check already holds a live graph and, usually,
    the daemon's bus, and opening a second connection to the same Memgraph to ask it
    whether it is reachable would be absurd. It is the same `own_graph`/`own_bus`
    bookkeeping health.py did by hand, expressed once and structurally.

    AsyncExitStack rather than nested `async with` because both connections are
    conditional: if the bus fails to open, the stack still unwinds a graph it created,
    which a hand-written try/finally over two optional objects gets wrong more often
    than not.

    Reachability stays the caller's business -- `atlas index` exits 1 on an unreachable
    graph while the MCP server degrades to query-only, and folding that choice in here
    would force one of them to fight it.
    """
    async with AsyncExitStack() as stack:
        if graph is None:
            graph = await stack.enter_async_context(await create_graph_client(settings))
        if bus is None and with_bus:
            bus = await stack.enter_async_context(await create_event_bus(settings))
        yield Backends(graph=graph, bus=bus)


def _sqlite_queue_path(settings: AtlasSettings) -> Path:
    data_dir = settings.project_root / settings.backend.sqlite_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "queue.sqlite3"


def _sqlite_graph_path(settings: AtlasSettings) -> Path:
    data_dir = settings.project_root / settings.backend.sqlite_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "graph.sqlite3"


async def create_event_bus(settings: AtlasSettings) -> EventBus | SqliteEventBus:
    """Build the event bus selected by ``settings.backend.queue``.

    - ``"sqlite"``: always build a :class:`SqliteEventBus` under
      ``project_root / backend.sqlite_data_dir``.
    - ``"valkey"``: always build a real :class:`EventBus`; unreachable Valkey
      fails loudly (no silent fallback for an explicit choice).
    - ``"auto"`` (default): probe a real :class:`EventBus` via ``ping()``;
      fall back to :class:`SqliteEventBus` with a logged warning if
      unreachable.
    """
    choice = settings.backend.queue
    project_name = derive_project_name(settings.project_root)

    if choice == "sqlite":
        return SqliteEventBus(_sqlite_queue_path(settings))

    if choice == "valkey":
        return EventBus(settings.redis, project_name=project_name)

    bus = EventBus(settings.redis, project_name=project_name)
    try:
        await bus.ping()
    except Exception:
        await bus.close()
        logger.warning("Valkey unreachable — falling back to in-process SQLite event queue")
        return SqliteEventBus(_sqlite_queue_path(settings))
    return bus


def graph_backend_label(client: GraphClient | SqliteGraphClient, settings: AtlasSettings) -> str:
    """Human-readable ``"<backend> at <address>"`` for whichever graph backend *client* actually is.

    Used in place of hardcoded "Memgraph at ..." log/error messages at construction
    call sites, so they stay honest once the backend can also be a SqliteGraphClient.
    """
    if isinstance(client, SqliteGraphClient):
        return f"SQLite (embedded) at {_sqlite_graph_path(settings)}"
    return f"Memgraph at {settings.memgraph.host}:{settings.memgraph.port}"


def queue_backend_label(client: EventBus | SqliteEventBus, settings: AtlasSettings) -> str:
    """Human-readable ``"<backend> at <address>"`` for whichever queue backend *client* actually is.

    Used in place of hardcoded "Valkey at ..." log/error messages at construction
    call sites, so they stay honest once the backend can also be a SqliteEventBus.
    """
    if isinstance(client, SqliteEventBus):
        return f"SQLite (embedded) at {_sqlite_queue_path(settings)}"
    return f"Valkey at {settings.redis.host}:{settings.redis.port}"


async def create_graph_client(settings: AtlasSettings) -> GraphClient | SqliteGraphClient:
    """Build the graph client selected by ``settings.backend.graph``.

    - ``"sqlite"``: always build a :class:`SqliteGraphClient` under
      ``project_root / backend.sqlite_data_dir``.
    - ``"memgraph"``: always build a real :class:`GraphClient`; unreachable
      Memgraph fails loudly (no silent fallback for an explicit choice).
    - ``"auto"`` (default): probe a real :class:`GraphClient` via ``ping()``;
      fall back to :class:`SqliteGraphClient` with a logged warning if
      unreachable.
    """
    choice = settings.backend.graph
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
