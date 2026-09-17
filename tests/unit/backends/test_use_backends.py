"""`use_backends` — the one place a process acquires connections.

Its whole contract is an ownership rule: close what it opened, never what it was handed.
Get that backwards in either direction and you either leak a connection per call or
close one out from under the caller still using it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import SecretStr

from code_atlas.backends import QueueConnections, create_event_bus, queue_backend_label, use_backends
from code_atlas.backends.postgres_queue import PostgresEventBus
from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.backends.sqlite_queue import SqliteEventBus
from code_atlas.events import EventBus
from code_atlas.search.ratelimit import PostgresRateLimiter, RateLimiter, SqliteRateLimiter
from code_atlas.settings import AtlasSettings, BackendSettings, PostgresSettings, RedisSettings

if TYPE_CHECKING:
    from pathlib import Path


def _settings(tmp_path: Path) -> AtlasSettings:
    """Embedded backends only — no Memgraph, no Valkey, no network."""
    return AtlasSettings(project_root=tmp_path, backend=BackendSettings(graph={"sqlite": {}}, queue={"sqlite": {}}))


class TestOpenBackends:
    async def test_it_opens_both_and_closes_both(self, tmp_path: Path) -> None:
        async with use_backends(_settings(tmp_path)) as backends:
            assert await backends.graph.ping() is True
            assert backends.bus is not None
            await backends.bus.ping()
            graph, bus = backends.graph, backends.bus

        # Narrowed rather than cast: these settings force the embedded backends, and
        # `_conn` exists only there -- saying so keeps the checker honest about which
        # half of the union this test is actually exercising.
        assert isinstance(graph, SqliteGraphClient)
        assert isinstance(bus, SqliteEventBus)
        assert graph._conn is None, "a graph it opened must be closed"
        assert bus._conn is None, "a bus it opened must be closed"

    async def test_with_bus_false_opens_no_queue_connection(self, tmp_path: Path) -> None:
        """`atlas search` and `atlas ui` never publish. A queue connection they will not
        use is just a connection to leak -- but a search still embeds its query, so it
        still gets the limiter."""
        async with use_backends(_settings(tmp_path), with_bus=False) as backends:
            assert backends.bus is None
            assert isinstance(backends.limiter, SqliteRateLimiter)
            assert await backends.graph.ping() is True

    async def test_a_reused_graph_is_left_open(self, tmp_path: Path) -> None:
        """The MCP health check holds a live graph. Closing it to answer "are you
        reachable" would end the session that asked."""
        settings = _settings(tmp_path)
        async with SqliteGraphClient(tmp_path / "mine.sqlite3") as mine:
            await mine.ping()

            async with use_backends(settings, graph=mine) as backends:
                assert backends.graph is mine, "should have reused, not opened a second one"

            assert mine._conn is not None, "closed a connection it did not open"

    async def test_a_reused_bus_is_left_open(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        async with SqliteEventBus(tmp_path / "q.sqlite3") as mine:
            await mine.ping()

            async with use_backends(settings, bus=mine) as backends:
                assert backends.bus is mine

            assert mine._conn is not None, "closed a bus it did not open"

    async def test_a_reused_limiter_is_left_open(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        async with SqliteRateLimiter(tmp_path / "rl.sqlite3") as mine:
            await mine._get_conn()

            async with use_backends(settings, limiter=mine) as backends:
                assert backends.limiter is mine

            assert mine._conn is not None, "closed a limiter it did not open"

    async def test_a_reused_client_does_not_stop_the_other_being_opened(self, tmp_path: Path) -> None:
        """The mixed case, which is the one the MCP server actually hits: it holds a
        graph, and may or may not have a bus depending on --no-index."""
        settings = _settings(tmp_path)
        async with SqliteGraphClient(tmp_path / "g.sqlite3") as mine:
            await mine.ping()

            async with use_backends(settings, graph=mine) as backends:
                assert backends.graph is mine
                assert backends.bus is not None
                opened_bus = backends.bus

            assert mine._conn is not None, "reused graph was closed"
            assert isinstance(opened_bus, SqliteEventBus)
            assert opened_bus._conn is None, "opened bus was not closed"


class TestPostgresQueueSelection:
    async def test_declared_postgres_builds_the_postgres_bus_without_connecting(self, tmp_path: Path) -> None:
        """Construction is lazy, so selection needs no server; the label names the address only."""
        settings = AtlasSettings(
            project_root=tmp_path,
            backend=BackendSettings(
                graph={"sqlite": {}},
                queue={"postgres": PostgresSettings(host="pg.local", port=5999, password=SecretStr("s3cret"))},
            ),
        )
        connections = QueueConnections(settings)
        bus = await create_event_bus(settings, connections)

        assert isinstance(bus, PostgresEventBus)
        assert connections.postgres()._pool is None, "selecting a backend must not open a connection"
        label = queue_backend_label(bus, settings)
        assert label == "Postgres at pg.local:5999/atlas"
        assert "s3cret" not in label


class TestOneConnectionPerService:
    """The bus and the rate limiter borrow the same connection; neither opens its own."""

    async def test_valkey_bus_and_limiter_share_one_client(self, tmp_path: Path) -> None:
        settings = AtlasSettings(
            project_root=tmp_path,
            backend=BackendSettings(graph={"sqlite": {}}, queue={"valkey": RedisSettings(host="valkey.invalid")}),
        )
        async with use_backends(settings) as backends:
            assert isinstance(backends.bus, EventBus)
            assert isinstance(backends.limiter, RateLimiter)
            # Nothing connects to build either: the client is created, not dialled.
            assert backends.limiter._acquire_script.registered_client is backends.bus._redis

    async def test_postgres_bus_and_limiter_share_one_pool_holder(self, tmp_path: Path) -> None:
        settings = AtlasSettings(
            project_root=tmp_path,
            backend=BackendSettings(graph={"sqlite": {}}, queue={"postgres": PostgresSettings(host="pg.invalid")}),
        )
        async with use_backends(settings) as backends:
            assert isinstance(backends.bus, PostgresEventBus)
            assert isinstance(backends.limiter, PostgresRateLimiter)
            assert backends.limiter._connections is backends.bus._connections

    async def test_the_limiter_alone_opens_no_bus(self, tmp_path: Path) -> None:
        """Pacing without a bus: `atlas search` embeds its query and never publishes."""
        settings = AtlasSettings(
            project_root=tmp_path,
            backend=BackendSettings(graph={"sqlite": {}}, queue={"postgres": PostgresSettings(host="pg.invalid")}),
        )
        async with use_backends(settings, with_bus=False) as backends:
            assert backends.bus is None
            assert isinstance(backends.limiter, PostgresRateLimiter)
