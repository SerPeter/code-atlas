"""`use_backends` — the one place a process acquires connections.

Its whole contract is an ownership rule: close what it opened, never what it was handed.
Get that backwards in either direction and you either leak a connection per call or
close one out from under the caller still using it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from code_atlas.backends import use_backends
from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.backends.sqlite_queue import SqliteEventBus
from code_atlas.settings import AtlasSettings, BackendSettings

if TYPE_CHECKING:
    from pathlib import Path


def _settings(tmp_path: Path) -> AtlasSettings:
    """Embedded backends only — no Memgraph, no Valkey, no network."""
    return AtlasSettings(project_root=tmp_path, backend=BackendSettings(graph="sqlite", queue="sqlite"))


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
        use is just a connection to leak."""
        async with use_backends(_settings(tmp_path), with_bus=False) as backends:
            assert backends.bus is None
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
