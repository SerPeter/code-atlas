"""Integration tests for the hybrid search module."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from code_atlas.search.engine import SearchType, hybrid_search

pytestmark = pytest.mark.integration


class TestHybridSearchIntegration:
    """Integration tests requiring a live Memgraph instance."""

    @pytest.fixture
    async def seeded_graph(self, graph_client, settings):
        """Seed a few entities for search testing."""
        await graph_client.ensure_schema()

        # Create entities of different labels
        dim = graph_client._dimension
        dummy_vec = [0.1] * dim

        await graph_client.execute_write(
            "CREATE (n:Callable {uid: 'proj:mod.get_user', project_name: 'proj', "
            "name: 'get_user', qualified_name: 'mod.get_user', file_path: 'mod.py', "
            "kind: 'function', line_start: 10, line_end: 20, "
            "visibility: 'public', docstring: 'Get a user by ID', signature: 'def get_user(uid):', "
            "tags: [], embedding: $vec})",
            {"vec": dummy_vec},
        )
        await graph_client.execute_write(
            "CREATE (n:TypeDef {uid: 'proj:mod.UserService', project_name: 'proj', "
            "name: 'UserService', qualified_name: 'mod.UserService', file_path: 'mod.py', "
            "kind: 'class', line_start: 30, line_end: 50, "
            "visibility: 'public', docstring: 'Handles user operations', signature: 'class UserService:', "
            "tags: [], embedding: $vec})",
            {"vec": dummy_vec},
        )
        await graph_client.execute_write(
            "CREATE (n:Callable {uid: 'other:lib.get_user', project_name: 'other', "
            "name: 'get_user', qualified_name: 'lib.get_user', file_path: 'lib.py', "
            "kind: 'function', line_start: 1, line_end: 5, "
            "visibility: 'public', docstring: 'Another get_user', signature: 'def get_user():', "
            "tags: [], embedding: $vec})",
            {"vec": dummy_vec},
        )

        return graph_client

    async def test_graph_only(self, seeded_graph, settings):
        """Graph-only search finds entities by name matching."""
        results = await hybrid_search(
            graph=seeded_graph,
            embed=None,
            settings=settings.search,
            query="get_user",
            search_types=[SearchType.GRAPH],
            limit=10,
        )
        assert len(results) >= 2
        uids = [r.uid for r in results]
        assert "proj:mod.get_user" in uids
        assert "other:lib.get_user" in uids

    async def test_scope_filtering(self, seeded_graph, settings):
        """Scope filters results to a single project."""
        results = await hybrid_search(
            graph=seeded_graph,
            embed=None,
            settings=settings.search,
            query="get_user",
            search_types=[SearchType.GRAPH],
            limit=10,
            scope="proj",
        )
        uids = [r.uid for r in results]
        assert "proj:mod.get_user" in uids
        assert "other:lib.get_user" not in uids

    async def test_sources_provenance(self, seeded_graph, settings):
        """Each result tracks which channels found it."""
        results = await hybrid_search(
            graph=seeded_graph,
            embed=None,
            settings=settings.search,
            query="UserService",
            search_types=[SearchType.GRAPH],
            limit=10,
        )
        assert len(results) >= 1
        svc = next(r for r in results if r.uid == "proj:mod.UserService")
        assert "graph" in svc.sources
        assert svc.sources["graph"] >= 1

    async def test_hybrid_with_mock_embed(self, seeded_graph, settings):
        """Hybrid search with mocked embedding client."""
        dim = seeded_graph._dimension
        mock_embed = AsyncMock()
        mock_embed.embed_one = AsyncMock(return_value=[0.1] * dim)

        results = await hybrid_search(
            graph=seeded_graph,
            embed=mock_embed,
            settings=settings.search,
            query="get_user",
            limit=10,
        )
        # Should have results from multiple channels
        assert len(results) >= 1
        # At least one result should have graph source
        any_graph = any("graph" in r.sources for r in results)
        assert any_graph

    async def test_embed_failure_graceful(self, seeded_graph, settings):
        """If embedding fails, vector channel is skipped but others work."""
        mock_embed = AsyncMock()
        mock_embed.embed_one = AsyncMock(side_effect=RuntimeError("TEI down"))

        results = await hybrid_search(
            graph=seeded_graph,
            embed=mock_embed,
            settings=settings.search,
            query="get_user",
            limit=10,
        )
        # Should still get results from graph and/or bm25
        assert len(results) >= 1
        # No result should have vector source
        assert not any("vector" in r.sources for r in results)

    async def test_filter_excludes_test_entity(self, seeded_graph, settings):
        """Test entities are excluded by default filters."""
        dim = seeded_graph._dimension
        dummy_vec = [0.1] * dim

        # Seed a test entity
        await seeded_graph.execute_write(
            "CREATE (n:Callable {uid: 'proj:tests.test_mod.test_get_user', project_name: 'proj', "
            "name: 'test_get_user', qualified_name: 'tests.test_mod.test_get_user', "
            "file_path: 'tests/test_mod.py', kind: 'function', line_start: 1, line_end: 5, "
            "visibility: 'public', docstring: 'Test get_user', signature: 'def test_get_user():', "
            "tags: [], embedding: $vec})",
            {"vec": dummy_vec},
        )

        # Default: test entity excluded
        results = await hybrid_search(
            graph=seeded_graph,
            embed=None,
            settings=settings.search,
            query="test_get_user",
            search_types=[SearchType.GRAPH],
            limit=10,
        )
        uids = [r.uid for r in results]
        assert "proj:tests.test_mod.test_get_user" not in uids

        # Override: include tests
        results_with_tests = await hybrid_search(
            graph=seeded_graph,
            embed=None,
            settings=settings.search,
            query="test_get_user",
            search_types=[SearchType.GRAPH],
            limit=10,
            exclude_tests=False,
        )
        uids_with_tests = [r.uid for r in results_with_tests]
        assert "proj:tests.test_mod.test_get_user" in uids_with_tests
