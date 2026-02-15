"""Integration tests for the embeddings module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from code_atlas.search.embeddings import EmbedClient
from code_atlas.settings import AtlasSettings
from tests.conftest import TEST_DRAIN_TIMEOUT_S

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Model lock tests (integration-level, using graph_client fixture)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestModelLock:
    async def test_first_run_sets_config(self, graph_client):
        await graph_client.ensure_schema()

        result = await graph_client.get_embedding_config()
        assert result is None

        await graph_client.set_embedding_config("nomic-ai/nomic-embed-code", 768)
        result = await graph_client.get_embedding_config()
        assert result == ("nomic-ai/nomic-embed-code", 768)

    async def test_model_mismatch_detected(self, graph_client):
        await graph_client.ensure_schema()
        await graph_client.set_embedding_config("old-model", 768)

        stored = await graph_client.get_embedding_config()
        assert stored is not None
        stored_model, _ = stored
        assert stored_model == "old-model"
        assert stored_model != "new-model"

    async def test_clear_embeddings(self, graph_client):
        await graph_client.ensure_schema()
        # Create a test node with a correctly-dimensioned embedding (matches vector index)
        dim = graph_client._dimension
        await graph_client.execute_write(
            "CREATE (n:Module {uid: 'test:mod', qualified_name: 'mod', project_name: 'test', "
            "name: 'mod', file_path: 'mod.py', content_hash: 'h', project_root: '/tmp', "
            "embedding: $emb})",
            {"emb": [0.1] * dim},
        )
        # Clear all embeddings
        await graph_client.clear_all_embeddings()
        records = await graph_client.execute("MATCH (n {uid: 'test:mod'}) RETURN n.embedding AS emb")
        assert records[0]["emb"] is None


# ---------------------------------------------------------------------------
# TEI integration tests (require Memgraph + Valkey + TEI)
# ---------------------------------------------------------------------------


def _write(root: Path, rel_path: str, content: str = "") -> Path:
    """Write a file at root/rel_path, creating parent dirs."""
    p = root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


@pytest.mark.tei
@pytest.mark.integration
class TestTEIIntegration:
    """Tests that require a real TEI embedding service (TaylorAI/gte-tiny)."""

    async def test_dimension_auto_detected(self, tei_settings):
        """EmbedClient.detect_dimension() returns the correct dim from TEI."""
        client = EmbedClient(tei_settings.embeddings)
        dim = await client.detect_dimension()
        # TaylorAI/gte-tiny is 384-dim
        assert dim == 384

    async def test_index_writes_embeddings(self, tmp_path, tei_graph_client, tei_event_bus, tei_settings):
        """Full index with real TEI produces non-null embeddings on nodes."""
        from code_atlas.indexing.orchestrator import index_project

        _write(tmp_path, "app.py", 'def greet(name: str) -> str:\n    """Greet someone."""\n    return f"Hi {name}"\n')

        settings = AtlasSettings(
            project_root=tmp_path,
            memgraph=tei_settings.memgraph,
            redis=tei_settings.redis,
            embeddings=tei_settings.embeddings,
        )
        await tei_graph_client.ensure_schema()
        await index_project(settings, tei_graph_client, tei_event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # Check that at least one entity has a non-null embedding
        records = await tei_graph_client.execute(
            "MATCH (n:Callable) WHERE n.embedding IS NOT NULL RETURN count(n) AS cnt"
        )
        assert records[0]["cnt"] >= 1

    async def test_vector_search_returns_results(self, tmp_path, tei_graph_client, tei_event_bus, tei_settings):
        """Vector search via TEI embeddings returns relevant results."""
        from code_atlas.indexing.orchestrator import index_project

        _write(
            tmp_path,
            "math_utils.py",
            "def add(a: int, b: int) -> int:\n"
            '    """Add two numbers together."""\n'
            "    return a + b\n"
            "\n"
            "def multiply(x: int, y: int) -> int:\n"
            '    """Multiply two numbers."""\n'
            "    return x * y\n",
        )

        settings = AtlasSettings(
            project_root=tmp_path,
            memgraph=tei_settings.memgraph,
            redis=tei_settings.redis,
            embeddings=tei_settings.embeddings,
        )
        await tei_graph_client.ensure_schema()
        await index_project(settings, tei_graph_client, tei_event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # Perform a vector search
        client = EmbedClient(tei_settings.embeddings)
        query_vec = await client.embed_one("add two numbers")
        results = await tei_graph_client.vector_search(query_vec, limit=5)
        assert len(results) > 0, "Vector search should return at least one result"
        # Results are {"node": Node, "similarity": float}
        names = [r["node"]["name"] for r in results]
        assert "add" in names
