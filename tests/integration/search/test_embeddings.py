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
            "CREATE (n:Module:Entity {uid: 'test:mod', qualified_name: 'mod', project_name: 'test', "
            "name: 'mod', file_path: 'mod.py', content_hash: 'h', project_root: '/tmp', "
            "embedding: $emb})",
            {"emb": [0.1] * dim},
        )
        # Clear all embeddings
        await graph_client.clear_embeddings()
        records = await graph_client.execute("MATCH (n {uid: 'test:mod'}) RETURN n.embedding AS emb")
        assert records[0]["emb"] is None

    async def _make_embedded(self, graph_client, project: str, uid: str) -> None:
        dim = graph_client._dimension
        await graph_client.execute_write(
            "CREATE (n:Module:Entity {uid: $uid, qualified_name: $uid, project_name: $p, "
            "name: 'mod', file_path: 'mod.py', content_hash: 'h', project_root: '/tmp', "
            "embedding: $emb, embed_hash: 'eh'})",
            {"uid": uid, "p": project, "emb": [0.1] * dim},
        )

    async def test_clearing_one_project_leaves_the_others_embedded(self, graph_client):
        """The ATL-135 defect, at the storage layer.

        A model change belongs to one project. Clearing database-wide for it destroyed
        every other project's vectors, and said nothing.
        """
        await graph_client.ensure_schema()
        await self._make_embedded(graph_client, "test-alpha", "test-alpha:mod")
        await self._make_embedded(graph_client, "test-beta", "test-beta:mod")

        cleared = await graph_client.clear_embeddings("test-alpha")
        assert cleared == 1

        rows = await graph_client.execute(
            "MATCH (n:Entity) WHERE n.uid IN ['test-alpha:mod', 'test-beta:mod'] "
            "RETURN n.uid AS uid, n.embedding AS emb, n.embed_hash AS h ORDER BY uid"
        )
        by_uid = {r["uid"]: r for r in rows}
        assert by_uid["test-alpha:mod"]["emb"] is None
        assert by_uid["test-alpha:mod"]["h"] is None
        assert by_uid["test-beta:mod"]["emb"] is not None, "beta lost its vectors to alpha's model change"

    async def test_a_scoped_clear_reaches_sub_projects(self, graph_client):
        """Monorepo sub-projects are stored as "{root}/{sub}" and share the root's model."""
        await graph_client.ensure_schema()
        await self._make_embedded(graph_client, "test-root", "test-root:mod")
        await self._make_embedded(graph_client, "test-root/core", "test-root/core:mod")
        await self._make_embedded(graph_client, "test-rooted", "test-rooted:mod")

        cleared = await graph_client.clear_embeddings("test-root")
        assert cleared == 2, "the sub-project must be cleared with its root"

        rows = await graph_client.execute("MATCH (n:Entity {uid: 'test-rooted:mod'}) RETURN n.embedding AS emb")
        assert rows[0]["emb"] is not None, "prefix matching must not catch a differently-named project"

    async def test_project_embedding_model_is_recorded_per_project(self, graph_client):
        await graph_client.ensure_schema()
        for project in ("test-alpha", "test-beta"):
            await graph_client.execute_write(
                "CREATE (p:Project:Entity {uid: $uid, name: $name, project_name: $name, "
                "qualified_name: $name, file_path: '', content_hash: 'h'})",
                {"uid": f"{project}:project", "name": project},
            )

        assert await graph_client.get_project_embedding_model("test-alpha") is None

        await graph_client.set_project_embedding_model("test-alpha", "model-a")
        await graph_client.set_project_embedding_model("test-beta", "model-b")

        assert await graph_client.get_project_embedding_model("test-alpha") == "model-a"
        assert await graph_client.get_project_embedding_model("test-beta") == "model-b"
        assert await graph_client.get_embedding_models_by_project() == {
            "test-alpha": "model-a",
            "test-beta": "model-b",
        }

    async def test_counts_by_project_report_what_a_clear_would_destroy(self, graph_client):
        await graph_client.ensure_schema()
        await self._make_embedded(graph_client, "test-alpha", "test-alpha:mod")
        await self._make_embedded(graph_client, "test-beta", "test-beta:a")
        await self._make_embedded(graph_client, "test-beta", "test-beta:b")

        counts = await graph_client.count_embeddings_by_project()
        assert counts["test-alpha"] == 1
        assert counts["test-beta"] == 2

    async def _embedded(self, graph_client, uid: str, ehash: str, model: str, project: str = "test") -> None:
        dim = graph_client._dimension
        await graph_client.execute_write(
            "CREATE (n:Module:Entity {uid: $uid, qualified_name: $uid, project_name: $p, name: 'mod', "
            "file_path: 'mod.py', content_hash: 'h', project_root: '/tmp'})",
            {"uid": uid, "p": project},
        )
        await graph_client.write_embeddings_and_hashes([(uid, [0.3] * dim, ehash)], labels=["Module"], model=model)

    async def test_dedup_lookup_finds_a_vector_another_node_already_has(self, graph_client):
        """The whole point of ADR-0036: the graph is the dedup layer."""
        await graph_client.ensure_schema()
        await self._embedded(graph_client, "test:a", "shared-hash", "model-a")

        found = await graph_client.find_embeddings_by_hash(["shared-hash", "absent-hash"], "model-a")

        assert set(found) == {"shared-hash"}
        assert len(found["shared-hash"]) == graph_client._dimension

    async def test_dedup_lookup_crosses_projects(self, graph_client):
        """Cross-project dedup was the deleted cache's one real service; the graph
        keeps it, because every project shares one Memgraph."""
        await graph_client.ensure_schema()
        await self._embedded(graph_client, "test-alpha:a", "shared-hash", "model-a", project="test-alpha")

        found = await graph_client.find_embeddings_by_hash(["shared-hash"], "model-a")

        assert "shared-hash" in found

    async def test_dedup_lookup_refuses_a_different_models_vector(self, graph_client):
        """The failure this predicate exists to prevent.

        Two models produced 1536-dimensional vectors in this very database, so copying
        without the filter mixes embedding spaces and *nothing downstream can tell* --
        no dimension error, no exception, just silently meaningless distances.
        """
        await graph_client.ensure_schema()
        await self._embedded(graph_client, "test:a", "shared-hash", "model-a")

        found = await graph_client.find_embeddings_by_hash(["shared-hash"], "model-b")

        assert found == {}, "a vector from another model's space must not be copied"

    async def test_dedup_lookup_ignores_vectors_with_no_model_stamp(self, graph_client):
        """Legacy vectors predate the stamp, so their space is unknown.

        Dedup copies data, so it takes the strict half of the asymmetry: unknown means
        excluded here, even though search still ranks them.
        """
        await graph_client.ensure_schema()
        dim = graph_client._dimension
        await graph_client.execute_write(
            "CREATE (n:Module:Entity {uid: 'test:legacy', qualified_name: 'legacy', project_name: 'test', "
            "name: 'mod', file_path: 'mod.py', content_hash: 'h', project_root: '/tmp', "
            "embedding: $emb, embed_hash: 'legacy-hash'})",
            {"emb": [0.4] * dim},
        )

        found = await graph_client.find_embeddings_by_hash(["legacy-hash"], "model-a")

        assert found == {}

    async def test_the_model_is_stamped_on_the_vector_it_made(self, graph_client):
        """Two models at the same dimension are indistinguishable without the stamp --
        measured coexisting on the production graph at 1536d (ATL-135)."""
        await graph_client.ensure_schema()
        dim = graph_client._dimension
        await graph_client.execute_write(
            "CREATE (n:Module:Entity {uid: 'test:stamped', qualified_name: 'stamped', "
            "project_name: 'test', name: 'mod', file_path: 'mod.py', content_hash: 'h', "
            "project_root: '/tmp'})"
        )
        await graph_client.write_embeddings_and_hashes(
            [("test:stamped", [0.2] * dim, "hash-1")],
            labels=["Module"],
            model="openai/text-embedding-3-small",
        )

        rows = await graph_client.execute(
            "MATCH (n:Entity {uid: 'test:stamped'}) RETURN n.embed_model AS m, n.embed_hash AS h"
        )
        assert rows[0]["m"] == "openai/text-embedding-3-small"
        assert rows[0]["h"] == "hash-1"


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
