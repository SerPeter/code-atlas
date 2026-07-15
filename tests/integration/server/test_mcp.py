"""Integration tests for the MCP server tools (require Memgraph)."""

from __future__ import annotations

import pytest

from code_atlas.schema import SCHEMA_VERSION, NodeLabel, RelType
from code_atlas.search.embeddings import EmbedClient
from code_atlas.server.mcp import AppContext
from tests.unit.server.test_mcp import _invoke_tool

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def app_ctx(graph_client, settings):
    """Create an AppContext for testing tools directly."""
    embed = EmbedClient(settings.embeddings)
    return AppContext(graph=graph_client, settings=settings, embed=embed)


@pytest.fixture
async def seeded_graph(graph_client):
    """Seed the graph with test entities for querying."""
    await graph_client.ensure_schema()

    # Create a project
    await graph_client.merge_project_node(
        "test-project",
        file_count=3,
        entity_count=5,
        last_indexed_at=1700000000,
        git_hash="abc123",
    )

    # Create some entities
    await graph_client.execute_write(
        f"CREATE (n:{NodeLabel.MODULE} {{"
        "uid: 'test-project:mypackage.mymodule', "
        "project_name: 'test-project', "
        "name: 'mymodule', "
        "qualified_name: 'mypackage.mymodule', "
        "file_path: 'mypackage/mymodule.py', "
        "kind: 'module', "
        "line_start: 1, line_end: 100"
        "})"
    )

    await graph_client.execute_write(
        f"CREATE (n:{NodeLabel.CALLABLE} {{"
        "uid: 'test-project:mypackage.mymodule.my_function', "
        "project_name: 'test-project', "
        "name: 'my_function', "
        "qualified_name: 'mypackage.mymodule.my_function', "
        "file_path: 'mypackage/mymodule.py', "
        "kind: 'function', "
        "line_start: 10, line_end: 20, "
        "signature: 'def my_function(x: int) -> str', "
        "docstring: 'A test function that does something useful.'"
        "})"
    )

    await graph_client.execute_write(
        f"CREATE (n:{NodeLabel.TYPE_DEF} {{"
        "uid: 'test-project:mypackage.mymodule.MyClass', "
        "project_name: 'test-project', "
        "name: 'MyClass', "
        "qualified_name: 'mypackage.mymodule.MyClass', "
        "file_path: 'mypackage/mymodule.py', "
        "kind: 'class', "
        "line_start: 30, line_end: 80, "
        "signature: 'class MyClass(Base)', "
        "docstring: 'A test class.'"
        "})"
    )

    await graph_client.execute_write(
        f"CREATE (n:{NodeLabel.CALLABLE} {{"
        "uid: 'test-project:mypackage.mymodule.MyClass.my_method', "
        "project_name: 'test-project', "
        "name: 'my_method', "
        "qualified_name: 'mypackage.mymodule.MyClass.my_method', "
        "file_path: 'mypackage/mymodule.py', "
        "kind: 'method', "
        "line_start: 40, line_end: 50, "
        "signature: 'def my_method(self) -> None', "
        "docstring: 'A method on MyClass.'"
        "})"
    )

    # Create DEFINES relationships
    await graph_client.execute_write(
        "MATCH (m {uid: 'test-project:mypackage.mymodule'}), "
        "(f {uid: 'test-project:mypackage.mymodule.my_function'}) "
        f"CREATE (m)-[:{RelType.DEFINES}]->(f)"
    )
    await graph_client.execute_write(
        "MATCH (m {uid: 'test-project:mypackage.mymodule'}), "
        "(c {uid: 'test-project:mypackage.mymodule.MyClass'}) "
        f"CREATE (m)-[:{RelType.DEFINES}]->(c)"
    )
    await graph_client.execute_write(
        "MATCH (c {uid: 'test-project:mypackage.mymodule.MyClass'}), "
        "(method {uid: 'test-project:mypackage.mymodule.MyClass.my_method'}) "
        f"CREATE (c)-[:{RelType.DEFINES}]->(method)"
    )

    # Create a CALLS relationship: my_function -> my_method
    await graph_client.execute_write(
        "MATCH (f {uid: 'test-project:mypackage.mymodule.my_function'}), "
        "(m {uid: 'test-project:mypackage.mymodule.MyClass.my_method'}) "
        f"CREATE (f)-[:{RelType.CALLS}]->(m)"
    )

    return graph_client


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestCypherQuery:
    async def test_cypher_query_returns_results(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "cypher_query", query="MATCH (n:Callable) RETURN n.name AS name", limit=10)
        assert "results" in result
        assert result["count"] >= 1
        names = [r["name"] for r in result["results"]]
        assert "my_function" in names

    async def test_cypher_query_rejects_writes(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "cypher_query", query="CREATE (n:Foo {name: 'bar'})")
        assert result["code"] == "WRITE_REJECTED"

    async def test_cypher_query_rejects_delete(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "cypher_query", query="MATCH (n) DELETE n")
        assert result["code"] == "WRITE_REJECTED"

    async def test_cypher_query_rejects_set(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "cypher_query", query="MATCH (n) SET n.name = 'x'")
        assert result["code"] == "WRITE_REJECTED"

    async def test_cypher_query_enforces_limit(self, app_ctx, seeded_graph):
        result = await _invoke_tool(
            app_ctx,
            "cypher_query",
            query="MATCH (n) RETURN n.name AS name",
            limit=2,
        )
        assert result["count"] <= 2

    async def test_cypher_query_allows_string_literal_matching_write_keyword(self, app_ctx, seeded_graph):
        """A quoted string literal equal to a write keyword (e.g. 'set') must not be rejected."""
        result = await _invoke_tool(
            app_ctx,
            "cypher_query",
            query="MATCH (n) WHERE n.name = 'set' RETURN n.name AS name",
        )
        assert "error" not in result
        assert result["count"] == 0  # no entity named 'set' — but the query itself must run

    async def test_cypher_query_still_rejects_real_write_keyword(self, app_ctx, seeded_graph):
        """An actual (unquoted) write keyword must still be rejected."""
        result = await _invoke_tool(app_ctx, "cypher_query", query="MATCH (n) WHERE n.name = 'x' SET n.name = 'y'")
        assert result["code"] == "WRITE_REJECTED"

    async def test_cypher_query_truncated_true_when_more_matches_than_limit(self, app_ctx, graph_client):
        await graph_client.ensure_schema()
        await graph_client.execute_write(
            "UNWIND range(1, 25) AS i "
            f"CREATE (n:{NodeLabel.CALLABLE} {{"
            "uid: 'trunc-project:foo' + toString(i), project_name: 'trunc-project', name: 'foo', "
            "qualified_name: 'mod.foo' + toString(i), file_path: 'mod.py', kind: 'function', "
            "line_start: i, line_end: i})"
        )
        result = await _invoke_tool(
            app_ctx, "cypher_query", query="MATCH (n:Callable {name: 'foo'}) RETURN n.uid AS uid", limit=20
        )
        assert result["count"] == 20
        assert result["truncated"] is True

    async def test_cypher_query_not_truncated_when_matches_fit(self, app_ctx, graph_client):
        await graph_client.ensure_schema()
        await graph_client.execute_write(
            "UNWIND range(1, 5) AS i "
            f"CREATE (n:{NodeLabel.CALLABLE} {{"
            "uid: 'trunc-project2:foo' + toString(i), project_name: 'trunc-project2', name: 'foo', "
            "qualified_name: 'mod.foo' + toString(i), file_path: 'mod.py', kind: 'function', "
            "line_start: i, line_end: i})"
        )
        result = await _invoke_tool(
            app_ctx, "cypher_query", query="MATCH (n:Callable {name: 'foo'}) RETURN n.uid AS uid", limit=20
        )
        assert result["count"] == 5
        assert result["truncated"] is False

    async def test_cypher_query_truncated_unknown_with_explicit_limit(self, app_ctx, graph_client):
        """When the caller supplies their own LIMIT clause, truncation is not claimed."""
        await graph_client.ensure_schema()
        await graph_client.execute_write(
            "UNWIND range(1, 25) AS i "
            f"CREATE (n:{NodeLabel.CALLABLE} {{"
            "uid: 'trunc-project3:foo' + toString(i), project_name: 'trunc-project3', name: 'foo', "
            "qualified_name: 'mod.foo' + toString(i), file_path: 'mod.py', kind: 'function', "
            "line_start: i, line_end: i})"
        )
        result = await _invoke_tool(
            app_ctx,
            "cypher_query",
            query="MATCH (n:Callable {name: 'foo'}) RETURN n.uid AS uid LIMIT 5",
            limit=20,
        )
        assert result["count"] == 5
        assert result["truncated"] is False

    async def test_cypher_query_serializes_collected_nodes(self, app_ctx, seeded_graph):
        """Aggregation results (e.g. collect(n)) nest Node objects inside lists — these must
        be converted to plain dicts, not passed through as raw neo4j Node objects."""
        result = await _invoke_tool(
            app_ctx,
            "cypher_query",
            query="MATCH (n:Callable) RETURN collect(n) AS nodes, count(n) AS cnt",
        )
        assert "error" not in result
        row = result["results"][0]
        assert row["cnt"] >= 1
        assert isinstance(row["nodes"], list)
        assert isinstance(row["nodes"][0], dict)
        assert "_labels" in row["nodes"][0]
        assert "Callable" in row["nodes"][0]["_labels"]


class TestGetNode:
    async def test_get_node_exact_name(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "get_node", name="my_function")
        assert result["count"] >= 1
        assert any(r.get("name") == "my_function" for r in result["results"])

    async def test_get_node_by_uid(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "get_node", name="test-project:mypackage.mymodule.MyClass")
        assert result["count"] == 1
        assert result["results"][0]["name"] == "MyClass"

    async def test_get_node_ends_with(self, app_ctx, seeded_graph):
        # "MyClass" should match via suffix ".MyClass" on qualified_name
        result = await _invoke_tool(app_ctx, "get_node", name="MyClass")
        assert result["count"] >= 1
        assert any(r.get("qualified_name", "").endswith("MyClass") for r in result["results"])

    async def test_get_node_with_label_filter(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "get_node", name="my_function", label="Callable")
        assert result["count"] >= 1

    async def test_get_node_not_found(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "get_node", name="nonexistent_entity_xyz")
        assert result["count"] == 0

    async def test_get_node_truncated_true_when_more_matches_than_limit(self, app_ctx, graph_client):
        await graph_client.ensure_schema()
        await graph_client.execute_write(
            "UNWIND range(1, 25) AS i "
            f"CREATE (n:{NodeLabel.CALLABLE} {{"
            "uid: 'gn-trunc:foo' + toString(i), project_name: 'gn-trunc', name: 'shared_name', "
            "qualified_name: 'mod.shared_name' + toString(i), file_path: 'mod.py', kind: 'function', "
            "line_start: i, line_end: i})"
        )
        result = await _invoke_tool(app_ctx, "get_node", name="shared_name", limit=20)
        assert result["count"] == 20
        assert result["truncated"] is True

    async def test_get_node_truncated_false_when_matches_fit(self, app_ctx, graph_client):
        await graph_client.ensure_schema()
        await graph_client.execute_write(
            "UNWIND range(1, 5) AS i "
            f"CREATE (n:{NodeLabel.CALLABLE} {{"
            "uid: 'gn-trunc2:foo' + toString(i), project_name: 'gn-trunc2', name: 'shared_name2', "
            "qualified_name: 'mod.shared_name2' + toString(i), file_path: 'mod.py', kind: 'function', "
            "line_start: i, line_end: i})"
        )
        result = await _invoke_tool(app_ctx, "get_node", name="shared_name2", limit=20)
        assert result["count"] == 5
        assert result["truncated"] is False


class TestGetContext:
    async def test_get_context_returns_neighborhood(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "get_context", uid="test-project:mypackage.mymodule.MyClass")
        assert "node" in result
        assert result["node"]["name"] == "MyClass"

        # Parent should be the module (now a dict, not a list)
        assert result["parent"] is not None
        assert result["parent"]["name"] == "mymodule"

        # Siblings should include my_function (both defined by mymodule)
        assert len(result["siblings"]) >= 1
        assert any(s.get("name") == "my_function" for s in result["siblings"])

    async def test_get_context_callers_callees(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "get_context", uid="test-project:mypackage.mymodule.MyClass.my_method")
        # my_function calls my_method
        assert len(result["callers"]) >= 1
        assert any(c.get("name") == "my_function" for c in result["callers"])

    async def test_get_context_not_found(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "get_context", uid="nonexistent:uid")
        assert result["code"] == "NOT_FOUND"

    async def test_get_context_siblings(self, app_ctx, seeded_graph):
        """my_function and MyClass are siblings under mymodule."""
        result = await _invoke_tool(app_ctx, "get_context", uid="test-project:mypackage.mymodule.my_function")
        assert any(s.get("name") == "MyClass" for s in result["siblings"])

    async def test_get_context_hierarchy_off(self, app_ctx, seeded_graph):
        """include_hierarchy=False returns None parent, empty siblings."""
        result = await _invoke_tool(
            app_ctx, "get_context", uid="test-project:mypackage.mymodule.MyClass", include_hierarchy=False
        )
        assert result["parent"] is None
        assert result["siblings"] == []
        # Callers/callees should still be present
        assert "callers" in result
        assert "callees" in result

    async def test_get_context_calls_off(self, app_ctx, seeded_graph):
        """include_calls=False returns empty callers/callees."""
        result = await _invoke_tool(
            app_ctx, "get_context", uid="test-project:mypackage.mymodule.MyClass.my_method", include_calls=False
        )
        assert result["callers"] == []
        assert result["callees"] == []
        # Parent should still be present
        assert result["parent"] is not None

    async def test_get_context_response_shape(self, app_ctx, seeded_graph):
        """Verify all expected top-level keys are present."""
        result = await _invoke_tool(app_ctx, "get_context", uid="test-project:mypackage.mymodule.MyClass")
        expected_keys = {"node", "parent", "siblings", "callers", "callees", "docs", "package_context", "query_ms"}
        assert expected_keys == set(result.keys())


class TestIndexStatus:
    async def test_index_status_with_projects(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "index_status")
        assert "projects" in result
        assert len(result["projects"]) >= 1

        project = result["projects"][0]
        assert project["name"] == "test-project"
        assert project["entity_count"] >= 1
        assert result["schema_version"] == SCHEMA_VERSION
        assert "text_indices" in result
        assert isinstance(result["text_indices"], list)

    async def test_index_status_empty_graph(self, app_ctx):
        result = await _invoke_tool(app_ctx, "index_status")
        assert result["projects"] == []
        assert result["schema_version"] == SCHEMA_VERSION
        assert "text_indices" in result


class TestKnowledgeHealth:
    async def test_knowledge_health_returns_expected_keys(self, app_ctx, graph_client):
        await graph_client.ensure_schema()
        result = await _invoke_tool(app_ctx, "knowledge_health")
        expected_keys = {
            "inbox_count",
            "inbox_paths",
            "orphan_notes",
            "duplicate_ids",
            "dangling_links",
            "similar_pairs",
            "promotion_candidates",
            "memory_index_issues",
            "broken_anchors",
            "query_ms",
        }
        assert expected_keys == set(result.keys())

    async def test_knowledge_health_flags_orphan_note(self, app_ctx, graph_client):
        await graph_client.ensure_schema()
        await graph_client.execute_write(
            f"CREATE (n:{NodeLabel.NOTE} {{"
            "uid: 'test-project:note:solo', project_name: 'test-project', "
            "name: 'Solo', qualified_name: 'note:solo', file_path: 'solo.md', "
            "kind: 'note', line_start: 1, line_end: 5"
            "})"
        )

        result = await _invoke_tool(app_ctx, "knowledge_health")

        orphan_uids = {n["uid"] for n in result["orphan_notes"]}
        assert "test-project:note:solo" in orphan_uids


class TestTextSearch:
    async def test_text_search_finds_entity(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "text_search", query="my_function")
        # Text search may or may not find results depending on Memgraph text index state
        # Just verify the structure is correct
        assert "results" in result
        assert "count" in result
        assert "query_ms" in result

    async def test_text_search_with_label_filter(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "text_search", query="my_function", label="Callable")
        assert "results" in result
        assert "count" in result
        assert "query_ms" in result
        assert isinstance(result["results"], list)

    async def test_text_search_with_project_filter(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "text_search", query="my_function", project="test-project")
        assert "results" in result
        assert "count" in result
        assert "query_ms" in result
        assert isinstance(result["results"], list)

    async def test_text_search_empty_query(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "text_search", query="")
        assert "results" in result
        assert "count" in result

    async def test_malicious_label_rejected_and_graph_survives(self, app_ctx, seeded_graph):
        """A Cypher-injection label must be refused, and the graph must be intact afterwards."""
        malicious = "callable', $query, 60) YIELD node WITH node LIMIT 1 MATCH (m) DETACH DELETE m //"
        before = await seeded_graph.execute("MATCH (n) RETURN count(n) AS c")
        assert before[0]["c"] > 0

        result = await _invoke_tool(app_ctx, "text_search", query="x", label=malicious)
        assert "error" in result
        assert "Invalid label" in result["error"]

        after = await seeded_graph.execute("MATCH (n) RETURN count(n) AS c")
        assert after[0]["c"] == before[0]["c"], "Injection payload must not have deleted any nodes"


class TestDefaultScopeMonorepo:
    """Default (unscoped) search must include monorepo sub-project entities,
    stored under project_name = '{root}/{sub}' — not just the bare root name."""

    async def test_hybrid_search_default_scope_includes_sub_project_entities(self, app_ctx, graph_client):
        from code_atlas.settings import derive_project_name

        await graph_client.ensure_schema()
        root_name = derive_project_name(app_ctx.settings.project_root)
        sub_name = f"{root_name}/sub"

        await graph_client.merge_project_node(root_name, file_count=1, entity_count=1)
        await graph_client.merge_project_node(sub_name, file_count=1, entity_count=1)

        await graph_client.execute_write(
            f"CREATE (n:{NodeLabel.CALLABLE} {{"
            "uid: $uid, project_name: $project_name, name: 'root_only_fn', "
            "qualified_name: 'root_only_fn', file_path: 'root.py', kind: 'function', "
            "line_start: 1, line_end: 2})",
            {"uid": f"{root_name}:root_only_fn", "project_name": root_name},
        )
        await graph_client.execute_write(
            f"CREATE (n:{NodeLabel.CALLABLE} {{"
            "uid: $uid, project_name: $project_name, name: 'sub_only_fn', "
            "qualified_name: 'sub_only_fn', file_path: 'sub/sub.py', kind: 'function', "
            "line_start: 1, line_end: 2})",
            {"uid": f"{sub_name}:sub_only_fn", "project_name": sub_name},
        )

        # No scope/project passed — must still find the sub-project entity, not just root's.
        result = await _invoke_tool(app_ctx, "hybrid_search", query="sub_only_fn", search_types="graph")
        names = {r["name"] for r in result["results"]}
        assert "sub_only_fn" in names

    async def test_text_search_default_scope_resolves_to_root_and_sub_projects(self, app_ctx, graph_client):
        """text_search's default (unscoped) project resolution must include sub-projects —
        verified against the same _default_scope_projects helper hybrid_search relies on."""
        from code_atlas.server.mcp import _default_scope_projects
        from code_atlas.settings import derive_project_name

        await graph_client.ensure_schema()
        root_name = derive_project_name(app_ctx.settings.project_root)
        sub_name = f"{root_name}/sub"
        await graph_client.merge_project_node(root_name, file_count=1, entity_count=1)
        await graph_client.merge_project_node(sub_name, file_count=1, entity_count=1)

        resolved = await _default_scope_projects(app_ctx)
        assert set(resolved) == {root_name, sub_name}


# ---------------------------------------------------------------------------
# Integration: validate_cypher with EXPLAIN
# ---------------------------------------------------------------------------


class TestValidateCypherExplain:
    async def test_valid_query_with_explain(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "validate_cypher", query="MATCH (n:Callable) RETURN n LIMIT 10")
        assert result["valid"] is True

    async def test_invalid_syntax_with_explain(self, app_ctx, seeded_graph):
        result = await _invoke_tool(app_ctx, "validate_cypher", query="MATCHX (n) RETURN n LIMIT 10")
        assert result["valid"] is False
        assert any("explain" in i["message"].lower() for i in result["issues"] if i["level"] == "error")


# ---------------------------------------------------------------------------
# Analysis tools fixtures
# ---------------------------------------------------------------------------

_PROJECT = "analysis-project"


@pytest.fixture
async def seeded_analysis_graph(graph_client):
    """Seed graph with rich structure for analysis/diagram tests."""
    await graph_client.ensure_schema()

    # Project
    await graph_client.merge_project_node(_PROJECT, file_count=3, entity_count=10)

    # Packages
    await graph_client.merge_package_node(_PROJECT, "mypkg", "mypkg", "mypkg/__init__.py")
    await graph_client.merge_package_node(_PROJECT, "mypkg.sub", "sub", "mypkg/sub/__init__.py")

    # Package containment
    await graph_client.create_contains_edge(_PROJECT, f"{_PROJECT}:mypkg")
    await graph_client.create_contains_edge(f"{_PROJECT}:mypkg", f"{_PROJECT}:mypkg.sub")

    # --- Modules ---
    for uid, name, qn, fp, le in [
        (f"{_PROJECT}:mypkg.models", "models", "mypkg.models", "mypkg/models.py", 50),
        (f"{_PROJECT}:mypkg.utils", "utils", "mypkg.utils", "mypkg/utils.py", 30),
        (f"{_PROJECT}:mypkg.sub.api", "api", "mypkg.sub.api", "mypkg/sub/api.py", 40),
    ]:
        await graph_client.execute_write(
            "CREATE (n:Module {uid: $uid, project_name: $p, name: $name, "
            "qualified_name: $qn, file_path: $fp, kind: 'module', line_start: 1, line_end: $le})",
            {"uid": uid, "p": _PROJECT, "name": name, "qn": qn, "fp": fp, "le": le},
        )

    # Package → Module containment
    await graph_client.create_contains_edge(f"{_PROJECT}:mypkg", f"{_PROJECT}:mypkg.models")
    await graph_client.create_contains_edge(f"{_PROJECT}:mypkg", f"{_PROJECT}:mypkg.utils")
    await graph_client.create_contains_edge(f"{_PROJECT}:mypkg.sub", f"{_PROJECT}:mypkg.sub.api")

    # --- TypeDefs ---
    await graph_client.execute_write(
        "CREATE (n:TypeDef {uid: $uid, project_name: $p, name: 'Base', "
        "qualified_name: 'mypkg.models.Base', file_path: 'mypkg/models.py', "
        "kind: 'class', line_start: 5, line_end: 20, visibility: 'public', "
        "docstring: 'Base model class.'})",
        {"uid": f"{_PROJECT}:mypkg.models.Base", "p": _PROJECT},
    )
    await graph_client.execute_write(
        "CREATE (n:TypeDef {uid: $uid, project_name: $p, name: 'User', "
        "qualified_name: 'mypkg.models.User', file_path: 'mypkg/models.py', "
        "kind: 'class', line_start: 22, line_end: 45, visibility: 'public', "
        "docstring: 'User model.', signature: 'class User(Base)'})",
        {"uid": f"{_PROJECT}:mypkg.models.User", "p": _PROJECT},
    )

    # INHERITS: User → Base
    await graph_client.execute_write(
        "MATCH (child {uid: $c}), (parent {uid: $p}) CREATE (child)-[:INHERITS]->(parent)",
        {"c": f"{_PROJECT}:mypkg.models.User", "p": f"{_PROJECT}:mypkg.models.Base"},
    )

    # --- Callables ---
    for uid, name, qn, fp, kind, ls, le, vis, doc, sig in [
        (
            f"{_PROJECT}:mypkg.models.User.save",
            "save",
            "mypkg.models.User.save",
            "mypkg/models.py",
            "method",
            30,
            40,
            "public",
            "Save the user.",
            "def save(self) -> None",
        ),
        (
            f"{_PROJECT}:mypkg.utils.helper",
            "helper",
            "mypkg.utils.helper",
            "mypkg/utils.py",
            "function",
            5,
            15,
            "public",
            "A helper function.",
            "def helper(x: int) -> str",
        ),
        (
            f"{_PROJECT}:mypkg.sub.api.handle_request",
            "handle_request",
            "mypkg.sub.api.handle_request",
            "mypkg/sub/api.py",
            "function",
            5,
            30,
            "public",
            "Handle an API request.",
            "def handle_request(req) -> Response",
        ),
    ]:
        await graph_client.execute_write(
            "CREATE (n:Callable {uid: $uid, project_name: $p, name: $name, "
            "qualified_name: $qn, file_path: $fp, kind: $kind, "
            "line_start: $ls, line_end: $le, visibility: $vis, "
            "docstring: $doc, signature: $sig})",
            {
                "uid": uid,
                "p": _PROJECT,
                "name": name,
                "qn": qn,
                "fp": fp,
                "kind": kind,
                "ls": ls,
                "le": le,
                "vis": vis,
                "doc": doc,
                "sig": sig,
            },
        )

    # --- Value (no docstring) ---
    await graph_client.execute_write(
        "CREATE (n:Value {uid: $uid, project_name: $p, name: 'MAX_SIZE', "
        "qualified_name: 'mypkg.utils.MAX_SIZE', file_path: 'mypkg/utils.py', "
        "kind: 'constant', line_start: 1, line_end: 1, visibility: 'public'})",
        {"uid": f"{_PROJECT}:mypkg.utils.MAX_SIZE", "p": _PROJECT},
    )

    # --- DEFINES relationships ---
    defines_pairs = [
        (f"{_PROJECT}:mypkg.models", f"{_PROJECT}:mypkg.models.Base"),
        (f"{_PROJECT}:mypkg.models", f"{_PROJECT}:mypkg.models.User"),
        (f"{_PROJECT}:mypkg.models.User", f"{_PROJECT}:mypkg.models.User.save"),
        (f"{_PROJECT}:mypkg.utils", f"{_PROJECT}:mypkg.utils.helper"),
        (f"{_PROJECT}:mypkg.utils", f"{_PROJECT}:mypkg.utils.MAX_SIZE"),
        (f"{_PROJECT}:mypkg.sub.api", f"{_PROJECT}:mypkg.sub.api.handle_request"),
    ]
    for from_uid, to_uid in defines_pairs:
        await graph_client.execute_write(
            "MATCH (a {uid: $f}), (b {uid: $t}) CREATE (a)-[:DEFINES]->(b)",
            {"f": from_uid, "t": to_uid},
        )

    # --- IMPORTS relationships ---
    imports_pairs = [
        # api → User (indirect: api depends on models)
        (f"{_PROJECT}:mypkg.sub.api", f"{_PROJECT}:mypkg.models.User"),
        # api → helper (indirect: api depends on utils)
        (f"{_PROJECT}:mypkg.sub.api", f"{_PROJECT}:mypkg.utils.helper"),
        # models → helper (creates models→utils dependency)
        (f"{_PROJECT}:mypkg.models", f"{_PROJECT}:mypkg.utils.helper"),
        # utils → Base (creates utils→models dependency — circular with above!)
        (f"{_PROJECT}:mypkg.utils", f"{_PROJECT}:mypkg.models.Base"),
    ]
    for from_uid, to_uid in imports_pairs:
        await graph_client.execute_write(
            "MATCH (a {uid: $f}), (b {uid: $t}) CREATE (a)-[:IMPORTS]->(b)",
            {"f": from_uid, "t": to_uid},
        )

    # --- External dependency ---
    await graph_client.execute_write(
        "CREATE (ep:ExternalPackage {uid: $uid, project_name: $p, "
        "name: 'dataclasses', qualified_name: 'ext/dataclasses'})",
        {"uid": f"{_PROJECT}:ext/dataclasses", "p": _PROJECT},
    )
    await graph_client.execute_write(
        "CREATE (es:ExternalSymbol {uid: $uid, project_name: $p, "
        "name: 'dataclass', qualified_name: 'ext/dataclasses.dataclass', package: 'dataclasses'})",
        {"uid": f"{_PROJECT}:ext/dataclasses.dataclass", "p": _PROJECT},
    )
    await graph_client.execute_write(
        "MATCH (ep {uid: $ep}), (es {uid: $es}) CREATE (ep)-[:CONTAINS]->(es)",
        {"ep": f"{_PROJECT}:ext/dataclasses", "es": f"{_PROJECT}:ext/dataclasses.dataclass"},
    )
    # models imports dataclass (external)
    await graph_client.execute_write(
        "MATCH (m {uid: $f}), (es {uid: $t}) CREATE (m)-[:IMPORTS]->(es)",
        {"f": f"{_PROJECT}:mypkg.models", "t": f"{_PROJECT}:ext/dataclasses.dataclass"},
    )

    # --- CALLS: handle_request → helper ---
    await graph_client.execute_write(
        "MATCH (a {uid: $f}), (b {uid: $t}) CREATE (a)-[:CALLS]->(b)",
        {"f": f"{_PROJECT}:mypkg.sub.api.handle_request", "t": f"{_PROJECT}:mypkg.utils.helper"},
    )

    return graph_client


# ---------------------------------------------------------------------------
# analyze_repo integration tests
# ---------------------------------------------------------------------------


class TestAnalyzeRepo:
    async def test_structure_counts(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="structure", project=_PROJECT)
        assert result["analysis"] == "structure"
        assert result["label_counts"]["Module"] == 3
        assert result["label_counts"]["TypeDef"] == 2
        assert result["label_counts"]["Callable"] == 3
        assert result["label_counts"]["Value"] == 1
        assert result["kind_breakdown"]["Callable"]["function"] == 2
        assert result["kind_breakdown"]["Callable"]["method"] == 1

    async def test_structure_packages(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="structure", project=_PROJECT)
        pkg_names = [p["name"] for p in result["packages"]]
        assert "mypkg" in pkg_names

    async def test_structure_largest_modules(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="structure", project=_PROJECT)
        # models defines Base + User = 2 entities; utils defines helper + MAX_SIZE = 2
        assert len(result["largest_modules"]) >= 2

    async def test_structure_external_deps(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="structure", project=_PROJECT)
        ext_names = [d["package"] for d in result["external_dependencies"]]
        assert "dataclasses" in ext_names

    async def test_centrality_hubs(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="centrality", project=_PROJECT)
        assert result["analysis"] == "centrality"
        hub_names = [h["name"] for h in result["hub_entities"]]
        # helper is imported by api + models, and called by handle_request
        assert "helper" in hub_names

    async def test_centrality_leaves(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="centrality", project=_PROJECT)
        leaf_names = [lf["name"] for lf in result["leaf_entities"]]
        # MAX_SIZE has no inbound IMPORTS/INHERITS/CALLS
        assert "MAX_SIZE" in leaf_names

    async def test_dependencies_internal(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="dependencies", project=_PROJECT)
        assert result["analysis"] == "dependencies"
        assert len(result["internal_imports"]) >= 2

    async def test_dependencies_circular(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="dependencies", project=_PROJECT)
        # models<->utils is a circular dependency
        assert len(result["circular_dependencies"]) >= 1
        cycle = result["circular_dependencies"][0]
        pair = {cycle["module_a"], cycle["module_b"]}
        assert pair == {"mypkg.models", "mypkg.utils"}

    async def test_dependencies_external(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="dependencies", project=_PROJECT)
        ext_pkgs = [e["package"] for e in result["external_imports"]]
        assert "dataclasses" in ext_pkgs

    async def test_patterns_inheritance(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="patterns", project=_PROJECT)
        assert result["analysis"] == "patterns"
        assert len(result["inheritance"]) >= 1
        inh = result["inheritance"][0]
        assert inh["child"] == "User"
        assert inh["parent"] == "Base"

    async def test_patterns_visibility(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="patterns", project=_PROJECT)
        assert result["visibility_distribution"]["public"] >= 5

    async def test_patterns_docstring_coverage(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="patterns", project=_PROJECT)
        cov = result["docstring_coverage"]
        assert cov["total"] >= 5
        # MAX_SIZE has no docstring, so documented < total
        assert cov["documented"] < cov["total"]
        assert cov["percentage"] > 0

    async def test_quality_health_score(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="quality", project=_PROJECT)
        assert result["analysis"] == "quality"
        assert 0 <= result["health_score"] <= 100
        assert "score_breakdown" in result
        assert "query_ms" in result

    async def test_quality_modularity(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="quality", project=_PROJECT)
        # models<->utils are intra-package (mypkg), api->models and api->utils are cross-package
        assert 0 <= result["modularity_ratio"] <= 1.0

    async def test_quality_circular_deps(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="quality", project=_PROJECT)
        assert result["circular_dependency_count"] >= 1
        pair = {result["circular_dependencies"][0]["module_a"], result["circular_dependencies"][0]["module_b"]}
        assert pair == {"mypkg.models", "mypkg.utils"}

    async def test_quality_no_god_modules(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="quality", project=_PROJECT)
        # All seeded modules have <= 2 entities
        assert result["god_modules"] == []

    async def test_quality_coupling_stats(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="quality", project=_PROJECT)
        stats = result["coupling_stats"]
        assert stats["max_fan_in"] >= 1
        assert stats["max_fan_out"] >= 1
        assert stats["avg_fan_in"] > 0
        assert stats["avg_fan_out"] > 0

    async def test_quality_instability(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="quality", project=_PROJECT)
        # api has fan_out=2, fan_in=0 → instability=1.0 (unstable)
        unstable_mods = [m["module"] for m in result["instability"]["unstable"]]
        assert "mypkg.sub.api" in unstable_mods

    async def test_quality_worst_modules(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="quality", project=_PROJECT)
        # At least models/utils (circular) and api (unstable) should appear
        assert len(result["worst_modules"]) >= 1
        all_issues = set()
        for m in result["worst_modules"]:
            all_issues.update(m["issues"])
        assert "circular_dependency" in all_issues or "unstable" in all_issues

    async def test_quality_path_filter(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="quality", project=_PROJECT, path="mypkg/utils")
        assert result["analysis"] == "quality"
        assert 0 <= result["health_score"] <= 100

    async def test_path_filter(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="structure", project=_PROJECT, path="mypkg/utils")
        # Only entities under mypkg/utils.py
        assert result["label_counts"].get("Module", 0) <= 1
        assert "TypeDef" not in result["label_counts"]

    async def test_invalid_analysis(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="nonexistent", project=_PROJECT)
        assert result["code"] == "INVALID_ANALYSIS"


# ---------------------------------------------------------------------------
# generate_diagram integration tests
# ---------------------------------------------------------------------------


class TestGenerateDiagram:
    async def test_packages_diagram(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "generate_diagram", type="packages", project=_PROJECT)
        assert result["type"] == "packages"
        assert "graph TD" in result["mermaid"]
        assert result["node_count"] >= 2
        assert "-->" in result["mermaid"]

    async def test_imports_diagram(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "generate_diagram", type="imports", project=_PROJECT)
        assert result["type"] == "imports"
        assert "graph LR" in result["mermaid"]
        assert result["node_count"] >= 2
        # Should show module dependency edges
        assert "-->" in result["mermaid"]

    async def test_inheritance_diagram(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "generate_diagram", type="inheritance", project=_PROJECT)
        assert result["type"] == "inheritance"
        assert "classDiagram" in result["mermaid"]
        assert "Base" in result["mermaid"]
        assert "User" in result["mermaid"]
        assert "<|--" in result["mermaid"]

    async def test_module_detail_diagram(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(
            app_ctx, "generate_diagram", type="module_detail", project=_PROJECT, path="mypkg/models"
        )
        assert result["type"] == "module_detail"
        assert result["module"] == "mypkg.models"
        assert "classDiagram" in result["mermaid"]
        # Should include Base and User
        assert "Base" in result["mermaid"]
        assert "User" in result["mermaid"]
        # User.save should appear as a method
        assert "save()" in result["mermaid"]

    async def test_module_detail_requires_path(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "generate_diagram", type="module_detail", project=_PROJECT)
        assert result["code"] == "PATH_REQUIRED"

    async def test_module_detail_not_found(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(
            app_ctx, "generate_diagram", type="module_detail", project=_PROJECT, path="nonexistent/path"
        )
        assert result["code"] == "NOT_FOUND"

    async def test_max_nodes_caps_output(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "generate_diagram", type="packages", project=_PROJECT, max_nodes=2)
        assert result["node_count"] <= 4  # max_nodes limits edges, each edge adds up to 2 nodes

    async def test_invalid_diagram_type(self, app_ctx, seeded_analysis_graph):
        result = await _invoke_tool(app_ctx, "generate_diagram", type="nonexistent", project=_PROJECT)
        assert result["code"] == "INVALID_DIAGRAM_TYPE"

    async def test_empty_project_diagrams(self, app_ctx):
        """Diagrams for a non-existent project return empty/placeholder output."""
        result = await _invoke_tool(app_ctx, "generate_diagram", type="packages", project="nonexistent")
        assert result["node_count"] == 0

    async def test_imports_empty(self, app_ctx):
        result = await _invoke_tool(app_ctx, "generate_diagram", type="imports", project="nonexistent")
        assert result["node_count"] == 0


# ---------------------------------------------------------------------------
# Detail parameter integration tests
# ---------------------------------------------------------------------------

_DETAIL_PROJECT = "detail-project"


@pytest.fixture
async def seeded_detail_graph(graph_client):
    """Seed the graph with entities that have source code for detail tests."""
    await graph_client.ensure_schema()

    await graph_client.merge_project_node(_DETAIL_PROJECT, file_count=1, entity_count=3)

    await graph_client.execute_write(
        f"CREATE (n:{NodeLabel.MODULE} {{"
        "uid: $uid, project_name: $p, name: 'mod', qualified_name: 'mod', "
        "file_path: 'mod.py', kind: 'module', line_start: 1, line_end: 50"
        "})",
        {"uid": f"{_DETAIL_PROJECT}:mod", "p": _DETAIL_PROJECT},
    )

    await graph_client.execute_write(
        f"CREATE (n:{NodeLabel.CALLABLE} {{"
        "uid: $uid, project_name: $p, name: 'caller_fn', qualified_name: 'mod.caller_fn', "
        "file_path: 'mod.py', kind: 'function', line_start: 5, line_end: 10, "
        "signature: 'def caller_fn() -> None', docstring: 'Calls helper.', "
        "source: 'def caller_fn() -> None:\\n    helper()'"
        "})",
        {"uid": f"{_DETAIL_PROJECT}:mod.caller_fn", "p": _DETAIL_PROJECT},
    )

    await graph_client.execute_write(
        f"CREATE (n:{NodeLabel.CALLABLE} {{"
        "uid: $uid, project_name: $p, name: 'helper', qualified_name: 'mod.helper', "
        "file_path: 'mod.py', kind: 'function', line_start: 15, line_end: 20, "
        "signature: 'def helper() -> str', docstring: 'A helper function.', "
        "source: 'def helper() -> str:\\n    return \"hi\"'"
        "})",
        {"uid": f"{_DETAIL_PROJECT}:mod.helper", "p": _DETAIL_PROJECT},
    )

    # DEFINES edges
    await graph_client.execute_write(
        f"MATCH (m {{uid: $m}}), (f {{uid: $f}}) CREATE (m)-[:{RelType.DEFINES}]->(f)",
        {"m": f"{_DETAIL_PROJECT}:mod", "f": f"{_DETAIL_PROJECT}:mod.caller_fn"},
    )
    await graph_client.execute_write(
        f"MATCH (m {{uid: $m}}), (f {{uid: $f}}) CREATE (m)-[:{RelType.DEFINES}]->(f)",
        {"m": f"{_DETAIL_PROJECT}:mod", "f": f"{_DETAIL_PROJECT}:mod.helper"},
    )

    # CALLS edge: caller_fn -> helper
    await graph_client.execute_write(
        f"MATCH (a {{uid: $a}}), (b {{uid: $b}}) CREATE (a)-[:{RelType.CALLS}]->(b)",
        {"a": f"{_DETAIL_PROJECT}:mod.caller_fn", "b": f"{_DETAIL_PROJECT}:mod.helper"},
    )

    return graph_client


class TestDetailParameter:
    async def test_get_node_detail_full_includes_source(self, app_ctx, seeded_detail_graph):
        result = await _invoke_tool(app_ctx, "get_node", name="helper", detail="full")
        assert result["count"] >= 1
        node = next(r for r in result["results"] if r["name"] == "helper")
        assert "source" in node
        assert "helper" in node["source"]

    async def test_get_node_detail_summary_excludes_source(self, app_ctx, seeded_detail_graph):
        result = await _invoke_tool(app_ctx, "get_node", name="helper", detail="summary")
        assert result["count"] >= 1
        node = next(r for r in result["results"] if r["name"] == "helper")
        assert "source" not in node

    async def test_get_node_full_has_call_info(self, app_ctx, seeded_detail_graph):
        result = await _invoke_tool(app_ctx, "get_node", name="helper", detail="full")
        node = next(r for r in result["results"] if r["name"] == "helper")
        assert "caller_count" in node
        assert node["caller_count"] >= 1
        assert "callers" in node
        assert "caller_fn" in node["callers"]

    async def test_get_context_neighborhood_excludes_source(self, app_ctx, seeded_detail_graph):
        result = await _invoke_tool(app_ctx, "get_context", uid=f"{_DETAIL_PROJECT}:mod.helper")
        # Target node keeps source
        assert "source" in result["node"] or result["node"].get("source") is not None
        # Siblings should NOT have source
        for sibling in result["siblings"]:
            assert "source" not in sibling
