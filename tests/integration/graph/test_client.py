"""Integration tests for GraphClient and schema application.

Requires a running Memgraph instance (docker compose up -d memgraph).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from code_atlas.graph.client import QueryTimeoutError
from code_atlas.parsing.ast import ParsedEntity, ParsedRelationship, parse_file
from code_atlas.schema import SCHEMA_VERSION, NodeLabel, RelType

if TYPE_CHECKING:
    from code_atlas.graph.client import GraphClient


pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------


async def test_ping(graph_client: GraphClient):
    assert await graph_client.ping() is True


# ---------------------------------------------------------------------------
# Schema lifecycle
# ---------------------------------------------------------------------------


async def test_ensure_schema_fresh_db(graph_client: GraphClient):
    """On a clean database, ensure_schema creates the version node and DDL."""
    await graph_client.ensure_schema()

    version = await graph_client.get_schema_version()
    assert version == SCHEMA_VERSION


async def test_ensure_schema_idempotent(graph_client: GraphClient):
    """Calling ensure_schema twice succeeds without error."""
    await graph_client.ensure_schema()
    await graph_client.ensure_schema()

    version = await graph_client.get_schema_version()
    assert version == SCHEMA_VERSION


async def test_ensure_schema_rejects_downgrade(graph_client: GraphClient):
    """A newer DB version should raise RuntimeError."""
    await graph_client.ensure_schema()

    # Manually bump the stored version higher than code version
    await graph_client.execute_write(
        f"MATCH (sv:{NodeLabel.SCHEMA_VERSION}) SET sv.version = $v",
        {"v": SCHEMA_VERSION + 100},
    )

    with pytest.raises(RuntimeError, match="newer than code"):
        await graph_client.ensure_schema()


async def test_set_schema_version_no_duplicate_nodes(graph_client: GraphClient):
    """Migrating from an older version updates the existing SchemaVersion node in place."""
    await graph_client.ensure_schema()

    # Simulate an old database: force the stored version below current
    await graph_client.execute_write(f"MATCH (sv:{NodeLabel.SCHEMA_VERSION}) SET sv.version = 1")

    await graph_client.ensure_schema()  # migrates 1 → SCHEMA_VERSION

    records = await graph_client.execute(f"MATCH (sv:{NodeLabel.SCHEMA_VERSION}) RETURN count(sv) AS cnt")
    assert records[0]["cnt"] == 1
    assert await graph_client.get_schema_version() == SCHEMA_VERSION


async def test_get_schema_version_returns_max_across_duplicates(graph_client: GraphClient):
    """Defensive: duplicate SchemaVersion nodes (pre-fix damage) resolve to the max version."""
    await graph_client.execute_write(f"CREATE (:{NodeLabel.SCHEMA_VERSION} {{version: 1}})")
    await graph_client.execute_write(f"CREATE (:{NodeLabel.SCHEMA_VERSION} {{version: $v}})", {"v": SCHEMA_VERSION})

    assert await graph_client.get_schema_version() == SCHEMA_VERSION


async def test_duplicate_schema_versions_collapsed_preserving_embedding_config(graph_client: GraphClient):
    """Migration collapses duplicate SchemaVersion nodes into one, keeping the embedding config."""
    # Pre-fix damage: stale duplicate carries the embedding config, newer node has none
    await graph_client.execute_write(
        f"CREATE (:{NodeLabel.SCHEMA_VERSION} {{version: 1, embedding_model: 'old-model', embedding_dimension: 768}})"
    )
    await graph_client.execute_write(f"CREATE (:{NodeLabel.SCHEMA_VERSION} {{version: 2}})")

    await graph_client.ensure_schema()  # migrates to SCHEMA_VERSION, collapsing duplicates

    records = await graph_client.execute(
        f"MATCH (sv:{NodeLabel.SCHEMA_VERSION}) "
        "RETURN sv.version AS v, sv.embedding_model AS m, sv.embedding_dimension AS d"
    )
    assert len(records) == 1
    assert records[0]["v"] == SCHEMA_VERSION
    assert (records[0]["m"], records[0]["d"]) == ("old-model", 768)
    assert await graph_client.get_embedding_config() == ("old-model", 768)


async def test_get_embedding_config_reads_canonical_node_across_duplicates(graph_client: GraphClient):
    """With duplicates still present, the config comes from the highest-version node, not an arbitrary one."""
    await graph_client.execute_write(
        f"CREATE (:{NodeLabel.SCHEMA_VERSION} {{version: 1, embedding_model: 'stale', embedding_dimension: 384}})"
    )
    await graph_client.execute_write(
        f"CREATE (:{NodeLabel.SCHEMA_VERSION} {{version: $v, embedding_model: 'current', embedding_dimension: 1024}})",
        {"v": SCHEMA_VERSION},
    )

    assert await graph_client.get_embedding_config() == ("current", 1024)


async def test_migration_v3_clears_freshness_markers(graph_client: GraphClient):
    """Migrating from v2 (through v3 and v4) clears Module.file_hash and Project.git_hash for a re-parse."""
    await graph_client.ensure_schema()

    await graph_client.execute_write(
        f"CREATE (:{NodeLabel.MODULE} {{"
        "  uid: 'mig3:mod', project_name: 'mig3', name: 'mod',"
        "  qualified_name: 'mod', file_path: 'mod.py', kind: 'module',"
        "  content_hash: 'ch', file_hash: 'stale'"
        "})"
    )
    await graph_client.execute_write(
        f"CREATE (:{NodeLabel.PROJECT} {{uid: 'mig3', project_name: 'mig3', name: 'mig3', git_hash: 'oldcommit'}})"
    )
    # Force the stored version back to 2 (pre-v3 database)
    await graph_client.execute_write(f"MATCH (sv:{NodeLabel.SCHEMA_VERSION}) SET sv.version = 2")

    await graph_client.ensure_schema()

    mod = await graph_client.execute("MATCH (n {uid: 'mig3:mod'}) RETURN n.file_hash AS fh")
    assert mod[0]["fh"] is None
    proj = await graph_client.execute("MATCH (p {uid: 'mig3'}) RETURN p.git_hash AS gh")
    assert proj[0]["gh"] is None
    assert await graph_client.get_schema_version() == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Node creation
# ---------------------------------------------------------------------------


async def test_create_code_nodes(graph_client: GraphClient):
    """Create TypeDef, Callable, and Value nodes with all expected properties."""
    await graph_client.ensure_schema()

    await graph_client.execute_write(
        f"CREATE (n:{NodeLabel.TYPE_DEF} {{"
        "  uid: $uid, project_name: $project, name: $name,"
        "  qualified_name: $qn, kind: $kind, file_path: $fp,"
        "  line_start: 10, line_end: 50, visibility: 'public',"
        "  tags: ['abstract', 'generic'], content_hash: 'abc123'"
        "})",
        {
            "uid": "proj:mod.MyClass",
            "project": "proj",
            "name": "MyClass",
            "qn": "mod.MyClass",
            "kind": "class",
            "fp": "src/mod.py",
        },
    )

    await graph_client.execute_write(
        f"CREATE (n:{NodeLabel.CALLABLE} {{"
        "  uid: $uid, project_name: $project, name: $name,"
        "  qualified_name: $qn, kind: $kind, file_path: $fp,"
        "  line_start: 15, line_end: 30, visibility: 'public',"
        "  tags: ['async'], content_hash: 'def456'"
        "})",
        {
            "uid": "proj:mod.MyClass.do_work",
            "project": "proj",
            "name": "do_work",
            "qn": "mod.MyClass.do_work",
            "kind": "method",
            "fp": "src/mod.py",
        },
    )

    await graph_client.execute_write(
        f"CREATE (n:{NodeLabel.VALUE} {{"
        "  uid: $uid, project_name: $project, name: $name,"
        "  qualified_name: $qn, kind: $kind, file_path: $fp,"
        "  line_start: 1, line_end: 1, content_hash: 'ghi789'"
        "})",
        {
            "uid": "proj:mod.MAX_SIZE",
            "project": "proj",
            "name": "MAX_SIZE",
            "qn": "mod.MAX_SIZE",
            "kind": "constant",
            "fp": "src/mod.py",
        },
    )

    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.TYPE_DEF} {{project_name: 'proj'}}) RETURN n.name AS name"
    )
    assert len(records) == 1
    assert records[0]["name"] == "MyClass"


async def test_create_relationships(graph_client: GraphClient):
    """Create a Project->Package->Module->TypeDef->Callable chain."""
    await graph_client.ensure_schema()

    # Create nodes individually with parameters, then link them
    nodes = [
        (
            NodeLabel.PROJECT,
            {
                "uid": "proj:proj",
                "project_name": "proj",
                "name": "proj",
                "qualified_name": "proj",
                "file_path": ".",
                "kind": "project",
                "content_hash": "p1",
            },
        ),
        (
            NodeLabel.PACKAGE,
            {
                "uid": "proj:proj.api",
                "project_name": "proj",
                "name": "api",
                "qualified_name": "proj.api",
                "file_path": "src/api",
                "kind": "package",
                "content_hash": "p2",
            },
        ),
        (
            NodeLabel.MODULE,
            {
                "uid": "proj:proj.api.auth",
                "project_name": "proj",
                "name": "auth",
                "qualified_name": "proj.api.auth",
                "file_path": "src/api/auth.py",
                "kind": "module",
                "content_hash": "p3",
            },
        ),
        (
            NodeLabel.TYPE_DEF,
            {
                "uid": "proj:proj.api.auth.UserService",
                "project_name": "proj",
                "name": "UserService",
                "qualified_name": "proj.api.auth.UserService",
                "file_path": "src/api/auth.py",
                "kind": "class",
                "content_hash": "p4",
            },
        ),
        (
            NodeLabel.CALLABLE,
            {
                "uid": "proj:proj.api.auth.UserService.validate",
                "project_name": "proj",
                "name": "validate",
                "qualified_name": "proj.api.auth.UserService.validate",
                "file_path": "src/api/auth.py",
                "kind": "method",
                "content_hash": "p5",
            },
        ),
    ]

    for label, props in nodes:
        await graph_client.execute_write(f"CREATE (n:{label} $props)", {"props": props})

    # Create relationships
    rels = [
        ("proj:proj", RelType.CONTAINS, "proj:proj.api"),
        ("proj:proj.api", RelType.CONTAINS, "proj:proj.api.auth"),
        ("proj:proj.api.auth", RelType.DEFINES, "proj:proj.api.auth.UserService"),
        ("proj:proj.api.auth.UserService", RelType.DEFINES, "proj:proj.api.auth.UserService.validate"),
    ]
    for from_uid, rel_type, to_uid in rels:
        await graph_client.execute_write(
            f"MATCH (a {{uid: $from_uid}}), (b {{uid: $to_uid}}) CREATE (a)-[:{rel_type}]->(b)",
            {"from_uid": from_uid, "to_uid": to_uid},
        )

    records = await graph_client.execute(
        f"MATCH (p:{NodeLabel.PROJECT} {{project_name: 'proj'}})"
        f"-[:{RelType.CONTAINS}*1..2]->"
        f"(m:{NodeLabel.MODULE})"
        " RETURN m.name AS name"
    )
    assert len(records) == 1
    assert records[0]["name"] == "auth"


async def test_project_isolation(graph_client: GraphClient):
    """Two projects with same entity names should be independent."""
    await graph_client.ensure_schema()

    for project in ("alpha", "beta"):
        await graph_client.execute_write(
            f"CREATE (n:{NodeLabel.CALLABLE} {{"
            "  uid: $uid, project_name: $project, name: 'main',"
            "  qualified_name: $qn, kind: 'function', file_path: 'main.py',"
            "  content_hash: $hash"
            "})",
            {
                "uid": f"{project}:main",
                "project": project,
                "qn": "main",
                "hash": f"h_{project}",
            },
        )

    alpha = await graph_client.execute(f"MATCH (n:{NodeLabel.CALLABLE} {{project_name: 'alpha'}}) RETURN n.uid AS uid")
    beta = await graph_client.execute(f"MATCH (n:{NodeLabel.CALLABLE} {{project_name: 'beta'}}) RETURN n.uid AS uid")
    assert len(alpha) == 1
    assert len(beta) == 1
    assert alpha[0]["uid"] != beta[0]["uid"]


async def test_uniqueness_constraint_enforced(graph_client: GraphClient):
    """Duplicate uid on the same label should be rejected."""
    await graph_client.ensure_schema()

    await graph_client.execute_write(
        f"CREATE (:{NodeLabel.MODULE} {{"
        "  uid: 'dup:test', project_name: 'dup', name: 'test',"
        "  qualified_name: 'test', file_path: 'test.py', kind: 'module',"
        "  content_hash: 'h1'"
        "})"
    )

    with pytest.raises(Exception, match="uid"):
        await graph_client.execute_write(
            f"CREATE (:{NodeLabel.MODULE} {{"
            "  uid: 'dup:test', project_name: 'dup', name: 'test2',"
            "  qualified_name: 'test2', file_path: 'test2.py', kind: 'module',"
            "  content_hash: 'h2'"
            "})"
        )


async def test_tags_query(graph_client: GraphClient):
    """Nodes with tags can be queried with WHERE ... IN n.tags."""
    await graph_client.ensure_schema()

    await graph_client.execute_write(
        f"CREATE (:{NodeLabel.CALLABLE} {{"
        "  uid: 'tag:proj.async_func', project_name: 'tag', name: 'async_func',"
        "  qualified_name: 'proj.async_func', kind: 'function', file_path: 'f.py',"
        "  tags: ['async', 'generator'], content_hash: 'th1'"
        "})"
    )
    await graph_client.execute_write(
        f"CREATE (:{NodeLabel.CALLABLE} {{"
        "  uid: 'tag:proj.sync_func', project_name: 'tag', name: 'sync_func',"
        "  qualified_name: 'proj.sync_func', kind: 'function', file_path: 'f.py',"
        "  tags: [], content_hash: 'th2'"
        "})"
    )

    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.CALLABLE} {{project_name: 'tag'}}) WHERE 'async' IN n.tags RETURN n.name AS name"
    )
    assert len(records) == 1
    assert records[0]["name"] == "async_func"


# ---------------------------------------------------------------------------
# Vector indices
# ---------------------------------------------------------------------------


async def test_vector_index_created(graph_client: GraphClient):
    """After ensure_schema, vector indices should be visible."""
    await graph_client.ensure_schema()

    info = await graph_client.get_vector_index_info()
    index_names = {r.get("index_name", r.get("name", "")) for r in info}
    # At minimum, the Callable index should exist
    assert "vec_callable" in index_names


async def test_write_and_search_embeddings(graph_client: GraphClient):
    """Write embedding vectors to nodes, then search via the vector index."""
    await graph_client.ensure_schema()

    # Create a Callable node
    await graph_client.execute_write(
        f"CREATE (:{NodeLabel.CALLABLE} {{"
        "  uid: 'vec:proj.my_func', project_name: 'vec', name: 'my_func',"
        "  qualified_name: 'proj.my_func', kind: 'function', file_path: 'f.py',"
        "  content_hash: 'v1'"
        "})"
    )

    # Write embedding (dimension must match vector index)
    dim = graph_client._dimension
    vector = [1.0] + [0.0] * (dim - 1)
    await graph_client.write_embeddings([("vec:proj.my_func", vector)])

    # Search with same vector — should find our node with high similarity
    records = await graph_client.execute(
        "CALL vector_search.search('vec_callable', 5, $vector) "
        "YIELD node, similarity "
        "RETURN node.uid AS uid, similarity",
        {"vector": vector},
    )
    assert len(records) >= 1
    assert records[0]["uid"] == "vec:proj.my_func"
    assert records[0]["similarity"] > 0.9


async def test_vector_search_scope_filter(graph_client: GraphClient):
    """Two projects with embeddings — verify scope isolation via post-filter."""
    await graph_client.ensure_schema()

    dim = graph_client._dimension
    vector_a = [1.0] + [0.0] * (dim - 1)
    vector_b = [0.0, 1.0] + [0.0] * (dim - 2)

    for project, vec, name in [("alpha", vector_a, "func_a"), ("beta", vector_b, "func_b")]:
        uid = f"{project}:{name}"
        await graph_client.execute_write(
            f"CREATE (:{NodeLabel.CALLABLE} {{"
            "  uid: $uid, project_name: $project, name: $name,"
            "  qualified_name: $qn, kind: 'function', file_path: 'f.py',"
            "  content_hash: $hash"
            "})",
            {
                "uid": uid,
                "project": project,
                "name": name,
                "qn": name,
                "hash": f"h_{project}",
            },
        )
        await graph_client.write_embeddings([(uid, vec)])

    # Search both — should get two results
    records = await graph_client.execute(
        "CALL vector_search.search('vec_callable', 10, $vector) "
        "YIELD node, similarity "
        "RETURN node.uid AS uid, node.project_name AS project_name, similarity",
        {"vector": vector_a},
    )
    assert len(records) == 2

    # Python-side scope filter (mirrors what mcp_server.vector_search does)
    alpha_only = [r for r in records if r["project_name"] == "alpha"]
    assert len(alpha_only) == 1
    assert alpha_only[0]["uid"] == "alpha:func_a"


async def test_vector_search_threshold(graph_client: GraphClient):
    """Threshold filter discards low-similarity results."""
    await graph_client.ensure_schema()

    dim = graph_client._dimension
    vector_a = [1.0] + [0.0] * (dim - 1)
    vector_b = [0.0, 1.0] + [0.0] * (dim - 2)

    for name, vec in [("near", vector_a), ("far", vector_b)]:
        uid = f"thresh:{name}"
        await graph_client.execute_write(
            f"CREATE (:{NodeLabel.CALLABLE} {{"
            "  uid: $uid, project_name: 'thresh', name: $name,"
            "  qualified_name: $qn, kind: 'function', file_path: 'f.py',"
            "  content_hash: $hash"
            "})",
            {"uid": uid, "name": name, "qn": name, "hash": f"h_{name}"},
        )
        await graph_client.write_embeddings([(uid, vec)])

    # Search with vector_a — both results returned
    records = await graph_client.execute(
        "CALL vector_search.search('vec_callable', 10, $vector) "
        "YIELD node, similarity "
        "RETURN node.uid AS uid, similarity",
        {"vector": vector_a},
    )
    assert len(records) == 2

    # Apply threshold — only high-similarity result survives
    high_sim = [r for r in records if r["similarity"] >= 0.9]
    assert len(high_sim) == 1
    assert high_sim[0]["uid"] == "thresh:near"


# ---------------------------------------------------------------------------
# Text (BM25) search
# ---------------------------------------------------------------------------


async def test_text_search_method(graph_client: GraphClient):
    """GraphClient.text_search() returns a list of dicts."""
    await graph_client.ensure_schema()

    await graph_client.execute_write(
        f"CREATE (:{NodeLabel.CALLABLE} {{"
        "  uid: 'txt:proj.search_func', project_name: 'txt', name: 'search_func',"
        "  qualified_name: 'proj.search_func', kind: 'function', file_path: 'f.py',"
        "  docstring: 'A function that searches things.'"
        "})"
    )

    results = await graph_client.text_search("search_func")
    assert isinstance(results, list)


async def test_text_search_with_project_filter(graph_client: GraphClient):
    """text_search with project filter returns only matching project."""
    await graph_client.ensure_schema()

    for project in ("alpha", "beta"):
        await graph_client.execute_write(
            f"CREATE (:{NodeLabel.CALLABLE} {{"
            "  uid: $uid, project_name: $project, name: 'common_func',"
            "  qualified_name: $qn, kind: 'function', file_path: 'f.py',"
            "  docstring: 'A common function.'"
            "})",
            {
                "uid": f"{project}:common_func",
                "project": project,
                "qn": f"{project}.common_func",
            },
        )

    results = await graph_client.text_search("common_func", project="alpha")
    assert isinstance(results, list)
    # All returned results (if any) should belong to alpha
    for r in results:
        node = r.get("node") or r.get("n")
        if node and hasattr(node, "get"):
            assert node.get("project_name") == "alpha"


async def test_text_search_escapes_syntax_special_chars(graph_client: GraphClient):
    """BM25 queries containing raw Tantivy syntax characters (parens, brackets,
    colons, quotes) must not crash the text index into an empty result set —
    common in code-shaped queries like signatures or generics (client.py:1690).
    """
    await graph_client.ensure_schema()

    await graph_client.execute_write(
        f"CREATE (:{NodeLabel.CALLABLE} {{"
        "  uid: 'bm25esc:proj.embed_batch', project_name: 'bm25esc', name: 'embed_batch',"
        "  qualified_name: 'proj.embed_batch', kind: 'function', file_path: 'f.py',"
        "  docstring: 'embed_batch(texts) processes a dict[str, Any] of embeddings.'"
        "})"
    )

    for query in ["embed_batch(texts)", "dict[str, Any]", 'unbalanced "quote', "std::vector"]:
        results = await graph_client.text_search(query, project="bm25esc")
        assert isinstance(results, list), f"query {query!r} should not raise"

    for query in ["embed_batch(texts)", "dict[str, Any]"]:
        results = await graph_client.text_search(query, project="bm25esc")
        names = [node.get("name") for r in results if (node := (r.get("node") or r.get("n"))) is not None]
        assert "embed_batch" in names, f"query {query!r} returned no match: {results}"


async def test_get_text_index_info(graph_client: GraphClient):
    """get_text_index_info() returns a list (may be empty without indices)."""
    await graph_client.ensure_schema()
    info = await graph_client.get_text_index_info()
    assert isinstance(info, list)


# ---------------------------------------------------------------------------
# Embed data preservation across upsert
# ---------------------------------------------------------------------------


async def test_upsert_preserves_embed_data(graph_client: GraphClient):
    """embed_hash+embedding survive a DELETE+CREATE upsert cycle when QN matches."""
    await graph_client.ensure_schema()

    project = "etest"
    fp = "src/mod.py"

    entities = [
        ParsedEntity(
            name="my_func",
            qualified_name=f"{project}:mod.my_func",
            label=NodeLabel.CALLABLE,
            kind="function",
            line_start=1,
            line_end=5,
            file_path=fp,
            docstring="A test function.",
            signature="my_func()",
        ),
    ]

    # First upsert — creates the node
    await graph_client.upsert_file_entities(project, fp, entities, [])

    # Manually set embed_hash + embedding on the node (must match vector index dimension)
    embed_hash = "abc123"
    dim = graph_client._dimension
    embedding = [0.1] * dim
    await graph_client.execute_write(
        "MATCH (n {qualified_name: 'mod.my_func', project_name: $p}) SET n.embed_hash = $hash, n.embedding = $vec",
        {"p": project, "hash": embed_hash, "vec": embedding},
    )

    # Second upsert — same entity (simulates re-parse with no rename)
    await graph_client.upsert_file_entities(project, fp, entities, [])

    # Verify embed data was preserved
    records = await graph_client.execute(
        "MATCH (n {qualified_name: 'mod.my_func', project_name: $p}) RETURN n.embed_hash AS hash, n.embedding AS vec",
        {"p": project},
    )
    assert len(records) == 1
    assert records[0]["hash"] == embed_hash
    assert records[0]["vec"] == embedding


async def test_upsert_drops_embed_for_removed_entity(graph_client: GraphClient):
    """Embed data is NOT carried forward when an entity is removed from the file."""
    await graph_client.ensure_schema()

    project = "etest2"
    fp = "src/mod2.py"

    # Two entities initially
    entities = [
        ParsedEntity(
            name="func_a",
            qualified_name=f"{project}:mod2.func_a",
            label=NodeLabel.CALLABLE,
            kind="function",
            line_start=1,
            line_end=3,
            file_path=fp,
        ),
        ParsedEntity(
            name="func_b",
            qualified_name=f"{project}:mod2.func_b",
            label=NodeLabel.CALLABLE,
            kind="function",
            line_start=5,
            line_end=8,
            file_path=fp,
        ),
    ]

    await graph_client.upsert_file_entities(project, fp, entities, [])

    # Set embed data on both (must match vector index dimension)
    dim = graph_client._dimension
    for qn in ("mod2.func_a", "mod2.func_b"):
        await graph_client.execute_write(
            "MATCH (n {qualified_name: $qn, project_name: $p}) SET n.embed_hash = 'hash_' + $qn, n.embedding = $vec",
            {"qn": qn, "p": project, "vec": [1.0] * dim},
        )

    # Re-upsert with only func_a (func_b was deleted from the file)
    await graph_client.upsert_file_entities(project, fp, [entities[0]], [])

    # func_a should still have its embed data
    records_a = await graph_client.execute(
        "MATCH (n {qualified_name: 'mod2.func_a', project_name: $p}) RETURN n.embed_hash AS hash",
        {"p": project},
    )
    assert len(records_a) == 1
    assert records_a[0]["hash"] == "hash_mod2.func_a"

    # func_b should no longer exist
    records_b = await graph_client.execute(
        "MATCH (n {qualified_name: 'mod2.func_b', project_name: $p}) RETURN n",
        {"p": project},
    )
    assert len(records_b) == 0


async def test_write_embed_hashes(graph_client: GraphClient):
    """write_embed_hashes correctly sets embed_hash on nodes."""
    await graph_client.ensure_schema()

    await graph_client.execute_write(
        f"CREATE (:{NodeLabel.CALLABLE} {{"
        "  uid: 'eh:proj.func', project_name: 'eh', name: 'func',"
        "  qualified_name: 'proj.func', kind: 'function', file_path: 'f.py'"
        "})"
    )

    await graph_client.write_embed_hashes([("eh:proj.func", "deadbeef")])

    records = await graph_client.execute("MATCH (n {qualified_name: 'proj.func'}) RETURN n.embed_hash AS hash")
    assert records[0]["hash"] == "deadbeef"


async def test_read_entity_texts_includes_embed_fields(graph_client: GraphClient):
    """read_entity_texts returns embed_hash and embedding fields."""
    await graph_client.ensure_schema()

    dim = graph_client._dimension
    embedding = [0.1] * dim
    await graph_client.execute_write(
        f"CREATE (:{NodeLabel.CALLABLE} {{"
        "  uid: 'ret:proj.func', project_name: 'ret', name: 'func',"
        "  qualified_name: 'proj.func', kind: 'function', file_path: 'f.py',"
        "  embed_hash: 'abc', embedding: $emb, content_hash: 'ch'"
        "})",
        {"emb": embedding},
    )

    results = await graph_client.read_entity_texts(["ret:proj.func"])
    assert len(results) == 1
    assert results[0]["uid"] == "ret:proj.func"
    assert results[0]["embed_hash"] == "abc"
    assert results[0]["has_embedding"] is True


# ---------------------------------------------------------------------------
# DOCUMENTS edge creation
# ---------------------------------------------------------------------------


async def test_upsert_with_documents_rels(graph_client: GraphClient):
    """Full flow: pre-create a code entity, upsert markdown with DOCUMENTS rels, verify edges."""
    await graph_client.ensure_schema()

    project = "doclink"

    # Pre-create a code entity (the target of the doc link)
    await graph_client.execute_write(
        f"CREATE (:{NodeLabel.CALLABLE} {{"
        "  uid: $uid, project_name: $project, name: $name,"
        "  qualified_name: $qn, kind: 'function', file_path: 'src/auth.py'"
        "})",
        {
            "uid": f"{project}:auth.validate_token",
            "project": project,
            "name": "validate_token",
            "qn": "auth.validate_token",
        },
    )

    # Now upsert a markdown file with a DOCUMENTS relationship targeting that entity
    doc_fp = "docs/auth.md"
    section_qn = f"{project}:{doc_fp} > Auth > validate_token"
    entities = [
        ParsedEntity(
            name="auth.md",
            qualified_name=f"{project}:{doc_fp}",
            label=NodeLabel.DOC_FILE,
            kind="doc_file",
            line_start=1,
            line_end=10,
            file_path=doc_fp,
        ),
        ParsedEntity(
            name="validate_token",
            qualified_name=section_qn,
            label=NodeLabel.DOC_SECTION,
            kind="section",
            line_start=3,
            line_end=8,
            file_path=doc_fp,
            header_level=2,
            header_path="Auth > validate_token",
        ),
    ]
    relationships = [
        ParsedRelationship(
            from_qualified_name=f"{project}:{doc_fp}",
            rel_type=RelType.CONTAINS,
            to_name=section_qn,
        ),
        ParsedRelationship(
            from_qualified_name=section_qn,
            rel_type=RelType.DOCUMENTS,
            to_name="validate_token",
            properties={"link_type": "explicit", "confidence": 0.9, "is_file_ref": False},
        ),
    ]

    await graph_client.upsert_file_entities(project, doc_fp, entities, relationships)

    # Verify DOCUMENTS edge was created with correct properties
    records = await graph_client.execute(
        f"MATCH (d:{NodeLabel.DOC_SECTION})-[r:{RelType.DOCUMENTS}]->(c:{NodeLabel.CALLABLE}) "
        "RETURN d.name AS doc_name, c.name AS code_name, r.link_type AS link_type, r.confidence AS confidence"
    )
    assert len(records) == 1
    assert records[0]["doc_name"] == "validate_token"
    assert records[0]["code_name"] == "validate_token"
    assert records[0]["link_type"] == "explicit"
    assert records[0]["confidence"] == 0.9


# ---------------------------------------------------------------------------
# Note vault round-trip (Phase 1 — knowledge convergence)
# ---------------------------------------------------------------------------


async def test_note_round_trip_links_to_and_derived_from(graph_client: GraphClient):
    """A vault markdown file parses to a Note node whose LINKS_TO/DERIVED_FROM edges
    actually materialize in the graph — the RelType-routing fix under test."""
    await graph_client.ensure_schema()
    project = "notevault"

    parsed_b = parse_file("docs/notes/b.md", b"---\nid: b\nkind: note\n---\n\nTarget note.\n", project)
    assert parsed_b is not None
    await graph_client.upsert_file_entities(project, parsed_b.file_path, parsed_b.entities, parsed_b.relationships)

    parsed_a = parse_file(
        "docs/notes/a.md",
        b"---\nid: a\nkind: note\nderived_from: [b]\n---\n\nSee [[b]] for context.\n",
        project,
    )
    assert parsed_a is not None
    await graph_client.upsert_file_entities(project, parsed_a.file_path, parsed_a.entities, parsed_a.relationships)

    note_records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE} {{project_name: $p}}) RETURN n.uid AS uid ORDER BY uid", {"p": project}
    )
    assert {r["uid"] for r in note_records} == {f"{project}:note:a", f"{project}:note:b"}

    links = await graph_client.execute(
        f"MATCH (a:{NodeLabel.NOTE} {{uid: $a}})-[:{RelType.LINKS_TO}]->"
        f"(b:{NodeLabel.NOTE} {{uid: $b}}) RETURN count(*) AS cnt",
        {"a": f"{project}:note:a", "b": f"{project}:note:b"},
    )
    assert links[0]["cnt"] == 1

    derived = await graph_client.execute(
        f"MATCH (a:{NodeLabel.NOTE} {{uid: $a}})-[:{RelType.DERIVED_FROM}]->"
        f"(b:{NodeLabel.NOTE} {{uid: $b}}) RETURN count(*) AS cnt",
        {"a": f"{project}:note:a", "b": f"{project}:note:b"},
    )
    assert derived[0]["cnt"] == 1

    # Backlinks are queryable in the reverse direction too — the point of the exercise.
    backlinks = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE} {{uid: $b}})<-[:{RelType.LINKS_TO}]-(m:{NodeLabel.NOTE}) RETURN m.uid AS uid",
        {"b": f"{project}:note:b"},
    )
    assert backlinks[0]["uid"] == f"{project}:note:a"


async def test_unresolved_wikilink_creates_no_phantom_edge(graph_client: GraphClient):
    """A wikilink to a note that doesn't exist creates no edge and no phantom node."""
    await graph_client.ensure_schema()
    project = "notevault2"

    parsed = parse_file("docs/notes/a.md", b"---\nid: a\nkind: note\n---\n\nSee [[does-not-exist]].\n", project)
    assert parsed is not None
    await graph_client.upsert_file_entities(project, parsed.file_path, parsed.entities, parsed.relationships)

    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE} {{project_name: $p}})-[:{RelType.LINKS_TO}]->() RETURN count(*) AS cnt",
        {"p": project},
    )
    assert records[0]["cnt"] == 0
    phantom = await graph_client.execute(
        "MATCH (n {uid: $uid}) RETURN count(n) AS cnt", {"uid": f"{project}:note:does-not-exist"}
    )
    assert phantom[0]["cnt"] == 0


# ---------------------------------------------------------------------------
# Explicit anchors + staleness (Phase 3 — anchors + staleness)
# ---------------------------------------------------------------------------


def _split_anchor_rels(
    relationships: list[ParsedRelationship],
) -> tuple[list[ParsedRelationship], list[ParsedRelationship]]:
    """Mirror consumers.py's ``_is_anchor`` split — anchor-type DOCUMENTS rels are
    resolved separately via ``resolve_anchors``, not through the immediate
    ``_create_relationships``/``_create_doc_links`` path (see indexing/consumers.py)."""
    anchor_rels = [
        r for r in relationships if r.rel_type == RelType.DOCUMENTS and r.properties.get("link_type") == "anchor"
    ]
    other_rels = [r for r in relationships if r not in anchor_rels]
    return other_rels, anchor_rels


async def test_anchor_uid_form_resolves_with_hash(graph_client: GraphClient):
    """A uid-form anchor resolves directly and captures the target's content_hash."""
    await graph_client.ensure_schema()
    project = "anchor_uid"
    fp = "src/foo.py"
    target = _make_entity(project, "Bar", fp, label=NodeLabel.TYPE_DEF, kind="class", content_hash="v1")
    await graph_client.upsert_file_entities(project, fp, [target], [])

    parsed = parse_file(
        "docs/notes/a.md",
        f"---\nid: a\nkind: note\nanchors: [{project}:src.foo.Bar]\n---\n\nBody.\n".encode(),
        project,
    )
    assert parsed is not None
    other, anchors = _split_anchor_rels(parsed.relationships)
    await graph_client.upsert_file_entities(project, parsed.file_path, parsed.entities, other)
    await graph_client.resolve_anchors(anchors)

    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE})-[r:{RelType.DOCUMENTS} {{link_type: 'anchor'}}]->(b) "
        "RETURN r.anchor_hash AS hash, r.stale AS stale, b.name AS name"
    )
    assert len(records) == 1
    assert records[0]["hash"] == "v1"
    assert records[0]["stale"] is False
    assert records[0]["name"] == "Bar"


async def test_anchor_uid_form_unresolved_records_on_note(graph_client: GraphClient):
    """A uid anchor with no matching target is recorded unresolved, not a phantom edge."""
    await graph_client.ensure_schema()
    project = "anchor_uid_miss"

    parsed = parse_file(
        "docs/notes/a.md",
        f"---\nid: a\nkind: note\nanchors: [{project}:src.foo.Missing]\n---\n\nBody.\n".encode(),
        project,
    )
    assert parsed is not None
    other, anchors = _split_anchor_rels(parsed.relationships)
    await graph_client.upsert_file_entities(project, parsed.file_path, parsed.entities, other)
    await graph_client.resolve_anchors(anchors)

    note_uid = f"{project}:note:a"
    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE} {{uid: $uid}}) RETURN n.unresolved_anchors AS unresolved", {"uid": note_uid}
    )
    assert records[0]["unresolved"] == [f"{project}:src.foo.Missing"]

    edges = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE} {{uid: $uid}})-[:{RelType.DOCUMENTS}]->() RETURN count(*) AS cnt",
        {"uid": note_uid},
    )
    assert edges[0]["cnt"] == 0


async def test_anchor_bare_path_form_resolves_to_module(graph_client: GraphClient):
    """A bare relative path anchor resolves to the file's Module node within the note's own project."""
    await graph_client.ensure_schema()
    project = "anchor_path"
    fp = "src/watcher.py"
    module = ParsedEntity(
        name="watcher",
        qualified_name=f"{project}:src.watcher",
        label=NodeLabel.MODULE,
        kind="module",
        line_start=1,
        line_end=20,
        file_path=fp,
        content_hash="mod_v1",
    )
    await graph_client.upsert_file_entities(project, fp, [module], [])

    parsed = parse_file(
        "docs/notes/a.md",
        f"---\nid: a\nkind: note\nanchors: [{fp}]\n---\n\nBody.\n".encode(),
        project,
    )
    assert parsed is not None
    other, anchors = _split_anchor_rels(parsed.relationships)
    await graph_client.upsert_file_entities(project, parsed.file_path, parsed.entities, other)
    await graph_client.resolve_anchors(anchors)

    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE})-[r:{RelType.DOCUMENTS} {{link_type: 'anchor'}}]->(b:{NodeLabel.MODULE}) "
        "RETURN r.anchor_hash AS hash"
    )
    assert len(records) == 1
    assert records[0]["hash"] == "mod_v1"


async def test_anchor_symbol_refinement_resolves_to_callable(graph_client: GraphClient):
    """A #Symbol suffix narrows a path anchor to a specific entity within the file."""
    await graph_client.ensure_schema()
    project = "anchor_symbol"
    fp = "src/watcher.py"
    module = ParsedEntity(
        name="watcher",
        qualified_name=f"{project}:src.watcher",
        label=NodeLabel.MODULE,
        kind="module",
        line_start=1,
        line_end=20,
        file_path=fp,
        content_hash="mod_v1",
    )
    file_watcher = _make_entity(project, "FileWatcher", fp, label=NodeLabel.TYPE_DEF, kind="class", content_hash="c_v1")
    await graph_client.upsert_file_entities(project, fp, [module, file_watcher], [])

    parsed = parse_file(
        "docs/notes/a.md",
        f"---\nid: a\nkind: note\nanchors: [{fp}#FileWatcher]\n---\n\nBody.\n".encode(),
        project,
    )
    assert parsed is not None
    other, anchors = _split_anchor_rels(parsed.relationships)
    await graph_client.upsert_file_entities(project, parsed.file_path, parsed.entities, other)
    await graph_client.resolve_anchors(anchors)

    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE})-[r:{RelType.DOCUMENTS} {{link_type: 'anchor'}}]->(b) "
        "RETURN b.name AS name, labels(b) AS labels"
    )
    assert len(records) == 1
    assert records[0]["name"] == "FileWatcher"
    assert NodeLabel.TYPE_DEF.value in records[0]["labels"]


async def test_anchor_ambiguous_symbol_falls_back_to_file_level(graph_client: GraphClient):
    """An ambiguous #Symbol (two same-named entities in one file) falls back to the
    file-level anchor rather than failing outright or guessing (decision Q9)."""
    await graph_client.ensure_schema()
    project = "anchor_amb_sym"
    fp = "src/dup.py"
    module = ParsedEntity(
        name="dup",
        qualified_name=f"{project}:src.dup",
        label=NodeLabel.MODULE,
        kind="module",
        line_start=1,
        line_end=20,
        file_path=fp,
        content_hash="mod_v1",
    )
    dup1 = ParsedEntity(
        name="Thing",
        qualified_name=f"{project}:src.dup.Thing#1",
        label=NodeLabel.TYPE_DEF,
        kind="class",
        line_start=2,
        line_end=5,
        file_path=fp,
        content_hash="d1",
    )
    dup2 = ParsedEntity(
        name="Thing",
        qualified_name=f"{project}:src.dup.Thing#2",
        label=NodeLabel.TYPE_DEF,
        kind="class",
        line_start=7,
        line_end=10,
        file_path=fp,
        content_hash="d2",
    )
    await graph_client.upsert_file_entities(project, fp, [module, dup1, dup2], [])

    parsed = parse_file(
        "docs/notes/a.md",
        f"---\nid: a\nkind: note\nanchors: [{fp}#Thing]\n---\n\nBody.\n".encode(),
        project,
    )
    assert parsed is not None
    other, anchors = _split_anchor_rels(parsed.relationships)
    await graph_client.upsert_file_entities(project, parsed.file_path, parsed.entities, other)
    await graph_client.resolve_anchors(anchors)

    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE})-[r:{RelType.DOCUMENTS} {{link_type: 'anchor'}}]->(b) RETURN labels(b) AS labels"
    )
    assert len(records) == 1
    assert NodeLabel.MODULE.value in records[0]["labels"]

    note_records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE} {{uid: $uid}}) RETURN n.unresolved_anchors AS unresolved",
        {"uid": f"{project}:note:a"},
    )
    assert note_records[0]["unresolved"] == []


async def test_anchor_project_prefixed_path_cross_project(graph_client: GraphClient):
    """A project-prefixed path anchor resolves in the NAMED project, not the note's own."""
    await graph_client.ensure_schema()
    target_project = "anchor_target_proj"
    note_project = "anchor_note_proj"
    fp = "src/shared.py"
    module = ParsedEntity(
        name="shared",
        qualified_name=f"{target_project}:src.shared",
        label=NodeLabel.MODULE,
        kind="module",
        line_start=1,
        line_end=10,
        file_path=fp,
        content_hash="shared_v1",
    )
    await graph_client.upsert_file_entities(target_project, fp, [module], [])

    parsed = parse_file(
        "docs/notes/a.md",
        f"---\nid: a\nkind: note\nanchors: [{target_project}:{fp}]\n---\n\nBody.\n".encode(),
        note_project,
    )
    assert parsed is not None
    other, anchors = _split_anchor_rels(parsed.relationships)
    await graph_client.upsert_file_entities(note_project, parsed.file_path, parsed.entities, other)
    await graph_client.resolve_anchors(anchors)

    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE} {{project_name: $np}})-[r:{RelType.DOCUMENTS} {{link_type: 'anchor'}}]->"
        f"(b:{NodeLabel.MODULE} {{project_name: $tp}}) RETURN r.anchor_hash AS hash",
        {"np": note_project, "tp": target_project},
    )
    assert len(records) == 1
    assert records[0]["hash"] == "shared_v1"


async def test_anchor_absolute_path_via_root_path(graph_client: GraphClient):
    """An absolute-path anchor resolves via longest-prefix match against Project.root_path."""
    await graph_client.ensure_schema()
    project = "anchor_abs"
    root = "/repo/anchor_abs"
    await graph_client.merge_project_node(project, root_path=root)

    fp = "src/thing.py"
    module = ParsedEntity(
        name="thing",
        qualified_name=f"{project}:src.thing",
        label=NodeLabel.MODULE,
        kind="module",
        line_start=1,
        line_end=10,
        file_path=fp,
        content_hash="thing_v1",
    )
    await graph_client.upsert_file_entities(project, fp, [module], [])

    parsed = parse_file(
        "docs/notes/a.md",
        f"---\nid: a\nkind: note\nanchors: ['{root}/{fp}']\n---\n\nBody.\n".encode(),
        project,
    )
    assert parsed is not None
    other, anchors = _split_anchor_rels(parsed.relationships)
    await graph_client.upsert_file_entities(project, parsed.file_path, parsed.entities, other)
    await graph_client.resolve_anchors(anchors)

    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE})-[r:{RelType.DOCUMENTS} {{link_type: 'anchor'}}]->(b:{NodeLabel.MODULE}) "
        "RETURN r.anchor_hash AS hash"
    )
    assert len(records) == 1
    assert records[0]["hash"] == "thing_v1"


async def test_anchor_unresolved_cleared_when_target_appears(graph_client: GraphClient):
    """A note whose anchor now resolves is cleared, not left with a stale failure list."""
    await graph_client.ensure_schema()
    project = "anchor_heal"

    parsed = parse_file(
        "docs/notes/a.md",
        f"---\nid: a\nkind: note\nanchors: [{project}:src.foo.Bar]\n---\n\nBody.\n".encode(),
        project,
    )
    assert parsed is not None
    other, anchors = _split_anchor_rels(parsed.relationships)
    await graph_client.upsert_file_entities(project, parsed.file_path, parsed.entities, other)
    await graph_client.resolve_anchors(anchors)

    note_uid = f"{project}:note:a"
    before = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE} {{uid: $uid}}) RETURN n.unresolved_anchors AS unresolved", {"uid": note_uid}
    )
    assert before[0]["unresolved"] == [f"{project}:src.foo.Bar"]

    target = _make_entity(project, "Bar", "src/foo.py", label=NodeLabel.TYPE_DEF, kind="class", content_hash="v1")
    await graph_client.upsert_file_entities(project, "src/foo.py", [target], [])

    # Re-run resolution for the same anchor rel (mirrors the retry-on-appearance pass).
    await graph_client.resolve_anchors(anchors)

    after = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE} {{uid: $uid}}) RETURN n.unresolved_anchors AS unresolved", {"uid": note_uid}
    )
    assert after[0]["unresolved"] == []

    edges = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE} {{uid: $uid}})-[:{RelType.DOCUMENTS} {{link_type: 'anchor'}}]->() "
        "RETURN count(*) AS cnt",
        {"uid": note_uid},
    )
    assert edges[0]["cnt"] == 1


async def test_invalidate_stale_anchors_marks_stale_on_content_change(graph_client: GraphClient):
    """Editing an anchored entity's content flags the anchoring note's edge stale."""
    await graph_client.ensure_schema()
    project = "anchor_stale"
    fp = "src/foo.py"
    target = _make_entity(project, "Bar", fp, label=NodeLabel.TYPE_DEF, kind="class", content_hash="v1")
    await graph_client.upsert_file_entities(project, fp, [target], [])

    parsed = parse_file(
        "docs/notes/a.md",
        f"---\nid: a\nkind: note\nanchors: [{project}:src.foo.Bar]\n---\n\nBody.\n".encode(),
        project,
    )
    assert parsed is not None
    other, anchors = _split_anchor_rels(parsed.relationships)
    await graph_client.upsert_file_entities(project, parsed.file_path, parsed.entities, other)
    await graph_client.resolve_anchors(anchors)

    target_uid = f"{project}:src.foo.Bar"

    marked = await graph_client.invalidate_stale_anchors({target_uid})
    assert marked == 0

    edited = _make_entity(project, "Bar", fp, label=NodeLabel.TYPE_DEF, kind="class", content_hash="v2")
    await graph_client.upsert_file_entities(project, fp, [edited], [])

    marked = await graph_client.invalidate_stale_anchors({target_uid})
    assert marked == 1

    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE})-[r:{RelType.DOCUMENTS} {{link_type: 'anchor'}}]->(b) RETURN r.stale AS stale"
    )
    assert records[0]["stale"] is True


async def test_invalidate_stale_anchors_empty_input_is_noop(graph_client: GraphClient):
    """Guard the fast paths: no changed uids and unrelated uids both mark nothing."""
    await graph_client.ensure_schema()
    assert await graph_client.invalidate_stale_anchors(set()) == 0
    assert await graph_client.invalidate_stale_anchors({"nonexistent:uid"}) == 0


async def test_delete_anchored_entity_marks_note_broken(graph_client: GraphClient):
    """Deleting an anchor's target sets has_broken_anchors in the same delete statement."""
    await graph_client.ensure_schema()
    project = "anchor_delete"
    fp = "src/foo.py"
    target = _make_entity(project, "Bar", fp, label=NodeLabel.TYPE_DEF, kind="class", content_hash="v1")
    await graph_client.upsert_file_entities(project, fp, [target], [])

    parsed = parse_file(
        "docs/notes/a.md",
        f"---\nid: a\nkind: note\nanchors: [{project}:src.foo.Bar]\n---\n\nBody.\n".encode(),
        project,
    )
    assert parsed is not None
    other, anchors = _split_anchor_rels(parsed.relationships)
    await graph_client.upsert_file_entities(project, parsed.file_path, parsed.entities, other)
    await graph_client.resolve_anchors(anchors)

    note_uid = f"{project}:note:a"
    before = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE} {{uid: $uid}}) RETURN n.has_broken_anchors AS broken", {"uid": note_uid}
    )
    assert not before[0]["broken"]

    await graph_client.delete_file_entities(project, fp)

    after = await graph_client.execute(
        f"MATCH (n:{NodeLabel.NOTE} {{uid: $uid}}) RETURN n.has_broken_anchors AS broken", {"uid": note_uid}
    )
    assert after[0]["broken"] is True

    gone = await graph_client.execute("MATCH (n {uid: $uid}) RETURN count(n) AS cnt", {"uid": f"{project}:src.foo.Bar"})
    assert gone[0]["cnt"] == 0


async def test_create_doc_links_ambiguous_match_creates_no_edge(graph_client: GraphClient):
    """Two same-named callables in a project: a heuristic doc ref is left unresolved,
    not fanned out into one edge per candidate (the multi-link bug this fix closes)."""
    await graph_client.ensure_schema()
    project = "doclink_ambiguous"
    fp1, fp2 = "src/a.py", "src/b.py"
    dup1 = _make_entity(project, "process", fp1, content_hash="p1")
    dup2 = _make_entity(project, "process", fp2, content_hash="p2")
    await graph_client.upsert_file_entities(project, fp1, [dup1], [])
    await graph_client.upsert_file_entities(project, fp2, [dup2], [])

    doc_fp = "docs/architecture.md"
    doc_entities = [
        ParsedEntity(
            name="architecture.md",
            qualified_name=f"{project}:docs/architecture.md",
            label=NodeLabel.DOC_FILE,
            kind="doc_file",
            line_start=1,
            line_end=5,
            file_path=doc_fp,
        ),
        ParsedEntity(
            name="Overview",
            qualified_name=f"{project}:docs/architecture.md > Overview",
            label=NodeLabel.DOC_SECTION,
            kind="section",
            line_start=1,
            line_end=5,
            file_path=doc_fp,
        ),
    ]
    doc_rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:docs/architecture.md > Overview",
            rel_type=RelType.DOCUMENTS,
            to_name="process",
            properties={"link_type": "symbol_mention", "confidence": 0.8},
        ),
    ]
    await graph_client.upsert_file_entities(project, doc_fp, doc_entities, doc_rels)

    records = await graph_client.execute(
        f"MATCH (:{NodeLabel.DOC_SECTION} {{project_name: $p}})-[:{RelType.DOCUMENTS}]->() RETURN count(*) AS cnt",
        {"p": project},
    )
    assert records[0]["cnt"] == 0


async def test_create_doc_links_excludes_note_and_docsection_targets(graph_client: GraphClient):
    """A heuristic doc ref never lands on another Note, even on an exact name match."""
    await graph_client.ensure_schema()
    project = "doclink_excl"

    note = ParsedEntity(
        name="helper",
        qualified_name=f"{project}:note:helper-note",
        label=NodeLabel.NOTE,
        kind="note",
        line_start=1,
        line_end=1,
        file_path="docs/notes/helper-note.md",
    )
    await graph_client.upsert_file_entities(project, "docs/notes/helper-note.md", [note], [])

    doc_fp = "docs/architecture.md"
    doc_entities = [
        ParsedEntity(
            name="architecture.md",
            qualified_name=f"{project}:docs/architecture.md",
            label=NodeLabel.DOC_FILE,
            kind="doc_file",
            line_start=1,
            line_end=5,
            file_path=doc_fp,
        ),
        ParsedEntity(
            name="Overview",
            qualified_name=f"{project}:docs/architecture.md > Overview",
            label=NodeLabel.DOC_SECTION,
            kind="section",
            line_start=1,
            line_end=5,
            file_path=doc_fp,
        ),
    ]
    doc_rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:docs/architecture.md > Overview",
            rel_type=RelType.DOCUMENTS,
            to_name="helper",
            properties={"link_type": "symbol_mention", "confidence": 0.8},
        ),
    ]
    await graph_client.upsert_file_entities(project, doc_fp, doc_entities, doc_rels)

    records = await graph_client.execute(
        f"MATCH (:{NodeLabel.DOC_SECTION} {{project_name: $p}})-[:{RelType.DOCUMENTS}]->() RETURN count(*) AS cnt",
        {"p": project},
    )
    assert records[0]["cnt"] == 0


# ---------------------------------------------------------------------------
# Delta upsert
# ---------------------------------------------------------------------------


def _make_entity(
    project: str,
    name: str,
    fp: str,
    *,
    label: NodeLabel = NodeLabel.CALLABLE,
    kind: str = "function",
    line_start: int = 1,
    line_end: int = 5,
    docstring: str | None = None,
    signature: str | None = None,
    content_hash: str = "",
) -> ParsedEntity:
    """Helper to build a ParsedEntity with content_hash."""
    module = fp.replace("/", ".").removesuffix(".py")
    return ParsedEntity(
        name=name,
        qualified_name=f"{project}:{module}.{name}",
        label=label,
        kind=kind,
        line_start=line_start,
        line_end=line_end,
        file_path=fp,
        docstring=docstring,
        signature=signature,
        content_hash=content_hash,
    )


async def test_upsert_delta_no_changes(graph_client: GraphClient):
    """Upserting the same entities twice returns all unchanged, no graph writes."""
    await graph_client.ensure_schema()

    project = "delta1"
    fp = "src/mod.py"
    entities = [
        _make_entity(project, "func_a", fp, content_hash="aaa111"),
        _make_entity(project, "func_b", fp, line_start=6, line_end=10, content_hash="bbb222"),
    ]

    result1 = await graph_client.upsert_file_entities(project, fp, entities, [])
    assert len(result1.added) == 2
    assert len(result1.unchanged) == 0

    result2 = await graph_client.upsert_file_entities(project, fp, entities, [])
    assert len(result2.unchanged) == 2
    assert result2.added == []
    assert result2.modified == []
    assert result2.deleted == []


async def test_upsert_delta_added_entity(graph_client: GraphClient):
    """Adding a new entity to a file is classified as 'added'."""
    await graph_client.ensure_schema()

    project = "delta2"
    fp = "src/mod.py"
    entity_a = _make_entity(project, "func_a", fp, content_hash="aaa111")

    await graph_client.upsert_file_entities(project, fp, [entity_a], [])

    entity_b = _make_entity(project, "func_b", fp, line_start=6, line_end=10, content_hash="bbb222")
    result = await graph_client.upsert_file_entities(project, fp, [entity_a, entity_b], [])
    assert "src.mod.func_b" in result.added
    assert "src.mod.func_a" in result.unchanged
    assert result.deleted == []
    assert result.modified == []

    # Verify both nodes exist in graph
    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.CALLABLE} {{project_name: $p, file_path: $f}}) RETURN n.name AS name",
        {"p": project, "f": fp},
    )
    names = {r["name"] for r in records}
    assert names == {"func_a", "func_b"}


async def test_upsert_delta_deleted_entity(graph_client: GraphClient):
    """Removing an entity from a file is classified as 'deleted'."""
    await graph_client.ensure_schema()

    project = "delta3"
    fp = "src/mod.py"
    entities = [
        _make_entity(project, "func_a", fp, content_hash="aaa111"),
        _make_entity(project, "func_b", fp, line_start=6, line_end=10, content_hash="bbb222"),
    ]

    await graph_client.upsert_file_entities(project, fp, entities, [])

    # Re-upsert with only func_a
    result = await graph_client.upsert_file_entities(project, fp, [entities[0]], [])
    assert "src.mod.func_b" in result.deleted
    assert "src.mod.func_a" in result.unchanged
    assert result.added == []
    assert result.modified == []

    # Verify func_b is gone
    records = await graph_client.execute(
        f"MATCH (n {{project_name: $p, file_path: $f}}) "
        f"WHERE NOT n:{NodeLabel.PACKAGE} AND NOT n:{NodeLabel.PROJECT} "
        "RETURN n.name AS name",
        {"p": project, "f": fp},
    )
    names = {r["name"] for r in records}
    assert names == {"func_a"}


async def test_upsert_delta_modified_entity(graph_client: GraphClient):
    """Changing an entity's content_hash is classified as 'modified'."""
    await graph_client.ensure_schema()

    project = "delta4"
    fp = "src/mod.py"
    entity_v1 = _make_entity(project, "func_a", fp, docstring="Version 1", content_hash="hash_v1")

    await graph_client.upsert_file_entities(project, fp, [entity_v1], [])

    entity_v2 = _make_entity(project, "func_a", fp, docstring="Version 2", content_hash="hash_v2")
    result = await graph_client.upsert_file_entities(project, fp, [entity_v2], [])
    assert "src.mod.func_a" in result.modified
    assert result.added == []
    assert result.deleted == []
    assert result.unchanged == []

    # Verify the node was updated in graph
    records = await graph_client.execute(
        "MATCH (n {uid: $uid}) RETURN n.docstring AS doc, n.content_hash AS hash",
        {"uid": f"{project}:src.mod.func_a"},
    )
    assert len(records) == 1
    assert records[0]["doc"] == "Version 2"
    assert records[0]["hash"] == "hash_v2"


async def test_upsert_delta_preserves_embeddings_unchanged(graph_client: GraphClient):
    """Unchanged entities keep their embed_hash and embedding without save/restore."""
    await graph_client.ensure_schema()

    project = "delta5"
    fp = "src/mod.py"
    entity = _make_entity(project, "func_a", fp, content_hash="stable_hash")

    await graph_client.upsert_file_entities(project, fp, [entity], [])

    # Set embed data
    dim = graph_client._dimension
    embedding = [0.5] * dim
    await graph_client.execute_write(
        "MATCH (n {uid: $uid}) SET n.embed_hash = $hash, n.embedding = $vec",
        {"uid": f"{project}:src.mod.func_a", "hash": "embed_abc", "vec": embedding},
    )

    # Re-upsert with same content_hash — should be unchanged
    result = await graph_client.upsert_file_entities(project, fp, [entity], [])
    assert "src.mod.func_a" in result.unchanged

    # Verify embed data is still there (never deleted because entity was unchanged)
    records = await graph_client.execute(
        "MATCH (n {uid: $uid}) RETURN n.embed_hash AS hash, n.embedding AS vec",
        {"uid": f"{project}:src.mod.func_a"},
    )
    assert len(records) == 1
    assert records[0]["hash"] == "embed_abc"
    assert records[0]["vec"] == embedding


async def test_upsert_updates_positions_on_shift(graph_client: GraphClient):
    """Adding an entity above existing ones updates their line_start/line_end."""
    await graph_client.ensure_schema()

    project = "delta_pos"
    fp = "src/mod.py"

    # Initial: func_b at lines 1-5
    entity_b = _make_entity(project, "func_b", fp, line_start=1, line_end=5, content_hash="bbb")
    await graph_client.upsert_file_entities(project, fp, [entity_b], [])

    # Set embed data on func_b — should be preserved after position update
    dim = graph_client._dimension
    await graph_client.execute_write(
        "MATCH (n {uid: $uid}) SET n.embed_hash = $hash, n.embedding = $vec",
        {"uid": f"{project}:src.mod.func_b", "hash": "emb_b", "vec": [0.5] * dim},
    )

    # Now add func_a above it — func_b shifts to lines 6-10
    entity_a = _make_entity(project, "func_a", fp, line_start=1, line_end=4, content_hash="aaa")
    entity_b_shifted = _make_entity(project, "func_b", fp, line_start=6, line_end=10, content_hash="bbb")

    result = await graph_client.upsert_file_entities(project, fp, [entity_a, entity_b_shifted], [])
    assert "src.mod.func_a" in result.added
    assert "src.mod.func_b" in result.unchanged

    # Verify func_b's position was updated
    records = await graph_client.execute(
        "MATCH (n {uid: $uid}) RETURN n.line_start AS ls, n.line_end AS le, n.embed_hash AS hash",
        {"uid": f"{project}:src.mod.func_b"},
    )
    assert len(records) == 1
    assert records[0]["ls"] == 6
    assert records[0]["le"] == 10
    # Embed data preserved — no re-embedding needed
    assert records[0]["hash"] == "emb_b"


async def test_upsert_body_only_edit_classified_modified(graph_client: GraphClient):
    """A body-only edit (same signature/docstring) is classified modified, not unchanged."""
    await graph_client.ensure_schema()

    project = "delta_body"
    fp = "src/mod.py"

    parsed1 = parse_file(fp, b"def f():\n    return 1\n", project)
    assert parsed1 is not None
    await graph_client.upsert_file_entities(project, fp, parsed1.entities, parsed1.relationships)

    parsed2 = parse_file(fp, b"def f():\n    return 2\n", project)
    assert parsed2 is not None
    result = await graph_client.upsert_file_entities(project, fp, parsed2.entities, parsed2.relationships)

    assert "mod.f" in result.modified
    assert "mod.f" not in result.unchanged

    records = await graph_client.execute("MATCH (n {uid: $uid}) RETURN n.source AS src", {"uid": f"{project}:mod.f"})
    assert len(records) == 1
    assert "return 2" in records[0]["src"]


async def test_upsert_shifted_positions_updated_on_modified_only(graph_client: GraphClient):
    """Position shifts apply for unchanged entities even when nothing was added/deleted."""
    await graph_client.ensure_schema()

    project = "delta_shift_mod"
    fp = "src/mod.py"

    entities_v1 = [
        _make_entity(project, "func_a", fp, line_start=1, line_end=5, content_hash="a1"),
        _make_entity(project, "func_b", fp, line_start=7, line_end=11, content_hash="b"),
    ]
    await graph_client.upsert_file_entities(project, fp, entities_v1, [])

    # func_a grows (modified); func_b is unchanged but shifted down
    entities_v2 = [
        _make_entity(project, "func_a", fp, line_start=1, line_end=8, content_hash="a2"),
        _make_entity(project, "func_b", fp, line_start=10, line_end=14, content_hash="b"),
    ]
    result = await graph_client.upsert_file_entities(project, fp, entities_v2, [])
    assert "src.mod.func_a" in result.modified
    assert "src.mod.func_b" in result.unchanged

    records = await graph_client.execute(
        "MATCH (n {uid: $uid}) RETURN n.line_start AS ls, n.line_end AS le",
        {"uid": f"{project}:src.mod.func_b"},
    )
    assert (records[0]["ls"], records[0]["le"]) == (10, 14)
    records_a = await graph_client.execute(
        "MATCH (n {uid: $uid}) RETURN n.line_end AS le",
        {"uid": f"{project}:src.mod.func_a"},
    )
    assert records_a[0]["le"] == 8


async def test_upsert_positions_updated_on_pure_shift(graph_client: GraphClient):
    """A pure position shift (e.g. comment inserted above) still updates stored positions."""
    await graph_client.ensure_schema()

    project = "delta_shift_pure"
    fp = "src/mod.py"

    await graph_client.upsert_file_entities(
        project, fp, [_make_entity(project, "func_a", fp, line_start=2, line_end=6, content_hash="a")], []
    )
    result = await graph_client.upsert_file_entities(
        project, fp, [_make_entity(project, "func_a", fp, line_start=4, line_end=8, content_hash="a")], []
    )
    assert result.unchanged == ["src.mod.func_a"]
    assert result.modified == []

    records = await graph_client.execute(
        "MATCH (n {uid: $uid}) RETURN n.line_start AS ls, n.line_end AS le",
        {"uid": f"{project}:src.mod.func_a"},
    )
    assert (records[0]["ls"], records[0]["le"]) == (4, 8)


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------


def _setup_module_node(project: str, fp: str) -> tuple[list[ParsedEntity], str]:
    """Helper: create a Module entity for import resolution tests.

    Returns (entities, module_uid).
    """
    module_qn = fp.replace("/", ".").removesuffix(".py")
    uid = f"{project}:{module_qn}"
    entity = ParsedEntity(
        name=fp.rsplit("/", 1)[-1].removesuffix(".py"),
        qualified_name=uid,
        label=NodeLabel.MODULE,
        kind="module",
        line_start=1,
        line_end=1,
        file_path=fp,
        content_hash="mod_hash",
    )
    return [entity], uid


async def test_resolve_imports_external(graph_client: GraphClient):
    """Module with external-only imports creates ExternalPackage + ExternalSymbol nodes."""
    await graph_client.ensure_schema()

    project = "imp_ext"
    fp = "src/app.py"
    entities, mod_uid = _setup_module_node(project, fp)
    await graph_client.upsert_file_entities(project, fp, entities, [])

    import_rels = [
        ParsedRelationship(from_qualified_name=mod_uid, rel_type=RelType.IMPORTS, to_name="loguru.logger"),
    ]
    await graph_client.resolve_imports(project, import_rels)

    # ExternalPackage created
    pkg_records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.EXTERNAL_PACKAGE} {{project_name: $p}}) RETURN n.uid AS uid, n.name AS name",
        {"p": project},
    )
    assert len(pkg_records) == 1
    assert pkg_records[0]["name"] == "loguru"

    # ExternalSymbol created
    sym_records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.EXTERNAL_SYMBOL} {{project_name: $p}}) "
        "RETURN n.uid AS uid, n.name AS name, n.package AS pkg",
        {"p": project},
    )
    assert len(sym_records) == 1
    assert sym_records[0]["name"] == "logger"
    assert sym_records[0]["pkg"] == "loguru"

    # CONTAINS edge: ExternalPackage → ExternalSymbol
    contains = await graph_client.execute(
        f"MATCH (p:{NodeLabel.EXTERNAL_PACKAGE})-[:{RelType.CONTAINS}]->(s:{NodeLabel.EXTERNAL_SYMBOL}) "
        "RETURN p.name AS pkg, s.name AS sym",
    )
    assert len(contains) == 1

    # IMPORTS edge: Module → ExternalSymbol
    imports = await graph_client.execute(
        f"MATCH (m:{NodeLabel.MODULE})-[:{RelType.IMPORTS}]->(s:{NodeLabel.EXTERNAL_SYMBOL}) "
        "RETURN m.name AS mod, s.name AS sym",
    )
    assert len(imports) == 1
    assert imports[0]["sym"] == "logger"


async def test_resolve_imports_internal(graph_client: GraphClient):
    """Module importing from another indexed module creates IMPORTS edge to internal entity."""
    await graph_client.ensure_schema()

    project = "imp_int"

    # Create two modules: app.py imports utils.helper
    fp_utils = "src/utils.py"
    utils_entities = [
        ParsedEntity(
            name="utils",
            qualified_name=f"{project}:src.utils",
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=1,
            file_path=fp_utils,
            content_hash="utils_hash",
        ),
        ParsedEntity(
            name="helper",
            qualified_name=f"{project}:src.utils.helper",
            label=NodeLabel.CALLABLE,
            kind="function",
            line_start=2,
            line_end=5,
            file_path=fp_utils,
            content_hash="helper_hash",
        ),
    ]
    await graph_client.upsert_file_entities(project, fp_utils, utils_entities, [])

    fp_app = "src/app.py"
    app_entities, app_uid = _setup_module_node(project, fp_app)
    await graph_client.upsert_file_entities(project, fp_app, app_entities, [])

    import_rels = [
        ParsedRelationship(from_qualified_name=app_uid, rel_type=RelType.IMPORTS, to_name="src.utils.helper"),
    ]
    await graph_client.resolve_imports(project, import_rels)

    # IMPORTS edge to internal entity (no External* nodes created)
    imports = await graph_client.execute(
        f"MATCH (m:{NodeLabel.MODULE} {{name: 'app'}})-[:{RelType.IMPORTS}]->(c:{NodeLabel.CALLABLE}) "
        "RETURN c.name AS name",
    )
    assert len(imports) == 1
    assert imports[0]["name"] == "helper"

    # No external nodes
    ext = await graph_client.execute(
        f"MATCH (n:{NodeLabel.EXTERNAL_PACKAGE} {{project_name: $p}}) RETURN n", {"p": project}
    )
    assert len(ext) == 0


async def test_resolve_imports_mixed(graph_client: GraphClient):
    """Both internal and external imports are correctly classified."""
    await graph_client.ensure_schema()

    project = "imp_mix"

    # Internal target
    fp_utils = "src/utils.py"
    utils_entities = [
        ParsedEntity(
            name="utils",
            qualified_name=f"{project}:src.utils",
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=1,
            file_path=fp_utils,
            content_hash="u_hash",
        ),
    ]
    await graph_client.upsert_file_entities(project, fp_utils, utils_entities, [])

    fp_app = "src/app.py"
    app_entities, app_uid = _setup_module_node(project, fp_app)
    await graph_client.upsert_file_entities(project, fp_app, app_entities, [])

    import_rels = [
        ParsedRelationship(from_qualified_name=app_uid, rel_type=RelType.IMPORTS, to_name="src.utils"),
        ParsedRelationship(from_qualified_name=app_uid, rel_type=RelType.IMPORTS, to_name="pydantic.BaseModel"),
    ]
    await graph_client.resolve_imports(project, import_rels)

    # Internal import
    internal = await graph_client.execute(
        f"MATCH (:{NodeLabel.MODULE} {{name: 'app'}})-[:{RelType.IMPORTS}]->(n:{NodeLabel.MODULE}) "
        "RETURN n.name AS name",
    )
    assert len(internal) == 1
    assert internal[0]["name"] == "utils"

    # External import
    external = await graph_client.execute(
        f"MATCH (:{NodeLabel.MODULE} {{name: 'app'}})-[:{RelType.IMPORTS}]->(n:{NodeLabel.EXTERNAL_SYMBOL}) "
        "RETURN n.name AS name",
    )
    assert len(external) == 1
    assert external[0]["name"] == "BaseModel"


async def test_resolve_imports_idempotent(graph_client: GraphClient):
    """Re-running resolve_imports does not duplicate External* nodes (MERGE)."""
    await graph_client.ensure_schema()

    project = "imp_idem"
    fp = "src/app.py"
    entities, mod_uid = _setup_module_node(project, fp)
    await graph_client.upsert_file_entities(project, fp, entities, [])

    import_rels = [
        ParsedRelationship(from_qualified_name=mod_uid, rel_type=RelType.IMPORTS, to_name="requests.get"),
    ]

    # Run twice
    await graph_client.resolve_imports(project, import_rels)
    await graph_client.resolve_imports(project, import_rels)

    # Should still be exactly one ExternalPackage and one ExternalSymbol
    pkgs = await graph_client.execute(
        f"MATCH (n:{NodeLabel.EXTERNAL_PACKAGE} {{project_name: $p}}) RETURN n", {"p": project}
    )
    assert len(pkgs) == 1

    syms = await graph_client.execute(
        f"MATCH (n:{NodeLabel.EXTERNAL_SYMBOL} {{project_name: $p}}) RETURN n", {"p": project}
    )
    assert len(syms) == 1


async def test_resolve_imports_bare_package(graph_client: GraphClient):
    """Bare package import (e.g. `import os`) creates IMPORTS edge to ExternalPackage directly."""
    await graph_client.ensure_schema()

    project = "imp_bare"
    fp = "src/app.py"
    entities, mod_uid = _setup_module_node(project, fp)
    await graph_client.upsert_file_entities(project, fp, entities, [])

    import_rels = [
        ParsedRelationship(from_qualified_name=mod_uid, rel_type=RelType.IMPORTS, to_name="os"),
    ]
    await graph_client.resolve_imports(project, import_rels)

    # IMPORTS edge points directly to ExternalPackage (no ExternalSymbol)
    imports = await graph_client.execute(
        f"MATCH (m:{NodeLabel.MODULE})-[:{RelType.IMPORTS}]->(p:{NodeLabel.EXTERNAL_PACKAGE}) RETURN p.name AS name",
    )
    assert len(imports) == 1
    assert imports[0]["name"] == "os"

    # No ExternalSymbol for bare import
    syms = await graph_client.execute(
        f"MATCH (n:{NodeLabel.EXTERNAL_SYMBOL} {{project_name: $p}}) RETURN n", {"p": project}
    )
    assert len(syms) == 0


async def test_external_package_versions(graph_client: GraphClient):
    """Version property is set on ExternalPackage nodes."""
    await graph_client.ensure_schema()

    project = "imp_ver"
    fp = "src/app.py"
    entities, mod_uid = _setup_module_node(project, fp)
    await graph_client.upsert_file_entities(project, fp, entities, [])

    import_rels = [
        ParsedRelationship(from_qualified_name=mod_uid, rel_type=RelType.IMPORTS, to_name="loguru.logger"),
        ParsedRelationship(from_qualified_name=mod_uid, rel_type=RelType.IMPORTS, to_name="pydantic.BaseModel"),
    ]
    await graph_client.resolve_imports(project, import_rels)

    # Set versions
    versions = {"loguru": "~=0.7", "pydantic": ">=2.0,<3.0"}
    await graph_client.update_external_package_versions(project, versions)

    # Verify
    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.EXTERNAL_PACKAGE} {{project_name: $p}}) "
        "RETURN n.name AS name, n.version AS version ORDER BY n.name",
        {"p": project},
    )
    assert len(records) == 2
    version_map = {r["name"]: r["version"] for r in records}
    assert version_map["loguru"] == "~=0.7"
    assert version_map["pydantic"] == ">=2.0,<3.0"


async def test_resolve_imports_prefix_fallback_reexport(graph_client: GraphClient):
    """Imports of re-exported names resolve to the closest containing module, not External* stubs."""
    await graph_client.ensure_schema()

    project = "imp_prefix"
    fp_utils = "pkg/utils.py"
    utils_entities = [
        ParsedEntity(
            name="utils",
            qualified_name=f"{project}:pkg.utils",
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=5,
            file_path=fp_utils,
            content_hash="u_hash",
        ),
    ]
    await graph_client.upsert_file_entities(project, fp_utils, utils_entities, [])

    fp_app = "pkg/app.py"
    app_entities = [
        ParsedEntity(
            name="app",
            qualified_name=f"{project}:pkg.app",
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=5,
            file_path=fp_app,
            content_hash="a_hash",
        ),
    ]
    await graph_client.upsert_file_entities(project, fp_app, app_entities, [])

    # 'reexported_helper' is not a stored entity — only the containing module is
    import_rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:pkg.app",
            rel_type=RelType.IMPORTS,
            to_name="pkg.utils.reexported_helper",
        ),
    ]
    await graph_client.resolve_imports(project, import_rels)

    imports = await graph_client.execute(
        f"MATCH (:{NodeLabel.MODULE} {{name: 'app'}})-[:{RelType.IMPORTS}]->(n:{NodeLabel.MODULE}) "
        "RETURN n.name AS name",
    )
    assert len(imports) == 1
    assert imports[0]["name"] == "utils"

    ext = await graph_client.execute(
        f"MATCH (n {{project_name: $p}}) "
        f"WHERE n:{NodeLabel.EXTERNAL_PACKAGE} OR n:{NodeLabel.EXTERNAL_SYMBOL} RETURN n",
        {"p": project},
    )
    assert len(ext) == 0


async def test_resolve_imports_prefix_fallback_python_only(graph_client: GraphClient):
    """The dotted-prefix fallback must not fire for non-Python importers.

    A C# ``using System.Collections.Generic`` must classify as external even
    when the project happens to contain a module named 'System' — dotted import
    paths in non-Python languages live in a different namespace than the
    path-derived qualified_names, so a prefix hit would be a misclassification.
    """
    await graph_client.ensure_schema()

    project = "imp_lang_gate"
    for name, qn, fp in (("System", "System", "System.cs"), ("App", "src.App", "src/App.cs")):
        entities = [
            ParsedEntity(
                name=name,
                qualified_name=f"{project}:{qn}",
                label=NodeLabel.MODULE,
                kind="module",
                line_start=1,
                line_end=5,
                file_path=fp,
                content_hash=f"{name}_hash",
            ),
        ]
        await graph_client.upsert_file_entities(project, fp, entities, [])

    import_rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:src.App",
            rel_type=RelType.IMPORTS,
            to_name="System.Collections.Generic",
        ),
    ]
    await graph_client.resolve_imports(project, import_rels)

    internal = await graph_client.execute(
        f"MATCH (:{NodeLabel.MODULE} {{uid: $u}})-[:{RelType.IMPORTS}]->(m:{NodeLabel.MODULE}) RETURN m.name AS name",
        {"u": f"{project}:src.App"},
    )
    assert internal == []

    ext = await graph_client.execute(
        f"MATCH (:{NodeLabel.MODULE} {{uid: $u}})-[:{RelType.IMPORTS}]->(s:{NodeLabel.EXTERNAL_SYMBOL}) "
        "RETURN s.qualified_name AS qn",
        {"u": f"{project}:src.App"},
    )
    assert [r["qn"] for r in ext] == ["ext/System.Collections.Generic"]


async def test_resolve_imports_from_package_node(graph_client: GraphClient):
    """IMPORTS edges from __init__.py (Package-labeled) source nodes are created."""
    await graph_client.ensure_schema()

    project = "imp_pkg"
    pkg_entity = ParsedEntity(
        name="pkg",
        qualified_name=f"{project}:pkg",
        label=NodeLabel.PACKAGE,
        kind="package",
        line_start=1,
        line_end=2,
        file_path="pkg/__init__.py",
        content_hash="p_hash",
    )
    await graph_client.upsert_file_entities(project, "pkg/__init__.py", [pkg_entity], [])

    for mod_name in ("mod", "types"):
        entities = [
            ParsedEntity(
                name=mod_name,
                qualified_name=f"{project}:pkg.{mod_name}",
                label=NodeLabel.MODULE,
                kind="module",
                line_start=1,
                line_end=5,
                file_path=f"pkg/{mod_name}.py",
                content_hash=f"{mod_name}_hash",
            ),
        ]
        await graph_client.upsert_file_entities(project, f"pkg/{mod_name}.py", entities, [])

    import_rels = [
        ParsedRelationship(from_qualified_name=f"{project}:pkg", rel_type=RelType.IMPORTS, to_name="pkg.mod"),
        ParsedRelationship(
            from_qualified_name=f"{project}:pkg",
            rel_type=RelType.IMPORTS,
            to_name="pkg.types",
            properties={"type_only": True},
        ),
    ]
    await graph_client.resolve_imports(project, import_rels)

    normal = await graph_client.execute(
        f"MATCH (p:{NodeLabel.PACKAGE} {{uid: $u}})-[:{RelType.IMPORTS}]->(m:{NodeLabel.MODULE} {{name: 'mod'}}) "
        "RETURN count(*) AS cnt",
        {"u": f"{project}:pkg"},
    )
    assert normal[0]["cnt"] == 1

    type_only = await graph_client.execute(
        f"MATCH (p:{NodeLabel.PACKAGE} {{uid: $u}})-[e:{RelType.IMPORTS}]->(m:{NodeLabel.MODULE} {{name: 'types'}}) "
        "RETURN e.type_only AS to",
        {"u": f"{project}:pkg"},
    )
    assert len(type_only) == 1
    assert type_only[0]["to"] is True


async def test_src_layout_end_to_end_import_resolution(graph_client: GraphClient):
    """Full src-layout chain: parse → upsert → resolve produces internal Package→Callable IMPORTS."""
    await graph_client.ensure_schema()

    project = "imp_srclayout"
    parsed_init = parse_file("src/mypkg/__init__.py", b"from .util import helper\n", project)
    parsed_util = parse_file("src/mypkg/util.py", b"def helper():\n    return 1\n", project)
    assert parsed_init is not None
    assert parsed_util is not None

    import_rels = [r for r in parsed_init.relationships if r.rel_type == RelType.IMPORTS]
    assert import_rels, "parser should emit an IMPORTS rel for the relative import"

    await graph_client.upsert_file_entities(
        project, "src/mypkg/__init__.py", parsed_init.entities, parsed_init.relationships
    )
    await graph_client.upsert_file_entities(
        project, "src/mypkg/util.py", parsed_util.entities, parsed_util.relationships
    )

    await graph_client.resolve_imports(project, import_rels)

    imports = await graph_client.execute(
        f"MATCH (p:{NodeLabel.PACKAGE} {{project_name: $proj, name: 'mypkg'}})-[:{RelType.IMPORTS}]->"
        f"(c:{NodeLabel.CALLABLE}) RETURN c.name AS name",
        {"proj": project},
    )
    assert len(imports) == 1
    assert imports[0]["name"] == "helper"

    ext = await graph_client.execute(
        f"MATCH (n {{project_name: $proj}}) "
        f"WHERE n:{NodeLabel.EXTERNAL_PACKAGE} OR n:{NodeLabel.EXTERNAL_SYMBOL} RETURN n",
        {"proj": project},
    )
    assert len(ext) == 0


# ---------------------------------------------------------------------------
# Cross-project import resolution
# ---------------------------------------------------------------------------


async def test_cross_project_resolution_rewires_all_symbols(graph_client: GraphClient):
    """Cross-project resolution rewires EVERY matched symbol, not just one row per query."""
    await graph_client.ensure_schema()

    proj_app, proj_lib = "xp_app", "xp_lib"

    # Project A: module importing two symbols from libpkg
    fp_app = "src/app.py"
    app_entities, app_uid = _setup_module_node(proj_app, fp_app)
    await graph_client.upsert_file_entities(proj_app, fp_app, app_entities, [])
    import_rels = [
        ParsedRelationship(from_qualified_name=app_uid, rel_type=RelType.IMPORTS, to_name="libpkg.func_one"),
        ParsedRelationship(from_qualified_name=app_uid, rel_type=RelType.IMPORTS, to_name="libpkg.func_two"),
    ]
    await graph_client.resolve_imports(proj_app, import_rels)

    # Project B: the real libpkg package with both callables
    await graph_client.merge_package_node(proj_lib, "libpkg", "libpkg", "libpkg/__init__.py")
    lib_entities = [
        ParsedEntity(
            name="mod",
            qualified_name=f"{proj_lib}:libpkg.mod",
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=10,
            file_path="libpkg/mod.py",
            content_hash="m",
        ),
        ParsedEntity(
            name="func_one",
            qualified_name=f"{proj_lib}:libpkg.mod.func_one",
            label=NodeLabel.CALLABLE,
            kind="function",
            line_start=1,
            line_end=2,
            file_path="libpkg/mod.py",
            content_hash="f1",
        ),
        ParsedEntity(
            name="func_two",
            qualified_name=f"{proj_lib}:libpkg.mod.func_two",
            label=NodeLabel.CALLABLE,
            kind="function",
            line_start=4,
            line_end=5,
            file_path="libpkg/mod.py",
            content_hash="f2",
        ),
    ]
    await graph_client.upsert_file_entities(proj_lib, "libpkg/mod.py", lib_entities, [])

    rewired = await graph_client.resolve_cross_project_imports([proj_app, proj_lib])
    assert rewired == 2

    edges = await graph_client.execute(
        f"MATCH (m:{NodeLabel.MODULE} {{project_name: $a}})-[:{RelType.IMPORTS}]->"
        f"(c:{NodeLabel.CALLABLE} {{project_name: $b}}) "
        "RETURN c.name AS name ORDER BY c.name",
        {"a": proj_app, "b": proj_lib},
    )
    assert [e["name"] for e in edges] == ["func_one", "func_two"]

    stubs = await graph_client.execute(
        f"MATCH (n:{NodeLabel.EXTERNAL_SYMBOL} {{project_name: $a}}) RETURN n",
        {"a": proj_app},
    )
    assert len(stubs) == 0


# ---------------------------------------------------------------------------
# CALLS resolution
# ---------------------------------------------------------------------------


async def test_resolve_calls_via_import(graph_client: GraphClient):
    """Module imports a function, calls it → CALLS edge created to imported target."""
    await graph_client.ensure_schema()

    project = "call_imp"

    # Module A defines helper
    fp_utils = "src/utils.py"
    utils_entities = [
        ParsedEntity(
            name="utils",
            qualified_name=f"{project}:src.utils",
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=10,
            file_path=fp_utils,
            content_hash="u_hash",
        ),
        ParsedEntity(
            name="helper",
            qualified_name=f"{project}:src.utils.helper",
            label=NodeLabel.CALLABLE,
            kind="function",
            line_start=2,
            line_end=5,
            file_path=fp_utils,
            content_hash="h_hash",
        ),
    ]
    await graph_client.upsert_file_entities(project, fp_utils, utils_entities, [])

    # Module B imports helper and has a function that calls it
    fp_app = "src/app.py"
    app_entities = [
        ParsedEntity(
            name="app",
            qualified_name=f"{project}:src.app",
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=10,
            file_path=fp_app,
            content_hash="a_hash",
        ),
        ParsedEntity(
            name="main",
            qualified_name=f"{project}:src.app.main",
            label=NodeLabel.CALLABLE,
            kind="function",
            line_start=3,
            line_end=8,
            file_path=fp_app,
            content_hash="m_hash",
        ),
    ]
    await graph_client.upsert_file_entities(project, fp_app, app_entities, [])

    # Create IMPORTS edge: app module → helper
    import_rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:src.app",
            rel_type=RelType.IMPORTS,
            to_name="src.utils.helper",
        ),
    ]
    await graph_client.resolve_imports(project, import_rels)

    # Now resolve calls: main() calls helper()
    call_rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:src.app.main",
            rel_type=RelType.CALLS,
            to_name="helper",
        ),
    ]
    await graph_client.resolve_calls(project, call_rels)

    # Verify CALLS edge: main → helper
    records = await graph_client.execute(
        f"MATCH (a:{NodeLabel.CALLABLE})-[:{RelType.CALLS}]->(b:{NodeLabel.CALLABLE}) "
        "RETURN a.name AS caller, b.name AS callee",
    )
    assert len(records) == 1
    assert records[0]["caller"] == "main"
    assert records[0]["callee"] == "helper"


async def test_resolve_calls_same_class(graph_client: GraphClient):
    """Method calls sibling method → resolves to sibling."""
    await graph_client.ensure_schema()

    project = "call_sib"
    fp = "src/mod.py"

    entities = [
        ParsedEntity(
            name="mod",
            qualified_name=f"{project}:src.mod",
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=20,
            file_path=fp,
            content_hash="mod_hash",
        ),
        ParsedEntity(
            name="MyClass",
            qualified_name=f"{project}:src.mod.MyClass",
            label=NodeLabel.TYPE_DEF,
            kind="class",
            line_start=2,
            line_end=18,
            file_path=fp,
            content_hash="cls_hash",
        ),
        ParsedEntity(
            name="process",
            qualified_name=f"{project}:src.mod.MyClass.process",
            label=NodeLabel.CALLABLE,
            kind="method",
            line_start=3,
            line_end=8,
            file_path=fp,
            content_hash="proc_hash",
        ),
        ParsedEntity(
            name="validate",
            qualified_name=f"{project}:src.mod.MyClass.validate",
            label=NodeLabel.CALLABLE,
            kind="method",
            line_start=10,
            line_end=15,
            file_path=fp,
            content_hash="val_hash",
        ),
    ]
    rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:src.mod",
            rel_type=RelType.DEFINES,
            to_name=f"{project}:src.mod.MyClass",
        ),
        ParsedRelationship(
            from_qualified_name=f"{project}:src.mod.MyClass",
            rel_type=RelType.DEFINES,
            to_name=f"{project}:src.mod.MyClass.process",
        ),
        ParsedRelationship(
            from_qualified_name=f"{project}:src.mod.MyClass",
            rel_type=RelType.DEFINES,
            to_name=f"{project}:src.mod.MyClass.validate",
        ),
    ]
    await graph_client.upsert_file_entities(project, fp, entities, rels)

    # process() calls validate()
    call_rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:src.mod.MyClass.process",
            rel_type=RelType.CALLS,
            to_name="validate",
        ),
    ]
    await graph_client.resolve_calls(project, call_rels)

    records = await graph_client.execute(
        f"MATCH (a:{NodeLabel.CALLABLE})-[:{RelType.CALLS}]->(b:{NodeLabel.CALLABLE}) "
        "RETURN a.name AS caller, b.name AS callee",
    )
    assert len(records) == 1
    assert records[0]["caller"] == "process"
    assert records[0]["callee"] == "validate"


async def test_resolve_calls_unresolved_skipped(graph_client: GraphClient):
    """Call to builtin (e.g. 'print') → no edge created, no crash."""
    await graph_client.ensure_schema()

    project = "call_unres"
    fp = "src/mod.py"

    entities = [
        ParsedEntity(
            name="mod",
            qualified_name=f"{project}:src.mod",
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=10,
            file_path=fp,
            content_hash="mod_hash",
        ),
        ParsedEntity(
            name="func",
            qualified_name=f"{project}:src.mod.func",
            label=NodeLabel.CALLABLE,
            kind="function",
            line_start=2,
            line_end=5,
            file_path=fp,
            content_hash="f_hash",
        ),
    ]
    await graph_client.upsert_file_entities(project, fp, entities, [])

    call_rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:src.mod.func",
            rel_type=RelType.CALLS,
            to_name="print",
        ),
    ]
    await graph_client.resolve_calls(project, call_rels)

    # No CALLS edges should exist
    records = await graph_client.execute(
        f"MATCH ()-[r:{RelType.CALLS}]->() RETURN count(r) AS cnt",
    )
    assert records[0]["cnt"] == 0


async def test_resolve_calls_deduplication(graph_client: GraphClient):
    """Same call from same caller twice → single CALLS edge."""
    await graph_client.ensure_schema()

    project = "call_dedup"
    fp = "src/mod.py"

    entities = [
        ParsedEntity(
            name="mod",
            qualified_name=f"{project}:src.mod",
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=10,
            file_path=fp,
            content_hash="mod_hash",
        ),
        ParsedEntity(
            name="caller_func",
            qualified_name=f"{project}:src.mod.caller_func",
            label=NodeLabel.CALLABLE,
            kind="function",
            line_start=2,
            line_end=5,
            file_path=fp,
            content_hash="cf_hash",
        ),
        ParsedEntity(
            name="target_func",
            qualified_name=f"{project}:src.mod.target_func",
            label=NodeLabel.CALLABLE,
            kind="function",
            line_start=7,
            line_end=10,
            file_path=fp,
            content_hash="tf_hash",
        ),
    ]
    await graph_client.upsert_file_entities(project, fp, entities, [])

    # Two CALLS rels pointing to the same target from same caller
    call_rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:src.mod.caller_func",
            rel_type=RelType.CALLS,
            to_name="target_func",
        ),
        ParsedRelationship(
            from_qualified_name=f"{project}:src.mod.caller_func",
            rel_type=RelType.CALLS,
            to_name="target_func",
        ),
    ]
    await graph_client.resolve_calls(project, call_rels)

    # Only one CALLS edge
    records = await graph_client.execute(
        f"MATCH (a)-[r:{RelType.CALLS}]->(b) RETURN a.name AS caller, b.name AS callee",
    )
    assert len(records) == 1
    assert records[0]["caller"] == "caller_func"
    assert records[0]["callee"] == "target_func"


# ---------------------------------------------------------------------------
# IMPLEMENTS resolution (bare names vs uids)
# ---------------------------------------------------------------------------


async def _count_implements(graph_client: GraphClient, project: str, impl_name: str, iface_name: str) -> int:
    """Count (:TypeDef {impl_name})-[:IMPLEMENTS]->(:TypeDef {iface_name}) edges in *project*."""
    records = await graph_client.execute(
        f"MATCH (a:{NodeLabel.TYPE_DEF} {{project_name: $p, name: $impl}})-[:{RelType.IMPLEMENTS}]->"
        f"(b:{NodeLabel.TYPE_DEF} {{project_name: $p, name: $iface}}) RETURN count(*) AS cnt",
        {"p": project, "impl": impl_name, "iface": iface_name},
    )
    return records[0]["cnt"]


async def test_implements_bare_name_resolved(graph_client: GraphClient):
    """Parser-emitted bare-name IMPLEMENTS resolves to the same-project TypeDef by name."""
    await graph_client.ensure_schema()

    project = "impl_bare"
    fp = "src/log.py"
    entities = [
        _make_entity(project, "Logger", fp, label=NodeLabel.TYPE_DEF, kind="interface", content_hash="ifc"),
        _make_entity(
            project,
            "FileLogger",
            fp,
            label=NodeLabel.TYPE_DEF,
            kind="class",
            line_start=6,
            line_end=10,
            content_hash="cls",
        ),
    ]
    rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:src.log.FileLogger",
            rel_type=RelType.IMPLEMENTS,
            to_name="Logger",
        ),
    ]
    await graph_client.upsert_file_entities(project, fp, entities, rels)

    assert await _count_implements(graph_client, project, "FileLogger", "Logger") == 1


async def test_implements_bare_name_cross_file(graph_client: GraphClient):
    """IMPLEMENTS resolves across files — the interface lives in another module."""
    await graph_client.ensure_schema()

    project = "impl_xfile"
    iface = _make_entity(
        project, "Logger", "src/contracts.py", label=NodeLabel.TYPE_DEF, kind="interface", content_hash="i"
    )
    await graph_client.upsert_file_entities(project, "src/contracts.py", [iface], [])

    impl = _make_entity(
        project, "FileLogger", "src/service.py", label=NodeLabel.TYPE_DEF, kind="class", content_hash="c"
    )
    rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:src.service.FileLogger",
            rel_type=RelType.IMPLEMENTS,
            to_name="Logger",
        ),
    ]
    await graph_client.upsert_file_entities(project, "src/service.py", [impl], rels)

    assert await _count_implements(graph_client, project, "FileLogger", "Logger") == 1


async def test_implements_ambiguous_name_fans_out(graph_client: GraphClient):
    """Ambiguous names fan out to all same-project matches; other projects get no edge."""
    await graph_client.ensure_schema()

    project = "impl_fan"
    for fp in ("src/log_a.py", "src/log_b.py"):
        iface = _make_entity(project, "Logger", fp, label=NodeLabel.TYPE_DEF, kind="interface", content_hash=f"i_{fp}")
        await graph_client.upsert_file_entities(project, fp, [iface], [])

    # Same-named TypeDef in a DIFFERENT project must not receive an edge
    other = "impl_fan_other"
    other_iface = _make_entity(
        other, "Logger", "src/log.py", label=NodeLabel.TYPE_DEF, kind="interface", content_hash="oi"
    )
    await graph_client.upsert_file_entities(other, "src/log.py", [other_iface], [])

    impl = _make_entity(project, "FileLogger", "src/svc.py", label=NodeLabel.TYPE_DEF, kind="class", content_hash="c")
    rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:src.svc.FileLogger",
            rel_type=RelType.IMPLEMENTS,
            to_name="Logger",
        ),
    ]
    await graph_client.upsert_file_entities(project, "src/svc.py", [impl], rels)

    assert await _count_implements(graph_client, project, "FileLogger", "Logger") == 2

    cross = await graph_client.execute(
        f"MATCH (:{NodeLabel.TYPE_DEF} {{project_name: $p}})-[:{RelType.IMPLEMENTS}]->"
        f"(b:{NodeLabel.TYPE_DEF} {{project_name: $o}}) RETURN count(*) AS cnt",
        {"p": project, "o": other},
    )
    assert cross[0]["cnt"] == 0


async def test_implements_unresolved_name_no_edge_no_error(graph_client: GraphClient):
    """Behavior pin: unresolvable interface names (external/stdlib) silently create no edge."""
    await graph_client.ensure_schema()

    project = "impl_unres"
    impl = _make_entity(project, "Task", "src/task.py", label=NodeLabel.TYPE_DEF, kind="class", content_hash="t")
    rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:src.task.Task",
            rel_type=RelType.IMPLEMENTS,
            to_name="Runnable",
        ),
    ]
    await graph_client.upsert_file_entities(project, "src/task.py", [impl], rels)

    records = await graph_client.execute(f"MATCH ()-[r:{RelType.IMPLEMENTS}]->() RETURN count(r) AS cnt")
    assert records[0]["cnt"] == 0


async def test_implements_uid_shape_detector_path_unbroken(graph_client: GraphClient):
    """uid-shaped IMPLEMENTS (detector-emitted, Callable→Callable) still flows through the uid path."""
    await graph_client.ensure_schema()

    project = "impl_uid"
    base = ParsedEntity(
        name="save",
        qualified_name=f"{project}:src.base.Base.save",
        label=NodeLabel.CALLABLE,
        kind="method",
        line_start=2,
        line_end=4,
        file_path="src/base.py",
        content_hash="b",
    )
    await graph_client.upsert_file_entities(project, "src/base.py", [base], [])

    child = ParsedEntity(
        name="save",
        qualified_name=f"{project}:src.app.Child.save",
        label=NodeLabel.CALLABLE,
        kind="method",
        line_start=2,
        line_end=4,
        file_path="src/app.py",
        content_hash="c",
    )
    rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:src.app.Child.save",
            rel_type=RelType.IMPLEMENTS,
            to_name=f"{project}:src.base.Base.save",
        ),
    ]
    await graph_client.upsert_file_entities(project, "src/app.py", [child], rels)

    records = await graph_client.execute(
        f"MATCH (a:{NodeLabel.CALLABLE} {{uid: $c}})-[:{RelType.IMPLEMENTS}]->(b:{NodeLabel.CALLABLE} {{uid: $b}}) "
        "RETURN count(*) AS cnt",
        {"c": f"{project}:src.app.Child.save", "b": f"{project}:src.base.Base.save"},
    )
    assert records[0]["cnt"] == 1


async def _parse_upsert_count_implements(
    graph_client: GraphClient,
    project: str,
    path: str,
    source: bytes,
    impl_name: str,
    iface_name: str,
) -> int:
    """Parse a real source file, upsert it, and count the resulting IMPLEMENTS edges."""
    parsed = parse_file(path, source, project)
    assert parsed is not None
    await graph_client.upsert_file_entities(project, path, parsed.entities, parsed.relationships)
    return await _count_implements(graph_client, project, impl_name, iface_name)


async def test_implements_e2e_typescript(graph_client: GraphClient):
    pytest.importorskip("tree_sitter_typescript")
    await graph_client.ensure_schema()
    cnt = await _parse_upsert_count_implements(
        graph_client,
        "impl_e2e_ts",
        "src/logger.ts",
        b"interface Logger { log(msg: string): void }\nclass FileLogger implements Logger { log(msg: string) {} }\n",
        "FileLogger",
        "Logger",
    )
    assert cnt == 1


async def test_implements_e2e_java(graph_client: GraphClient):
    pytest.importorskip("tree_sitter_java")
    await graph_client.ensure_schema()
    cnt = await _parse_upsert_count_implements(
        graph_client,
        "impl_e2e_java",
        "src/Example.java",
        b"interface PaymentHandler {}\npublic class OrderService implements PaymentHandler {}\n",
        "OrderService",
        "PaymentHandler",
    )
    assert cnt == 1


async def test_implements_e2e_csharp(graph_client: GraphClient):
    pytest.importorskip("tree_sitter_c_sharp")
    await graph_client.ensure_schema()
    cnt = await _parse_upsert_count_implements(
        graph_client,
        "impl_e2e_cs",
        "src/Example.cs",
        b"public interface IStore {}\npublic class Store : IStore {}\n",
        "Store",
        "IStore",
    )
    assert cnt == 1


async def test_implements_e2e_php(graph_client: GraphClient):
    pytest.importorskip("tree_sitter_php")
    await graph_client.ensure_schema()
    cnt = await _parse_upsert_count_implements(
        graph_client,
        "impl_e2e_php",
        "src/example.php",
        b"<?php\ninterface Cacheable {}\nclass User implements Cacheable {}\n",
        "User",
        "Cacheable",
    )
    assert cnt == 1


async def test_implements_e2e_rust(graph_client: GraphClient):
    pytest.importorskip("tree_sitter_rust")
    await graph_client.ensure_schema()
    cnt = await _parse_upsert_count_implements(
        graph_client,
        "impl_e2e_rust",
        "src/example.rs",
        b"struct Foo;\ntrait Bar {}\nimpl Bar for Foo {}\n",
        "Foo",
        "Bar",
    )
    assert cnt == 1


# ---------------------------------------------------------------------------
# Member DEFINES resolution (cross-file parent types)
# ---------------------------------------------------------------------------


def _module_entity(project: str, module_qn: str, fp: str) -> ParsedEntity:
    """Helper: a Module entity for member-DEFINES tests."""
    return ParsedEntity(
        name=module_qn.rsplit(".", 1)[-1],
        qualified_name=f"{project}:{module_qn}",
        label=NodeLabel.MODULE,
        kind="module",
        line_start=1,
        line_end=20,
        file_path=fp,
        content_hash=f"mod_{module_qn}",
    )


def _typedef_entity(project: str, module_qn: str, name: str, fp: str) -> ParsedEntity:
    """Helper: a TypeDef entity for member-DEFINES tests."""
    return ParsedEntity(
        name=name,
        qualified_name=f"{project}:{module_qn}.{name}",
        label=NodeLabel.TYPE_DEF,
        kind="struct",
        line_start=2,
        line_end=5,
        file_path=fp,
        content_hash=f"td_{module_qn}.{name}",
    )


def _method_entity(project: str, qualified_name: str, name: str, fp: str) -> ParsedEntity:
    """Helper: a method Callable entity for member-DEFINES tests."""
    return ParsedEntity(
        name=name,
        qualified_name=qualified_name,
        label=NodeLabel.CALLABLE,
        kind="method",
        line_start=3,
        line_end=8,
        file_path=fp,
        content_hash=f"c_{qualified_name}",
    )


async def test_resolve_member_defines_cross_file(graph_client: GraphClient):
    """A Go-style method attaches to its receiver TypeDef declared in another same-package file."""
    await graph_client.ensure_schema()

    project = "member_x"

    server_entities = [
        _module_entity(project, "pkg.server", "pkg/server.go"),
        _typedef_entity(project, "pkg.server", "Server", "pkg/server.go"),
    ]
    await graph_client.upsert_file_entities(project, "pkg/server.go", server_entities, [])

    member_uid = f"{project}:pkg.routes.Server.Routes"
    routes_entities = [
        _module_entity(project, "pkg.routes", "pkg/routes.go"),
        _method_entity(project, member_uid, "Routes", "pkg/routes.go"),
    ]
    await graph_client.upsert_file_entities(project, "pkg/routes.go", routes_entities, [])

    member_rel = ParsedRelationship(
        from_qualified_name=f"{project}:pkg.routes",
        rel_type=RelType.DEFINES,
        to_name=member_uid,
        properties={"parent_type_name": "Server", "parent_scope": "package"},
    )
    await graph_client.resolve_member_defines(project, [member_rel])

    type_edges = await graph_client.execute(
        f"MATCH (t:{NodeLabel.TYPE_DEF} {{uid: $t}})-[:{RelType.DEFINES}]->(c {{uid: $m}}) RETURN count(*) AS cnt",
        {"t": f"{project}:pkg.server.Server", "m": member_uid},
    )
    assert type_edges[0]["cnt"] == 1

    module_edges = await graph_client.execute(
        f"MATCH (m:{NodeLabel.MODULE})-[:{RelType.DEFINES}]->(c {{uid: $m}}) RETURN count(*) AS cnt",
        {"m": member_uid},
    )
    assert module_edges[0]["cnt"] == 0


async def test_resolve_member_defines_project_wide_unique(graph_client: GraphClient):
    """C++ header/impl split: a project-wide unique TypeDef wins when no parent_scope restricts it."""
    await graph_client.ensure_schema()

    project = "member_cpp"

    header_entities = [
        _module_entity(project, "include.widget", "include/widget.hpp"),
        _typedef_entity(project, "include.widget", "Widget", "include/widget.hpp"),
    ]
    await graph_client.upsert_file_entities(project, "include/widget.hpp", header_entities, [])

    member_uid = f"{project}:src.widget.Widget.draw"
    impl_entities = [
        _module_entity(project, "src.widget", "src/widget.cpp"),
        _method_entity(project, member_uid, "draw", "src/widget.cpp"),
    ]
    await graph_client.upsert_file_entities(project, "src/widget.cpp", impl_entities, [])

    member_rel = ParsedRelationship(
        from_qualified_name=f"{project}:src.widget",
        rel_type=RelType.DEFINES,
        to_name=member_uid,
        properties={"parent_type_name": "Widget"},
    )
    await graph_client.resolve_member_defines(project, [member_rel])

    type_edges = await graph_client.execute(
        f"MATCH (t:{NodeLabel.TYPE_DEF} {{uid: $t}})-[:{RelType.DEFINES}]->(c {{uid: $m}}) RETURN count(*) AS cnt",
        {"t": f"{project}:include.widget.Widget", "m": member_uid},
    )
    assert type_edges[0]["cnt"] == 1


async def test_resolve_member_defines_ambiguous_falls_back_to_module(graph_client: GraphClient):
    """Ambiguous parent names produce a Module fallback edge, never a guessed type edge."""
    await graph_client.ensure_schema()

    project = "member_amb"

    # Two same-named 'Server' TypeDefs in different directories + a unique 'Router' in a/
    a_entities = [
        _module_entity(project, "a.x", "a/x.cpp"),
        _typedef_entity(project, "a.x", "Server", "a/x.cpp"),
        _typedef_entity(project, "a.x", "Router", "a/x.cpp"),
    ]
    await graph_client.upsert_file_entities(project, "a/x.cpp", a_entities, [])
    b_entities = [
        _module_entity(project, "b.y", "b/y.cpp"),
        _typedef_entity(project, "b.y", "Server", "b/y.cpp"),
    ]
    await graph_client.upsert_file_entities(project, "b/y.cpp", b_entities, [])

    member_uid = f"{project}:c.z.Server.draw"
    member2_uid = f"{project}:c.z.Router.route"
    c_entities = [
        _module_entity(project, "c.z", "c/z.cpp"),
        _method_entity(project, member_uid, "draw", "c/z.cpp"),
        _method_entity(project, member2_uid, "route", "c/z.cpp"),
    ]
    await graph_client.upsert_file_entities(project, "c/z.cpp", c_entities, [])

    rels = [
        # Ambiguous project-wide (two Servers) — must fall back to the module
        ParsedRelationship(
            from_qualified_name=f"{project}:c.z",
            rel_type=RelType.DEFINES,
            to_name=member_uid,
            properties={"parent_type_name": "Server"},
        ),
        # Unique project-wide but parent_scope='package' forbids cross-directory guessing
        ParsedRelationship(
            from_qualified_name=f"{project}:c.z",
            rel_type=RelType.DEFINES,
            to_name=member2_uid,
            properties={"parent_type_name": "Router", "parent_scope": "package"},
        ),
    ]
    await graph_client.resolve_member_defines(project, rels)

    for uid in (member_uid, member2_uid):
        type_edges = await graph_client.execute(
            f"MATCH (:{NodeLabel.TYPE_DEF})-[:{RelType.DEFINES}]->(c {{uid: $m}}) RETURN count(*) AS cnt",
            {"m": uid},
        )
        assert type_edges[0]["cnt"] == 0

        module_edges = await graph_client.execute(
            f"MATCH (m:{NodeLabel.MODULE} {{uid: $mod}})-[:{RelType.DEFINES}]->(c {{uid: $m}}) RETURN count(*) AS cnt",
            {"mod": f"{project}:c.z", "m": uid},
        )
        assert module_edges[0]["cnt"] == 1


async def test_resolve_member_defines_same_dir_wins(graph_client: GraphClient):
    """The same-directory rung beats project-wide ambiguity (Go package rule)."""
    await graph_client.ensure_schema()

    project = "member_dir"

    pkg_entities = [
        _module_entity(project, "pkg.a", "pkg/a.go"),
        _typedef_entity(project, "pkg.a", "Server", "pkg/a.go"),
    ]
    await graph_client.upsert_file_entities(project, "pkg/a.go", pkg_entities, [])
    other_entities = [
        _module_entity(project, "other.b", "other/b.go"),
        _typedef_entity(project, "other.b", "Server", "other/b.go"),
    ]
    await graph_client.upsert_file_entities(project, "other/b.go", other_entities, [])

    member_uid = f"{project}:pkg.c.Server.Handle"
    c_entities = [
        _module_entity(project, "pkg.c", "pkg/c.go"),
        _method_entity(project, member_uid, "Handle", "pkg/c.go"),
    ]
    await graph_client.upsert_file_entities(project, "pkg/c.go", c_entities, [])

    member_rel = ParsedRelationship(
        from_qualified_name=f"{project}:pkg.c",
        rel_type=RelType.DEFINES,
        to_name=member_uid,
        properties={"parent_type_name": "Server", "parent_scope": "package"},
    )
    await graph_client.resolve_member_defines(project, [member_rel])

    same_dir = await graph_client.execute(
        f"MATCH (t:{NodeLabel.TYPE_DEF} {{uid: $t}})-[:{RelType.DEFINES}]->(c {{uid: $m}}) RETURN count(*) AS cnt",
        {"t": f"{project}:pkg.a.Server", "m": member_uid},
    )
    assert same_dir[0]["cnt"] == 1

    other_dir = await graph_client.execute(
        f"MATCH (t:{NodeLabel.TYPE_DEF} {{uid: $t}})-[:{RelType.DEFINES}]->(c {{uid: $m}}) RETURN count(*) AS cnt",
        {"t": f"{project}:other.b.Server", "m": member_uid},
    )
    assert other_dir[0]["cnt"] == 0

    module_edges = await graph_client.execute(
        f"MATCH (m:{NodeLabel.MODULE})-[:{RelType.DEFINES}]->(c {{uid: $m}}) RETURN count(*) AS cnt",
        {"m": member_uid},
    )
    assert module_edges[0]["cnt"] == 0


async def test_member_defines_survives_parent_file_reupsert(graph_client: GraphClient):
    """Re-upserting the parent type's file must not destroy resolved cross-file member edges."""
    await graph_client.ensure_schema()

    project = "member_reup"

    server_td = _typedef_entity(project, "pkg.server", "Server", "pkg/server.go")
    server_entities = [_module_entity(project, "pkg.server", "pkg/server.go"), server_td]
    server_rels = [
        ParsedRelationship(
            from_qualified_name=f"{project}:pkg.server",
            rel_type=RelType.DEFINES,
            to_name=f"{project}:pkg.server.Server",
        )
    ]
    await graph_client.upsert_file_entities(project, "pkg/server.go", server_entities, server_rels)

    member_uid = f"{project}:pkg.routes.Server.Routes"
    routes_entities = [
        _module_entity(project, "pkg.routes", "pkg/routes.go"),
        _method_entity(project, member_uid, "Routes", "pkg/routes.go"),
    ]
    await graph_client.upsert_file_entities(project, "pkg/routes.go", routes_entities, [])

    member_rel = ParsedRelationship(
        from_qualified_name=f"{project}:pkg.routes",
        rel_type=RelType.DEFINES,
        to_name=member_uid,
        properties={"parent_type_name": "Server", "parent_scope": "package"},
    )
    await graph_client.resolve_member_defines(project, [member_rel])

    # Edit server.go (content change) and re-upsert: the file-scoped rel delete
    # must preserve the cross-file member edge it did not create.
    server_entities_v2 = [
        _module_entity(project, "pkg.server", "pkg/server.go"),
        replace(server_td, content_hash="td_v2"),
    ]
    await graph_client.upsert_file_entities(project, "pkg/server.go", server_entities_v2, server_rels)

    type_edges = await graph_client.execute(
        f"MATCH (t:{NodeLabel.TYPE_DEF} {{uid: $t}})-[:{RelType.DEFINES}]->(c {{uid: $m}}) RETURN count(*) AS cnt",
        {"t": f"{project}:pkg.server.Server", "m": member_uid},
    )
    assert type_edges[0]["cnt"] == 1

    # The type file's own same-file DEFINES edge is recreated normally
    own_edges = await graph_client.execute(
        f"MATCH (m:{NodeLabel.MODULE} {{uid: $mod}})-[:{RelType.DEFINES}]->(t {{uid: $t}}) RETURN count(*) AS cnt",
        {"mod": f"{project}:pkg.server", "t": f"{project}:pkg.server.Server"},
    )
    assert own_edges[0]["cnt"] == 1


async def test_resolve_member_defines_rerun_replaces_stale_parent(graph_client: GraphClient):
    """Re-resolution is authoritative: a previously-resolved type edge is dropped on fallback."""
    await graph_client.ensure_schema()

    project = "member_stale"

    a_entities = [_module_entity(project, "a.x", "a/x.cpp"), _typedef_entity(project, "a.x", "Widget", "a/x.cpp")]
    await graph_client.upsert_file_entities(project, "a/x.cpp", a_entities, [])

    member_uid = f"{project}:c.z.Widget.draw"
    c_entities = [_module_entity(project, "c.z", "c/z.cpp"), _method_entity(project, member_uid, "draw", "c/z.cpp")]
    await graph_client.upsert_file_entities(project, "c/z.cpp", c_entities, [])

    member_rel = ParsedRelationship(
        from_qualified_name=f"{project}:c.z",
        rel_type=RelType.DEFINES,
        to_name=member_uid,
        properties={"parent_type_name": "Widget"},
    )
    # First pass: project-wide unique — resolves to a/x.cpp's Widget
    await graph_client.resolve_member_defines(project, [member_rel])

    # A second same-named TypeDef appears — re-resolution becomes ambiguous
    b_entities = [_module_entity(project, "b.y", "b/y.cpp"), _typedef_entity(project, "b.y", "Widget", "b/y.cpp")]
    await graph_client.upsert_file_entities(project, "b/y.cpp", b_entities, [])
    await graph_client.resolve_member_defines(project, [member_rel])

    type_edges = await graph_client.execute(
        f"MATCH (:{NodeLabel.TYPE_DEF})-[:{RelType.DEFINES}]->(c {{uid: $m}}) RETURN count(*) AS cnt",
        {"m": member_uid},
    )
    assert type_edges[0]["cnt"] == 0

    module_edges = await graph_client.execute(
        f"MATCH (m:{NodeLabel.MODULE} {{uid: $mod}})-[:{RelType.DEFINES}]->(c {{uid: $m}}) RETURN count(*) AS cnt",
        {"mod": f"{project}:c.z", "m": member_uid},
    )
    assert module_edges[0]["cnt"] == 1


# ---------------------------------------------------------------------------
# Query timeout
# ---------------------------------------------------------------------------


async def test_execute_raises_query_timeout(graph_client: GraphClient):
    """Setting an impossibly short timeout triggers QueryTimeoutError."""
    graph_client._query_timeout_s = 0.0001
    with pytest.raises(QueryTimeoutError):
        await graph_client.execute("RETURN 1 AS n")


async def test_execute_write_raises_query_timeout(graph_client: GraphClient):
    """Setting an impossibly short write timeout triggers QueryTimeoutError."""
    original = graph_client._write_timeout_s
    graph_client._write_timeout_s = 0.0001
    try:
        with pytest.raises(QueryTimeoutError):
            await graph_client.execute_write("CREATE (n:_Tmp {x: 1})")
    finally:
        graph_client._write_timeout_s = original


# ---------------------------------------------------------------------------
# Overload uids (S6)
# ---------------------------------------------------------------------------


async def test_upsert_java_overloads_creates_distinct_nodes(graph_client: GraphClient):
    """S6 e2e: Java overloads survive upsert as distinct Callable nodes.

    Before the parser fix both overloads shared one qualified_name and
    ``_classify_file``'s qn-keyed dicts collapsed them to a single node.
    """
    pytest.importorskip("tree_sitter_java")
    await graph_client.ensure_schema()

    project = "test_ovl_java"
    path = "src/Example.java"
    source = (
        b"class A {\n"
        b"    void process(Order o) { helperOne(); }\n"
        b"    void process(java.util.List<Order> os, String... rest) { helperTwo(); }\n"
        b"}\n"
    )
    parsed = parse_file(path, source, project)
    assert parsed is not None
    await graph_client.upsert_file_entities(project, path, parsed.entities, parsed.relationships)

    records = await graph_client.execute(
        f"MATCH (c:{NodeLabel.CALLABLE} {{project_name: $p, name: 'process'}}) RETURN c.uid AS uid",
        {"p": project},
    )
    uids = {r["uid"] for r in records}
    assert len(records) == 2
    assert uids == {
        f"{project}:Example.A.process(Order)",
        f"{project}:Example.A.process(List<Order>,String[])",
    }

    defines = await graph_client.execute(
        f"MATCH (:{NodeLabel.TYPE_DEF} {{project_name: $p, name: 'A'}})-[:{RelType.DEFINES}]->"
        f"(c:{NodeLabel.CALLABLE}) RETURN c.uid AS uid",
        {"p": project},
    )
    assert uids <= {r["uid"] for r in defines}


# ---------------------------------------------------------------------------
# Deletion — foreign inbound edges and Package file_hash staleness
# ---------------------------------------------------------------------------


async def test_delete_file_entities_clears_package_file_hash(graph_client: GraphClient):
    """Deleting a file's entities also clears its surviving Package node's
    stale file_hash (client.py:638) — otherwise an identically re-created
    __init__.py is silently skipped forever by the AST consumer's hash gate.
    """
    await graph_client.ensure_schema()
    project = "pkgdel1"
    fp = "src/pkg/__init__.py"

    # Package node for __init__.py (created by orchestrator/parser), carrying
    # a stored file_hash from the last successful parse.
    await graph_client.merge_package_node(project, "src.pkg", "pkg", fp)
    await graph_client.execute_write(
        f"MATCH (n:{NodeLabel.PACKAGE} {{project_name: $p, file_path: $f}}) SET n.file_hash = 'stale_hash'",
        {"p": project, "f": fp},
    )

    # A Callable defined in the same __init__.py.
    entity = _make_entity(project, "helper", fp, content_hash="c1")
    await graph_client.upsert_file_entities(project, fp, [entity], [])

    deleted = await graph_client.delete_file_entities(project, fp)
    assert "src.pkg.__init__.helper" in deleted

    # The Package node itself must survive (the directory hierarchy still
    # needs it) but its stale file_hash must be cleared.
    pkg = await graph_client.execute(
        f"MATCH (n:{NodeLabel.PACKAGE} {{project_name: $p, file_path: $f}}) RETURN n.file_hash AS fh",
        {"p": project, "f": fp},
    )
    assert len(pkg) == 1
    assert pkg[0]["fh"] is None

    # The hash gate must see this file as unseen, so a re-created __init__.py
    # with identical content is reprocessed instead of silently skipped.
    hashes = await graph_client.get_batch_file_hashes(project, [fp])
    assert hashes[fp] is None


async def test_recreate_relationships_package_defines_no_duplicate(graph_client: GraphClient):
    """Re-indexing __init__.py must not accumulate duplicate DEFINES edges
    from its Package-labeled module entity (client.py:1992) — the relationship
    delete phase excludes Package-sourced rels, so the create phase must MERGE
    rather than CREATE to stay idempotent across re-indexes.
    """
    await graph_client.ensure_schema()
    project = "pkgdup1"
    fp = "src/pkg/__init__.py"
    pkg_uid = f"{project}:src.pkg"
    helper_uid = f"{project}:src.pkg.__init__.helper"

    await graph_client.merge_package_node(project, "src.pkg", "pkg", fp)

    rels = [
        ParsedRelationship(
            from_qualified_name=pkg_uid,
            rel_type=RelType.DEFINES,
            to_name=helper_uid,
        )
    ]

    async def _defines_count() -> int:
        records = await graph_client.execute(
            f"MATCH (:{NodeLabel.PACKAGE} {{uid: $pkg}})-[:{RelType.DEFINES}]->"
            f"(:{NodeLabel.CALLABLE} {{uid: $helper}}) RETURN count(*) AS cnt",
            {"pkg": pkg_uid, "helper": helper_uid},
        )
        return records[0]["cnt"]

    helper_v1 = _make_entity(project, "helper", fp, content_hash="v1")
    await graph_client.upsert_file_entities(project, fp, [helper_v1], rels)
    assert await _defines_count() == 1

    # Re-index with helper's content changed — forces relationship recreation
    # (the fast no-op path only triggers when nothing in the file changed).
    helper_v2 = _make_entity(project, "helper", fp, content_hash="v2")
    await graph_client.upsert_file_entities(project, fp, [helper_v2], rels)
    assert await _defines_count() == 1


async def test_batch_delete_preserves_foreign_inbound_edges(graph_client: GraphClient):
    """Deleting an entity that another, untouched file still references must
    not destroy that cross-file edge (client.py:1967) — if the entity
    reappears later (e.g. a brief comment-out/comment-back-in edit), the
    referencing file is never re-parsed and could never recreate it.
    """
    await graph_client.ensure_schema()
    project = "delfk1"

    # File A defines func_b — the entity that will be removed then restored.
    func_b = _make_entity(project, "func_b", "src/mod_a.py", content_hash="")

    await graph_client.upsert_file_entities(project, "src/mod_a.py", [func_b], [])

    # File B defines func_c, which OVERRIDES func_b — a uid-based cross-file
    # rel that is never re-derived by re-parsing file A alone.
    func_c = _make_entity(project, "func_c", "src/mod_b.py", content_hash="")
    rel = ParsedRelationship(
        from_qualified_name=func_c.qualified_name,
        rel_type=RelType.OVERRIDES,
        to_name=func_b.qualified_name,
    )
    await graph_client.upsert_file_entities(project, "src/mod_b.py", [func_c], [rel])

    async def _edge_count() -> int:
        records = await graph_client.execute(
            f"MATCH (:{NodeLabel.CALLABLE} {{uid: $c}})-[:{RelType.OVERRIDES}]->"
            f"(:{NodeLabel.CALLABLE} {{uid: $b}}) RETURN count(*) AS cnt",
            {"c": func_c.qualified_name, "b": func_b.qualified_name},
        )
        return records[0]["cnt"]

    assert await _edge_count() == 1

    # File A is re-parsed without func_b (e.g. temporarily commented out).
    # File B is never touched, so nothing could recreate the edge if it were
    # destroyed here.
    await graph_client.upsert_file_entities(project, "src/mod_a.py", [], [])
    assert await _edge_count() == 1

    # func_b reappears with identical content.
    await graph_client.upsert_file_entities(project, "src/mod_a.py", [func_b], [])
    assert await _edge_count() == 1


async def test_batch_delete_removes_member_with_cross_file_defines_edge(graph_client: GraphClient):
    """A genuinely deleted cross-file member (S5) must still be fully removed.

    Unlike the foreign OVERRIDES case above, a member's own incoming DEFINES
    edge from its (foreign-file) parent TypeDef is created by
    resolve_member_defines keyed off the MEMBER's own file — so it is always
    re-resolved whenever the member reappears, regardless of whether the
    parent's file is touched. Preserving the node here (as if it were an
    OVERRIDES-style edge) would make a deleted member an undeletable zombie.
    """
    await graph_client.ensure_schema()
    project = "member_del"

    server_entities = [
        _module_entity(project, "pkg.server", "pkg/server.go"),
        _typedef_entity(project, "pkg.server", "Server", "pkg/server.go"),
    ]
    await graph_client.upsert_file_entities(project, "pkg/server.go", server_entities, [])

    member_uid = f"{project}:pkg.routes.Server.Routes"
    routes_entities = [
        _module_entity(project, "pkg.routes", "pkg/routes.go"),
        _method_entity(project, member_uid, "Routes", "pkg/routes.go"),
    ]
    await graph_client.upsert_file_entities(project, "pkg/routes.go", routes_entities, [])

    member_rel = ParsedRelationship(
        from_qualified_name=f"{project}:pkg.routes",
        rel_type=RelType.DEFINES,
        to_name=member_uid,
        properties={"parent_type_name": "Server", "parent_scope": "package"},
    )
    await graph_client.resolve_member_defines(project, [member_rel])

    async def _node_count() -> int:
        records = await graph_client.execute(
            f"MATCH (c:{NodeLabel.CALLABLE} {{uid: $m}}) RETURN count(*) AS cnt",
            {"m": member_uid},
        )
        return records[0]["cnt"]

    assert await _node_count() == 1

    # Routes is removed from pkg/routes.go (routes.go itself is reprocessed —
    # the module entity remains, but the method is gone).
    await graph_client.upsert_file_entities(
        project, "pkg/routes.go", [_module_entity(project, "pkg.routes", "pkg/routes.go")], []
    )

    assert await _node_count() == 0


async def test_batch_delete_same_call_mutual_referrer_both_removed(graph_client: GraphClient):
    """A same-batch mutual reference must not needlessly preserve either side.

    _batch_delete_entities snapshots "foreign inbound edge" once before any
    deletes. If func_a's only foreign referrer is func_b, and func_b is ALSO
    being deleted in this same call (both files reprocessed to empty in one
    upsert_batch_entities call), func_b's edge to func_a is gone by the time
    the call completes either way (DETACH DELETE of func_b, or func_b's own
    outgoing edges being stripped) -- so func_a must be fully removed too,
    not kept around as an edge-stripped zombie node that would still surface
    in vector/text search and direct uid lookups after both source files are gone.
    """
    await graph_client.ensure_schema()
    project = "delfk2"

    func_a = _make_entity(project, "func_a", "src/mod_a.py", content_hash="")
    await graph_client.upsert_file_entities(project, "src/mod_a.py", [func_a], [])

    func_b = _make_entity(project, "func_b", "src/mod_b.py", content_hash="")
    rel = ParsedRelationship(
        from_qualified_name=func_b.qualified_name,
        rel_type=RelType.OVERRIDES,
        to_name=func_a.qualified_name,
    )
    await graph_client.upsert_file_entities(project, "src/mod_b.py", [func_b], [rel])

    async def _node_count(uid: str) -> int:
        records = await graph_client.execute("MATCH (n {uid: $uid}) RETURN count(*) AS cnt", {"uid": uid})
        return records[0]["cnt"]

    assert await _node_count(func_a.qualified_name) == 1
    assert await _node_count(func_b.qualified_name) == 1

    # Both files re-processed to empty in ONE batched call: func_a's only
    # foreign referrer (func_b) is being deleted in this SAME call.
    await graph_client.upsert_batch_entities(
        project,
        {
            "src/mod_a.py": ([], []),
            "src/mod_b.py": ([], []),
        },
    )

    assert await _node_count(func_a.qualified_name) == 0, "func_a preserved as a zombie node"
    assert await _node_count(func_b.qualified_name) == 0
