"""Integration tests for GraphClient and schema application.

Requires a running Memgraph instance (docker compose up -d memgraph).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from code_atlas.parser import ParsedEntity, ParsedRelationship
from code_atlas.schema import SCHEMA_VERSION, NodeLabel, RelType

if TYPE_CHECKING:
    from code_atlas.graph import GraphClient


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

    # Write a 768-dim embedding (simple pattern: 1.0 at index 0, rest 0)
    dim = 768
    vector = [1.0] + [0.0] * (dim - 1)
    await graph_client.write_embeddings([("proj.my_func", vector)])

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

    dim = 768
    vector_a = [1.0] + [0.0] * (dim - 1)
    vector_b = [0.0, 1.0] + [0.0] * (dim - 2)

    for project, vec, name in [("alpha", vector_a, "func_a"), ("beta", vector_b, "func_b")]:
        await graph_client.execute_write(
            f"CREATE (:{NodeLabel.CALLABLE} {{"
            "  uid: $uid, project_name: $project, name: $name,"
            "  qualified_name: $qn, kind: 'function', file_path: 'f.py',"
            "  content_hash: $hash"
            "})",
            {
                "uid": f"{project}:{name}",
                "project": project,
                "name": name,
                "qn": name,
                "hash": f"h_{project}",
            },
        )
        await graph_client.write_embeddings([(name, vec)])

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

    dim = 768
    vector_a = [1.0] + [0.0] * (dim - 1)
    vector_b = [0.0, 1.0] + [0.0] * (dim - 2)

    for name, vec in [("near", vector_a), ("far", vector_b)]:
        await graph_client.execute_write(
            f"CREATE (:{NodeLabel.CALLABLE} {{"
            "  uid: $uid, project_name: 'thresh', name: $name,"
            "  qualified_name: $qn, kind: 'function', file_path: 'f.py',"
            "  content_hash: $hash"
            "})",
            {"uid": f"thresh:{name}", "name": name, "qn": name, "hash": f"h_{name}"},
        )
        await graph_client.write_embeddings([(name, vec)])

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

    await graph_client.write_embed_hashes([("proj.func", "deadbeef")])

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

    results = await graph_client.read_entity_texts(["proj.func"])
    assert len(results) == 1
    assert results[0]["embed_hash"] == "abc"
    assert results[0]["embedding"] == embedding


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
