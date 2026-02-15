"""Integration tests for GraphClient and schema application.

Requires a running Memgraph instance (docker compose up -d memgraph).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from code_atlas.graph.client import QueryTimeoutError
from code_atlas.parsing.ast import ParsedEntity, ParsedRelationship
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

    dim = graph_client._dimension
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

    dim = graph_client._dimension
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
