"""Unit tests for SqliteGraphClient — the in-process fallback graph engine (no infrastructure needed)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import aiosqlite

from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.parsing.ast import ParsedEntity, ParsedRelationship
from code_atlas.parsing.detectors import PropertyEnrichment
from code_atlas.schema import GLOBAL_PROJECT, SCHEMA_VERSION, NodeLabel, RelType, Visibility
from code_atlas.server.analysis import _analyze_communities

if TYPE_CHECKING:
    from pathlib import Path


def _entity(
    name: str,
    qualified_name: str,
    *,
    project: str = "proj",
    label: NodeLabel = NodeLabel.CALLABLE,
    kind: str | None = None,
    file_path: str = "mod.py",
    docstring: str | None = None,
    signature: str | None = None,
    visibility: str = Visibility.PUBLIC,
    content_hash: str = "h1",
) -> ParsedEntity:
    return ParsedEntity(
        name=name,
        qualified_name=f"{project}:{qualified_name}",
        label=label,
        kind=kind if kind is not None else ("function" if label == NodeLabel.CALLABLE else "class"),
        line_start=1,
        line_end=5,
        file_path=file_path,
        docstring=docstring,
        signature=signature,
        visibility=visibility,
        content_hash=content_hash,
    )


async def _insert_edge(
    client: SqliteGraphClient, from_uid: str, to_uid: str, rel_type: str, props: dict[str, Any] | None = None
) -> None:
    """Directly insert an edge row — used for CALLS/IMPORTS/CO_CHANGES_WITH edges,
    which ``upsert_file_entities`` doesn't create (those are wired by dedicated
    resolvers/CLI commands in production, not by parsing).
    """
    conn = await client._get_conn()
    await conn.execute(
        "INSERT INTO edges(from_uid, to_uid, rel_type, props_json) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(from_uid, to_uid, rel_type) DO UPDATE SET props_json = excluded.props_json",
        (from_uid, to_uid, rel_type, json.dumps(props or {})),
    )
    await conn.commit()


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------


class TestSchemaBootstrap:
    async def test_ensure_schema_fresh_db(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")

        assert await client.get_schema_version() is None
        await client.ensure_schema()

        assert await client.get_schema_version() == SCHEMA_VERSION
        await client.close()

    async def test_ensure_schema_idempotent(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.ensure_schema()

        assert await client.get_schema_version() == SCHEMA_VERSION
        await client.close()

    async def test_ping(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        assert await client.ping() is True
        await client.close()


# ---------------------------------------------------------------------------
# Upsert + get round-trip
# ---------------------------------------------------------------------------


class TestUpsertRoundTrip:
    async def test_upsert_then_graph_search_round_trip(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()

        entity = _entity("my_func", "mod.my_func", docstring="A test function.", signature="my_func()")
        result = await client.upsert_file_entities("proj", "mod.py", [entity], [])

        assert result.added == ["mod.my_func"]
        assert await client.count_entities("proj") == 1

        found = await client.graph_search("my_func", project="proj")
        assert len(found) == 1
        assert found[0]["node"]["uid"] == "proj:mod.my_func"
        assert found[0]["node"]["docstring"] == "A test function."
        assert found[0]["node"]["_labels"] == ["Callable"]
        await client.close()

    async def test_reupsert_unchanged_entity_reports_no_changes(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        entity = _entity("my_func", "mod.my_func")

        await client.upsert_file_entities("proj", "mod.py", [entity], [])
        result = await client.upsert_file_entities("proj", "mod.py", [entity], [])

        assert result.added == []
        assert result.modified == []
        assert result.unchanged == ["mod.my_func"]
        await client.close()

    async def test_delete_file_entities_removes_node(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        entity = _entity("my_func", "mod.my_func")
        await client.upsert_file_entities("proj", "mod.py", [entity], [])

        deleted = await client.delete_file_entities("proj", "mod.py")

        assert deleted == ["mod.my_func"]
        assert await client.count_entities("proj") == 0
        await client.close()

    async def test_upsert_creates_uid_routed_relationship(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()

        parent = _entity("MyClass", "mod.MyClass", label=NodeLabel.TYPE_DEF)
        child = _entity("method", "mod.MyClass.method")
        rel = ParsedRelationship(
            from_qualified_name="proj:mod.MyClass",
            rel_type=RelType.DEFINES,
            to_name="proj:mod.MyClass.method",
        )

        await client.upsert_file_entities("proj", "mod.py", [parent, child], [rel])

        lookup, _typedefs = await client.build_resolution_lookup("proj")
        assert lookup.caller_to_parent.get("proj:mod.MyClass.method") == "proj:mod.MyClass"
        await client.close()


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


class TestVectorSearch:
    async def test_vector_search_returns_nearest_neighbor_first(self, tmp_path: Path) -> None:
        dim = 4
        client = SqliteGraphClient(tmp_path / "graph.sqlite3", dimension=dim)
        await client.ensure_schema()

        close = _entity("close_func", "mod.close_func")
        far = _entity("far_func", "mod.far_func")
        await client.upsert_file_entities("proj", "mod.py", [close, far], [])

        await client.write_embeddings(
            [
                ("proj:mod.close_func", [1.0, 0.0, 0.0, 0.0]),
                ("proj:mod.far_func", [0.0, 0.0, 0.0, 1.0]),
            ]
        )

        results = await client.vector_search([0.9, 0.1, 0.0, 0.0], project="proj", limit=5)

        assert len(results) == 2
        assert results[0]["node"]["uid"] == "proj:mod.close_func"
        assert results[0]["similarity"] > results[1]["similarity"]
        await client.close()

    async def test_read_embed_hashes_reflects_written_embedding(self, tmp_path: Path) -> None:
        dim = 4
        client = SqliteGraphClient(tmp_path / "graph.sqlite3", dimension=dim)
        await client.ensure_schema()
        entity = _entity("my_func", "mod.my_func")
        await client.upsert_file_entities("proj", "mod.py", [entity], [])

        await client.write_embeddings_and_hashes([("proj:mod.my_func", [1.0, 0.0, 0.0, 0.0], "embedhash1")])

        hashes = await client.read_embed_hashes(["proj:mod.my_func"])
        assert hashes["proj:mod.my_func"] == ("embedhash1", True)
        await client.close()


# ---------------------------------------------------------------------------
# Text (BM25) search
# ---------------------------------------------------------------------------


class TestTextSearch:
    async def test_text_search_ranks_exact_match_above_unrelated(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()

        target = _entity("frobnicate", "mod.frobnicate", docstring="Frobnicate the given widget.")
        unrelated = _entity("compute_stats", "mod.compute_stats", docstring="Compute aggregate statistics.")
        await client.upsert_file_entities("proj", "mod.py", [target, unrelated], [])

        results = await client.text_search("frobnicate", project="proj", limit=5)

        assert len(results) >= 1
        assert results[0]["node"]["uid"] == "proj:mod.frobnicate"
        await client.close()

    async def test_text_search_no_match_returns_empty(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        entity = _entity("my_func", "mod.my_func", docstring="Nothing special.")
        await client.upsert_file_entities("proj", "mod.py", [entity], [])

        results = await client.text_search("zzz_nonexistent_term", project="proj")

        assert results == []
        await client.close()


# ---------------------------------------------------------------------------
# Communities guard
# ---------------------------------------------------------------------------


class TestCommunitiesGuard:
    async def test_communities_analysis_returns_unsupported_error(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()

        result = await _analyze_communities(client, "proj", "", 10)

        # Assert the contract, not the prose: the exact wording explains WHY the
        # backend cannot run it, and that reason has already changed once (it was
        # MAGE/Leiden; it is now the two raw Cypher reads the clustering feeds on).
        # Pinning the full string made a correct explanation update look like a
        # regression.
        assert result["analysis"] == "communities"
        assert "unsupported on the sqlite backend" in result["error"]
        await client.close()


# ---------------------------------------------------------------------------
# Constructor injection (dependency injection)
# ---------------------------------------------------------------------------


class TestConstructorInjection:
    async def test_injected_fake_connection_is_used_directly(self, tmp_path: Path) -> None:
        """A fake/mock connection passed at construction is returned as-is by
        ``_get_conn`` — no ``aiosqlite.connect``/PRAGMA/schema bootstrap runs on it.
        """
        fake_conn = AsyncMock()
        client = SqliteGraphClient(tmp_path / "unused.sqlite3", conn=fake_conn)

        conn = await client._get_conn()

        assert conn is fake_conn
        fake_conn.execute.assert_not_called()

    async def test_injected_real_connection_is_used_instead_of_opening_new_one(self, tmp_path: Path) -> None:
        """A real, already-open connection passed at construction is used as-is —
        settings-based construction (opening ``db_path`` itself) never runs.
        """
        conn = await aiosqlite.connect(tmp_path / "real.sqlite3")
        client = SqliteGraphClient(tmp_path / "never-opened.sqlite3", conn=conn)

        with patch("code_atlas.backends.sqlite_graph.aiosqlite.connect") as mock_connect:
            assert await client.ping() is True
            mock_connect.assert_not_called()

        await client.close()
        assert not (tmp_path / "never-opened.sqlite3").exists()

    async def test_no_injected_connection_falls_back_to_settings_based_construction(self, tmp_path: Path) -> None:
        """Default (no *conn* passed) behavior is unchanged — the client opens its own
        connection at *db_path* lazily on first use.
        """
        db_path = tmp_path / "graph.sqlite3"
        client = SqliteGraphClient(db_path)

        assert await client.ping() is True
        assert db_path.exists()
        await client.close()


# ---------------------------------------------------------------------------
# Analysis / diagram queries (server/analysis.py, via graph/protocol.py's
# GraphBackend) — SQL ports of GraphClient's analysis methods.
# ---------------------------------------------------------------------------


class TestNodeExists:
    async def test_true_for_existing_uid_false_otherwise(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("f", "mod.f")], [])

        assert await client.node_exists("proj:mod.f") is True
        assert await client.node_exists("proj:mod.missing") is False
        await client.close()


class TestTracePathBetween:
    async def test_found_path_reports_hops_with_confidence(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        a, b, c = _entity("a", "mod.a"), _entity("b", "mod.b"), _entity("c", "mod.c")
        await client.upsert_file_entities("proj", "mod.py", [a, b, c], [])
        await _insert_edge(
            client, "proj:mod.a", "proj:mod.b", "CALLS", {"confidence": "resolved", "strategy": "import"}
        )
        await _insert_edge(
            client, "proj:mod.b", "proj:mod.c", "CALLS", {"confidence": "resolved", "strategy": "import"}
        )

        result = await client.trace_path_between("proj:mod.a", "proj:mod.c", 6, ("CALLS",))

        assert result["from_exists"] is True
        assert result["to_exists"] is True
        assert result["found"] is True
        assert result["hop_count"] == 2
        assert result["hops"][0]["edge_type"] == "CALLS"
        assert result["hops"][0]["confidence"] == "resolved"
        assert result["hops"][1]["to"]["uid"] == "proj:mod.c"
        await client.close()

    async def test_missing_node_reports_exists_false(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("a", "mod.a")], [])

        result = await client.trace_path_between("proj:mod.a", "proj:mod.missing", 6, ("CALLS",))

        assert result["from_exists"] is True
        assert result["to_exists"] is False
        assert result["found"] is False
        await client.close()

    async def test_no_path_within_max_depth(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        a, b, c = _entity("a", "mod.a"), _entity("b", "mod.b"), _entity("c", "mod.c")
        await client.upsert_file_entities("proj", "mod.py", [a, b, c], [])
        await _insert_edge(client, "proj:mod.a", "proj:mod.b", "CALLS")
        await _insert_edge(client, "proj:mod.b", "proj:mod.c", "CALLS")

        result = await client.trace_path_between("proj:mod.a", "proj:mod.c", 1, ("CALLS",))

        assert result["found"] is False
        await client.close()


class TestComputeBlastRadius:
    async def test_callers_direction_flags_ambiguous_entries(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        target = _entity("target", "mod.target")
        resolved_caller = _entity("resolved_caller", "mod.resolved_caller")
        ambiguous_caller = _entity("ambiguous_caller", "mod.ambiguous_caller")
        await client.upsert_file_entities("proj", "mod.py", [target, resolved_caller, ambiguous_caller], [])
        await _insert_edge(client, "proj:mod.resolved_caller", "proj:mod.target", "CALLS", {"confidence": "resolved"})
        await _insert_edge(client, "proj:mod.ambiguous_caller", "proj:mod.target", "CALLS", {"confidence": "ambiguous"})

        # direction_kind "in" == callers (incoming edges, matching blast_radius's _BLAST_DIRECTIONS)
        results = await client.compute_blast_radius("proj:mod.target", "in", ("CALLS",), 3)

        by_uid = {r["uid"]: r for r in results}
        assert by_uid["proj:mod.resolved_caller"]["ambiguous_only"] is False
        assert by_uid["proj:mod.ambiguous_caller"]["ambiguous_only"] is True
        assert all(r["min_depth"] == 1 for r in results)
        await client.close()

    async def test_no_reachable_nodes_returns_empty(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("solo", "mod.solo")], [])

        results = await client.compute_blast_radius("proj:mod.solo", "out", ("CALLS",), 3)

        assert results == []
        await client.close()


class TestGetStructureOverview:
    async def test_counts_packages_largest_modules_and_external_deps(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        module = _entity("mod", "pkg.mod", label=NodeLabel.MODULE, file_path="pkg/mod.py")
        f1 = _entity("f1", "pkg.mod.f1", file_path="pkg/mod.py")
        f2 = _entity("f2", "pkg.mod.f2", file_path="pkg/mod.py")
        defines = [
            ParsedRelationship(from_qualified_name="proj:pkg.mod", rel_type=RelType.DEFINES, to_name="proj:pkg.mod.f1"),
            ParsedRelationship(from_qualified_name="proj:pkg.mod", rel_type=RelType.DEFINES, to_name="proj:pkg.mod.f2"),
        ]
        await client.upsert_file_entities("proj", "pkg/mod.py", [module, f1, f2], defines)
        await client.merge_package_node("proj", "pkg", "pkg", "pkg/")
        await client.create_contains_edge("proj:pkg", "proj:pkg.mod")
        await client.resolve_imports(
            "proj",
            [ParsedRelationship(from_qualified_name="proj:pkg.mod", rel_type=RelType.IMPORTS, to_name="requests")],
        )

        data = await client.get_structure_overview("proj", "", 20)

        label_counts = {r["label"] for r in data["counts"]}
        assert {"Module", "Callable", "Package", "ExternalPackage"} <= label_counts
        assert data["packages"] == [{"package": "pkg", "qn": "pkg", "modules": 1}]
        assert data["largest_modules"][0]["module"] == "mod"
        assert data["largest_modules"][0]["entities"] == 2
        assert data["external_deps"][0]["package"] == "requests"
        assert data["external_deps"][0]["imported_by"] == 1
        await client.close()

    async def test_path_scope_filters_to_matching_files(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        in_scope = _entity("a", "pkg.in_scope.a", file_path="pkg/in_scope/a.py")
        out_scope = _entity("b", "pkg.out_scope.b", file_path="pkg/out_scope/b.py")
        await client.upsert_file_entities("proj", "pkg/in_scope/a.py", [in_scope], [])
        await client.upsert_file_entities("proj", "pkg/out_scope/b.py", [out_scope], [])

        data = await client.get_structure_overview("proj", "pkg/in_scope", 20)

        assert sum(r["cnt"] for r in data["counts"]) == 1
        await client.close()


class TestGetCentralityData:
    async def test_hub_entity_and_leaf_entity(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        hub, caller, leaf = _entity("hub", "mod.hub"), _entity("caller", "mod.caller"), _entity("leaf", "mod.leaf")
        await client.upsert_file_entities("proj", "mod.py", [hub, caller, leaf], [])
        await _insert_edge(client, "proj:mod.caller", "proj:mod.hub", "CALLS")

        data = await client.get_centrality_data("proj", "", 20)

        assert "mod.hub" in {r["qn"] for r in data["hubs"]}
        leaf_qns = {r["qn"] for r in data["leaves"]}
        assert "mod.leaf" in leaf_qns
        assert "mod.hub" not in leaf_qns
        await client.close()


class TestGetModuleImportEdges:
    async def test_direct_module_edge(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        mod_a = _entity("a", "pkg.a", label=NodeLabel.MODULE, file_path="pkg/a.py")
        mod_b = _entity("b", "pkg.b", label=NodeLabel.MODULE, file_path="pkg/b.py")
        await client.upsert_file_entities("proj", "pkg/a.py", [mod_a], [])
        await client.upsert_file_entities("proj", "pkg/b.py", [mod_b], [])
        await _insert_edge(client, "proj:pkg.a", "proj:pkg.b", "IMPORTS")

        data = await client.get_module_import_edges("proj", "")

        assert {"from_mod": "pkg.a", "to_mod": "pkg.b"} in data["direct"]
        await client.close()

    async def test_indirect_edge_via_entity_import(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        mod_a = _entity("a", "pkg.a", label=NodeLabel.MODULE, file_path="pkg/a.py")
        mod_b = _entity("b", "pkg.b", label=NodeLabel.MODULE, file_path="pkg/b.py")
        helper = _entity("helper", "pkg.b.helper", file_path="pkg/b.py")
        defines = [
            ParsedRelationship(from_qualified_name="proj:pkg.b", rel_type=RelType.DEFINES, to_name="proj:pkg.b.helper")
        ]
        await client.upsert_file_entities("proj", "pkg/a.py", [mod_a], [])
        await client.upsert_file_entities("proj", "pkg/b.py", [mod_b, helper], defines)
        await _insert_edge(client, "proj:pkg.a", "proj:pkg.b.helper", "IMPORTS")

        data = await client.get_module_import_edges("proj", "")

        assert {"from_mod": "pkg.a", "to_mod": "pkg.b"} in data["indirect"]
        await client.close()


class TestGetDependencyExternalCounts:
    async def test_counts_grouped_by_package(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        mod = _entity("a", "pkg.a", label=NodeLabel.MODULE, file_path="pkg/a.py")
        await client.upsert_file_entities("proj", "pkg/a.py", [mod], [])
        await client.resolve_imports(
            "proj",
            [
                ParsedRelationship(from_qualified_name="proj:pkg.a", rel_type=RelType.IMPORTS, to_name="requests"),
                ParsedRelationship(
                    from_qualified_name="proj:pkg.a", rel_type=RelType.IMPORTS, to_name="requests.Session"
                ),
            ],
        )

        data = await client.get_dependency_external_counts("proj", "")

        pkg_counts = {r["package"]: r["cnt"] for r in data["ext_packages"]}
        sym_counts = {r["package"]: r["cnt"] for r in data["ext_symbols"]}
        assert pkg_counts.get("requests") == 1
        assert sym_counts.get("requests") == 1
        await client.close()


class TestGetQualityData:
    async def test_entities_and_either_side_path_scope(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        mod_a = _entity("a", "pkg.a", label=NodeLabel.MODULE, file_path="pkg/a.py")
        mod_b = _entity("b", "pkg.b", label=NodeLabel.MODULE, file_path="pkg/b.py")
        fn = _entity("fn", "pkg.a.fn", file_path="pkg/a.py")
        defines = [
            ParsedRelationship(from_qualified_name="proj:pkg.a", rel_type=RelType.DEFINES, to_name="proj:pkg.a.fn")
        ]
        await client.upsert_file_entities("proj", "pkg/a.py", [mod_a, fn], defines)
        await client.upsert_file_entities("proj", "pkg/b.py", [mod_b], [])
        await _insert_edge(client, "proj:pkg.b", "proj:pkg.a", "IMPORTS")

        # Scoped to pkg/a only — must still see the inbound edge from out-of-scope pkg.b
        # (either-side match), matching GraphClient.get_quality_data's fan-in requirement.
        data = await client.get_quality_data("proj", "pkg/a")

        assert data["entities"] == [{"module": "pkg.a", "file_path": "pkg/a.py", "entity_count": 1}]
        assert {"from_mod": "pkg.b", "to_mod": "pkg.a"} in data["direct"]
        await client.close()


class TestGetPatternsData:
    async def test_inheritance_and_enum_members(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        base = _entity("Base", "mod.Base", label=NodeLabel.TYPE_DEF)
        child = _entity("Child", "mod.Child", label=NodeLabel.TYPE_DEF)
        color = _entity("Color", "mod.Color", label=NodeLabel.TYPE_DEF, kind="enum")
        red = _entity("RED", "mod.Color.RED", label=NodeLabel.VALUE)
        rels = [
            ParsedRelationship(from_qualified_name="proj:mod.Child", rel_type=RelType.INHERITS, to_name="Base"),
            ParsedRelationship(
                from_qualified_name="proj:mod.Color", rel_type=RelType.DEFINES, to_name="proj:mod.Color.RED"
            ),
        ]
        await client.upsert_file_entities("proj", "mod.py", [base, child, color, red], rels)

        data = await client.get_patterns_data("proj", "", 20)

        assert data["inheritance"] == [
            {"child": "Child", "child_qn": "mod.Child", "parent": "Base", "parent_qn": "mod.Base"}
        ]
        assert data["enums"] == [{"name": "Color", "qn": "mod.Color", "file_path": "mod.py", "members": 1}]
        await client.close()

    async def test_visibility_distribution_and_docstring_coverage(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        documented = _entity("documented_fn", "mod.documented_fn", docstring="Has docs.")
        undocumented = _entity("bare_fn", "mod.bare_fn", visibility=Visibility.PRIVATE)
        await client.upsert_file_entities("proj", "mod.py", [documented, undocumented], [])

        data = await client.get_patterns_data("proj", "", 20)

        vis_counts = {r["visibility"]: r["cnt"] for r in data["visibility"]}
        assert vis_counts == {"public": 1, "private": 1}
        assert data["docstring"] == [{"total": 2, "documented": 1}]
        await client.close()

    async def test_detected_patterns(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        handler = _entity("handle_x", "mod.handle_x")
        target = _entity("Target", "mod.Target", label=NodeLabel.TYPE_DEF)
        rel = ParsedRelationship(
            from_qualified_name="proj:mod.handle_x", rel_type=RelType.HANDLES_ROUTE, to_name="proj:mod.Target"
        )
        await client.upsert_file_entities("proj", "mod.py", [handler, target], [rel])

        data = await client.get_patterns_data("proj", "", 20)

        assert data["detected_patterns"] == [
            {"pattern_type": "HANDLES_ROUTE", "name": "handle_x", "qn": "mod.handle_x", "target_name": "Target"}
        ]
        await client.close()


class TestGetDeadCodeCandidates:
    async def test_excludes_called_and_dunder_entities(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        dead = _entity("dead_fn", "mod.dead_fn")
        called = _entity("called_fn", "mod.called_fn")
        caller = _entity("caller_fn", "mod.caller_fn")
        dunder = _entity("__init__", "mod.Widget.__init__")
        await client.upsert_file_entities("proj", "mod.py", [dead, called, caller, dunder], [])
        await _insert_edge(client, "proj:mod.caller_fn", "proj:mod.called_fn", "CALLS")

        candidates = await client.get_dead_code_candidates("proj", "")

        names = {c["name"] for c in candidates}
        assert "dead_fn" in names
        assert "called_fn" not in names
        assert "__init__" not in names
        await client.close()


class TestGetComplexityHotspots:
    async def test_sorted_by_loc_span_descending(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        small = ParsedEntity(
            name="small",
            qualified_name="proj:mod.small",
            label=NodeLabel.CALLABLE,
            kind="function",
            line_start=1,
            line_end=5,
            file_path="mod.py",
        )
        big = ParsedEntity(
            name="big",
            qualified_name="proj:mod.big",
            label=NodeLabel.CALLABLE,
            kind="function",
            line_start=1,
            line_end=105,
            file_path="mod.py",
        )
        await client.upsert_file_entities("proj", "mod.py", [small, big], [])

        hotspots = await client.get_complexity_hotspots("proj", "", 20)

        assert [h["name"] for h in hotspots] == ["big", "small"]
        assert hotspots[0]["loc_span"] == 104
        await client.close()


class TestGetGitSignalsData:
    async def test_hotspots_bus_factor_and_co_change(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        mod_a = _entity("a", "pkg.a", label=NodeLabel.MODULE, file_path="pkg/a.py")
        mod_b = _entity("b", "pkg.b", label=NodeLabel.MODULE, file_path="pkg/b.py")
        await client.upsert_file_entities("proj", "pkg/a.py", [mod_a], [])
        await client.upsert_file_entities("proj", "pkg/b.py", [mod_b], [])
        await client.apply_property_enrichments(
            [
                PropertyEnrichment(
                    qualified_name="proj:pkg.a",
                    properties={"git_commit_count": 10, "git_author_count": 1, "git_days_since_last_commit": 2.0},
                ),
                PropertyEnrichment(
                    qualified_name="proj:pkg.b",
                    properties={"git_commit_count": 5, "git_author_count": 3, "git_days_since_last_commit": 1.0},
                ),
            ]
        )
        await _insert_edge(client, "proj:pkg.a", "proj:pkg.b", "CO_CHANGES_WITH", {"count": 4})

        data = await client.get_git_signals_data("proj", "", 20, 1)

        hotspot_counts = {r["qn"]: r["commit_count"] for r in data["hotspots"]}
        assert hotspot_counts == {"pkg.a": 10, "pkg.b": 5}
        assert {r["qn"] for r in data["bus_factor"]} == {"pkg.a"}
        assert data["co_change"][0]["a_qn"] == "pkg.a"
        assert data["co_change"][0]["count"] == 4
        await client.close()


class TestWriteGitFileSignals:
    async def test_writes_signal_properties_and_returns_matched_count(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        mod_a = _entity("a", "pkg.a", label=NodeLabel.MODULE, file_path="pkg/a.py")
        mod_b = _entity("b", "pkg.b", label=NodeLabel.MODULE, file_path="pkg/b.py")
        await client.upsert_file_entities("proj", "pkg/a.py", [mod_a], [])
        await client.upsert_file_entities("proj", "pkg/b.py", [mod_b], [])

        matched = await client.write_git_file_signals(
            "proj",
            "Module",
            [
                {"fp": "pkg/a.py", "cc": 10, "ac": 2, "days": 1.5},
                {"fp": "pkg/b.py", "cc": 3, "ac": 1, "days": 4.0},
                {"fp": "pkg/deleted.py", "cc": 1, "ac": 1, "days": 9.0},  # no matching node
            ],
        )

        assert matched == 2
        data = await client.get_git_signals_data("proj", "", 20, 5)
        by_qn = {r["qn"]: r for r in data["hotspots"]}
        assert by_qn["pkg.a"]["commit_count"] == 10
        assert by_qn["pkg.a"]["author_count"] == 2
        assert by_qn["pkg.b"]["commit_count"] == 3
        await client.close()

    async def test_empty_items_returns_zero_without_touching_db(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()

        assert await client.write_git_file_signals("proj", "Module", []) == 0
        await client.close()


class TestWriteCoChangeEdges:
    async def test_creates_edge_between_matched_files_and_returns_count(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        mod_a = _entity("a", "pkg.a", label=NodeLabel.MODULE, file_path="pkg/a.py")
        mod_b = _entity("b", "pkg.b", label=NodeLabel.MODULE, file_path="pkg/b.py")
        await client.upsert_file_entities("proj", "pkg/a.py", [mod_a], [])
        await client.upsert_file_entities("proj", "pkg/b.py", [mod_b], [])

        created = await client.write_co_change_edges(
            "proj",
            [
                {"a": "pkg/a.py", "b": "pkg/b.py", "cnt": 4},
                {"a": "pkg/a.py", "b": "pkg/deleted.py", "cnt": 3},  # one side unmatched — skipped
            ],
        )

        assert created == 1
        data = await client.get_git_signals_data("proj", "", 20, 5)
        assert data["co_change"] == [
            {"a_qn": "pkg.a", "a_path": "pkg/a.py", "b_qn": "pkg.b", "b_path": "pkg/b.py", "count": 4}
        ]
        await client.close()

    async def test_rewriting_a_pair_updates_count_not_a_duplicate_edge(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        mod_a = _entity("a", "pkg.a", label=NodeLabel.MODULE, file_path="pkg/a.py")
        mod_b = _entity("b", "pkg.b", label=NodeLabel.MODULE, file_path="pkg/b.py")
        await client.upsert_file_entities("proj", "pkg/a.py", [mod_a], [])
        await client.upsert_file_entities("proj", "pkg/b.py", [mod_b], [])

        await client.write_co_change_edges("proj", [{"a": "pkg/a.py", "b": "pkg/b.py", "cnt": 4}])
        created_again = await client.write_co_change_edges("proj", [{"a": "pkg/a.py", "b": "pkg/b.py", "cnt": 9}])

        assert created_again == 1
        data = await client.get_git_signals_data("proj", "", 20, 5)
        assert len(data["co_change"]) == 1
        assert data["co_change"][0]["count"] == 9
        await client.close()

    async def test_empty_pairs_returns_zero_without_touching_db(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()

        assert await client.write_co_change_edges("proj", []) == 0
        await client.close()


class TestGetDiagramPackages:
    async def test_package_to_module_edge(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        module = _entity("mod", "pkg.mod", label=NodeLabel.MODULE, file_path="pkg/mod.py")
        await client.upsert_file_entities("proj", "pkg/mod.py", [module], [])
        await client.merge_package_node("proj", "pkg", "pkg", "pkg/")
        await client.create_contains_edge("proj:pkg", "proj:pkg.mod")

        records = await client.get_diagram_packages("proj", "", 30)

        assert records == [
            {
                "parent_qn": "pkg",
                "parent_name": "pkg",
                "child_label": "Module",
                "child_qn": "pkg.mod",
                "child_name": "mod",
            }
        ]
        await client.close()


class TestGetDiagramInheritance:
    async def test_child_parent_edge(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        base = _entity("Base", "mod.Base", label=NodeLabel.TYPE_DEF)
        child = _entity("Child", "mod.Child", label=NodeLabel.TYPE_DEF, kind="class")
        rel = ParsedRelationship(from_qualified_name="proj:mod.Child", rel_type=RelType.INHERITS, to_name="Base")
        await client.upsert_file_entities("proj", "mod.py", [base, child], [rel])

        records = await client.get_diagram_inheritance("proj", "", 30)

        assert records == [
            {
                "child_name": "Child",
                "child_qn": "mod.Child",
                "child_kind": "class",
                "parent_name": "Base",
                "parent_qn": "mod.Base",
            }
        ]
        await client.close()


class TestGetDiagramModuleDetail:
    async def test_module_entities_methods_and_inheritance(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        module = _entity("mod", "pkg.mod", label=NodeLabel.MODULE, file_path="pkg/mod.py")
        base = _entity("Base", "pkg.mod.Base", label=NodeLabel.TYPE_DEF)
        widget = _entity("Widget", "pkg.mod.Widget", label=NodeLabel.TYPE_DEF)
        method = _entity("run", "pkg.mod.Widget.run")
        rels = [
            ParsedRelationship(
                from_qualified_name="proj:pkg.mod", rel_type=RelType.DEFINES, to_name="proj:pkg.mod.Base"
            ),
            ParsedRelationship(
                from_qualified_name="proj:pkg.mod", rel_type=RelType.DEFINES, to_name="proj:pkg.mod.Widget"
            ),
            ParsedRelationship(
                from_qualified_name="proj:pkg.mod.Widget",
                rel_type=RelType.DEFINES,
                to_name="proj:pkg.mod.Widget.run",
            ),
            ParsedRelationship(from_qualified_name="proj:pkg.mod.Widget", rel_type=RelType.INHERITS, to_name="Base"),
        ]
        await client.upsert_file_entities("proj", "pkg/mod.py", [module, base, widget, method], rels)

        detail = await client.get_diagram_module_detail("proj", "pkg/mod", 30)

        assert detail is not None
        assert detail["module"]["qn"] == "pkg.mod"
        assert {e["qn"] for e in detail["entities"]} == {"pkg.mod.Base", "pkg.mod.Widget"}
        assert detail["methods"] == [
            {"class_qn": "pkg.mod.Widget", "class_name": "Widget", "name": "run", "vis": "public", "kind": "function"}
        ]
        assert detail["inherits"] == [
            {"child_qn": "pkg.mod.Widget", "child_name": "Widget", "parent_qn": "pkg.mod.Base", "parent_name": "Base"}
        ]
        await client.close()

    async def test_no_matching_module_returns_none(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()

        detail = await client.get_diagram_module_detail("proj", "nonexistent/path", 30)

        assert detail is None
        await client.close()


# ---------------------------------------------------------------------------
# Context expansion / navigation (search/engine.py's expand_context) —
# SQL ports of GraphClient's expand_context methods.
# ---------------------------------------------------------------------------


class TestGetEntityByUid:
    async def test_returns_node_or_none(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("f", "mod.f")], [])

        node = await client.get_entity_by_uid("proj:mod.f")
        assert node is not None
        assert node["uid"] == "proj:mod.f"
        assert await client.get_entity_by_uid("proj:mod.missing") is None
        await client.close()

    async def test_label_mismatch_returns_none(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("f", "mod.f")], [])

        assert await client.get_entity_by_uid("proj:mod.f", label="TypeDef") is None
        assert await client.get_entity_by_uid("proj:mod.f", label="Callable") is not None
        await client.close()


class TestGetDefiningParentAndSiblings:
    async def test_parent_and_siblings(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        parent = _entity("MyClass", "mod.MyClass", label=NodeLabel.TYPE_DEF)
        m1 = _entity("m1", "mod.MyClass.m1")
        m2 = _entity("m2", "mod.MyClass.m2")
        rels = [
            ParsedRelationship(
                from_qualified_name="proj:mod.MyClass", rel_type=RelType.DEFINES, to_name="proj:mod.MyClass.m1"
            ),
            ParsedRelationship(
                from_qualified_name="proj:mod.MyClass", rel_type=RelType.DEFINES, to_name="proj:mod.MyClass.m2"
            ),
        ]
        await client.upsert_file_entities("proj", "mod.py", [parent, m1, m2], rels)

        p = await client.get_defining_parent("proj:mod.MyClass.m1")
        assert p is not None
        assert p["uid"] == "proj:mod.MyClass"

        siblings = await client.get_sibling_entities("proj:mod.MyClass.m1", 10)
        assert {s["uid"] for s in siblings} == {"proj:mod.MyClass.m2"}

        assert await client.get_defining_parent("proj:mod.MyClass") is None
        await client.close()


class TestGetPackageDocstring:
    async def test_walks_up_to_nearest_module(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        module = _entity("mod", "pkg.mod", label=NodeLabel.MODULE, docstring="Module docs.")
        cls = _entity("MyClass", "pkg.mod.MyClass", label=NodeLabel.TYPE_DEF)
        method = _entity("method", "pkg.mod.MyClass.method")
        rels = [
            ParsedRelationship(
                from_qualified_name="proj:pkg.mod", rel_type=RelType.DEFINES, to_name="proj:pkg.mod.MyClass"
            ),
            ParsedRelationship(
                from_qualified_name="proj:pkg.mod.MyClass",
                rel_type=RelType.DEFINES,
                to_name="proj:pkg.mod.MyClass.method",
            ),
        ]
        await client.upsert_file_entities("proj", "pkg/mod.py", [module, cls, method], rels)

        assert await client.get_package_docstring("proj:pkg.mod.MyClass.method") == "Module docs."
        await client.close()

    async def test_no_enclosing_module_returns_none(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("f", "mod.f")], [])

        assert await client.get_package_docstring("proj:mod.f") is None
        await client.close()


class TestGetCallersAndCallees:
    async def test_multi_hop_traversal_filters_by_callable_label(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        a, b, c = _entity("a", "mod.a"), _entity("b", "mod.b"), _entity("c", "mod.c")
        await client.upsert_file_entities("proj", "mod.py", [a, b, c], [])
        await _insert_edge(client, "proj:mod.a", "proj:mod.b", "CALLS")
        await _insert_edge(client, "proj:mod.b", "proj:mod.c", "CALLS")

        callers = await client.get_callers("proj:mod.c", "", 2, 10)
        assert {r["uid"] for r in callers} == {"proj:mod.a", "proj:mod.b"}

        callees = await client.get_callees("proj:mod.a", "", 2, 10)
        assert {r["uid"] for r in callees} == {"proj:mod.b", "proj:mod.c"}
        await client.close()

    async def test_label_mismatch_on_target_returns_empty(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        a, b = _entity("a", "mod.a"), _entity("b", "mod.b")
        await client.upsert_file_entities("proj", "mod.py", [a, b], [])
        await _insert_edge(client, "proj:mod.a", "proj:mod.b", "CALLS")

        assert await client.get_callers("proj:mod.b", "TypeDef", 1, 10) == []
        await client.close()


class TestGetLinkedDocs:
    async def test_returns_docs_with_anchor_metadata(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        target = _entity("f", "mod.f")
        doc = _entity("doc", "note:doc", label=NodeLabel.NOTE, kind="note", file_path="doc.md")
        await client.upsert_file_entities("proj", "mod.py", [target], [])
        await client.upsert_file_entities("proj", "doc.md", [doc], [])
        await _insert_edge(
            client,
            "proj:note:doc",
            "proj:mod.f",
            "DOCUMENTS",
            {"link_type": "anchor", "stale": False, "anchor_hash": "h1"},
        )

        docs = await client.get_linked_docs("proj:mod.f", "", 10)

        assert len(docs) == 1
        assert docs[0]["node"]["uid"] == "proj:note:doc"
        assert docs[0]["link_type"] == "anchor"
        assert docs[0]["stale"] is False
        assert docs[0]["anchor_hash"] == "h1"
        await client.close()


class TestGetNodeExactMatches:
    async def test_matches_by_uid_and_name(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("my_func", "mod.my_func")], [])

        by_uid = await client.get_node_exact_matches("proj:mod.my_func", "", 5)
        assert any(r["n"]["uid"] == "proj:mod.my_func" for r in by_uid)

        by_name = await client.get_node_exact_matches("my_func", "", 5)
        assert any(r["n"]["uid"] == "proj:mod.my_func" for r in by_name)
        await client.close()


class TestGetNodePartialMatches:
    async def test_suffix_scores_highest(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("target_fn", "pkg.mod.target_fn")], [])

        results = await client.get_node_partial_matches("target_fn", "", 5)

        # The uid can legitimately appear in more than one branch (here: both the
        # suffix and contains branches) — callers (mcp.py's get_node) dedupe by
        # picking the max score per uid, so check that, not a single collapsed entry.
        best_score = max(r["_match_score"] for r in results if r["n"]["uid"] == "proj:pkg.mod.target_fn")
        assert best_score == 3  # suffix (.target_fn) is the highest-scored branch
        await client.close()


class TestGetLabelCounts:
    async def test_counts_grouped_by_label(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities(
            "proj",
            "mod.py",
            [_entity("f", "mod.f"), _entity("g", "mod.g"), _entity("C", "mod.C", label=NodeLabel.TYPE_DEF)],
            [],
        )

        counts = await client.get_label_counts()

        assert counts["Callable"] == 2
        assert counts["TypeDef"] == 1
        await client.close()


class TestGetProjectDependencyEdges:
    async def test_project_to_project_edges(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.merge_project_node("app")
        await client.merge_project_node("lib")
        await _insert_edge(client, "app", "lib", "DEPENDS_ON")

        edges = await client.get_project_dependency_edges()

        assert edges == [{"from_proj": "app", "to_proj": "lib"}]
        await client.close()


class TestGetExistingUids:
    async def test_returns_only_existing(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("f", "mod.f")], [])

        existing = await client.get_existing_uids(["proj:mod.f", "proj:mod.missing"])

        assert existing == {"proj:mod.f"}
        await client.close()

    async def test_empty_input_returns_empty_set(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        assert await client.get_existing_uids([]) == set()
        await client.close()


class TestGetOrphanNotes:
    async def test_note_without_links_to_is_orphan(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        orphan = _entity("orphan", "note:orphan", label=NodeLabel.NOTE, kind="note", file_path="orphan.md")
        linked_a = _entity("a", "note:a", label=NodeLabel.NOTE, kind="note", file_path="a.md")
        linked_b = _entity("b", "note:b", label=NodeLabel.NOTE, kind="note", file_path="b.md")
        await client.upsert_file_entities("proj", "orphan.md", [orphan], [])
        await client.upsert_file_entities("proj", "a.md", [linked_a], [])
        await client.upsert_file_entities("proj", "b.md", [linked_b], [])
        await _insert_edge(client, "proj:note:a", "proj:note:b", "LINKS_TO")

        orphans = await client.get_orphan_notes()

        assert {n["uid"] for n in orphans} == {"proj:note:orphan"}
        await client.close()


class TestGetBrokenAnchorNotes:
    async def test_unresolved_anchors_flagged(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        note = _entity("n", "note:n", label=NodeLabel.NOTE, kind="note", file_path="n.md")
        clean = _entity("clean", "note:clean", label=NodeLabel.NOTE, kind="note", file_path="clean.md")
        await client.upsert_file_entities("proj", "n.md", [note], [])
        await client.upsert_file_entities("proj", "clean.md", [clean], [])
        await client.apply_property_enrichments(
            [PropertyEnrichment(qualified_name="proj:note:n", properties={"unresolved_anchors": ["missing-target"]})]
        )

        broken = await client.get_broken_anchor_notes()

        assert len(broken) == 1
        assert broken[0]["uid"] == "proj:note:n"
        assert broken[0]["unresolved_anchors"] == ["missing-target"]
        await client.close()


class TestGetInboxNotePaths:
    async def test_draft_kind_and_inbox_path_included(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        draft = _entity("d", "note:d", label=NodeLabel.NOTE, kind="draft", file_path="d.md")
        inbox = _entity("i", "note:i", label=NodeLabel.NOTE, kind="note", file_path="wiki/inbox/i.md")
        settled = _entity("s", "note:s", label=NodeLabel.NOTE, kind="note", file_path="wiki/notes/s.md")
        await client.upsert_file_entities("proj", "d.md", [draft], [])
        await client.upsert_file_entities("proj", "wiki/inbox/i.md", [inbox], [])
        await client.upsert_file_entities("proj", "wiki/notes/s.md", [settled], [])

        paths = await client.get_inbox_note_paths()

        assert set(paths) == {"d.md", "wiki/inbox/i.md"}
        await client.close()


class TestGetNoteEmbeddings:
    async def test_returns_notes_with_embeddings_only(self, tmp_path: Path) -> None:
        dim = 3
        client = SqliteGraphClient(tmp_path / "graph.sqlite3", dimension=dim)
        await client.ensure_schema()
        with_emb = _entity("a", "note:a", label=NodeLabel.NOTE, kind="note", file_path="a.md")
        without_emb = _entity("b", "note:b", label=NodeLabel.NOTE, kind="note", file_path="b.md")
        await client.upsert_file_entities("proj", "a.md", [with_emb], [])
        await client.upsert_file_entities("proj", "b.md", [without_emb], [])
        await client.write_embeddings([("proj:note:a", [1.0, 2.0, 3.0])])

        rows = await client.get_note_embeddings()

        assert len(rows) == 1
        assert rows[0]["uid"] == "proj:note:a"
        assert rows[0]["project_name"] == "proj"
        assert rows[0]["embedding"] == [1.0, 2.0, 3.0]
        await client.close()


# ---------------------------------------------------------------------------
# Config references (EnvVar / ResourceFile) + reference-counted GC
# ---------------------------------------------------------------------------


def _env_ref(from_uid: str, name: str, **props: Any) -> ParsedRelationship:
    return ParsedRelationship(
        from_qualified_name=from_uid, rel_type=RelType.READS_ENV, to_name=name, properties=dict(props)
    )


def _file_ref(from_uid: str, path: str, **props: Any) -> ParsedRelationship:
    return ParsedRelationship(
        from_qualified_name=from_uid, rel_type=RelType.REFERENCES_FILE, to_name=path, properties=dict(props)
    )


async def _node_row(client: SqliteGraphClient, uid: str) -> Any:
    conn = await client._get_conn()
    cur = await conn.execute(
        "SELECT labels, project_name, name, qualified_name, props_json FROM nodes WHERE uid = ?", (uid,)
    )
    row = await cur.fetchone()
    await cur.close()
    return row


async def _scalar(client: SqliteGraphClient, sql: str, params: tuple[Any, ...] = ()) -> Any:
    conn = await client._get_conn()
    cur = await conn.execute(sql, params)
    row = await cur.fetchone()
    await cur.close()
    return row[0] if row else None


class TestResolveConfigRefs:
    async def test_creates_global_env_var_and_scoped_resource_file(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("f", "mod.f")], [])

        await client.resolve_config_refs(
            "proj", [_env_ref("proj:mod.f", "DATABASE_URL"), _file_ref("proj:mod.f", "./data/fixtures.json")]
        )

        env = await _node_row(client, "env/DATABASE_URL")
        assert env[:4] == ("EnvVar", GLOBAL_PROJECT, "DATABASE_URL", "env/DATABASE_URL")
        res = await _node_row(client, "proj:res/data/fixtures.json")
        assert res[:4] == ("ResourceFile", "proj", "fixtures.json", "res/data/fixtures.json")
        await client.close()

    async def test_stores_names_not_values(self, tmp_path: Path) -> None:
        """A default argument is a live-secret channel — nothing from the
        reference's properties may be persisted on the node, the edge, or the
        BM25 document.
        """
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("f", "mod.f")], [])
        secret = "sk-live-abc123"

        await client.resolve_config_refs("proj", [_env_ref("proj:mod.f", "API_KEY", default=secret, value=secret)])

        assert await _scalar(client, "SELECT props_json FROM nodes WHERE uid = 'env/API_KEY'") == "{}"
        assert await _scalar(client, "SELECT props_json FROM edges WHERE rel_type = 'READS_ENV'") == "{}"
        fts_text = await _scalar(client, "SELECT text FROM text_envvar WHERE uid = 'env/API_KEY'")
        assert secret not in fts_text
        await client.close()

    async def test_env_var_is_text_searchable_from_a_scoped_search(self, tmp_path: Path) -> None:
        """A project-scoped search must still surface the global node — the
        whole point of making these labels text-searchable.
        """
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("f", "mod.f")], [])
        await client.resolve_config_refs("proj", [_env_ref("proj:mod.f", "DATABASE_URL")])

        hits = await client.text_search("DATABASE_URL", project="proj")

        assert [h["node"]["uid"] for h in hits] == ["env/DATABASE_URL"]
        await client.close()

    async def test_reresolve_is_idempotent(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("f", "mod.f")], [])
        rels = [_env_ref("proj:mod.f", "X")]

        await client.resolve_config_refs("proj", rels)
        await client.resolve_config_refs("proj", rels)

        assert await _scalar(client, "SELECT COUNT(*) FROM edges WHERE rel_type = 'READS_ENV'") == 1
        await client.close()


class TestGcOrphanedReferenceNodes:
    async def test_keeps_referenced_nodes(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("f", "mod.f")], [])
        await client.resolve_config_refs(
            "proj", [_env_ref("proj:mod.f", "KEEP"), _file_ref("proj:mod.f", "data/keep.json")]
        )

        assert await client.gc_orphaned_reference_nodes() == 0

        assert await _node_row(client, "env/KEEP") is not None
        assert await _node_row(client, "proj:res/data/keep.json") is not None
        await client.close()

    async def test_sweeps_node_after_its_last_reference_disappears(self, tmp_path: Path) -> None:
        """The end-to-end reference-counting contract: re-upserting the file
        without the reference drops the edge (relationship recreation), and the
        sweep then reclaims the now-unreferenced node.
        """
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("f", "mod.f")], [])
        await client.resolve_config_refs("proj", [_env_ref("proj:mod.f", "GONE")])
        assert await _node_row(client, "env/GONE") is not None

        # Reparse the same file with changed content and no config refs at all.
        await client.upsert_file_entities("proj", "mod.py", [_entity("f", "mod.f", content_hash="h2")], [])

        assert await client.gc_orphaned_reference_nodes() == 1
        assert await _node_row(client, "env/GONE") is None
        await client.close()

    async def test_sweep_also_clears_the_fts_row(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("f", "mod.f")], [])
        await client.resolve_config_refs("proj", [_env_ref("proj:mod.f", "GONE")])
        await client.upsert_file_entities("proj", "mod.py", [_entity("f", "mod.f", content_hash="h2")], [])

        await client.gc_orphaned_reference_nodes()

        assert await client.text_search("GONE") == []
        await client.close()

    async def test_survives_while_one_of_two_referrers_remains(self, tmp_path: Path) -> None:
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "a.py", [_entity("f", "a.f", file_path="a.py")], [])
        await client.upsert_file_entities("proj", "b.py", [_entity("g", "b.g", file_path="b.py")], [])
        await client.resolve_config_refs("proj", [_env_ref("proj:a.f", "SHARED"), _env_ref("proj:b.g", "SHARED")])

        await client.upsert_file_entities(
            "proj", "a.py", [_entity("f", "a.f", file_path="a.py", content_hash="h2")], []
        )

        assert await client.gc_orphaned_reference_nodes() == 0
        assert await _node_row(client, "env/SHARED") is not None
        await client.close()

    async def test_project_deletion_orphans_the_global_env_var(self, tmp_path: Path) -> None:
        """``delete_project_data`` cannot reach a ``_global`` node — the sweep is
        what stops deleted projects from leaking env vars forever.
        """
        client = SqliteGraphClient(tmp_path / "graph.sqlite3")
        await client.ensure_schema()
        await client.upsert_file_entities("proj", "mod.py", [_entity("f", "mod.f")], [])
        await client.resolve_config_refs("proj", [_env_ref("proj:mod.f", "ORPHANED")])

        await client.delete_project_data("proj")
        assert await _node_row(client, "env/ORPHANED") is not None  # survived the project wipe

        assert await client.gc_orphaned_reference_nodes() == 1
        assert await _node_row(client, "env/ORPHANED") is None
        await client.close()
