"""Structural type contract for graph backends (Memgraph ``GraphClient`` and the
SQLite fallback ``SqliteGraphClient``).

Generalizes — doesn't replace — the narrow ``SearchGraph``/``GraphExecutor``
Protocols in :mod:`code_atlas.search.engine`, which remain valid subsets for
their specific call sites. ``GraphBackend`` covers the ~46 portable
``GraphClient`` methods identified as backend-agnostic (plain read/write CRUD,
resolution, embeddings, search) — the handful of Memgraph/MAGE-specific calls
(raw Cypher via ``execute``/``execute_write``, community detection) are *not*
part of this contract; callers that need those must branch on backend
capability themselves (see ``server/analysis.py``'s communities guard).

Like ``SearchGraph``/``GraphExecutor``, this Protocol is TYPE_CHECKING-only —
it exists purely for static structural typing, never for runtime isinstance
checks. Modules that want to type a graph handle against this contract import
it inside their own ``if TYPE_CHECKING:`` block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Protocol

    from code_atlas.graph.client import CallStats, UpsertResult, _AnchorLookup, _CallLookup
    from code_atlas.parsing.ast import ParsedEntity, ParsedRelationship
    from code_atlas.parsing.detectors import PropertyEnrichment

    class GraphBackend(Protocol):
        """Structural subset of ``GraphClient`` shared by every graph backend.

        Covers connection lifecycle, schema bootstrap, entity/relationship
        CRUD, cross-file resolution, embeddings, and search — everything
        downstream modules (``indexing/*``, ``server/analysis.py``,
        ``dream.py``, ``parsing/detectors.py``, ``search/guidance.py``) need
        from whatever graph object they're constructed with.
        """

        # -- Connection lifecycle / schema ---------------------------------

        async def ping(self) -> bool: ...

        async def execute(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...

        async def execute_write(self, query: str, params: dict[str, Any] | None = None) -> None: ...

        async def ensure_schema(self) -> None: ...

        async def get_schema_version(self) -> int | None: ...

        async def close(self) -> None: ...

        # -- Entity / relationship CRUD -------------------------------------

        async def upsert_file_entities(
            self,
            project_name: str,
            file_path: str,
            entities: list[ParsedEntity],
            relationships: list[ParsedRelationship],
        ) -> UpsertResult: ...

        async def upsert_batch_entities(
            self,
            project_name: str,
            file_data: dict[str, tuple[list[ParsedEntity], list[ParsedRelationship]]],
        ) -> dict[str, UpsertResult]: ...

        async def delete_file_entities(self, project_name: str, file_path: str) -> list[str]: ...

        async def merge_project_node(self, project_name: str, **metadata: Any) -> None: ...

        async def get_batch_file_hashes(self, project_name: str, file_paths: list[str]) -> dict[str, str | None]: ...

        async def set_batch_file_hashes(self, project_name: str, file_hashes: dict[str, str]) -> None: ...

        async def merge_package_node(
            self, project_name: str, qualified_name: str, name: str, file_path: str
        ) -> None: ...

        async def merge_package_batch(self, project_name: str, packages: list[tuple[str, str, str]]) -> None: ...

        async def create_contains_edge(self, from_uid: str, to_uid: str) -> None: ...

        async def delete_project_data(self, project_name: str) -> None: ...

        async def update_project_metadata(self, project_name: str, **metadata: Any) -> None: ...

        async def get_project_status(self, project_name: str | None = None) -> list[dict[str, Any]]: ...

        async def get_project_git_hash(self, project_name: str) -> str | None: ...

        async def get_project_file_paths(self, project_name: str) -> set[str]: ...

        async def count_entities(self, project_name: str) -> int: ...

        # -- Cross-file resolution -------------------------------------------

        async def resolve_imports(self, project_name: str, import_rels: list[ParsedRelationship]) -> None: ...

        async def resolve_calls(
            self,
            project_name: str,
            call_rels: list[ParsedRelationship],
            *,
            lookup: _CallLookup | None = None,
            name_to_typedefs: dict[str, list[tuple[str, str]]] | None = None,
        ) -> None: ...

        async def build_anchor_lookup(self) -> _AnchorLookup: ...

        async def resolve_anchors(
            self,
            anchor_rels: list[ParsedRelationship],
            *,
            lookup: _AnchorLookup | None = None,
        ) -> None: ...

        async def invalidate_stale_anchors(self, changed_uids: set[str]) -> int: ...

        async def resolve_type_refs(
            self,
            project_name: str,
            type_rels: list[ParsedRelationship],
            *,
            lookup: _CallLookup | None = None,
            name_to_typedefs: dict[str, list[tuple[str, str]]] | None = None,
        ) -> None: ...

        async def resolve_member_defines(
            self,
            project_name: str,
            member_rels: list[ParsedRelationship],
            *,
            lookup: _CallLookup | None = None,
            name_to_typedefs: dict[str, list[tuple[str, str]]] | None = None,
        ) -> None: ...

        async def build_resolution_lookup(
            self, project_name: str
        ) -> tuple[_CallLookup, dict[str, list[tuple[str, str]]]]: ...

        async def update_external_package_versions(self, project_name: str, versions: dict[str, str]) -> None: ...

        async def resolve_cross_project_imports(self, project_names: list[str]) -> int: ...

        async def create_depends_on_edges(self, project_names: list[str]) -> int: ...

        async def apply_property_enrichments(self, enrichments: list[PropertyEnrichment]) -> None: ...

        # -- Detector lookups (parsing/languages/*.py) -------------------------

        async def find_entity_uid(self, project_name: str, label: str, name: str) -> str | None: ...

        async def find_overridden_method(
            self, project_name: str, bases: list[str], method_name: str
        ) -> tuple[str, list[str]] | None: ...

        # -- Embeddings -------------------------------------------------------

        async def get_embedding_config(self) -> tuple[str, int] | None: ...

        async def set_embedding_config(self, model: str, dimension: int) -> None: ...

        async def read_entity_texts(
            self,
            uids: list[str],
            *,
            labels: list[str] | None = None,
            chunk_size: int = 200,
        ) -> list[dict[str, Any]]: ...

        async def read_embed_hashes(
            self, uids: list[str], *, labels: list[str] | None = None
        ) -> dict[str, tuple[str | None, bool]]: ...

        async def write_embeddings(
            self,
            items: list[tuple[str, list[float]]],
            chunk_size: int = 50,
            *,
            labels: list[str] | None = None,
        ) -> None: ...

        async def write_embed_hashes(
            self, items: list[tuple[str, str]], *, labels: list[str] | None = None
        ) -> None: ...

        async def write_embeddings_and_hashes(
            self,
            items: list[tuple[str, list[float], str]],
            *,
            labels: list[str] | None = None,
        ) -> None: ...

        async def clear_all_embeddings(self) -> None: ...

        # -- Search -------------------------------------------------------------

        async def graph_search(
            self,
            query: str,
            label: str = "",
            limit: int = 20,
            project: str = "",
            projects: list[str] | None = None,
        ) -> list[dict[str, Any]]: ...

        async def text_search(
            self,
            query: str,
            label: str = "",
            limit: int = 20,
            project: str = "",
            projects: list[str] | None = None,
        ) -> list[dict[str, Any]]: ...

        async def vector_search(
            self,
            vector: list[float],
            label: str = "",
            limit: int = 20,
            project: str = "",
            threshold: float = 0.0,
            projects: list[str] | None = None,
        ) -> list[dict[str, Any]]: ...

        async def get_vector_index_info(self) -> list[dict[str, Any]]: ...

        async def get_text_index_info(self) -> list[dict[str, Any]]: ...

        async def rebuild_vector_indices(self, dimension: int) -> None: ...

        async def batch_call_stats(self, uids: list[str], *, top_n: int = 5) -> dict[str, CallStats]: ...

        # -- Analysis / diagram queries (server/analysis.py) ------------------
        #
        # One method per analyze_repo sub-analysis / generate_diagram diagram
        # type (plus trace_path/blast_radius) — each returns the raw record
        # data that function needs, already backend-agnostic (plain dicts, no
        # Memgraph Node/Relationship objects). The Python-side business logic
        # (sorting, dedup, SCC cycle detection, health scoring, Mermaid
        # rendering) stays in analysis.py, unchanged, operating on this data
        # identically regardless of which backend produced it. Communities
        # (MAGE Leiden clustering) is deliberately NOT part of this contract —
        # see server/analysis.py's _analyze_communities isinstance guard.

        async def node_exists(self, uid: str) -> bool: ...

        async def trace_path_between(
            self, from_uid: str, to_uid: str, max_depth: int, edge_types: tuple[str, ...]
        ) -> dict[str, Any]: ...

        async def compute_blast_radius(
            self, uid: str, direction_kind: str, edge_types: tuple[str, ...], max_depth: int
        ) -> list[dict[str, Any]]: ...

        async def get_structure_overview(
            self, project: str, path: str, limit: int
        ) -> dict[str, list[dict[str, Any]]]: ...

        async def get_centrality_data(self, project: str, path: str, limit: int) -> dict[str, list[dict[str, Any]]]: ...

        async def get_module_import_edges(self, project: str, path: str) -> dict[str, list[dict[str, Any]]]: ...

        async def get_dependency_external_counts(self, project: str, path: str) -> dict[str, list[dict[str, Any]]]: ...

        async def get_quality_data(self, project: str, path: str) -> dict[str, list[dict[str, Any]]]: ...

        async def get_patterns_data(self, project: str, path: str, limit: int) -> dict[str, list[dict[str, Any]]]: ...

        async def get_dead_code_candidates(self, project: str, path: str) -> list[dict[str, Any]]: ...

        async def get_complexity_hotspots(self, project: str, path: str, limit: int) -> list[dict[str, Any]]: ...

        async def get_git_signals_data(
            self, project: str, path: str, limit: int, bus_factor_threshold: int
        ) -> dict[str, list[dict[str, Any]]]: ...

        async def get_diagram_packages(self, project: str, path: str, max_nodes: int) -> list[dict[str, Any]]: ...

        async def get_diagram_inheritance(self, project: str, path: str, max_nodes: int) -> list[dict[str, Any]]: ...

        async def get_diagram_module_detail(self, project: str, path: str, max_nodes: int) -> dict[str, Any] | None: ...

        # -- Context expansion / navigation (search/engine.py's expand_context) --

        async def get_entity_by_uid(self, uid: str, label: str = "") -> dict[str, Any] | None: ...

        async def get_defining_parent(self, uid: str) -> dict[str, Any] | None: ...

        async def get_sibling_entities(self, uid: str, limit: int) -> list[dict[str, Any]]: ...

        async def get_package_docstring(self, uid: str) -> str | None: ...

        async def get_callers(self, uid: str, label: str, call_depth: int, limit: int) -> list[dict[str, Any]]: ...

        async def get_callees(self, uid: str, label: str, call_depth: int, limit: int) -> list[dict[str, Any]]: ...

        async def get_linked_docs(self, uid: str, label: str, limit: int) -> list[dict[str, Any]]: ...

        # -- get_node cascade / status queries (server/mcp.py, cli.py) -----------

        async def get_node_exact_matches(self, name: str, label: str, limit: int) -> list[dict[str, Any]]: ...

        async def get_node_partial_matches(self, name: str, label: str, limit: int) -> list[dict[str, Any]]: ...

        async def get_label_counts(self) -> dict[str, int]: ...

        async def get_project_dependency_edges(self) -> list[dict[str, Any]]: ...

        # -- Dream-mode lint queries (dream.py) -----------------------------------

        async def get_existing_uids(self, uids: list[str]) -> set[str]: ...

        async def get_orphan_notes(self) -> list[dict[str, Any]]: ...

        async def get_broken_anchor_notes(self) -> list[dict[str, Any]]: ...

        async def get_inbox_note_paths(self) -> list[str]: ...

        async def get_note_embeddings(self) -> list[dict[str, Any]]: ...
