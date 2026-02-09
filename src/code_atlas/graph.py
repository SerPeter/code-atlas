"""Async Memgraph client for Code Atlas.

Handles connection lifecycle, schema application, and version management.
Uses the neo4j async driver (Bolt protocol) which is compatible with Memgraph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby
from operator import attrgetter
from typing import TYPE_CHECKING, Any

from loguru import logger
from neo4j import AsyncGraphDatabase

from code_atlas.schema import (
    _EMBEDDABLE_LABELS,
    _TEXT_SEARCHABLE_LABELS,
    SCHEMA_VERSION,
    NodeLabel,
    RelType,
    generate_drop_text_index_ddl,
    generate_drop_vector_index_ddl,
    generate_existence_constraint_ddl,
    generate_index_ddl,
    generate_text_index_ddl,
    generate_unique_constraint_ddl,
    generate_vector_index_ddl,
)

if TYPE_CHECKING:
    from neo4j import AsyncDriver

    from code_atlas.detectors import PropertyEnrichment
    from code_atlas.parser import ParsedEntity, ParsedRelationship
    from code_atlas.settings import AtlasSettings


@dataclass(frozen=True)
class UpsertResult:
    """Result of a delta-aware upsert for a single file."""

    added: list[str] = field(default_factory=list)  # qualified_names of new entities
    modified: list[str] = field(default_factory=list)  # qualified_names with changed content_hash
    deleted: list[str] = field(default_factory=list)  # qualified_names removed from file
    unchanged: list[str] = field(default_factory=list)  # qualified_names with matching content_hash


def _node_project_name(record: dict[str, Any]) -> str:
    """Extract project_name from a record containing a neo4j Node."""
    node = record.get("node") or record.get("n")
    if node is None:
        return ""
    if hasattr(node, "get"):
        return node.get("project_name", "")
    return ""


class GraphClient:
    """Async Memgraph client wrapping the neo4j Bolt driver.

    Follows the same lifecycle pattern as EventBus: construct → ping → use → close.
    """

    def __init__(self, settings: AtlasSettings) -> None:
        mg = settings.memgraph
        self._uri = f"bolt://{mg.host}:{mg.port}"
        auth = (mg.username, mg.password) if mg.username else None
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(self._uri, auth=auth)
        self._dimension = settings.embeddings.dimension

    async def ping(self) -> bool:
        """Health check — returns True if Memgraph is reachable."""
        records = await self.execute("RETURN 1 AS n")
        return len(records) == 1 and records[0]["n"] == 1

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a read query and return results as a list of dicts."""
        async with self._driver.session() as session:
            result = await session.run(query, params or {})  # type: ignore[arg-type]  # dynamic Cypher
            return [dict(record) async for record in result]

    async def execute_write(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Execute a write query.

        Consumes the result to ensure server-side errors (e.g. constraint
        violations) are raised instead of being silently dropped.
        """
        async with self._driver.session() as session:
            result = await session.run(query, params or {})  # type: ignore[arg-type]  # dynamic Cypher
            await result.consume()

    async def get_schema_version(self) -> int | None:
        """Read the current schema version from the SchemaVersion node."""
        records = await self.execute(f"MATCH (sv:{NodeLabel.SCHEMA_VERSION}) RETURN sv.version AS version")
        if not records:
            return None
        return records[0]["version"]

    async def ensure_schema(self) -> None:
        """Apply or migrate the graph schema.

        - Fresh DB (no version): apply all DDL, create version node.
        - Same version: no-op.
        - Older version: drop & recreate vector/text indices, bump version.
        - Newer version: raise RuntimeError (downgrade not supported).
        """
        stored = await self.get_schema_version()

        if stored is None:
            logger.info("Fresh database — applying schema v{}", SCHEMA_VERSION)
            await self._apply_full_schema()
            await self._set_schema_version(SCHEMA_VERSION)
            logger.info("Schema v{} applied successfully", SCHEMA_VERSION)

        elif stored == SCHEMA_VERSION:
            logger.debug("Schema v{} already current — no migration needed", SCHEMA_VERSION)

        elif stored < SCHEMA_VERSION:
            logger.info("Migrating schema v{} → v{}", stored, SCHEMA_VERSION)
            await self._migrate_indices()
            await self._set_schema_version(SCHEMA_VERSION)
            logger.info("Schema migrated to v{}", SCHEMA_VERSION)

        else:
            msg = (
                f"Database schema v{stored} is newer than code v{SCHEMA_VERSION}. "
                f"Downgrade is not supported — update your Code Atlas installation."
            )
            raise RuntimeError(msg)

    async def get_file_content_hashes(self, project_name: str, file_path: str) -> dict[str, tuple[str, int, int]]:
        """Return ``{uid: (content_hash, line_start, line_end)}`` for all non-structural nodes in a file."""
        records = await self.execute(
            f"MATCH (n {{project_name: $p, file_path: $f}}) "
            f"WHERE NOT n:{NodeLabel.PACKAGE} AND NOT n:{NodeLabel.PROJECT} "
            "RETURN n.uid AS uid, n.content_hash AS hash, n.line_start AS ls, n.line_end AS le",
            {"p": project_name, "f": file_path},
        )
        return {r["uid"]: (r["hash"] or "", r["ls"] or 0, r["le"] or 0) for r in records}

    async def upsert_file_entities(
        self,
        project_name: str,
        file_path: str,
        entities: list[ParsedEntity],
        relationships: list[ParsedRelationship],
    ) -> UpsertResult:
        """Delta-aware upsert: only write changed entities to the graph.

        Compares ``content_hash`` of new entities against stored values to
        classify each as added/modified/deleted/unchanged.  Unchanged entities
        are skipped entirely — their embed data is never touched.

        When entities are added or deleted, unchanged entities may have shifted
        line positions.  Their ``line_start``/``line_end`` are updated without
        invalidating embeddings.

        Returns an ``UpsertResult`` describing what changed.
        """
        # 1. Read old content hashes + positions
        old_data = await self.get_file_content_hashes(project_name, file_path)

        # 2. Build new hash map keyed on uid (= qualified_name including project prefix)
        new_hashes: dict[str, str] = {e.qualified_name: e.content_hash for e in entities}
        new_entity_map: dict[str, ParsedEntity] = {e.qualified_name: e for e in entities}

        old_uids = set(old_data)
        new_uids = set(new_hashes)

        added_uids = new_uids - old_uids
        deleted_uids = old_uids - new_uids
        common_uids = old_uids & new_uids
        modified_uids = {uid for uid in common_uids if old_data[uid][0] != new_hashes[uid]}
        unchanged_uids = common_uids - modified_uids

        # 3. Fast path: nothing changed → skip graph writes entirely
        if not added_uids and not deleted_uids and not modified_uids:
            logger.debug("Delta skip (no changes) for {}", file_path)
            return UpsertResult(
                unchanged=[self._strip_uid(uid) for uid in unchanged_uids],
            )

        # 4. Apply delta
        # 4a. Delete removed entity nodes
        if deleted_uids:
            await self._batch_delete_entities(list(deleted_uids))

        # 4b. Create new entity nodes
        if added_uids:
            added_entities = [new_entity_map[uid] for uid in added_uids]
            await self._batch_create_entities(project_name, added_entities)

        # 4c. Update modified entity nodes
        if modified_uids:
            modified_entities = [new_entity_map[uid] for uid in modified_uids]
            await self._batch_update_entities(modified_entities)

        # 4d. Update positions of unchanged entities that shifted
        #     (only when entity count changed — adds or deletes cause shifts)
        if unchanged_uids and (added_uids or deleted_uids):
            shifted = [
                new_entity_map[uid]
                for uid in unchanged_uids
                if (new_entity_map[uid].line_start, new_entity_map[uid].line_end)
                != (old_data[uid][1], old_data[uid][2])
            ]
            if shifted:
                await self._batch_update_positions(shifted)

        # 4e. Recreate ALL relationships for the file (delete old, create new).
        #     Relationships are cheap and context-dependent — simpler to rebuild.
        await self._recreate_file_relationships(project_name, file_path, relationships)

        result = UpsertResult(
            added=[self._strip_uid(uid) for uid in added_uids],
            modified=[self._strip_uid(uid) for uid in modified_uids],
            deleted=[self._strip_uid(uid) for uid in deleted_uids],
            unchanged=[self._strip_uid(uid) for uid in unchanged_uids],
        )

        logger.debug(
            "Upserted {} (added={}, modified={}, deleted={}, unchanged={}) for {}",
            len(entities),
            len(result.added),
            len(result.modified),
            len(result.deleted),
            len(result.unchanged),
            file_path,
        )
        return result

    @staticmethod
    def _strip_uid(uid: str) -> str:
        """Strip project prefix from uid to get qualified_name."""
        return uid.split(":", 1)[1] if ":" in uid else uid

    async def delete_file_entities(self, project_name: str, file_path: str) -> list[str]:
        """Delete all non-structural entity nodes for a file. Returns deleted uids."""
        old_data = await self.get_file_content_hashes(project_name, file_path)
        if old_data:
            await self._batch_delete_entities(list(old_data.keys()))
        return [self._strip_uid(uid) for uid in old_data]

    async def merge_project_node(self, project_name: str, **metadata: Any) -> None:
        """Create or update a Project node by uid."""
        uid = project_name
        props = {"uid": uid, "project_name": project_name, "name": project_name, **metadata}
        set_clause = ", ".join(f"n.{k} = ${k}" for k in props)
        await self.execute_write(
            f"MERGE (n:{NodeLabel.PROJECT} {{uid: $uid}}) SET {set_clause}",
            props,
        )

    async def merge_package_node(self, project_name: str, qualified_name: str, name: str, file_path: str) -> None:
        """Create or update a Package node by uid."""
        uid = f"{project_name}:{qualified_name}"
        await self.execute_write(
            f"MERGE (n:{NodeLabel.PACKAGE} {{uid: $uid}}) "
            f"SET n.project_name = $project_name, n.name = $name, "
            f"n.qualified_name = $qualified_name, n.file_path = $file_path",
            {
                "uid": uid,
                "project_name": project_name,
                "name": name,
                "qualified_name": qualified_name,
                "file_path": file_path,
            },
        )

    async def create_contains_edge(self, from_uid: str, to_uid: str) -> None:
        """Create an idempotent CONTAINS relationship between two nodes."""
        await self.execute_write(
            f"MATCH (a {{uid: $from_uid}}), (b {{uid: $to_uid}}) MERGE (a)-[:{RelType.CONTAINS}]->(b)",
            {"from_uid": from_uid, "to_uid": to_uid},
        )

    async def delete_project_data(self, project_name: str) -> None:
        """Delete all nodes belonging to a project (for full reindex)."""
        await self.execute_write(
            "MATCH (n {project_name: $project_name}) DETACH DELETE n",
            {"project_name": project_name},
        )

    async def update_project_metadata(self, project_name: str, **metadata: Any) -> None:
        """Update properties on the Project node."""
        uid = project_name
        set_clause = ", ".join(f"n.{k} = ${k}" for k in metadata)
        if not set_clause:
            return
        await self.execute_write(
            f"MATCH (n:{NodeLabel.PROJECT} {{uid: $uid}}) SET {set_clause}",
            {"uid": uid, **metadata},
        )

    async def get_project_status(self, project_name: str | None = None) -> list[dict[str, Any]]:
        """Query Project nodes for status display."""
        if project_name:
            return await self.execute(
                f"MATCH (n:{NodeLabel.PROJECT} {{uid: $uid}}) RETURN n",
                {"uid": project_name},
            )
        return await self.execute(f"MATCH (n:{NodeLabel.PROJECT}) RETURN n")

    async def get_project_git_hash(self, project_name: str) -> str | None:
        """Read stored git_hash from the Project node."""
        records = await self.execute(
            f"MATCH (n:{NodeLabel.PROJECT} {{uid: $uid}}) RETURN n.git_hash AS git_hash",
            {"uid": project_name},
        )
        if not records or records[0]["git_hash"] is None:
            return None
        return records[0]["git_hash"]

    async def get_project_file_paths(self, project_name: str) -> set[str]:
        """Return all distinct file_paths indexed for a project (non-structural nodes)."""
        records = await self.execute(
            f"MATCH (n {{project_name: $p}}) "
            f"WHERE NOT n:{NodeLabel.PACKAGE} AND NOT n:{NodeLabel.PROJECT} "
            f"AND NOT n:{NodeLabel.SCHEMA_VERSION} "
            "RETURN DISTINCT n.file_path AS fp",
            {"p": project_name},
        )
        return {r["fp"] for r in records if r["fp"]}

    async def count_entities(self, project_name: str) -> int:
        """Count all entity nodes (Module, TypeDef, Callable, Value, Package) for a project."""
        records = await self.execute(
            "MATCH (n {project_name: $project_name}) "
            f"WHERE n:{NodeLabel.MODULE} OR n:{NodeLabel.TYPE_DEF} OR n:{NodeLabel.CALLABLE} "
            f"OR n:{NodeLabel.VALUE} OR n:{NodeLabel.PACKAGE} "
            "RETURN count(n) AS cnt",
            {"project_name": project_name},
        )
        return records[0]["cnt"] if records else 0

    # -- Import resolution helpers ---------------------------------------------

    async def resolve_imports(
        self,
        project_name: str,
        import_rels: list[ParsedRelationship],
    ) -> None:
        """Resolve IMPORTS relationships after all files in a batch have been upserted.

        Classifies each import as internal (target exists in graph) or external
        (no match → create ExternalPackage/ExternalSymbol stubs), then creates
        IMPORTS edges for both.
        """
        if not import_rels:
            return

        # 1. Query all internal entity qualified_name → uid
        records = await self.execute(
            f"MATCH (n {{project_name: $p}}) "
            f"WHERE NOT n:{NodeLabel.EXTERNAL_PACKAGE} AND NOT n:{NodeLabel.EXTERNAL_SYMBOL} "
            f"AND NOT n:{NodeLabel.SCHEMA_VERSION} AND NOT n:{NodeLabel.PROJECT} "
            "RETURN n.qualified_name AS qn, n.uid AS uid",
            {"p": project_name},
        )
        internal_map: dict[str, str] = {r["qn"]: r["uid"] for r in records}

        # 2. Classify imports as internal or external
        import_edges: list[dict[str, str]] = []  # [{from_uid, to_uid}]
        ext_packages: dict[str, dict[str, str]] = {}  # top_level → {uid, name, qn, project_name}
        ext_symbols: dict[str, dict[str, str]] = {}  # dotted_path → {uid, name, qn, package, project_name}

        for rel in import_rels:
            to_name = rel.to_name
            from_uid = rel.from_qualified_name  # already project-prefixed uid

            # Check internal match
            if to_name in internal_map:
                import_edges.append({"from_uid": from_uid, "to_uid": internal_map[to_name]})
                continue

            # External import — derive top-level package
            top_level = to_name.split(".")[0]
            pkg_uid = f"{project_name}:ext/{top_level}"

            if top_level not in ext_packages:
                ext_packages[top_level] = {
                    "uid": pkg_uid,
                    "project_name": project_name,
                    "name": top_level,
                    "qualified_name": f"ext/{top_level}",
                }

            if to_name == top_level:
                # Bare package import (e.g. `import os`) → point to ExternalPackage
                import_edges.append({"from_uid": from_uid, "to_uid": pkg_uid})
            else:
                # Symbol import (e.g. `from loguru import logger`) → ExternalSymbol
                sym_uid = f"{project_name}:ext/{to_name}"
                sym_name = to_name.rsplit(".", 1)[-1]
                if to_name not in ext_symbols:
                    ext_symbols[to_name] = {
                        "uid": sym_uid,
                        "project_name": project_name,
                        "name": sym_name,
                        "qualified_name": f"ext/{to_name}",
                        "package": top_level,
                    }
                import_edges.append({"from_uid": from_uid, "to_uid": sym_uid})

        # 3. MERGE ExternalPackage nodes
        if ext_packages:
            await self.execute_write(
                f"UNWIND $packages AS pkg "
                f"MERGE (n:{NodeLabel.EXTERNAL_PACKAGE} {{uid: pkg.uid}}) "
                f"ON CREATE SET n.project_name = pkg.project_name, n.name = pkg.name, "
                f"n.qualified_name = pkg.qualified_name",
                {"packages": list(ext_packages.values())},
            )

        # 4. MERGE ExternalSymbol nodes
        if ext_symbols:
            await self.execute_write(
                f"UNWIND $symbols AS sym "
                f"MERGE (n:{NodeLabel.EXTERNAL_SYMBOL} {{uid: sym.uid}}) "
                f"ON CREATE SET n.project_name = sym.project_name, n.name = sym.name, "
                f"n.qualified_name = sym.qualified_name, n.package = sym.package",
                {"symbols": list(ext_symbols.values())},
            )

        # 5. CONTAINS edges (ExternalPackage → ExternalSymbol)
        contains_edges = [
            {"pkg_uid": f"{project_name}:ext/{sym['package']}", "sym_uid": sym["uid"]} for sym in ext_symbols.values()
        ]
        if contains_edges:
            await self.execute_write(
                f"UNWIND $edges AS e "
                f"MATCH (p:{NodeLabel.EXTERNAL_PACKAGE} {{uid: e.pkg_uid}}), "
                f"(s:{NodeLabel.EXTERNAL_SYMBOL} {{uid: e.sym_uid}}) "
                f"MERGE (p)-[:{RelType.CONTAINS}]->(s)",
                {"edges": contains_edges},
            )

        # 6. IMPORTS edges (both internal and external targets)
        if import_edges:
            await self.execute_write(
                f"UNWIND $rels AS r "
                "MATCH (a {uid: r.from_uid}), (b {uid: r.to_uid}) "
                f"CREATE (a)-[:{RelType.IMPORTS}]->(b)",
                {"rels": import_edges},
            )

        logger.debug(
            "Resolved {} imports ({} packages, {} symbols created)",
            len(import_rels),
            len(ext_packages),
            len(ext_symbols),
        )

    async def update_external_package_versions(
        self,
        project_name: str,
        versions: dict[str, str],
    ) -> None:
        """Set version properties on ExternalPackage nodes from dependency metadata."""
        if not versions:
            return
        params = [{"uid": f"{project_name}:ext/{pkg}", "version": ver} for pkg, ver in versions.items()]
        await self.execute_write(
            f"UNWIND $items AS item "
            f"MATCH (n:{NodeLabel.EXTERNAL_PACKAGE} {{uid: item.uid}}) "
            "SET n.version = item.version",
            {"items": params},
        )

    # -- Detector enrichment helpers -------------------------------------------

    async def apply_property_enrichments(self, enrichments: list[PropertyEnrichment]) -> None:
        """Apply property enrichments from detectors to existing entity nodes.

        Each enrichment SETs additional properties on a node matched by uid.
        Uses ``+=`` (map merge) so existing properties are preserved.
        """
        for enrichment in enrichments:
            if not enrichment.properties:
                continue
            await self.execute_write(
                "MATCH (n {uid: $uid}) SET n += $props",
                {"uid": enrichment.qualified_name, "props": enrichment.properties},
            )

    # -- Embedding helpers -----------------------------------------------------

    async def get_embedding_config(self) -> tuple[str, int] | None:
        """Read embedding model and dimension from the SchemaVersion node.

        Returns ``(model, dimension)`` or ``None`` if not yet configured.
        """
        records = await self.execute(
            f"MATCH (sv:{NodeLabel.SCHEMA_VERSION}) RETURN sv.embedding_model AS model, sv.embedding_dimension AS dim"
        )
        if not records or records[0]["model"] is None:
            return None
        return (records[0]["model"], records[0]["dim"])

    async def set_embedding_config(self, model: str, dimension: int) -> None:
        """Write embedding model and dimension to the SchemaVersion node."""
        await self.execute_write(
            f"MATCH (sv:{NodeLabel.SCHEMA_VERSION}) SET sv.embedding_model = $model, sv.embedding_dimension = $dim",
            {"model": model, "dim": dimension},
        )

    async def read_entity_texts(self, qualified_names: list[str]) -> list[dict[str, Any]]:
        """Batch-read entity properties needed for embedding.

        Returns list of dicts with keys: ``qualified_name``, ``name``,
        ``signature``, ``docstring``, ``kind``, ``_label``,
        ``embed_hash``, ``embedding``.
        """
        return await self.execute(
            "UNWIND $qns AS qn "
            "MATCH (n {qualified_name: qn}) "
            "RETURN n.qualified_name AS qualified_name, n.name AS name, "
            "n.signature AS signature, n.docstring AS docstring, "
            "n.kind AS kind, labels(n)[0] AS _label, "
            "n.embed_hash AS embed_hash, n.embedding AS embedding",
            {"qns": qualified_names},
        )

    async def write_embeddings(self, items: list[tuple[str, list[float]]]) -> None:
        """Batch-write embedding vectors to nodes by qualified_name."""
        if not items:
            return
        params = [{"qn": qn, "vector": vec} for qn, vec in items]
        await self.execute_write(
            "UNWIND $items AS item MATCH (n {qualified_name: item.qn}) SET n.embedding = item.vector",
            {"items": params},
        )

    async def write_embed_hashes(self, items: list[tuple[str, str]]) -> None:
        """Batch-write embed_hash values to nodes by qualified_name."""
        if not items:
            return
        params = [{"qn": qn, "hash": h} for qn, h in items]
        await self.execute_write(
            "UNWIND $items AS item MATCH (n {qualified_name: item.qn}) SET n.embed_hash = item.hash",
            {"items": params},
        )

    async def clear_all_embeddings(self) -> None:
        """Remove embedding vectors from all nodes."""
        await self.execute_write("MATCH (n) WHERE n.embedding IS NOT NULL REMOVE n.embedding")

    async def get_vector_index_info(self) -> list[dict[str, Any]]:
        """Query Memgraph for vector index metadata.

        Returns a list of dicts with keys like ``index_name``, ``label``,
        ``property``, ``dimension``, ``size``, etc.
        """
        try:
            return await self.execute("CALL vector_search.show_index_info() YIELD * RETURN *")
        except Exception as exc:
            logger.debug("Could not fetch vector index info: {}", exc)
            return []

    # -- Text (BM25) search helpers -------------------------------------------

    async def text_search(
        self,
        query: str,
        label: str = "",
        limit: int = 20,
        project: str = "",
    ) -> list[dict[str, Any]]:
        """BM25 text search across text indices.

        Queries one or all text indices, optionally post-filters by project,
        and returns results sorted by score descending.
        """
        indices = (
            [f"text_{label.lower()}"] if label else [f"text_{lbl.value.lower()}" for lbl in _TEXT_SEARCHABLE_LABELS]
        )
        fetch_limit = limit * 3 if project else limit

        all_results: list[dict[str, Any]] = []
        for index_name in indices:
            cypher = (
                f"CALL text_search.search('{index_name}', $query, {fetch_limit}) "
                "YIELD node, score "
                "RETURN node, score "
                f"ORDER BY score DESC LIMIT {fetch_limit}"
            )
            try:
                records = await self.execute(cypher, {"query": query})
                all_results.extend(records)
            except Exception as exc:
                logger.warning("Text search on {} failed: {}", index_name, exc)

        # Post-filter by project scope
        if project:
            all_results = [r for r in all_results if _node_project_name(r) == project]

        # Sort by score descending and truncate
        all_results.sort(key=lambda rec: rec.get("score", 0), reverse=True)
        return all_results[:limit]

    async def get_text_index_info(self) -> list[dict[str, Any]]:
        """Query Memgraph for text index metadata via SHOW INDEX INFO (Memgraph 3.7+ DDL).

        Filters the generic index listing to text indices (type starts with 'label_text').
        Returns a list of dicts with index_type, label, and name keys.
        """
        try:
            rows = await self.execute(
                "SHOW INDEX INFO"  # type: ignore[arg-type]
            )
            return [
                {
                    "index_type": r["index type"],
                    "label": r["label"],
                    "name": r["index type"].split("name: ")[-1].rstrip(")") if "name:" in r["index type"] else "",
                }
                for r in rows
                if str(r.get("index type", "")).startswith("label_text")
            ]
        except Exception as exc:
            logger.debug("Could not fetch text index info: {}", exc)
            return []

    # -- Vector search helpers -------------------------------------------------

    async def vector_search(
        self,
        vector: list[float],
        label: str = "",
        limit: int = 20,
        project: str = "",
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Semantic similarity search using pre-computed vector.

        Queries one or all vector indices, optionally post-filters by project
        and similarity threshold, and returns results sorted by similarity
        descending.  Returns ``[{"node": Node, "similarity": float}, ...]``.
        """
        indices = [f"vec_{label.lower()}"] if label else [f"vec_{lbl.value.lower()}" for lbl in _EMBEDDABLE_LABELS]
        filtering = bool(project) or threshold > 0.0
        fetch_limit = limit * 3 if filtering else limit

        all_results: list[dict[str, Any]] = []
        for index_name in indices:
            cypher = (
                f"CALL vector_search.search('{index_name}', {fetch_limit}, $vector) "
                "YIELD node, similarity "
                "RETURN node, similarity "
                f"ORDER BY similarity DESC LIMIT {fetch_limit}"
            )
            try:
                records = await self.execute(cypher, {"vector": vector})
                all_results.extend(records)
            except Exception as exc:
                logger.warning("Vector search on {} failed: {}", index_name, exc)

        if threshold > 0.0:
            all_results = [r for r in all_results if r.get("similarity", 0) >= threshold]
        if project:
            all_results = [r for r in all_results if _node_project_name(r) == project]

        all_results.sort(key=lambda rec: rec.get("similarity", 0), reverse=True)
        return all_results[:limit]

    # -- Graph (name-based) search helpers ------------------------------------

    async def graph_search(
        self,
        query: str,
        label: str = "",
        limit: int = 20,
        project: str = "",
    ) -> list[dict[str, Any]]:
        """Name-based graph search with scored matching.

        Three-stage matching with decreasing scores:
        - Exact name match: score 3.0
        - Suffix match (qualified_name ends with .query): score 2.0
        - Contains match (name or qualified_name contains query): score 1.0

        Deduplicates by uid, keeping highest score.
        Returns ``[{"node": Node, "score": float}, ...]``.
        """
        label_filter = f":{label}" if label else ""
        project_clause = " AND n.project_name = $project" if project else ""
        params: dict[str, Any] = {"query": query, "project": project}
        fetch_limit = limit * 3

        scored: dict[str, tuple[Any, float]] = {}

        # Stage 1: Exact name match (score 3.0)
        records = await self.execute(
            f"MATCH (n{label_filter}) WHERE n.name = $query{project_clause} RETURN n LIMIT {fetch_limit}",
            params,
        )
        for r in records:
            node = r["n"]
            uid = node.get("uid", "") if hasattr(node, "get") else ""
            if uid and (uid not in scored or scored[uid][1] < 3.0):
                scored[uid] = (node, 3.0)

        # Stage 2: Suffix match (score 2.0)
        suffix = f".{query}"
        records = await self.execute(
            f"MATCH (n{label_filter}) WHERE n.qualified_name ENDS WITH $suffix{project_clause} "
            f"RETURN n LIMIT {fetch_limit}",
            {**params, "suffix": suffix},
        )
        for r in records:
            node = r["n"]
            uid = node.get("uid", "") if hasattr(node, "get") else ""
            if uid and (uid not in scored or scored[uid][1] < 2.0):
                scored[uid] = (node, 2.0)

        # Stage 3: Contains match (score 1.0)
        records = await self.execute(
            f"MATCH (n{label_filter}) WHERE (n.qualified_name CONTAINS $query OR n.name CONTAINS $query)"
            f"{project_clause} RETURN n LIMIT {fetch_limit}",
            params,
        )
        for r in records:
            node = r["n"]
            uid = node.get("uid", "") if hasattr(node, "get") else ""
            if uid and uid not in scored:
                scored[uid] = (node, 1.0)

        # Build result list sorted by score descending
        results = [{"node": node, "score": score} for node, score in scored.values()]
        results.sort(key=lambda rec: rec["score"], reverse=True)
        return results[:limit]

    async def close(self) -> None:
        """Close the driver and release connections."""
        await self._driver.close()

    # -- Private helpers -----------------------------------------------------

    async def _batch_create_entities(self, project_name: str, entities: list[ParsedEntity]) -> None:
        """Batch-create entity nodes grouped by label."""
        sorted_entities = sorted(entities, key=attrgetter("label"))
        for label, group in groupby(sorted_entities, key=attrgetter("label")):
            entity_list = list(group)
            params = [
                {
                    "uid": e.qualified_name,
                    "project_name": project_name,
                    "name": e.name,
                    "qualified_name": (
                        e.qualified_name.split(":", 1)[1] if ":" in e.qualified_name else e.qualified_name
                    ),
                    "file_path": e.file_path,
                    "kind": e.kind,
                    "line_start": e.line_start,
                    "line_end": e.line_end,
                    "visibility": e.visibility,
                    "docstring": e.docstring,
                    "signature": e.signature,
                    "tags": e.tags,
                    "header_path": e.header_path,
                    "header_level": e.header_level,
                    "content_hash": e.content_hash,
                }
                for e in entity_list
            ]
            query = (
                f"UNWIND $entities AS e "
                f"CREATE (n:{label.value} {{"
                f"uid: e.uid, project_name: e.project_name, name: e.name, "
                f"qualified_name: e.qualified_name, file_path: e.file_path, "
                f"kind: e.kind, line_start: e.line_start, line_end: e.line_end, "
                f"visibility: e.visibility, docstring: e.docstring, "
                f"signature: e.signature, tags: e.tags, "
                f"header_path: e.header_path, header_level: e.header_level, "
                f"content_hash: e.content_hash"
                f"}})"
            )
            await self.execute_write(query, {"entities": params})

    async def _batch_update_entities(self, entities: list[ParsedEntity]) -> None:
        """Batch-update modified entity nodes by uid."""
        params = [
            {
                "uid": e.qualified_name,
                "name": e.name,
                "kind": e.kind,
                "line_start": e.line_start,
                "line_end": e.line_end,
                "visibility": e.visibility,
                "docstring": e.docstring,
                "signature": e.signature,
                "tags": e.tags,
                "header_path": e.header_path,
                "header_level": e.header_level,
                "content_hash": e.content_hash,
            }
            for e in entities
        ]
        await self.execute_write(
            "UNWIND $entities AS e "
            "MATCH (n {uid: e.uid}) "
            "SET n.name = e.name, n.kind = e.kind, "
            "n.line_start = e.line_start, n.line_end = e.line_end, "
            "n.visibility = e.visibility, n.docstring = e.docstring, "
            "n.signature = e.signature, n.tags = e.tags, "
            "n.header_path = e.header_path, n.header_level = e.header_level, "
            "n.content_hash = e.content_hash",
            {"entities": params},
        )

    async def _batch_update_positions(self, entities: list[ParsedEntity]) -> None:
        """Update only line_start/line_end for entities whose content didn't change."""
        params = [{"uid": e.qualified_name, "ls": e.line_start, "le": e.line_end} for e in entities]
        await self.execute_write(
            "UNWIND $entities AS e MATCH (n {uid: e.uid}) SET n.line_start = e.ls, n.line_end = e.le",
            {"entities": params},
        )

    async def _batch_delete_entities(self, uids: list[str]) -> None:
        """Delete entity nodes by uid."""
        await self.execute_write(
            "UNWIND $uids AS uid MATCH (n {uid: uid}) DETACH DELETE n",
            {"uids": uids},
        )

    async def _recreate_file_relationships(
        self,
        project_name: str,
        file_path: str,
        relationships: list[ParsedRelationship],
    ) -> None:
        """Delete all relationships originating from this file's entities, then recreate them."""
        # Delete existing relationships from file entities (excluding Package/Project)
        await self.execute_write(
            f"MATCH (n {{project_name: $p, file_path: $f}})-[r]->() "
            f"WHERE NOT n:{NodeLabel.PACKAGE} AND NOT n:{NodeLabel.PROJECT} "
            "DELETE r",
            {"p": project_name, "f": file_path},
        )

        # Recreate: uid-based rels (both ends known by uid)
        # Includes structural rels and detector-produced rels (both ends resolved to UIDs)
        uid_rel_types = {
            RelType.DEFINES,
            RelType.CONTAINS,
            RelType.OVERRIDES,
            RelType.TESTS,
            RelType.HANDLES_ROUTE,
            RelType.HANDLES_EVENT,
            RelType.HANDLES_COMMAND,
            RelType.REGISTERED_BY,
            RelType.INJECTED_INTO,
        }
        uid_rels = [r for r in relationships if r.rel_type in uid_rel_types]
        other_rels = [r for r in relationships if r.rel_type not in uid_rel_types]

        for rel_type, group in groupby(sorted(uid_rels, key=attrgetter("rel_type")), key=attrgetter("rel_type")):
            rels_list = list(group)
            has_props = any(r.properties for r in rels_list)
            if has_props:
                rel_params = [
                    {"from_uid": r.from_qualified_name, "to_uid": r.to_name, "props": r.properties} for r in rels_list
                ]
                await self.execute_write(
                    f"UNWIND $rels AS r MATCH (a {{uid: r.from_uid}}), (b {{uid: r.to_uid}}) "
                    f"CREATE (a)-[:{rel_type}]->(b)",
                    {"rels": rel_params},
                )
                # SET properties in a second pass (Memgraph doesn't support dynamic props in CREATE)
                for param in rel_params:
                    if param["props"]:
                        set_clause = ", ".join(f"e.{k} = $prop_{k}" for k in param["props"])
                        prop_params = {f"prop_{k}": v for k, v in param["props"].items()}
                        await self.execute_write(
                            f"MATCH (a {{uid: $from_uid}})-[e:{rel_type}]->(b {{uid: $to_uid}}) SET {set_clause}",
                            {"from_uid": param["from_uid"], "to_uid": param["to_uid"], **prop_params},
                        )
            else:
                rel_params = [{"from_uid": r.from_qualified_name, "to_uid": r.to_name} for r in rels_list]
                await self.execute_write(
                    f"UNWIND $rels AS r MATCH (a {{uid: r.from_uid}}), (b {{uid: r.to_uid}}) "
                    f"CREATE (a)-[:{rel_type}]->(b)",
                    {"rels": rel_params},
                )

        # Name-matched rels (IMPORTS are intentionally excluded — resolved post-batch)
        doc_rels = [r for r in other_rels if r.rel_type == RelType.DOCUMENTS]
        inherits_rels = [r for r in other_rels if r.rel_type == RelType.INHERITS]

        for rel in inherits_rels:
            await self.execute_write(
                f"MATCH (a {{uid: $from_uid}}), (b {{project_name: $project, name: $to_name}}) "
                f"CREATE (a)-[:{RelType.INHERITS}]->(b)",
                {"from_uid": rel.from_qualified_name, "project": project_name, "to_name": rel.to_name},
            )

        if doc_rels:
            await self._create_doc_links(project_name, doc_rels)

    async def _create_doc_links(self, project_name: str, doc_rels: list[ParsedRelationship]) -> None:
        """Create DOCUMENTS edges via batched name/path matching.

        Two batched queries: one for symbol-based links (exact name match),
        one for file-path-based links (suffix match on file_path).
        """
        symbol_params = []
        file_params = []
        for rel in doc_rels:
            props = rel.properties
            entry = {
                "from_uid": rel.from_qualified_name,
                "to_name": rel.to_name,
                "link_type": props.get("link_type", ""),
                "confidence": props.get("confidence", 0.0),
            }
            if props.get("is_file_ref"):
                file_params.append(entry)
            else:
                symbol_params.append(entry)

        created = 0
        if symbol_params:
            records = await self.execute(
                f"UNWIND $rels AS r "
                f"MATCH (a {{uid: r.from_uid}}), (b {{project_name: $project, name: r.to_name}}) "
                f"CREATE (a)-[e:{RelType.DOCUMENTS} {{link_type: r.link_type, confidence: r.confidence}}]->(b) "
                f"RETURN count(e) AS cnt",
                {"rels": symbol_params, "project": project_name},
            )
            created += records[0]["cnt"] if records else 0

        if file_params:
            records = await self.execute(
                f"UNWIND $rels AS r "
                f"MATCH (a {{uid: r.from_uid}}), (b {{project_name: $project}}) "
                f"WHERE b.file_path ENDS WITH r.to_name "
                f"CREATE (a)-[e:{RelType.DOCUMENTS} {{link_type: r.link_type, confidence: r.confidence}}]->(b) "
                f"RETURN count(e) AS cnt",
                {"rels": file_params, "project": project_name},
            )
            created += records[0]["cnt"] if records else 0

        attempted = len(doc_rels)
        if created < attempted:
            logger.debug(
                "DOCUMENTS links: {}/{} resolved for project {}",
                created,
                attempted,
                project_name,
            )

    async def _apply_full_schema(self) -> None:
        """Apply all constraints, indices, vector indices, and text indices."""
        stmts: list[str] = []
        stmts.extend(generate_unique_constraint_ddl())
        stmts.extend(generate_existence_constraint_ddl())
        stmts.extend(generate_index_ddl())
        stmts.extend(generate_vector_index_ddl(self._dimension))
        stmts.extend(generate_text_index_ddl())

        for stmt in stmts:
            await self._exec_ddl(stmt)

    async def _migrate_indices(self) -> None:
        """Drop and recreate vector/text indices (dimension may have changed)."""
        for stmt in generate_drop_vector_index_ddl():
            await self._exec_ddl(stmt)
        for stmt in generate_drop_text_index_ddl():
            await self._exec_ddl(stmt)
        for stmt in generate_vector_index_ddl(self._dimension):
            await self._exec_ddl(stmt)
        for stmt in generate_text_index_ddl():
            await self._exec_ddl(stmt)

    async def _set_schema_version(self, version: int) -> None:
        """Create or update the SchemaVersion singleton node."""
        await self.execute_write(
            f"MERGE (sv:{NodeLabel.SCHEMA_VERSION} {{version: $old_version}}) SET sv.version = $version",
            {"old_version": version, "version": version},
        )
        # Also handle fresh creation (no existing node to merge on)
        records = await self.execute(f"MATCH (sv:{NodeLabel.SCHEMA_VERSION}) RETURN count(sv) AS cnt")
        if records[0]["cnt"] == 0:
            await self.execute_write(
                f"CREATE (sv:{NodeLabel.SCHEMA_VERSION} {{version: $version}})",
                {"version": version},
            )

    async def _exec_ddl(self, stmt: str) -> None:
        """Execute a DDL statement, ignoring 'already exists' / 'doesn't exist' errors."""
        try:
            await self.execute_write(stmt)
        except Exception as exc:
            msg = str(exc).lower()
            # Memgraph raises errors for duplicate constraints/indices and missing drops
            if "already exists" in msg or "doesn't exist" in msg or "not found" in msg:
                logger.debug("DDL skipped (idempotent): {}", stmt.rstrip(";"))
            else:
                raise
