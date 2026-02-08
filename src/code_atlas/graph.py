"""Async Memgraph client for Code Atlas.

Handles connection lifecycle, schema application, and version management.
Uses the neo4j async driver (Bolt protocol) which is compatible with Memgraph.
"""

from __future__ import annotations

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

    from code_atlas.parser import ParsedEntity, ParsedRelationship
    from code_atlas.settings import AtlasSettings


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

    async def upsert_file_entities(
        self,
        project_name: str,
        file_path: str,
        entities: list[ParsedEntity],
        relationships: list[ParsedRelationship],
    ) -> None:
        """Replace all entities for a file and re-create relationships.

        Strategy: delete-and-recreate per file (simple v1 — delta indexing
        comes in epic 05-delta).

        Preserves ``embed_hash`` and ``embedding`` for entities whose
        ``qualified_name`` survives the upsert cycle.
        """
        # 0. Save embed data from existing nodes (before delete)
        old_embeds = await self.execute(
            "MATCH (n {project_name: $p, file_path: $f}) "
            "WHERE n.embed_hash IS NOT NULL "
            "RETURN n.qualified_name AS qn, n.embed_hash AS hash, n.embedding AS vec",
            {"p": project_name, "f": file_path},
        )
        embed_lookup: dict[str, tuple[str, list[float] | None]] = {r["qn"]: (r["hash"], r["vec"]) for r in old_embeds}

        # 1. Delete existing entity nodes for this (project_name, file_path).
        #    Exclude structural nodes (Package, Project) — they share the same
        #    file_path as __init__.py modules but are managed by the hierarchy step.
        await self.execute_write(
            f"MATCH (n {{project_name: $project_name, file_path: $file_path}}) "
            f"WHERE NOT n:{NodeLabel.PACKAGE} AND NOT n:{NodeLabel.PROJECT} "
            "DETACH DELETE n",
            {"project_name": project_name, "file_path": file_path},
        )

        # 2. Batch-create nodes grouped by label
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
                f"header_path: e.header_path, header_level: e.header_level"
                f"}})"
            )
            await self.execute_write(query, {"entities": params})  # dynamic Cypher

        # 3. Create relationships
        # uid-based rels: both ends are in this file
        uid_rel_types = {RelType.DEFINES, RelType.CONTAINS}
        uid_rels = [r for r in relationships if r.rel_type in uid_rel_types]
        other_rels = [r for r in relationships if r.rel_type not in uid_rel_types]

        # Batch-create uid-based rels grouped by type
        for rel_type, group in groupby(sorted(uid_rels, key=attrgetter("rel_type")), key=attrgetter("rel_type")):
            rel_params = [{"from_uid": r.from_qualified_name, "to_uid": r.to_name} for r in group]
            await self.execute_write(
                f"UNWIND $rels AS r MATCH (a {{uid: r.from_uid}}), (b {{uid: r.to_uid}}) CREATE (a)-[:{rel_type}]->(b)",
                {"rels": rel_params},
            )

        # Other relationships: best-effort name matching within the project
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

        # 4. Restore embed data for entities whose qualified_name survived
        if embed_lookup:
            new_qns = {
                e.qualified_name.split(":", 1)[1] if ":" in e.qualified_name else e.qualified_name for e in entities
            }
            restore = [
                {"qn": qn, "hash": h, "vec": v}
                for qn, (h, v) in embed_lookup.items()
                if qn in new_qns and v is not None
            ]
            if restore:
                await self.execute_write(
                    "UNWIND $items AS item "
                    "MATCH (n {qualified_name: item.qn, project_name: $p, file_path: $f}) "
                    "SET n.embed_hash = item.hash, n.embedding = item.vec",
                    {"items": restore, "p": project_name, "f": file_path},
                )

        logger.debug(
            "Upserted {} entities and {} relationships for {}",
            len(entities),
            len(relationships),
            file_path,
        )

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
