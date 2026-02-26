"""Async Memgraph client for Code Atlas.

Handles connection lifecycle, schema application, and version management.
Uses the neo4j async driver (Bolt protocol) which is compatible with Memgraph.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from operator import attrgetter
from typing import TYPE_CHECKING, Any

from loguru import logger
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import TransientError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from code_atlas.schema import (
    _EMBEDDABLE_LABELS,
    _ENTITY_LABELS,
    _TEXT_SEARCHABLE_LABELS,
    SCHEMA_VERSION,
    NodeLabel,
    RelType,
    generate_composite_index_ddl,
    generate_drop_text_index_ddl,
    generate_drop_vector_index_ddl,
    generate_existence_constraint_ddl,
    generate_index_ddl,
    generate_text_index_ddl,
    generate_unique_constraint_ddl,
    generate_vector_index_ddl,
)
from code_atlas.telemetry import get_tracer

if TYPE_CHECKING:
    from neo4j import AsyncDriver

    from code_atlas.parsing.ast import ParsedEntity, ParsedRelationship
    from code_atlas.parsing.detectors import PropertyEnrichment
    from code_atlas.settings import AtlasSettings

_tracer = get_tracer(__name__)

_VALID_LABELS: frozenset[str] = frozenset(lbl.value for lbl in NodeLabel)

# Entity labels that can appear as file-level nodes (excluding Package, Project).
_FILE_ENTITY_LABELS: tuple[NodeLabel, ...] = tuple(
    sorted(
        (lbl for lbl in _ENTITY_LABELS if lbl not in {NodeLabel.PACKAGE, NodeLabel.PROJECT}),
        key=lambda lbl: lbl.value,
    )
)


def _assert_valid_label(label: str) -> None:
    """Raise ValueError if *label* is not a known NodeLabel value.

    Prevents Cypher injection when labels are interpolated into queries.
    """
    if label not in _VALID_LABELS:
        msg = f"Invalid node label: {label!r}"
        raise ValueError(msg)


def _build_graph_search_query(
    label: str,
    project_clause: str,
    fetch_limit: int,
) -> str:
    """Build a UNION ALL Cypher query for the 3-stage graph search cascade.

    Collapses exact / suffix / contains matching into a single round-trip.
    """
    label_filter = f":{label}" if label else ""

    return (
        f"MATCH (n{label_filter}) WHERE n.name = $query{project_clause} "
        f"RETURN n AS node, 3.0 AS score LIMIT {fetch_limit} "
        f"UNION ALL "
        f"MATCH (n{label_filter}) WHERE n.qualified_name ENDS WITH $suffix{project_clause} "
        f"RETURN n AS node, 2.0 AS score LIMIT {fetch_limit} "
        f"UNION ALL "
        f"MATCH (n{label_filter}) WHERE (n.qualified_name CONTAINS $query OR n.name CONTAINS $query)"
        f"{project_clause} RETURN n AS node, 1.0 AS score LIMIT {fetch_limit}"
    )


class QueryTimeoutError(Exception):
    """Raised when a read query exceeds the configured timeout."""

    def __init__(self, timeout_s: float, query_prefix: str = "") -> None:
        self.timeout_s = timeout_s
        self.query_prefix = query_prefix
        super().__init__(f"Query timed out after {timeout_s}s: {query_prefix}")


@dataclass(frozen=True)
class UpsertResult:
    """Result of a delta-aware upsert for a single file."""

    added: list[str] = field(default_factory=list)  # qualified_names of new entities
    modified: list[str] = field(default_factory=list)  # qualified_names with changed content_hash
    deleted: list[str] = field(default_factory=list)  # qualified_names removed from file
    unchanged: list[str] = field(default_factory=list)  # qualified_names with matching content_hash
    modified_significance: dict[str, str] = field(default_factory=dict)  # qualified_name → Significance value


def _node_project_name(record: dict[str, Any]) -> str:
    """Extract project_name from a record containing a neo4j Node."""
    node = record.get("node") or record.get("n")
    if node is None:
        return ""
    if hasattr(node, "get"):
        return node.get("project_name", "")
    return ""


@dataclass(frozen=True)
class _CallLookup:
    """Pre-built lookup tables for CALLS resolution."""

    name_to_callables: dict[str, list[tuple[str, str, str]]]  # name → [(uid, file_path, vis)]
    import_map: dict[str, dict[str, str]]  # module_uid → {imported_name: target_uid}
    caller_to_parent: dict[str, str]  # callable_uid → parent TypeDef uid
    parent_children: dict[str, list[str]]  # parent_uid → [child_uids]
    uid_to_info: dict[str, tuple[str, str]]  # uid → (name, file_path)


def _resolve_one_call(project_name: str, rel: ParsedRelationship, lk: _CallLookup) -> str | None:
    """Resolve a single CALLS relationship to a target uid (or ``None``)."""
    caller_uid = rel.from_qualified_name
    bare_name = rel.to_name

    # Derive caller's module uid — find the longest module prefix in import_map
    caller_qn = caller_uid.split(":", 1)[1] if ":" in caller_uid else caller_uid
    parts = caller_qn.split(".")
    module_uid: str | None = None
    for i in range(len(parts) - 1, 0, -1):
        candidate = f"{project_name}:{'.'.join(parts[:i])}"
        if candidate in lk.import_map:
            module_uid = candidate
            break

    # Strategy 1: Import match
    if module_uid and bare_name in lk.import_map.get(module_uid, {}):
        return lk.import_map[module_uid][bare_name]

    # Strategy 2: Same-class sibling
    if caller_uid in lk.caller_to_parent:
        parent_uid = lk.caller_to_parent[caller_uid]
        for sibling_uid in lk.parent_children.get(parent_uid, []):
            if sibling_uid == caller_uid:
                continue
            sib_info = lk.uid_to_info.get(sibling_uid)
            if sib_info and sib_info[0] == bare_name:
                return sibling_uid

    # Strategy 3: Same-file match
    caller_info = lk.uid_to_info.get(caller_uid)
    caller_fp = caller_info[1] if caller_info else ""
    if caller_fp:
        for uid, fp, _vis in lk.name_to_callables.get(bare_name, []):
            if fp == caller_fp and uid != caller_uid:
                return uid

    # Strategy 4: Project-wide unique match — only when exactly 1 candidate exists.
    # Ambiguous names (run, close, get) are left unresolved to avoid false positives
    # from external attribute calls like asyncio.run(), session.run(), etc.
    candidates = lk.name_to_callables.get(bare_name, [])
    non_self = [uid for uid, _fp, _vis in candidates if uid != caller_uid]
    return non_self[0] if len(non_self) == 1 else None


@dataclass(frozen=True)
class CallStats:
    """Caller/callee statistics for a single entity."""

    caller_count: int = 0
    callee_count: int = 0
    caller_names: list[str] = field(default_factory=list)
    callee_names: list[str] = field(default_factory=list)


# Per-task transaction context — prevents leaking between concurrent asyncio tasks.
_active_tx_var: contextvars.ContextVar[Any] = contextvars.ContextVar("_active_tx_var", default=None)


class GraphClient:
    """Async Memgraph client wrapping the neo4j Bolt driver.

    Follows the same lifecycle pattern as EventBus: construct → ping → use → close.
    """

    def __init__(self, settings: AtlasSettings) -> None:
        mg = settings.memgraph
        self._uri = f"bolt://{mg.host}:{mg.port}"
        auth = (mg.username, mg.password) if mg.username else None
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(self._uri, auth=auth)
        self._dimension = settings.embeddings.dimension or 768
        self._embeddings_enabled = settings.embeddings.enabled
        self._query_timeout_s = mg.query_timeout_s
        self._write_timeout_s = mg.write_timeout_s

    async def ping(self) -> bool:
        """Health check — returns True if Memgraph is reachable."""
        records = await self.execute("RETURN 1 AS n")
        return len(records) == 1 and records[0]["n"] == 1

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a read query and return results as a list of dicts."""
        active_tx = _active_tx_var.get()
        if active_tx is not None:
            result = await active_tx.run(query, params or {})
            return [dict(record) async for record in result]
        with _tracer.start_as_current_span("graph.execute", attributes={"db.statement": query[:200]}):
            try:
                return await asyncio.wait_for(self._execute_inner(query, params), timeout=self._query_timeout_s)
            except TimeoutError:
                raise QueryTimeoutError(self._query_timeout_s, query[:120]) from None

    async def _execute_inner(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Inner execute without timeout — used by ``execute()``."""
        async with self._driver.session() as session:
            result = await session.run(query, params or {})  # type: ignore[arg-type]  # dynamic Cypher
            return [dict(record) async for record in result]

    async def execute_write(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Execute a write query with automatic retry on transient conflicts.

        Consumes the result to ensure server-side errors (e.g. constraint
        violations) are raised instead of being silently dropped.

        When called inside a managed transaction (``_active_tx_var`` is set),
        errors propagate directly — the managed transaction handles retries.
        """
        active_tx = _active_tx_var.get()
        if active_tx is not None:
            result = await active_tx.run(query, params or {})
            await result.consume()
            return
        await self._execute_write_with_retry(query, params)

    @retry(
        retry=retry_if_exception_type(TransientError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=2),
        before_sleep=lambda rs: logger.warning(
            "Transient conflict, retrying {} in {:.1f}s (attempt {}): {}",
            rs.fn.__qualname__,
            rs.next_action.sleep,
            rs.attempt_number,
            rs.outcome.exception(),
        ),
        reraise=True,
    )
    async def _execute_write_with_retry(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Standalone write with retry + timeout (not used inside managed transactions)."""
        with _tracer.start_as_current_span("graph.execute_write", attributes={"db.statement": query[:200]}):
            try:
                await asyncio.wait_for(self._execute_write_inner(query, params), timeout=self._write_timeout_s)
            except TimeoutError:
                raise QueryTimeoutError(self._write_timeout_s, query[:120]) from None

    async def _execute_write_inner(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Inner execute_write without timeout."""
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

    async def get_file_content_hashes(
        self, project_name: str, file_path: str
    ) -> dict[str, tuple[str, int, int, str | None, str | None, str]]:
        """Return ``{uid: (hash, line_start, line_end, sig, doc, label)}`` for non-structural nodes."""
        records = await self.execute(
            f"MATCH (n {{project_name: $p, file_path: $f}}) "
            f"WHERE NOT n:{NodeLabel.PACKAGE} AND NOT n:{NodeLabel.PROJECT} "
            "RETURN n.uid AS uid, n.content_hash AS hash, n.line_start AS ls, n.line_end AS le, "
            "n.signature AS sig, n.docstring AS doc, labels(n)[0] AS lbl",
            {"p": project_name, "f": file_path},
        )
        return {
            r["uid"]: (r["hash"] or "", r["ls"] or 0, r["le"] or 0, r["sig"], r["doc"], r["lbl"] or "") for r in records
        }

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

        All graph reads and writes run inside a single managed transaction
        for atomicity and reduced session overhead.  ``session.execute_write``
        auto-retries on ``TransientError`` (MVCC conflicts).

        Returns an ``UpsertResult`` describing what changed.
        """
        # Pre-compute new hash map outside the transaction (pure Python)
        new_hashes: dict[str, str] = {e.qualified_name: e.content_hash for e in entities}
        new_entity_map: dict[str, ParsedEntity] = {e.qualified_name: e for e in entities}

        async def _tx_fn(tx: Any) -> UpsertResult:
            token = _active_tx_var.set(tx)
            try:
                return await self._upsert_file_entities_inner(
                    project_name, file_path, entities, relationships, new_hashes, new_entity_map
                )
            finally:
                _active_tx_var.reset(token)

        async with self._driver.session() as session:
            return await session.execute_write(_tx_fn)

    async def _upsert_file_entities_inner(
        self,
        project_name: str,
        file_path: str,
        entities: list[ParsedEntity],
        relationships: list[ParsedRelationship],
        new_hashes: dict[str, str],
        new_entity_map: dict[str, ParsedEntity],
    ) -> UpsertResult:
        """Core upsert logic — runs inside a managed transaction via ``_active_tx_var``."""
        # 1. Read old content hashes + positions (routes through _active_tx_var)
        old_data = await self.get_file_content_hashes(project_name, file_path)

        old_uids = set(old_data)
        new_uids = set(new_hashes)

        added_uids = new_uids - old_uids
        deleted_uids = old_uids - new_uids
        common_uids = old_uids & new_uids
        modified_uids = {uid for uid in common_uids if old_data[uid][0] != new_hashes[uid]}
        unchanged_uids = common_uids - modified_uids

        # 2. Fast path: nothing changed → skip graph writes entirely
        if not added_uids and not deleted_uids and not modified_uids:
            logger.debug("Delta skip (no changes) for {}", file_path)
            return UpsertResult(
                unchanged=[self._strip_uid(uid) for uid in unchanged_uids],
            )

        # 3. Apply delta (all writes route through _active_tx_var)
        # 3a. Delete removed entity nodes
        if deleted_uids:
            uids_by_label: dict[str, list[str]] = defaultdict(list)
            for uid in deleted_uids:
                uids_by_label[old_data[uid][5]].append(uid)
            await self._batch_delete_entities(dict(uids_by_label))

        # 3b. Create new entity nodes
        if added_uids:
            added_entities = [new_entity_map[uid] for uid in added_uids]
            await self._batch_create_entities(project_name, added_entities)

        # 3c. Update modified entity nodes + compute per-entity significance
        mod_significance: dict[str, str] = {}
        if modified_uids:
            modified_entities = [new_entity_map[uid] for uid in modified_uids]
            await self._batch_update_entities(modified_entities)

            for uid in modified_uids:
                old_sig, old_doc = old_data[uid][3], old_data[uid][4]
                new_entity = new_entity_map[uid]
                qn = self._strip_uid(uid)
                if (new_entity.signature or "") != (old_sig or ""):
                    mod_significance[qn] = "HIGH"
                elif (new_entity.docstring or "") != (old_doc or ""):
                    mod_significance[qn] = "MODERATE"
                else:
                    mod_significance[qn] = "HIGH"

        # 3d. Update positions of unchanged entities that shifted
        if unchanged_uids and (added_uids or deleted_uids):
            shifted = [
                new_entity_map[uid]
                for uid in unchanged_uids
                if (new_entity_map[uid].line_start, new_entity_map[uid].line_end)
                != (old_data[uid][1], old_data[uid][2])
            ]
            if shifted:
                await self._batch_update_positions(shifted)

        # 3e. Recreate ALL relationships for the file (delete old, create new).
        uid_label_map = {e.qualified_name: e.label.value for e in entities}
        await self._recreate_file_relationships(
            project_name, file_path, relationships, uid_to_label=uid_label_map, skip_delete=not old_data
        )

        result = UpsertResult(
            added=[self._strip_uid(uid) for uid in added_uids],
            modified=[self._strip_uid(uid) for uid in modified_uids],
            deleted=[self._strip_uid(uid) for uid in deleted_uids],
            unchanged=[self._strip_uid(uid) for uid in unchanged_uids],
            modified_significance=mod_significance,
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
            uids_by_label: dict[str, list[str]] = defaultdict(list)
            for uid, data in old_data.items():
                uids_by_label[data[5]].append(uid)
            await self._batch_delete_entities(dict(uids_by_label))
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

    async def merge_package_batch(
        self,
        project_name: str,
        packages: list[tuple[str, str, str]],
    ) -> None:
        """Create/update Package nodes and CONTAINS edges in two batched queries.

        Each tuple is ``(qualified_name, name, file_path)``.  Parent UIDs are
        derived from the dotted *qualified_name* prefix; top-level packages
        point to the Project node.
        """
        if not packages:
            return

        params = []
        for qn, name, fp in packages:
            uid = f"{project_name}:{qn}"
            parent_qn = qn.rsplit(".", 1)[0] if "." in qn else None
            parent_uid = f"{project_name}:{parent_qn}" if parent_qn else project_name
            params.append(
                {
                    "uid": uid,
                    "project_name": project_name,
                    "name": name,
                    "qualified_name": qn,
                    "file_path": fp,
                    "parent_uid": parent_uid,
                }
            )

        await self.execute_write(
            f"UNWIND $pkgs AS p "
            f"MERGE (n:{NodeLabel.PACKAGE} {{uid: p.uid}}) "
            f"SET n.project_name = p.project_name, n.name = p.name, "
            f"n.qualified_name = p.qualified_name, n.file_path = p.file_path",
            {"pkgs": params},
        )
        await self.execute_write(
            f"UNWIND $pkgs AS p "
            f"MATCH (parent {{uid: p.parent_uid}}), (child {{uid: p.uid}}) "
            f"MERGE (parent)-[:{RelType.CONTAINS}]->(child)",
            {"pkgs": params},
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
        """Return all distinct file_paths indexed for a project.

        Includes Package nodes (from ``__init__.py``) so delta detection
        doesn't treat them as newly added on every re-index.
        """
        records = await self.execute(
            f"MATCH (n {{project_name: $p}}) "
            f"WHERE NOT n:{NodeLabel.PROJECT} AND NOT n:{NodeLabel.SCHEMA_VERSION} "
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

    async def resolve_imports(  # noqa: PLR0912
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
        import_edges: list[dict[str, Any]] = []  # [{from_uid, to_uid, type_only?}]
        ext_packages: dict[str, dict[str, str]] = {}  # top_level → {uid, name, qn, project_name}
        ext_symbols: dict[str, dict[str, str]] = {}  # dotted_path → {uid, name, qn, package, project_name}

        for rel in import_rels:
            to_name = rel.to_name
            from_uid = rel.from_qualified_name  # already project-prefixed uid
            is_type_only = rel.properties.get("type_only", False)

            # Check internal match
            if to_name in internal_map:
                edge: dict[str, Any] = {"from_uid": from_uid, "to_uid": internal_map[to_name]}
                if is_type_only:
                    edge["type_only"] = True
                import_edges.append(edge)
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
                edge = {"from_uid": from_uid, "to_uid": pkg_uid}
                if is_type_only:
                    edge["type_only"] = True
                import_edges.append(edge)
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
                edge = {"from_uid": from_uid, "to_uid": sym_uid}
                if is_type_only:
                    edge["type_only"] = True
                import_edges.append(edge)

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
        #    Split into type_only and normal for efficient batching
        normal_edges = [e for e in import_edges if not e.get("type_only")]
        type_only_edges = [e for e in import_edges if e.get("type_only")]

        if normal_edges:
            await self.execute_write(
                f"UNWIND $rels AS r "
                f"MATCH (a:{NodeLabel.MODULE} {{uid: r.from_uid}}), (b {{uid: r.to_uid}}) "
                f"MERGE (a)-[:{RelType.IMPORTS}]->(b)",
                {"rels": normal_edges},
            )
        if type_only_edges:
            await self.execute_write(
                f"UNWIND $rels AS r "
                f"MATCH (a:{NodeLabel.MODULE} {{uid: r.from_uid}}), (b {{uid: r.to_uid}}) "
                f"MERGE (a)-[e:{RelType.IMPORTS}]->(b) ON CREATE SET e.type_only = true",
                {"rels": type_only_edges},
            )

        logger.debug(
            "Resolved {} imports ({} packages, {} symbols created)",
            len(import_rels),
            len(ext_packages),
            len(ext_symbols),
        )

    async def resolve_calls(
        self,
        project_name: str,
        call_rels: list[ParsedRelationship],
        *,
        lookup: _CallLookup | None = None,
    ) -> None:
        """Resolve CALLS relationships after all files in a batch have been upserted.

        Each call rel has a bare name (e.g. ``"some_func"``) as ``to_name``.
        Resolution strategy (in priority order):
        1. **Import match** — caller's module imports something with that name.
        2. **Same-class sibling** — if caller is a method, check siblings in same TypeDef.
        3. **Same-file match** — any Callable with that name in the same file.
        4. **Project-wide match** — any Callable with that name (prefer public).
        5. **Unresolved** — skip silently (builtins, dynamic calls).
        """
        if not call_rels:
            return

        if lookup is None:
            lookup = await self._build_call_lookup(project_name)

        # Resolve each call
        edges: set[tuple[str, str]] = set()
        resolved = 0
        unresolved = 0
        for rel in call_rels:
            target_uid = _resolve_one_call(project_name, rel, lookup)
            if target_uid is not None:
                edges.add((rel.from_qualified_name, target_uid))
                resolved += 1
            else:
                unresolved += 1

        # Batch-create CALLS edges
        if edges:
            edge_params = [{"f": f, "t": t} for f, t in edges]
            await self.execute_write(
                f"UNWIND $rels AS r "
                f"MATCH (a:{NodeLabel.CALLABLE} {{uid: r.f}}), (b:{NodeLabel.CALLABLE} {{uid: r.t}}) "
                f"MERGE (a)-[:{RelType.CALLS}]->(b)",
                {"rels": edge_params},
            )

        logger.debug("Resolved {} CALLS edges ({} unresolved)", resolved, unresolved)

    async def resolve_type_refs(  # noqa: PLR0912
        self,
        project_name: str,
        type_rels: list[ParsedRelationship],
        *,
        lookup: _CallLookup | None = None,
        name_to_typedefs: dict[str, list[tuple[str, str]]] | None = None,
    ) -> None:
        """Resolve USES_TYPE relationships after all files in a batch have been upserted.

        Each type rel has a bare name (e.g. ``"MyClass"``) as ``to_name``.
        Resolution strategy:
        1. **Import match** — caller's module imports something with that name.
        2. **Same-file TypeDef** — a TypeDef with that name in the same file.
        3. **Project-wide TypeDef** — any TypeDef with that name (unique only).
        4. **Unresolved** — skip silently (builtins, generic types).
        """
        if not type_rels:
            return

        if lookup is None:
            lookup = await self._build_call_lookup(project_name)

        if name_to_typedefs is None:
            td_records = await self.execute(
                f"MATCH (n:{NodeLabel.TYPE_DEF} {{project_name: $p}}) "
                "RETURN n.name AS name, n.uid AS uid, n.file_path AS fp",
                {"p": project_name},
            )
            name_to_typedefs = {}
            for r in td_records:
                name_to_typedefs.setdefault(r["name"], []).append((r["uid"], r["fp"] or ""))

        edges: set[tuple[str, str]] = set()
        resolved = 0
        for rel in type_rels:
            from_uid = rel.from_qualified_name
            type_name = rel.to_name

            # Derive caller's file_path for same-file matching
            caller_info = lookup.uid_to_info.get(from_uid)
            caller_fp = caller_info[1] if caller_info else ""

            # Derive module uid for import matching
            caller_qn = from_uid.split(":", 1)[1] if ":" in from_uid else from_uid
            parts = caller_qn.split(".")
            module_uid: str | None = None
            for i in range(len(parts) - 1, 0, -1):
                candidate = f"{project_name}:{'.'.join(parts[:i])}"
                if candidate in lookup.import_map:
                    module_uid = candidate
                    break

            target_uid: str | None = None

            # Strategy 1: Import match
            if module_uid and type_name in lookup.import_map.get(module_uid, {}):
                target_uid = lookup.import_map[module_uid][type_name]

            # Strategy 2: Same-file TypeDef
            if target_uid is None and caller_fp:
                for uid, fp in name_to_typedefs.get(type_name, []):
                    if fp == caller_fp:
                        target_uid = uid
                        break

            # Strategy 3: Project-wide unique TypeDef
            if target_uid is None:
                candidates = name_to_typedefs.get(type_name, [])
                if len(candidates) == 1:
                    target_uid = candidates[0][0]

            if target_uid is not None:
                edges.add((from_uid, target_uid))
                resolved += 1

        if edges:
            edge_params = [{"f": f, "t": t} for f, t in edges]
            await self.execute_write(
                f"UNWIND $rels AS r "
                f"MATCH (a {{uid: r.f}}), (b:{NodeLabel.TYPE_DEF} {{uid: r.t}}) "
                f"MERGE (a)-[:{RelType.USES_TYPE}]->(b)",
                {"rels": edge_params},
            )

        logger.debug("Resolved {} USES_TYPE edges", resolved)

    async def _build_call_lookup(self, project_name: str) -> _CallLookup:
        """Build lookup tables needed for CALLS resolution."""
        # name → [(uid, file_path, visibility)]
        name_records = await self.execute(
            f"MATCH (n:{NodeLabel.CALLABLE} {{project_name: $p}}) "
            "RETURN n.name AS name, n.uid AS uid, n.file_path AS fp, n.visibility AS vis",
            {"p": project_name},
        )
        name_to_callables: dict[str, list[tuple[str, str, str]]] = {}
        uid_to_info: dict[str, tuple[str, str]] = {}
        for r in name_records:
            name_to_callables.setdefault(r["name"], []).append((r["uid"], r["fp"] or "", r["vis"] or "public"))
            uid_to_info[r["uid"]] = (r["name"], r["fp"] or "")

        # module_uid → {imported_name: target_uid}
        import_records = await self.execute(
            f"MATCH (m:{NodeLabel.MODULE} {{project_name: $p}})-[:{RelType.IMPORTS}]->(t) "
            "RETURN m.uid AS mod_uid, t.name AS name, t.uid AS uid",
            {"p": project_name},
        )
        import_map: dict[str, dict[str, str]] = {}
        for r in import_records:
            import_map.setdefault(r["mod_uid"], {})[r["name"]] = r["uid"]

        # caller_uid → parent TypeDef uid, parent → children
        parent_records = await self.execute(
            f"MATCH (td:{NodeLabel.TYPE_DEF} {{project_name: $p}})-[:{RelType.DEFINES}]->(c:{NodeLabel.CALLABLE}) "
            "RETURN td.uid AS td_uid, c.uid AS c_uid",
            {"p": project_name},
        )
        caller_to_parent: dict[str, str] = {}
        parent_children: dict[str, list[str]] = {}
        for r in parent_records:
            caller_to_parent[r["c_uid"]] = r["td_uid"]
            parent_children.setdefault(r["td_uid"], []).append(r["c_uid"])

        return _CallLookup(
            name_to_callables=name_to_callables,
            import_map=import_map,
            caller_to_parent=caller_to_parent,
            parent_children=parent_children,
            uid_to_info=uid_to_info,
        )

    async def build_resolution_lookup(self, project_name: str) -> tuple[_CallLookup, dict[str, list[tuple[str, str]]]]:
        """Build shared lookup tables for both CALLS and USES_TYPE resolution.

        Returns ``(call_lookup, name_to_typedefs)`` where *name_to_typedefs*
        maps ``name → [(uid, file_path)]`` for TypeDef nodes.  Building both
        in a single call saves 3 redundant graph queries vs. calling
        ``_build_call_lookup`` twice.
        """
        lookup = await self._build_call_lookup(project_name)

        td_records = await self.execute(
            f"MATCH (n:{NodeLabel.TYPE_DEF} {{project_name: $p}}) "
            "RETURN n.name AS name, n.uid AS uid, n.file_path AS fp",
            {"p": project_name},
        )
        name_to_typedefs: dict[str, list[tuple[str, str]]] = {}
        for r in td_records:
            name_to_typedefs.setdefault(r["name"], []).append((r["uid"], r["fp"] or ""))

        return lookup, name_to_typedefs

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

    # -- Cross-project import resolution helpers --------------------------------

    async def _resolve_cross_project_read_phase(
        self,
        project_names: list[str],
        pkg_to_project: dict[str, str],
    ) -> tuple[list[dict[str, str]], dict[str, str], dict[str, str]]:
        """Batch-read stubs and resolve real entities for cross-project imports.

        Returns ``(matched_eps, sym_to_real, ep_to_real_pkg)`` where
        *matched_eps* are the ExternalPackage stubs that matched a sibling
        package, *sym_to_real* maps ExternalSymbol uid → real entity uid, and
        *ep_to_real_pkg* maps ExternalPackage uid → real Package uid.
        """
        # Fetch ALL ExternalPackage stubs across all projects in one query
        all_ext_pkgs = await self.execute(
            f"MATCH (ep:{NodeLabel.EXTERNAL_PACKAGE}) "
            "WHERE ep.project_name IN $projects "
            "RETURN ep.name AS name, ep.uid AS uid, ep.project_name AS proj",
            {"projects": project_names},
        )

        # Filter to those matching a sibling package (not self)
        matched_eps: list[dict[str, str]] = [
            {"name": ep["name"], "uid": ep["uid"], "proj": ep["proj"], "target": target}
            for ep in all_ext_pkgs
            if (target := pkg_to_project.get(ep["name"])) is not None and target != ep["proj"]
        ]
        if not matched_eps:
            return [], {}, {}

        # Build fast lookup: (proj, pkg_name) → target_project
        ep_target_map: dict[tuple[str, str], str] = {(ep["proj"], ep["name"]): ep["target"] for ep in matched_eps}

        # Fetch ALL ExternalSymbol stubs for matched packages in one query
        ep_keys = [{"proj": ep["proj"], "pkg": ep["name"]} for ep in matched_eps]
        all_ext_syms = await self.execute(
            f"UNWIND $keys AS k "
            f"MATCH (es:{NodeLabel.EXTERNAL_SYMBOL} {{project_name: k.proj, package: k.pkg}}) "
            "RETURN es.name AS name, es.uid AS uid, "
            "es.project_name AS proj, es.package AS pkg",
            {"keys": ep_keys},
        )

        # Build lookup pairs for bulk entity resolution
        lookup_pairs = [
            {"name": es["name"], "target_project": target, "es_uid": es["uid"]}
            for es in all_ext_syms
            if (target := ep_target_map.get((es["proj"], es["pkg"])))
        ]

        pkg_rewire = [
            {"ep_uid": ep["uid"], "pkg_name": ep["name"], "target_project": ep["target"]} for ep in matched_eps
        ]

        # Bulk-resolve real entities for ExternalSymbols
        sym_to_real: dict[str, str] = {}
        if lookup_pairs:
            real_matches = await self.execute(
                "UNWIND $pairs AS p "
                "MATCH (n {project_name: p.target_project, name: p.name}) "
                f"WHERE NOT n:{NodeLabel.EXTERNAL_PACKAGE} AND NOT n:{NodeLabel.EXTERNAL_SYMBOL} "
                f"AND NOT n:{NodeLabel.PROJECT} AND NOT n:{NodeLabel.SCHEMA_VERSION} "
                "RETURN p.es_uid AS es_uid, n.uid AS real_uid LIMIT 1",
                {"pairs": lookup_pairs},
            )
            sym_to_real = {m["es_uid"]: m["real_uid"] for m in real_matches}

        # Bulk-resolve real Package nodes for bare package imports
        ep_to_real_pkg: dict[str, str] = {}
        if pkg_rewire:
            real_pkgs = await self.execute(
                "UNWIND $pairs AS p "
                f"MATCH (pkg:{NodeLabel.PACKAGE} {{project_name: p.target_project, name: p.pkg_name}}) "
                "RETURN p.ep_uid AS ep_uid, pkg.uid AS real_uid LIMIT 1",
                {"pairs": pkg_rewire},
            )
            ep_to_real_pkg = {m["ep_uid"]: m["real_uid"] for m in real_pkgs}

        return matched_eps, sym_to_real, ep_to_real_pkg

    async def resolve_cross_project_imports(self, project_names: list[str]) -> int:
        """Rewire ExternalPackage/ExternalSymbol stubs that match real entities in sibling projects.

        For each project, finds ExternalPackage stubs whose name matches a Package
        in a sibling project, then rewires IMPORTS edges from ExternalSymbol stubs
        to the real entity. Orphaned stubs (no remaining inbound edges) are deleted.

        Returns the total number of imports rewired.
        """
        if len(project_names) < 2:
            return 0

        # Build map: package_name → project_name for all projects
        records = await self.execute(
            f"MATCH (pkg:{NodeLabel.PACKAGE}) "
            "WHERE pkg.project_name IN $projects "
            "RETURN pkg.name AS name, pkg.project_name AS project, pkg.qualified_name AS qn",
            {"projects": project_names},
        )
        pkg_to_project: dict[str, str] = {}
        for r in records:
            top_name = r["qn"].split(".")[0] if r["qn"] else r["name"]
            if top_name not in pkg_to_project:
                pkg_to_project[top_name] = r["project"]

        if not pkg_to_project:
            return 0

        # Batch-read stubs and resolve real entities
        matched_eps, sym_to_real, ep_to_real_pkg = await self._resolve_cross_project_read_phase(
            project_names, pkg_to_project
        )
        if not matched_eps:
            return 0

        # Writes — per-stub for correctness
        rewired = 0
        for es_uid, real_uid in sym_to_real.items():
            await self.execute_write(
                f"MATCH (src:{NodeLabel.MODULE})-[r:{RelType.IMPORTS}]->"
                f"(es:{NodeLabel.EXTERNAL_SYMBOL} {{uid: $es_uid}}) "
                f"MATCH (real {{uid: $real_uid}}) "
                f"CREATE (src)-[:{RelType.IMPORTS}]->(real) "
                "DELETE r",
                {"es_uid": es_uid, "real_uid": real_uid},
            )
            rewired += 1

        for ep_uid, real_uid in ep_to_real_pkg.items():
            await self.execute_write(
                f"MATCH (src:{NodeLabel.MODULE})-[r:{RelType.IMPORTS}]->"
                f"(ep:{NodeLabel.EXTERNAL_PACKAGE} {{uid: $ep_uid}}) "
                f"MATCH (real {{uid: $real_uid}}) "
                f"CREATE (src)-[:{RelType.IMPORTS}]->(real) "
                "DELETE r",
                {"ep_uid": ep_uid, "real_uid": real_uid},
            )

        # Delete orphaned stubs
        for ep in matched_eps:
            await self.execute_write(
                f"MATCH (es:{NodeLabel.EXTERNAL_SYMBOL} {{project_name: $proj, package: $pkg}}) "
                f"WHERE NOT ()-[:{RelType.IMPORTS}]->(es) "
                "DETACH DELETE es",
                {"proj": ep["proj"], "pkg": ep["name"]},
            )
            await self.execute_write(
                f"MATCH (ep:{NodeLabel.EXTERNAL_PACKAGE} {{uid: $uid}}) "
                f"WHERE NOT ()-[:{RelType.IMPORTS}]->(ep) AND NOT (ep)-[:{RelType.CONTAINS}]->() "
                "DETACH DELETE ep",
                {"uid": ep["uid"]},
            )

        logger.debug(
            "Cross-project import resolution: {} imports rewired across {} projects", rewired, len(project_names)
        )
        return rewired

    async def create_depends_on_edges(self, project_names: list[str]) -> int:
        """Create DEPENDS_ON edges between Project nodes based on cross-project IMPORTS.

        Queries all IMPORTS edges where source and target have different project_names,
        then creates DEPENDS_ON between the corresponding Project nodes.

        Returns the count of DEPENDS_ON edges created.
        """
        if len(project_names) < 2:
            return 0

        # Delete existing DEPENDS_ON edges between these projects
        await self.execute_write(
            f"MATCH (a:{NodeLabel.PROJECT})-[r:{RelType.DEPENDS_ON}]->(b:{NodeLabel.PROJECT}) "
            "WHERE a.name IN $projects AND b.name IN $projects "
            "DELETE r",
            {"projects": project_names},
        )

        # Find all cross-project import pairs
        records = await self.execute(
            f"MATCH (src)-[:{RelType.IMPORTS}]->(tgt) "
            "WHERE src.project_name IN $projects AND tgt.project_name IN $projects "
            "AND src.project_name <> tgt.project_name "
            "RETURN DISTINCT src.project_name AS from_proj, tgt.project_name AS to_proj",
            {"projects": project_names},
        )

        if not records:
            return 0

        # Create DEPENDS_ON edges
        edges = [{"from_proj": r["from_proj"], "to_proj": r["to_proj"]} for r in records]
        await self.execute_write(
            f"UNWIND $edges AS e "
            f"MATCH (a:{NodeLabel.PROJECT} {{name: e.from_proj}}), "
            f"(b:{NodeLabel.PROJECT} {{name: e.to_proj}}) "
            f"CREATE (a)-[:{RelType.DEPENDS_ON}]->(b)",
            {"edges": edges},
        )

        logger.debug("Created {} DEPENDS_ON edge(s) between projects", len(edges))
        return len(edges)

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

    async def read_entity_texts(
        self,
        uids: list[str],
        *,
        labels: list[str] | None = None,
        chunk_size: int = 200,
    ) -> list[dict[str, Any]]:
        """Batch-read entity properties needed for embedding.

        ``uids`` must be full uid strings (``project:qualified_name``).
        If *labels* is provided (parallel to *uids*), entities are grouped
        by label so each query uses a label-constrained MATCH, hitting the
        per-label property index on ``uid`` instead of scanning all nodes.

        Queries are chunked to *chunk_size* uids per round-trip to avoid
        query timeouts on large batches.

        Returns list of dicts with keys: ``uid``, ``qualified_name``, ``name``,
        ``signature``, ``docstring``, ``kind``, ``_label``,
        ``embed_hash``, ``embedding``.
        """
        ret = (
            "RETURN n.uid AS uid, n.qualified_name AS qualified_name, n.name AS name, "
            "n.signature AS signature, n.docstring AS docstring, "
            "n.source AS source, "
            "n.kind AS kind, labels(n)[0] AS _label, "
            "n.embed_hash AS embed_hash, n.embedding AS embedding"
        )

        if labels is None or len(labels) != len(uids):
            # Fallback: label-free scan (slow on large graphs), chunked
            results: list[dict[str, Any]] = []
            for i in range(0, len(uids), chunk_size):
                chunk = uids[i : i + chunk_size]
                rows = await self.execute(
                    f"UNWIND $uids AS u MATCH (n) WHERE n.uid = u {ret}",
                    {"uids": chunk},
                )
                results.extend(rows)
            return results

        # Group uids by label → one indexed query per label, chunked
        by_label: dict[str, list[str]] = defaultdict(list)
        for uid, lbl in zip(uids, labels, strict=True):
            by_label[lbl].append(uid)

        results: list[dict[str, Any]] = []
        for lbl, group_uids in by_label.items():
            _assert_valid_label(lbl)
            for i in range(0, len(group_uids), chunk_size):
                chunk = group_uids[i : i + chunk_size]
                rows = await self.execute(
                    f"UNWIND $uids AS u MATCH (n:{lbl}) WHERE n.uid = u {ret}",
                    {"uids": chunk},
                )
                results.extend(rows)
        return results

    async def write_embeddings(
        self,
        items: list[tuple[str, list[float]]],
        chunk_size: int = 50,
        *,
        labels: list[str] | None = None,
    ) -> None:
        """Batch-write embedding vectors to nodes by uid.

        If *labels* is provided (parallel to *items*), writes are grouped by
        label for index-backed matching.
        """
        if not items:
            return

        if labels is not None and len(labels) == len(items):
            by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for (uid, vec), lbl in zip(items, labels, strict=True):
                by_label[lbl].append({"uid": uid, "vector": vec})
            for lbl, group in by_label.items():
                _assert_valid_label(lbl)
                for i in range(0, len(group), chunk_size):
                    chunk = group[i : i + chunk_size]
                    await self.execute_write(
                        f"UNWIND $items AS item MATCH (n:{lbl}) WHERE n.uid = item.uid SET n.embedding = item.vector",
                        {"items": chunk},
                    )
        else:
            for i in range(0, len(items), chunk_size):
                chunk_items = items[i : i + chunk_size]
                params = [{"uid": uid, "vector": vec} for uid, vec in chunk_items]
                await self.execute_write(
                    "UNWIND $items AS item MATCH (n) WHERE n.uid = item.uid SET n.embedding = item.vector",
                    {"items": params},
                )

    async def write_embed_hashes(self, items: list[tuple[str, str]], *, labels: list[str] | None = None) -> None:
        """Batch-write embed_hash values to nodes by uid.

        If *labels* is provided (parallel to *items*), writes are grouped by
        label for index-backed matching.
        """
        if not items:
            return

        if labels is not None and len(labels) == len(items):
            by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for (uid, h), lbl in zip(items, labels, strict=True):
                by_label[lbl].append({"uid": uid, "hash": h})
            for lbl, group in by_label.items():
                _assert_valid_label(lbl)
                await self.execute_write(
                    f"UNWIND $items AS item MATCH (n:{lbl}) WHERE n.uid = item.uid SET n.embed_hash = item.hash",
                    {"items": group},
                )
        else:
            params = [{"uid": uid, "hash": h} for uid, h in items]
            await self.execute_write(
                "UNWIND $items AS item MATCH (n) WHERE n.uid = item.uid SET n.embed_hash = item.hash",
                {"items": params},
            )

    async def clear_all_embeddings(self) -> None:
        """Remove embedding vectors and content hashes from all nodes."""
        await self.execute_write(
            "MATCH (n) WHERE n.embedding IS NOT NULL OR n.embed_hash IS NOT NULL REMOVE n.embedding, n.embed_hash"
        )

    async def rebuild_vector_indices(self, dimension: int) -> None:
        """Drop and recreate vector indices at the specified dimension."""
        for stmt in generate_drop_vector_index_ddl():
            await self._exec_ddl(stmt)
        if self._embeddings_enabled:
            for stmt in generate_vector_index_ddl(dimension):
                await self._exec_ddl(stmt)
        self._dimension = dimension

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
        projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 text search across text indices.

        Queries one or all text indices, optionally post-filters by project(s),
        and returns results sorted by score descending.
        """
        with _tracer.start_as_current_span("graph.text_search", attributes={"query": query[:100], "limit": limit}):
            # Backward compat: single project → projects list
            filter_projects = projects if projects is not None else ([project] if project else None)

            indices = (
                [f"text_{label.lower()}"] if label else [f"text_{lbl.value.lower()}" for lbl in _TEXT_SEARCHABLE_LABELS]
            )
            fetch_limit = limit * 3 if filter_projects else limit

            all_results: list[dict[str, Any]] = []
            for index_name in indices:
                cypher = (
                    f"CALL text_search.search_all('{index_name}', $query, {fetch_limit}) "
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
            if filter_projects:
                project_set = set(filter_projects)
                all_results = [r for r in all_results if _node_project_name(r) in project_set]

            # Sort by score descending and truncate
            all_results.sort(key=lambda rec: rec.get("score", 0), reverse=True)
            return all_results[:limit]

    async def get_text_index_info(self) -> list[dict[str, Any]]:
        """Query Memgraph for text index metadata via SHOW INDEX INFO (Memgraph 3.7+ DDL).

        Filters the generic index listing to text indices (type starts with 'label_text').
        Returns a list of dicts with index_type, label, and name keys.
        """
        try:
            rows = await self.execute("SHOW INDEX INFO")
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
        projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic similarity search using pre-computed vector.

        Queries one or all vector indices, optionally post-filters by project(s)
        and similarity threshold, and returns results sorted by similarity
        descending.  Returns ``[{"node": Node, "similarity": float}, ...]``.
        """
        with _tracer.start_as_current_span("graph.vector_search", attributes={"limit": limit}):
            filter_projects = projects if projects is not None else ([project] if project else None)

            indices = [f"vec_{label.lower()}"] if label else [f"vec_{lbl.value.lower()}" for lbl in _EMBEDDABLE_LABELS]
            filtering = bool(filter_projects) or threshold > 0.0
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
            if filter_projects:
                project_set = set(filter_projects)
                all_results = [r for r in all_results if _node_project_name(r) in project_set]

            all_results.sort(key=lambda rec: rec.get("similarity", 0), reverse=True)
            return all_results[:limit]

    # -- Graph (name-based) search helpers ------------------------------------

    async def graph_search(
        self,
        query: str,
        label: str = "",
        limit: int = 20,
        project: str = "",
        projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Name-based graph search with scored matching.

        Three-stage matching with decreasing scores:
        - Exact name match: score 3.0
        - Suffix match (qualified_name ends with .query): score 2.0
        - Contains match (name or qualified_name contains query): score 1.0

        Deduplicates by uid, keeping highest score.
        Returns ``[{"node": Node, "score": float}, ...]``.
        """
        with _tracer.start_as_current_span("graph.graph_search", attributes={"query": query[:100], "limit": limit}):
            return await self._graph_search_inner(query, label=label, limit=limit, project=project, projects=projects)

    async def _graph_search_inner(
        self,
        query: str,
        label: str = "",
        limit: int = 20,
        project: str = "",
        projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Inner implementation of graph_search.

        Uses a single UNION ALL query (one round-trip) instead of 3
        sequential execute() calls.
        """
        filter_projects = projects if projects is not None else ([project] if project else None)

        project_clause = " AND n.project_name IN $projects" if filter_projects else ""
        params: dict[str, Any] = {"query": query, "suffix": f".{query}", "projects": filter_projects or []}
        fetch_limit = limit * 3

        query_str = _build_graph_search_query(label, project_clause, fetch_limit)
        records = await self.execute(query_str, params)

        # Deduplicate by uid, keeping highest score
        scored: dict[str, tuple[Any, float]] = {}
        for r in records:
            node = r["node"]
            score: float = r["score"]
            uid = node.get("uid", "") if hasattr(node, "get") else ""
            if uid and (uid not in scored or scored[uid][1] < score):
                scored[uid] = (node, score)

        # Build result list sorted by score descending
        results = [{"node": node, "score": score} for node, score in scored.values()]
        results.sort(key=lambda rec: rec["score"], reverse=True)
        return results[:limit]

    async def batch_call_stats(self, uids: list[str], *, top_n: int = 5) -> dict[str, CallStats]:
        """Return caller/callee counts and top-N names per uid in a single round-trip."""
        if not uids:
            return {}
        records = await self.execute(
            "UNWIND $uids AS uid "
            "MATCH (n {uid: uid}) "
            f"OPTIONAL MATCH (caller)-[:{RelType.CALLS}]->(n) "
            "WITH uid, n, count(DISTINCT caller) AS cc, collect(DISTINCT caller.name)[0..$top_n] AS cn "
            f"OPTIONAL MATCH (n)-[:{RelType.CALLS}]->(callee) "
            "RETURN uid, cc, cn, count(DISTINCT callee) AS ec, collect(DISTINCT callee.name)[0..$top_n] AS en",
            {"uids": uids, "top_n": top_n},
        )
        return {
            r["uid"]: CallStats(
                caller_count=r["cc"],
                callee_count=r["ec"],
                caller_names=r["cn"] or [],
                callee_names=r["en"] or [],
            )
            for r in records
        }

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
                    "source": e.source,
                    "tags": e.tags,
                    "header_path": e.header_path,
                    "header_level": e.header_level,
                    "content_hash": e.content_hash,
                }
                for e in entity_list
            ]
            query = (
                f"UNWIND $entities AS e "
                f"MERGE (n:{label.value} {{uid: e.uid}}) "
                f"ON CREATE SET "
                f"n.project_name = e.project_name, n.name = e.name, "
                f"n.qualified_name = e.qualified_name, n.file_path = e.file_path, "
                f"n.kind = e.kind, n.line_start = e.line_start, n.line_end = e.line_end, "
                f"n.visibility = e.visibility, n.docstring = e.docstring, "
                f"n.signature = e.signature, n.source = e.source, n.tags = e.tags, "
                f"n.header_path = e.header_path, n.header_level = e.header_level, "
                f"n.content_hash = e.content_hash "
                f"ON MATCH SET "
                f"n.project_name = e.project_name, n.name = e.name, "
                f"n.qualified_name = e.qualified_name, n.file_path = e.file_path, "
                f"n.kind = e.kind, n.line_start = e.line_start, n.line_end = e.line_end, "
                f"n.visibility = e.visibility, n.docstring = e.docstring, "
                f"n.signature = e.signature, n.source = e.source, n.tags = e.tags, "
                f"n.header_path = e.header_path, n.header_level = e.header_level, "
                f"n.content_hash = e.content_hash"
            )
            await self.execute_write(query, {"entities": params})

    async def _batch_update_entities(self, entities: list[ParsedEntity]) -> None:
        """Batch-update modified entity nodes by uid, grouped by label."""
        sorted_entities = sorted(entities, key=attrgetter("label"))
        for label, group in groupby(sorted_entities, key=attrgetter("label")):
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
                    "source": e.source,
                    "tags": e.tags,
                    "header_path": e.header_path,
                    "header_level": e.header_level,
                    "content_hash": e.content_hash,
                }
                for e in group
            ]
            await self.execute_write(
                f"UNWIND $entities AS e "
                f"MATCH (n:{label.value} {{uid: e.uid}}) "
                "SET n.name = e.name, n.kind = e.kind, "
                "n.line_start = e.line_start, n.line_end = e.line_end, "
                "n.visibility = e.visibility, n.docstring = e.docstring, "
                "n.signature = e.signature, n.source = e.source, n.tags = e.tags, "
                "n.header_path = e.header_path, n.header_level = e.header_level, "
                "n.content_hash = e.content_hash",
                {"entities": params},
            )

    async def _batch_update_positions(self, entities: list[ParsedEntity]) -> None:
        """Update only line_start/line_end for entities whose content didn't change."""
        sorted_entities = sorted(entities, key=attrgetter("label"))
        for label, group in groupby(sorted_entities, key=attrgetter("label")):
            params = [{"uid": e.qualified_name, "ls": e.line_start, "le": e.line_end} for e in group]
            await self.execute_write(
                f"UNWIND $entities AS e MATCH (n:{label.value} {{uid: e.uid}}) "
                "SET n.line_start = e.ls, n.line_end = e.le",
                {"entities": params},
            )

    async def _batch_delete_entities(self, uids_by_label: dict[str, list[str]]) -> None:
        """Delete entity nodes by uid, grouped by label for index-backed matching."""
        for label, uids in uids_by_label.items():
            if label:
                _assert_valid_label(label)
                await self.execute_write(
                    f"UNWIND $uids AS uid MATCH (n:{label} {{uid: uid}}) DETACH DELETE n",
                    {"uids": uids},
                )
            else:
                await self.execute_write(
                    "UNWIND $uids AS uid MATCH (n {uid: uid}) DETACH DELETE n",
                    {"uids": uids},
                )

    async def _recreate_file_relationships(
        self,
        project_name: str,
        file_path: str,
        relationships: list[ParsedRelationship],
        uid_to_label: dict[str, str] | None = None,
        *,
        skip_delete: bool = False,
    ) -> None:
        """Delete all relationships originating from this file's entities, then recreate them."""
        # Delete existing relationships from file entities — one label-constrained query
        # per entity label, each hitting the composite (project_name, file_path) index.
        # Skipped when the file has no prior entities (fresh reindex / new file).
        if not skip_delete:
            for lbl in _FILE_ENTITY_LABELS:
                await self.execute_write(
                    f"MATCH (n:{lbl.value} {{project_name: $p, file_path: $f}})-[r]->() DELETE r",
                    {"p": project_name, "f": file_path},
                )

        # Recreate: uid-based rels (both ends known by uid)
        # Includes structural rels and detector-produced rels (both ends resolved to UIDs)
        uid_rel_types = {
            RelType.DEFINES,
            RelType.CONTAINS,
            RelType.OVERRIDES,
            RelType.IMPLEMENTS,
            RelType.EXPORTS,
            RelType.TESTS,
            RelType.HANDLES_ROUTE,
            RelType.HANDLES_EVENT,
            RelType.HANDLES_COMMAND,
            RelType.REGISTERED_BY,
            RelType.INJECTED_INTO,
        }
        uid_rels = [r for r in relationships if r.rel_type in uid_rel_types]
        other_rels = [r for r in relationships if r.rel_type not in uid_rel_types]

        _uid_lbl = uid_to_label or {}

        for rel_type, group in groupby(sorted(uid_rels, key=attrgetter("rel_type")), key=attrgetter("rel_type")):
            rels_list = list(group)
            has_props = any(r.properties for r in rels_list)

            # Group rels by (from_label, to_label) for index-backed matching
            by_label_pair: dict[tuple[str, str], list[ParsedRelationship]] = defaultdict(list)
            for r in rels_list:
                fl = _uid_lbl.get(r.from_qualified_name, "")
                tl = _uid_lbl.get(r.to_name, "")
                by_label_pair[(fl, tl)].append(r)

            for (fl, tl), sub_rels in by_label_pair.items():
                from_clause = f":{fl}" if fl else ""
                to_clause = f":{tl}" if tl else ""

                if has_props:
                    rel_params = [
                        {"from_uid": r.from_qualified_name, "to_uid": r.to_name, "props": r.properties}
                        for r in sub_rels
                    ]
                    await self.execute_write(
                        f"UNWIND $rels AS r MATCH (a{from_clause} {{uid: r.from_uid}}), "
                        f"(b{to_clause} {{uid: r.to_uid}}) CREATE (a)-[:{rel_type}]->(b)",
                        {"rels": rel_params},
                    )
                    # SET properties in a second pass (Memgraph doesn't support dynamic props in CREATE)
                    for param in rel_params:
                        if param["props"]:
                            rel_props: dict[str, Any] = param["props"]  # type: ignore[assignment]
                            set_clause = ", ".join(f"e.{k} = $prop_{k}" for k in rel_props)
                            prop_params = {f"prop_{k}": v for k, v in rel_props.items()}
                            await self.execute_write(
                                f"MATCH (a{from_clause} {{uid: $from_uid}})-[e:{rel_type}]->"
                                f"(b{to_clause} {{uid: $to_uid}}) SET {set_clause}",
                                {"from_uid": param["from_uid"], "to_uid": param["to_uid"], **prop_params},
                            )
                else:
                    rel_params = [{"from_uid": r.from_qualified_name, "to_uid": r.to_name} for r in sub_rels]
                    await self.execute_write(
                        f"UNWIND $rels AS r MATCH (a{from_clause} {{uid: r.from_uid}}), "
                        f"(b{to_clause} {{uid: r.to_uid}}) CREATE (a)-[:{rel_type}]->(b)",
                        {"rels": rel_params},
                    )

        # Name-matched rels (IMPORTS are intentionally excluded — resolved post-batch)
        doc_rels = [r for r in other_rels if r.rel_type == RelType.DOCUMENTS]
        inherits_rels = [r for r in other_rels if r.rel_type == RelType.INHERITS]

        if inherits_rels:
            inh_params = [
                {"from_uid": r.from_qualified_name, "project": project_name, "to_name": r.to_name}
                for r in inherits_rels
            ]
            await self.execute_write(
                f"UNWIND $rels AS r "
                f"MATCH (a:{NodeLabel.TYPE_DEF} {{uid: r.from_uid}}), "
                f"(b:{NodeLabel.TYPE_DEF} {{project_name: r.project, name: r.to_name}}) "
                f"CREATE (a)-[:{RelType.INHERITS}]->(b)",
                {"rels": inh_params},
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
                f"MATCH (a:{NodeLabel.DOC_SECTION} {{uid: r.from_uid}}), "
                f"(b {{project_name: $project, name: r.to_name}}) "
                f"CREATE (a)-[e:{RelType.DOCUMENTS} {{link_type: r.link_type, confidence: r.confidence}}]->(b) "
                f"RETURN count(e) AS cnt",
                {"rels": symbol_params, "project": project_name},
            )
            created += records[0]["cnt"] if records else 0

        if file_params:
            records = await self.execute(
                f"UNWIND $rels AS r "
                f"MATCH (a:{NodeLabel.DOC_SECTION} {{uid: r.from_uid}}), (b {{project_name: $project}}) "
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
        """Apply all constraints, indices, vector indices, and text indices.

        On a fresh database (no SchemaVersion node), vector/text indices from
        a previous session may still exist with stale internal state.  Drop
        them first so they are cleanly recreated at the current dimension.
        """
        # Drop stale search indices left over from a wiped database
        await self._drop_all_vector_indices()
        for stmt in generate_drop_text_index_ddl():
            await self._exec_ddl(stmt)

        for stmt in generate_unique_constraint_ddl():
            await self._exec_ddl(stmt)
        for stmt in generate_existence_constraint_ddl():
            await self._exec_ddl(stmt)
        for stmt in generate_index_ddl():
            await self._exec_ddl(stmt)
        for stmt in generate_composite_index_ddl():
            await self._exec_ddl(stmt)
        if self._embeddings_enabled:
            await self._create_vector_indices()
        for stmt in generate_text_index_ddl():
            await self._exec_ddl(stmt)

    async def _migrate_indices(self) -> None:
        """Drop and recreate vector/text indices (dimension may have changed).

        Also applies property and composite indices (idempotent via _exec_ddl).
        """
        for stmt in generate_index_ddl():
            await self._exec_ddl(stmt)
        for stmt in generate_composite_index_ddl():
            await self._exec_ddl(stmt)
        await self._drop_all_vector_indices()
        for stmt in generate_drop_text_index_ddl():
            await self._exec_ddl(stmt)
        if self._embeddings_enabled:
            await self._create_vector_indices()
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

    async def _drop_all_vector_indices(self) -> None:
        """Drop every vector index, waiting until Memgraph confirms removal.

        Memgraph's ``DROP VECTOR INDEX`` returns success before internal
        state is fully cleaned up.  A subsequent ``CREATE VECTOR INDEX`` at
        a different dimension will fail unless we verify that the catalogue
        is actually empty.  Poll with short delays to handle this.
        """
        for drop_stmt in generate_drop_vector_index_ddl():
            try:
                await self.execute_write(drop_stmt)
                logger.debug("Dropped: {}", drop_stmt.rstrip(";"))
            except Exception as exc:
                logger.debug("Drop skipped ({}): {}", exc, drop_stmt.rstrip(";"))

        # Poll the catalogue until all vector indices are gone
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                rows = await self.execute("CALL vector_search.show_index_info() YIELD index_name RETURN index_name")
            except Exception:
                # If the procedure itself fails, assume no indices
                break

            if not rows:
                break

            names = [r["index_name"] for r in rows if r.get("index_name")]
            logger.debug(
                "Vector indices still in catalogue (attempt {}/{}): {}",
                attempt + 1,
                max_attempts,
                names,
            )

            # Try dropping each remaining index
            for name in names:
                with contextlib.suppress(Exception):
                    await self.execute_write(f"DROP VECTOR INDEX {name};")

            await asyncio.sleep(0.3)

    async def _create_vector_indices(self) -> None:
        """Create all vector indices, retrying if Memgraph's async DROP hasn't settled.

        Memgraph's ``DROP VECTOR INDEX`` returns success before internal state
        is fully cleaned.  A ``CREATE`` at a different dimension will fail
        with a 'dimensions' error.  This method retries the entire batch of
        CREATE statements with backoff until Memgraph accepts them.
        """
        stmts = generate_vector_index_ddl(self._dimension)
        max_retries = 10
        for attempt in range(max_retries):
            failed = False
            for stmt in stmts:
                try:
                    await self.execute_write(stmt)
                except Exception as exc:
                    msg = str(exc).lower()
                    if "already exists" in msg:
                        continue
                    if "dimensions" in msg:
                        failed = True
                        break
                    raise
            if not failed:
                return
            # Dimension mismatch — old internal state not yet cleared
            logger.debug(
                "Vector index CREATE blocked by stale dimension (attempt {}/{}), waiting…",
                attempt + 1,
                max_retries,
            )
            await self._drop_all_vector_indices()
            await asyncio.sleep(0.5 * (attempt + 1))
        # Final attempt — let exceptions propagate
        for stmt in stmts:
            await self._exec_ddl(stmt)
