"""SQL support — tree-sitter parser for SQL scripts and DDL.

Extracts the DDL surface: ``CREATE TABLE`` (with its columns as members),
``CREATE VIEW`` (with a dependency edge per table it reads), ``CREATE INDEX``
(with an edge to the table it indexes), ``CREATE FUNCTION``/``PROCEDURE``, and
FOREIGN KEY constraints from both ``CREATE TABLE`` and ``ALTER TABLE``.

Grammar notes (measured, tree-sitter-sql ABI 15, DerekStride general SQL):
  - Root is ``program``; a well-formed file is ``statement`` + ``;`` pairs.
  - Parses ANSI DDL/DML, CTEs, MySQL backtick quoting and Postgres
    dollar-quoted bodies.
  - FAILS on T-SQL bracket quoting (``[dbo].[Users]``) and on PL/SQL. Those
    files still parse — they just come back with ERROR nodes in the tree.
  - There is **no** ``create_procedure`` rule. Every ``CREATE PROCEDURE``
    collapses into an ERROR node, and the routine's own name token is dropped
    from the tree entirely — see ``_recover_routines``.
  - Even plain ANSI ``CONSTRAINT <name> FOREIGN KEY ...`` splits into
    ERROR(``CONSTRAINT <name> FOREIGN``) + ``constraint``(``KEY (...) REFERENCES
    ...``), so FK detection keys on a ``keyword_references`` child rather than
    on ``keyword_foreign``.

Because of that the parse function MUST tolerate partial trees. Rejecting a
file that contains ERROR nodes would silently drop every T-SQL and Oracle
schema in the repo, so extraction walks for the statement types it recognises
and ignores the rest, never gating on ``root.has_error``.

The cost of that tolerance is bounded upstream, not here. tree-sitter-sql's
error recovery is superlinear on all-ERROR input — a synthetic T-SQL procedure
dump measures 1.0s at 128 KiB, 2.4s at 256 KiB, 7.3s at 512 KiB and 28s at
1023 KiB — so a committed dump can stall the AST consumer. The ceiling that
stops it is ``parsing.ast.DEFAULT_MAX_PARSE_BYTES`` (1 MiB, configurable as
``IndexSettings.max_parse_bytes``), and it *has* to live there: ``_parse_sql``
is handed an already-built ``root``, so by the time any check in this module
could run, the expensive recovery has already happened. Do not add a
size ceiling here — it would be dead weight over the real one.

Identity model — SQL objects are named in a *schema-wide* namespace, not a
per-file one, so their uids are ``{project}:sql.{table,view,index,function,
procedure}.{name}`` and deliberately do NOT include the file path (only the
per-file ``Module`` node does). This is what makes migrations work: an
``ALTER TABLE orders ADD CONSTRAINT ... REFERENCES users`` in
``003_add_fk.sql`` lands its edge on the very ``orders`` node that
``001_init.sql`` created. File-scoped uids would put the edge on a node that
does not exist, and it would be dropped in silence. Migration files are parsed
independently and are never ordered — no schema-evolution fold is attempted.

Identifiers are case-folded to lowercase (and stripped of ``"`` / ``` ` ``` /
``[]`` quoting) everywhere they are used as a name or a uid, because unquoted
SQL identifiers are case-insensitive: ``REFERENCES Users`` must resolve to
``CREATE TABLE users``, and the post-batch USES_TYPE resolver matches on the
node's ``name`` property verbatim. The original casing survives in
``signature``/``source``.

Cross-object edges use USES_TYPE, which is resolved post-batch against
TypeDef names (``GraphClient.resolve_type_refs``) rather than uid-routed. That
is what lets a FK reach a table declared in another file. The cost is that
USES_TYPE carries no edge properties, so the FK's *column* is recorded
structurally instead — as a column-level edge alongside the rolled-up
table-level one — plus a ``sql_references`` property on the column node so the
target column is still visible when the target table is not in the graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    node_text,
    register_language,
)
from code_atlas.schema import NodeLabel, RelType

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tree_sitter import Node

_EXTENSIONS = frozenset({".sql"})

_KIND_FILE = "sql_file"
_KIND_TABLE = "sql_table"
_KIND_VIEW = "sql_view"
_KIND_COLUMN = "sql_column"
_KIND_INDEX = "sql_index"

# Dialect quoting: ANSI double quotes, MySQL backticks, T-SQL brackets.
_QUOTES = "\"'`[]"

# A *quoted* column name parses as ``literal``, not ``identifier``
# (``"Id" INT`` -> literal), so both types count as a name token.
_NAME_NODES = frozenset({"identifier", "literal"})

_VIEW_NODES = frozenset({"create_view", "create_materialized_view"})

# Routine headers recovered from ERROR regions. MySQL's `DEFINER = user@host`
# and Postgres/Oracle's `OR REPLACE` sit between CREATE and the routine kind.
_ROUTINE_RE = re.compile(
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:DEFINER\s*=\s*\S+\s+)?"
    r"(?P<keyword>PROCEDURE|FUNCTION)\s+(?P<name>[\w$.\"`\[\]]+)",
    re.IGNORECASE,
)


def _module_qualified_name(file_path: str) -> str:
    """Convert a file path to a dotted qualified name, extension folded in.

    ``db/schema.sql`` -> ``db.schema_sql``;  ``a.b/x.sql`` -> ``a_b.x_sql``

    Unlike the code-language modules, the extension is *preserved* (its dot
    replaced) rather than stripped, because ``qualified_name`` IS the graph uid
    and a ``schema.sql`` beside a ``schema.py`` would otherwise claim the same
    node, the later upsert silently overwriting the earlier one.

    Dots are folded in *every* segment, not just the basename, for that same
    reason: ``.`` is the separator being built here, so a directory named
    ``a.b`` would fake a nesting level and make ``a.b/x.sql`` and ``a/b/x.sql``
    claim one uid.
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    return ".".join(part.replace(".", "_") for part in p.parts)


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------


def _walk(node: Node) -> Iterator[Node]:
    """Pre-order walk over *node* and its descendants, in document order."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _child(node: Node, node_type: str) -> Node | None:
    """First direct child of *node* with the given type."""
    return next((c for c in node.children if c.type == node_type), None)


def _clean_name(text: str) -> str:
    """Strip dialect quoting from one identifier token and case-fold it."""
    return text.strip().strip(_QUOTES).strip().lower()


def _dotted_name(text: str) -> str:
    """Clean a possibly schema-qualified raw name: ``[dbo].[Users]`` -> ``dbo.users``."""
    return ".".join(part for part in (_clean_name(p) for p in text.split(".")) if part)


def _object_name(ref: Node) -> str:
    """Dotted name of an ``object_reference``, built from its identifier children.

    Built from the children rather than from ``ref.text`` because T-SQL bracket
    quoting parses as identifier(``dbo``) + ERROR(``]``) + ``.`` + ERROR(``[``)
    + identifier(``Users``): the node's own text is the mangled ``dbo].[Users``,
    while the identifier children are clean.
    """
    parts = [_clean_name(node_text(c)) for c in ref.children if c.type in _NAME_NODES]
    return ".".join(p for p in parts if p)


def _bare_name(full: str) -> str:
    """Drop the schema qualifier: ``public.users`` -> ``users``."""
    return full.rsplit(".", 1)[-1]


def _ordered_columns(node: Node) -> list[str]:
    """Column names from a constraint's ``ordered_columns`` list."""
    columns = _child(node, "ordered_columns")
    if columns is None:
        return []
    return [name for name in (_clean_name(node_text(c)) for c in columns.children if c.type == "column") if name]


def _fk_reference(node: Node) -> tuple[str, list[str]] | None:
    """``(target_table, target_columns)`` for a FK-bearing node, else ``None``.

    Handles both shapes with one pass: an inline ``column_definition``
    (``tenant_id INT REFERENCES tenants (id)``) and a table-level
    ``constraint`` (``[FOREIGN] KEY (a) REFERENCES t (b)``). Detection keys on
    ``keyword_references``, because the named form
    (``CONSTRAINT fk FOREIGN KEY ...``) loses its ``keyword_foreign`` to an
    ERROR node.
    """
    children = node.children
    ref_index = next((i for i, c in enumerate(children) if c.type == "keyword_references"), None)
    if ref_index is None:
        return None
    target_index = next(
        (i for i in range(ref_index + 1, len(children)) if children[i].type == "object_reference"), None
    )
    if target_index is None:
        return None
    full = _object_name(children[target_index])
    if not full:
        return None
    # Target columns are the identifiers in the parens right after the table
    # reference; the first *named* non-identifier node (``ON DELETE ...``) ends
    # the list.
    columns: list[str] = []
    for child in children[target_index + 1 :]:
        if child.type in _NAME_NODES:
            columns.append(_clean_name(node_text(child)))
        elif child.is_named:
            break
    return full, [c for c in columns if c]


def _pk_columns(definitions: Node) -> set[str]:
    """Column names named by a table-level ``PRIMARY KEY (...)`` constraint."""
    names: set[str] = set()
    for node in _walk(definitions):
        if node.type == "constraint" and _child(node, "keyword_primary") is not None:
            names.update(_ordered_columns(node))
    return names


def _relation_names(query: Node) -> list[str]:
    """Bare table names read by a query, in order, CTE names excluded.

    A CTE is itself a ``relation`` where it is used, so without the exclusion
    every view over a ``WITH`` clause would claim a dependency on a table that
    does not exist.
    """
    cte_names: set[str] = set()
    names: list[str] = []
    for node in _walk(query):
        if node.type == "cte":
            identifier = _child(node, "identifier")
            if identifier is not None:
                cte_names.add(_clean_name(node_text(identifier)))
        elif node.type == "relation":
            reference = _child(node, "object_reference")
            if reference is not None:
                full = _object_name(reference)
                if full:
                    names.append(_bare_name(full))
    return [name for name in dict.fromkeys(names) if name not in cte_names]


def _index_name(node: Node) -> str:
    """Declared index name, or ``""`` for the unnamed ``CREATE INDEX ON t (...)`` form.

    Only identifiers *before* ``ON`` count — ``USING gin`` puts a bare
    identifier after the table reference in some dialects.
    """
    for child in node.children:
        if child.type == "keyword_on":
            break
        if child.type == "identifier":
            return _clean_name(node_text(child))
    return ""


def _outermost_errors(root: Node) -> list[Node]:
    """ERROR nodes that are not nested inside another ERROR node.

    Nested ERRORs cover the same bytes as their parent, which would make the
    same routine header match several times.
    """
    found: list[Node] = []
    for node in _walk(root):
        if node.type != "ERROR":
            continue
        if any(e.start_byte <= node.start_byte and node.end_byte <= e.end_byte for e in found):
            continue
        found.append(node)
    return found


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    """Accumulator threaded through the extractors."""

    project_name: str
    path: str
    module_uid: str
    entities: list[ParsedEntity] = field(default_factory=list)
    relationships: list[ParsedRelationship] = field(default_factory=list)
    uids: set[str] = field(default_factory=set)
    edges: set[tuple[str, str, str]] = field(default_factory=set)

    def object_uid(self, namespace: str, full_name: str) -> str:
        """Uid for a schema-namespaced SQL object (no file path — see module docstring)."""
        return f"{self.project_name}:sql.{namespace}.{full_name}"

    def claim(self, uid: str) -> str:
        """Reserve *uid*, appending ``#N`` if this file already used it.

        Two ``CREATE TABLE users`` in one file would otherwise collide on a
        single node, the second silently overwriting the first (markdown.py
        disambiguates duplicate section names the same way). Collisions
        *across* files are intentional: that is one logical table.
        """
        if uid not in self.uids:
            self.uids.add(uid)
            return uid
        suffix = 2
        while f"{uid}#{suffix}" in self.uids:
            suffix += 1
        unique = f"{uid}#{suffix}"
        self.uids.add(unique)
        return unique

    def defines(self, parent_uid: str, child_uid: str) -> None:
        if (RelType.DEFINES, parent_uid, child_uid) in self.edges:
            return
        self.edges.add((RelType.DEFINES, parent_uid, child_uid))
        self.relationships.append(
            ParsedRelationship(from_qualified_name=parent_uid, rel_type=RelType.DEFINES, to_name=child_uid)
        )

    def uses(self, from_uid: str, table_name: str) -> None:
        """Reference another SQL object by bare name (resolved post-batch)."""
        if not table_name or (RelType.USES_TYPE, from_uid, table_name) in self.edges:
            return
        self.edges.add((RelType.USES_TYPE, from_uid, table_name))
        self.relationships.append(
            ParsedRelationship(from_qualified_name=from_uid, rel_type=RelType.USES_TYPE, to_name=table_name)
        )


def _extract_column(node: Node, table_uid: str, pk_columns: set[str], columns: dict[str, str], ctx: _Ctx) -> None:
    """Emit one Value per ``column_definition``, plus its inline FK edges."""
    name_node = next((c for c in node.children if c.type in _NAME_NODES), None)
    if name_node is None:
        return
    name = _clean_name(node_text(name_node))
    if not name:
        return
    column_uid = ctx.claim(f"{table_uid}.{name}")
    columns.setdefault(name, column_uid)

    reference = _fk_reference(node)
    extra: dict[str, str] = {}
    if reference is not None:
        target, target_columns = reference
        # Kept on the node as well as on the edge: the edge is dropped when the
        # referenced table is not indexed (another schema, another service),
        # and this is the only surviving record of the FK in that case.
        extra["sql_references"] = f"{target}({', '.join(target_columns)})" if target_columns else target

    tags = ["primary_key"] if name in pk_columns or _child(node, "keyword_primary") is not None else []
    ctx.entities.append(
        ParsedEntity(
            name=name,
            qualified_name=column_uid,
            label=NodeLabel.VALUE,
            kind=_KIND_COLUMN,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=ctx.path,
            signature=" ".join(node_text(node).split()),
            tags=tags,
            extra_properties=extra,
        )
    )
    ctx.defines(table_uid, column_uid)
    if reference is not None:
        ctx.uses(column_uid, _bare_name(reference[0]))
        ctx.uses(table_uid, _bare_name(reference[0]))


def _extract_constraint(node: Node, table_uid: str, columns: dict[str, str], ctx: _Ctx) -> None:
    """Emit FK edges for a table-level ``constraint`` node.

    Non-FK constraints (PRIMARY KEY, UNIQUE, CHECK) produce no edges; the
    primary key surfaces as a tag on the column instead.
    """
    reference = _fk_reference(node)
    if reference is None:
        return
    target = _bare_name(reference[0])
    ctx.uses(table_uid, target)
    for local in _ordered_columns(node):
        # Falls back to the derived uid for a column this file never declared
        # (the ALTER TABLE case) — it resolves if the CREATE lives elsewhere in
        # the project, and is dropped if it does not.
        ctx.uses(columns.get(local, f"{table_uid}.{local}"), target)


def _extract_table(node: Node, ctx: _Ctx) -> None:
    """``CREATE TABLE`` -> TypeDef, its columns -> Values, its FKs -> USES_TYPE."""
    reference = _child(node, "object_reference")
    if reference is None:
        return
    full = _object_name(reference)
    if not full:
        return
    table_uid = ctx.claim(ctx.object_uid("table", full))
    ctx.entities.append(
        ParsedEntity(
            name=_bare_name(full),
            qualified_name=table_uid,
            label=NodeLabel.TYPE_DEF,
            kind=_KIND_TABLE,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=ctx.path,
            source=node_text(node),
        )
    )
    ctx.defines(ctx.module_uid, table_uid)

    definitions = _child(node, "column_definitions")
    if definitions is None:
        # CREATE TABLE ... AS SELECT, or a form the grammar could not recover.
        return
    pk_columns = _pk_columns(definitions)
    columns: dict[str, str] = {}
    # One pre-order pass: the grammar puts ``constraints`` last inside
    # ``column_definitions``, so every column is registered before the
    # table-level constraints that name it.
    for child in _walk(definitions):
        if child.type == "column_definition":
            _extract_column(child, table_uid, pk_columns, columns, ctx)
        elif child.type == "constraint":
            _extract_constraint(child, table_uid, columns, ctx)


def _extract_alter_table(node: Node, ctx: _Ctx) -> None:
    """FK constraints added by ``ALTER TABLE`` — the dominant form in migrations.

    No entity is emitted: the table node belongs to whichever file ran the
    ``CREATE``, and the schema-namespaced uid reaches it from here. Column
    additions are deliberately ignored — reconstructing a table from a chain of
    migrations is schema evolution, which this parser does not attempt.
    """
    reference = _child(node, "object_reference")
    if reference is None:
        return
    full = _object_name(reference)
    if not full:
        return
    table_uid = ctx.object_uid("table", full)
    for child in _walk(node):
        if child.type == "constraint":
            _extract_constraint(child, table_uid, {}, ctx)


def _extract_view(node: Node, ctx: _Ctx) -> None:
    """``CREATE [MATERIALIZED] VIEW`` -> TypeDef + one edge per table it reads."""
    reference = _child(node, "object_reference")
    if reference is None:
        return
    full = _object_name(reference)
    if not full:
        return
    view_uid = ctx.claim(ctx.object_uid("view", full))
    ctx.entities.append(
        ParsedEntity(
            name=_bare_name(full),
            qualified_name=view_uid,
            label=NodeLabel.TYPE_DEF,
            kind=_KIND_VIEW,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=ctx.path,
            source=node_text(node),
            tags=["materialized"] if node.type == "create_materialized_view" else [],
        )
    )
    ctx.defines(ctx.module_uid, view_uid)

    query = _child(node, "create_query")
    if query is None:
        return
    own_name = _bare_name(full)
    for name in _relation_names(query):
        if name != own_name:
            ctx.uses(view_uid, name)


def _extract_index(node: Node, ctx: _Ctx) -> None:
    """``CREATE INDEX`` -> Value + an edge to the table it indexes."""
    reference = _child(node, "object_reference")
    if reference is None:
        return
    full = _object_name(reference)
    if not full:
        return
    fields_node = _child(node, "index_fields")
    fields = (
        [name for name in (_clean_name(node_text(f)) for f in fields_node.children if f.type == "field") if name]
        if fields_node is not None
        else []
    )
    # An unnamed `CREATE INDEX ON t (a)` still needs a stable uid; Postgres'
    # own auto-naming convention is exactly this shape.
    name = _index_name(node) or "_".join([_bare_name(full), *fields, "idx"])
    # Index names are unique per schema in Postgres but only per table in
    # MySQL, so the table is part of the uid.
    index_uid = ctx.claim(f"{ctx.object_uid('index', full)}.{name}")
    ctx.entities.append(
        ParsedEntity(
            name=name,
            qualified_name=index_uid,
            label=NodeLabel.VALUE,
            kind=_KIND_INDEX,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=ctx.path,
            signature=" ".join(node_text(node).split()),
        )
    )
    ctx.defines(ctx.module_uid, index_uid)
    ctx.uses(index_uid, _bare_name(full))


def _extract_function(node: Node, ctx: _Ctx) -> None:
    """``CREATE FUNCTION`` -> Callable. The body is left opaque (see module docstring)."""
    reference = _child(node, "object_reference")
    if reference is None:
        return
    full = _object_name(reference)
    if not full:
        return
    function_uid = ctx.claim(ctx.object_uid("function", full))
    arguments = _child(node, "function_arguments")
    signature = _bare_name(full) + (" ".join(node_text(arguments).split()) if arguments is not None else "()")
    ctx.entities.append(
        ParsedEntity(
            name=_bare_name(full),
            qualified_name=function_uid,
            label=NodeLabel.CALLABLE,
            kind="sql_function",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=ctx.path,
            signature=signature,
            source=node_text(node),
        )
    )
    ctx.defines(ctx.module_uid, function_uid)


def _recover_routines(root: Node, ctx: _Ctx) -> None:
    """Regex-recover ``CREATE PROCEDURE``/``FUNCTION`` headers from ERROR regions.

    The grammar has no ``create_procedure`` rule at all, so every stored
    procedure — T-SQL, PL/SQL, MySQL — collapses into an ERROR node, and
    (measured) the routine's own name token is dropped from the tree. Tree
    extraction cannot see procedures, so the ERROR text itself is the only
    place left to look, and skipping it would mean returning nothing but a
    Module node for a whole class of real schema files.

    Scoped to ERROR ranges on purpose: a well-parsed ``create_function`` is
    already handled by the tree walk, and keeping the regex off healthy source
    stops it inventing entities out of comments and string literals. Entities
    recovered this way are tagged ``recovered`` so the degradation is visible
    in the graph rather than silent, and they span only the header line — the
    body's extent is exactly what the grammar failed to determine.
    """
    for error in _outermost_errors(root):
        text = node_text(error)
        for match in _ROUTINE_RE.finditer(text):
            full = _dotted_name(match.group("name"))
            if not full:
                continue
            namespace = "procedure" if match.group("keyword").lower() == "procedure" else "function"
            uid = ctx.object_uid(namespace, full)
            if uid in ctx.uids:
                # Already recovered, or already extracted from the tree.
                continue
            ctx.uids.add(uid)
            line = error.start_point[0] + 1 + text[: match.start()].count("\n")
            ctx.entities.append(
                ParsedEntity(
                    name=_bare_name(full),
                    qualified_name=uid,
                    label=NodeLabel.CALLABLE,
                    kind=f"sql_{namespace}",
                    line_start=line,
                    line_end=line,
                    file_path=ctx.path,
                    signature=" ".join(match.group(0).split()),
                    tags=["recovered"],
                )
            )
            ctx.defines(ctx.module_uid, uid)


def _parse_sql(path: str, source: bytes, root: Node, project_name: str) -> ParsedFile:
    """Extract entities from a SQL file.

    Deliberately does not inspect ``root.has_error`` — see the module docstring.
    """
    norm_path = path.replace("\\", "/")
    language = "sql"

    if not source.strip():
        # No Module node for an empty file — it would be an unsearchable stub
        # that still costs an embedding.
        return ParsedFile(file_path=norm_path, language=language, entities=[], relationships=[])

    module_uid = f"{project_name}:{_module_qualified_name(norm_path)}"
    ctx = _Ctx(project_name=project_name, path=norm_path, module_uid=module_uid)
    ctx.uids.add(module_uid)
    ctx.entities.append(
        ParsedEntity(
            name=PurePosixPath(norm_path).name,
            qualified_name=module_uid,
            label=NodeLabel.MODULE,
            kind=_KIND_FILE,
            line_start=1,
            line_end=root.end_point[0] + 1,
            file_path=norm_path,
        )
    )

    for node in _walk(root):
        node_type = node.type
        if node_type == "create_table":
            _extract_table(node, ctx)
        elif node_type in _VIEW_NODES:
            _extract_view(node, ctx)
        elif node_type == "create_index":
            _extract_index(node, ctx)
        elif node_type == "create_function":
            _extract_function(node, ctx)
        elif node_type == "alter_table":
            _extract_alter_table(node, ctx)

    _recover_routines(root, ctx)
    return ParsedFile(file_path=norm_path, language=language, entities=ctx.entities, relationships=ctx.relationships)


# ---------------------------------------------------------------------------
# Language registration
# ---------------------------------------------------------------------------

try:
    import tree_sitter_sql as _ts_sql
    from tree_sitter import Language, Query

    _SQL_LANGUAGE = Language(_ts_sql.language())
    _SQL_QUERY = Query(_SQL_LANGUAGE, "(program) @root")

    register_language(
        LanguageConfig(
            name="sql",
            extensions=_EXTENSIONS,
            language=_SQL_LANGUAGE,
            query=_SQL_QUERY,
            parse_func=_parse_sql,
            # `--` comments are `comment`; `/* */` blocks are `marginalia`.
            comment_node_types=frozenset({"comment", "marginalia"}),
        )
    )
except ImportError:
    pass
