"""TMDL — a Power BI semantic model, as entities rather than as bytes.

TMDL (Tabular Model Definition Language) is what a PBIP project stores its semantic model
in: one object per file under ``<Name>.SemanticModel/definition/``, line-oriented and
indentation-scoped. Before this module ``*.tmdl`` was not indexed at all: a model's
tables, measures and columns had no representation in the graph whatsoever, while the
report layer sitting on top of them was indexed and — being addressed by generated
identifiers rather than by name — unusable.

**No grammar.** TMDL's indentation scoping needs a tree-sitter external scanner and
Microsoft publishes no grammar, so this module registers with ``language=None`` and parses
text itself (ATL-168, ADR-0048's sibling hatch). It is the only such language.

**Names, not GUIDs.** ``lineageTag`` is a stable identifier and a useless one to search
for; it goes to ``extra_properties`` where it stays queryable. The uid is the name the
report author typed.

**Read what you recognise, ignore the rest.** Microsoft versions TMDL, and a keyword this
parser has never seen must cost a property, never a table. Every unknown line is skipped
silently; there is no strict mode and there should not be one, because the failure mode of
strictness here is an entire semantic model vanishing from the graph on a Power BI update.
"""

from __future__ import annotations

import hashlib
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    register_language,
)
from code_atlas.schema import NodeLabel, RelType, Visibility

# ---------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------

KIND_MODEL = "pbi_model"
KIND_DEFINITION = "pbi_definition"
KIND_TABLE = "pbi_table"
KIND_MEASURE = "pbi_measure"
KIND_COLUMN = "pbi_column"
KIND_EXPRESSION = "pbi_m_expression"
KIND_CALC_ITEM = "pbi_calc_item"
KIND_FUNCTION = "pbi_udf"
KIND_ROLE = "pbi_role"

_TAB_WIDTH = 4
"""Tabs are the TMDL convention, but exports and hand edits both produce spaces. Indent is
compared after expansion so a file that mixes them still nests correctly."""


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def unquote(name: str) -> str:
    """Strip TMDL quoting from an object name.

    A name is single-quoted only when it contains one of dot, equals, colon, single quote
    or whitespace -- so the same measure appears as ``measure 'Sales Amount'`` where it is
    defined and as ``[Sales Amount]`` where it is referenced. Normalising on the way in is
    what makes those two the same node; skip it and every CALLS edge to a name that needed
    quoting misses, silently.

    An embedded quote is doubled, TMDL's own escape.
    """
    name = name.strip()
    if len(name) >= 2 and name.startswith("'") and name.endswith("'"):
        return name[1:-1].replace("''", "'")
    return name


_QN_UNSAFE_RE = re.compile(r"[.:]+")


def _qn_segment(text: str) -> str:
    """A name, made safe for a dotted qualified name -- and kept distinct.

    Only two characters are folded, because only two actually break something: a dot ends
    the segment ``qualified_name`` is built from, and a colon ends the ``{project}:`` half
    of every uid. Whitespace, ``=`` and ``'`` are all safe downstream, and leaving them
    keeps the uid readable -- ``'Sales Amount'`` stays ``Sales Amount`` rather than becoming
    ``Sales_Amount``.

    Folding is lossy, so it can merge two different objects onto one uid: ``'A.B'`` and
    ``'A_B'`` both fold to ``A_B``. A merged node is worse than a missing one -- it carries
    an arbitrary winner's DAX and the union of both edge sets -- and for a table it takes
    every measure and column with it. So a folded name earns a suffix derived from the
    original.

    Derived from the name, never from arrival order: a first draft suffixed only the
    *second* of two colliding names, which meant reordering a file changed a uid and churned
    the whole subtree under it for an edit that changed nothing. Names that need no folding
    -- the overwhelming majority -- are untouched and carry no suffix.
    """
    text = text.strip()
    folded = _QN_UNSAFE_RE.sub("_", text)
    if not folded:
        return "_"
    if folded == text:
        return folded
    return f"{folded}~{hashlib.blake2s(text.encode(), digest_size=3).hexdigest()}"


def model_name_for(path: str) -> str:
    """The semantic model a ``.tmdl`` file belongs to, from its path alone.

    A table file says ``table Sales`` and never names its model, so the model has to come
    from the layout. PBIP puts every definition under ``<Name>.SemanticModel/definition/``,
    which is the reliable signal; the fallbacks below exist so a loose file or a
    non-standard export still gets a stable, non-colliding namespace rather than being
    dropped.

    It matters that this is path-only: ``parse_file`` is a pure function over one file, so
    reading ``model.tmdl`` to find the name is not available. The plan this implements
    assumed a model segment in the uid without noticing that.
    """
    parts = PurePosixPath(path).parts
    for part in parts:
        if part.lower().endswith(".semanticmodel"):
            return _qn_segment(part[: -len(".SemanticModel")])
    # `.../<Model>/definition/tables/X.tmdl` — the directory above `definition`.
    for i, part in enumerate(parts):
        if part.lower() == "definition" and i:
            return _qn_segment(parts[i - 1])
    # A bare `tables/X.tmdl`, or a file at the root: the grandparent, then nothing.
    if len(parts) >= 3:
        return _qn_segment(parts[-3])
    return "model"


# ---------------------------------------------------------------------------
# The block tree
# ---------------------------------------------------------------------------


@dataclass
class _Block:
    """One TMDL line and everything indented under it."""

    indent: int
    text: str
    line: int
    children: list[_Block] = field(default_factory=list)
    description: str = ""
    end_line: int = 0

    @property
    def last_line(self) -> int:
        return max(self.end_line, *(c.last_line for c in self.children)) if self.children else self.end_line


_HEADER_RE = re.compile(
    r"^(?P<kw>[A-Za-z_][A-Za-z0-9_]*)"  # table / measure / column / ...
    r"(?:\s+(?P<name>'(?:[^']|'')*'|[^\s=]+))?"  # the object name, quoted when it needs to be
    r"\s*(?:=\s*(?P<expr>.*))?$"  # `= <expression>`, possibly empty and continued below
)
_PROPERTY_RE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<value>.*)$")


def _property_value(raw: str) -> str:
    """Unwrap a TMDL text property value.

    Double quotes around a property value are optional and stripped on serialization; they
    are only *required* when the text has leading or trailing whitespace. Inside them, a
    double quote is escaped by doubling. Both real, both in the spec\'s own example
    (``sourceColumn: "Net Price"``, ``displayFolder: " My ""Amazing"" Measures"``), and
    without this the stored value keeps its quotes and no search for the bare name matches.
    """
    value = raw.strip()
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace('""', '"')
    return value


def _indent_of(line: str) -> int:
    stripped = line.lstrip(" \t")
    return len(line[: len(line) - len(stripped)].expandtabs(_TAB_WIDTH))


def parse_blocks(source: str) -> list[_Block]:
    """Build the indentation tree, attaching ``///`` descriptions to what follows them.

    Blank lines and ``//`` comments carry no structure and are dropped, but their line
    numbers are not reused, so every entity's reported span still matches the file.
    """
    roots: list[_Block] = []
    stack: list[_Block] = []
    pending_doc: list[str] = []

    for lineno, raw in enumerate(source.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            # A description attaches to the declaration it sits directly on top of: the
            # spec allows no whitespace between the block and the object type token. So a
            # blank line ends it, and without this a stray comment at the top of a file
            # becomes the description of whatever object happens to come first.
            pending_doc.clear()
            continue
        if stripped.startswith("///"):
            pending_doc.append(stripped[3:].strip())
            continue
        if stripped.startswith("//"):
            continue

        indent = _indent_of(raw)
        block = _Block(indent=indent, text=stripped, line=lineno, end_line=lineno)
        if pending_doc:
            block.description = "\n".join(pending_doc)
            pending_doc = []

        while stack and stack[-1].indent >= indent:
            stack.pop()
        if stack:
            stack[-1].children.append(block)
        else:
            roots.append(block)
        stack.append(block)

    return roots


def _raw_body(block: _Block, source_lines: list[str]) -> str:
    """The multi-line expression under *block*, dedented, or ``""``.

    The expression is the LEADING run of children, stopping at the first child indented
    less than the first one. That boundary is the whole point: TMDL puts a multi-line
    expression one level deeper than the properties, and the properties come *after* it --

        measure Quantity =
                var result = SUMX(...)
                return result
            formatString: #,##0

    so a naive "everything under this block" read swallows ``formatString`` into the DAX.
    A fixture written from the same assumption hides it, which is exactly what happened
    here before the syntax was checked against the specification.

    Taken from the source rather than rebuilt from the block tree, because DAX and M are
    whitespace-significant in ways the tree has already discarded, then dedented because
    TMDL strips outer indentation beyond the parent\'s level on read.
    """
    if not block.children:
        return ""
    # Bounded by the FIRST body line's indent, not by the declaration's. TMDL puts
    # properties one level under the declaration and the expression one level under
    # those, so the declaration's own indent is two levels too shallow to be a boundary
    # -- keying off it lets `formatString` back into the DAX.
    #
    # Keying off the first body line also costs nothing in indent-delta assumptions,
    # which matters: measured across real files the body sits +2 from the declaration in
    # 1,478 cases but also +1, +3 and +4, and Microsoft's own Sales.tmdl uses +2 for a
    # measure and +1 for a kpi in the same file. Nothing here assumes a delta.
    body_indent = block.children[0].indent
    last = block.children[0].last_line
    for child in block.children[1:]:
        if child.indent < body_indent:
            break
        last = child.last_line
    raw = source_lines[block.children[0].line - 1 : last]
    return textwrap.dedent("\n".join(line.expandtabs(_TAB_WIDTH) for line in raw)).rstrip()


def _has_default_property(block: _Block) -> bool:
    """Whether the declaration assigns a default property with ``=``."""
    header = _HEADER_RE.match(block.text)
    return bool(header and header.group("expr") is not None)


def _expression_of(block: _Block, source_lines: list[str]) -> str:
    """The right-hand side of ``name = ...``, inline or continued below."""
    match = _HEADER_RE.match(block.text)
    inline = (match.group("expr") or "").strip() if match else ""
    if inline and inline not in {"```", "m", "calculated", "calculationGroup"}:
        return inline
    body = _raw_body(block, source_lines)
    # A ``` fence is TMDL's multi-line form; the content is what matters, not the fence.
    return "\n".join(ln for ln in body.splitlines() if ln.strip() != "```").strip()


def properties_of(block: _Block) -> dict[str, str]:
    """``key: value`` children of *block*, plus bare flags read as ``"true"``.

    A bare keyword line (``isHidden``) is TMDL's boolean-true shorthand. Unknown keys are
    kept rather than filtered: they cost one map entry and they are what makes a property
    this parser has never heard of still queryable.
    """
    out: dict[str, str] = {}
    for child in block.children:
        if prop := _PROPERTY_RE.match(child.text):
            out[prop.group("key")] = _property_value(prop.group("value"))
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", child.text) and not child.children:
            out[child.text] = "true"
    return out


_OWN_HANDLER_KEYWORDS = frozenset(
    {
        "model",
        "table",
        "column",
        "measure",
        "partition",
        "calculationgroup",
        "calculationitem",
        "expression",
        "function",
        "role",
        "tablepermission",
    }
)
"""Constructs this module dispatches for itself, so nothing else may collect them."""


def expression_children(block: _Block, source_lines: list[str]) -> dict[str, str]:
    """``key -> expression`` for the ``key = <expr>`` children of *block*.

    TMDL has two child syntaxes and they are not interchangeable: ``:`` assigns a scalar
    property, ``=`` assigns an *expression*. :func:`properties_of` implements only the
    first, so every expression-valued child was invisible -- and that is one omission, not
    eight: ``formatStringDefinition``, ``detailRowsDefinition``,
    ``defaultDetailRowsDefinition``, ``targetExpression``, ``statusExpression``,
    ``trendExpression``, ``multipleOrEmptySelectionExpression`` and ``noSelectionExpression``
    are all this shape.

    Every one of them is DAX that names measures and columns, so each is a dependency the
    graph was reporting as absent. A dynamic format string in particular characteristically
    reaches into a lookup table nothing else in the model touches -- no second witness.

    Nested one level, so a ``kpi`` block's expression children are reached too; a KPI is a
    property of its measure rather than a thing anyone searches for, which is the same call
    the partition fold makes.
    """
    out: dict[str, str] = {}
    for child in block.children:
        header = _HEADER_RE.match(child.text)
        if not header:
            continue
        keyword = header.group("kw").lower()
        if header.group("expr") is not None and header.group("name") is None:
            out[header.group("kw")] = _expression_of(child, source_lines)
        elif not header.group("expr") and child.children and keyword not in _OWN_HANDLER_KEYWORDS:
            # A container that takes children rather than a value -- `kpi`,
            # `refreshPolicy`. Constructs with a handler of their own are excluded, or a
            # `calculationGroup` would be collected here AND by that handler, emitting
            # every one of its edges twice.
            out |= expression_children(child, source_lines)
    return out


# ---------------------------------------------------------------------------
# DAX references
# ---------------------------------------------------------------------------

_QUALIFIED_COLUMN_RE = re.compile(r"(?:'(?P<qtable>(?:[^']|'')*)'|(?P<btable>[A-Za-z_][\w]*))\[(?P<column>[^\]]+)\]")
_BARE_REF_RE = re.compile(r"\[(?P<name>[^\]]+)\]")


def dax_references(expression: str) -> tuple[set[str], set[tuple[str, str]]]:
    """``(measure names, (table, column) pairs)`` referenced by *expression*.

    Two regexes reach the whole payload. A real DAX parser buys precision that ranking will
    not notice, and the node model does not change if one ever arrives.

    Qualified columns are matched first and their spans excluded, because ``'Sales'[Amount]``
    also matches the bare-reference pattern — read naively, every column reference would
    mint a phantom measure named after the column.
    """
    columns: set[tuple[str, str]] = set()
    spans: list[tuple[int, int]] = []
    for match in _QUALIFIED_COLUMN_RE.finditer(expression):
        table = unquote(match.group("qtable") or "") if match.group("qtable") is not None else match.group("btable")
        columns.add((table, unquote(match.group("column"))))
        spans.append(match.span())

    measures = {
        unquote(match.group("name"))
        for match in _BARE_REF_RE.finditer(expression)
        if not any(start <= match.start() < end for start, end in spans)
    }
    return measures, columns


# ---------------------------------------------------------------------------
# Entity construction
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    project_name: str
    file_path: str
    model: str
    line_count: int = 1
    entities: list[ParsedEntity] = field(default_factory=list)
    relationships: list[ParsedRelationship] = field(default_factory=list)
    root_uid: str = ""
    _uids: set[str] = field(default_factory=set)
    recognised: bool = False
    """Whether any root handler fired.

    Not `len(entities) > 1`: a `model.tmdl` declaring nothing but the model itself has
    exactly one entity, and that entity IS the recognised thing. Counting declined it.
    """

    @property
    def model_qn(self) -> str:
        return f"pbi.model.{self.model}"

    def set_file_root(
        self, *, name: str, qualified_name: str, kind: str, docstring: str = "", extra: dict[str, object] | None = None
    ) -> str:
        """Declare the one entity that stands for this file.

        Every language emits exactly one ``Module`` (or ``DocFile``), and it is not a
        formality: that node is what carries ``file_hash``, and the per-file gate is what
        lets an unchanged file skip parsing entirely. A language that emits none re-parses
        every one of its files on every indexing pass, forever, with no error anywhere --
        the bug ``DocFile`` was added to fix for markdown.

        Spanning line 1 to the end of the file whatever the declaration's own line is,
        because it stands for the document rather than for the object inside it.
        """
        self.root_uid = self._distinct_uid(f"{self.project_name}:{qualified_name}", name)
        # Replaces, never appends. `parse_tmdl` sets a document root up front so tables
        # have something to hang off, and `model.tmdl` then declares the real thing --
        # two calls, and one file may only ever have one Module. A second one would be a
        # second `file_hash` holder for the same path, which is the gate's whole premise.
        root = ParsedEntity(
            name=name,
            qualified_name=self.root_uid,
            label=NodeLabel.MODULE,
            kind=kind,
            line_start=1,
            line_end=max(1, self.line_count),
            file_path=self.file_path,
            visibility=Visibility.PUBLIC,
            docstring=docstring,
            extra_properties=extra or {},
        )
        if self.entities and self.entities[0].label is NodeLabel.MODULE:
            self.entities[0] = root
        else:
            self.entities.insert(0, root)
        return self.root_uid

    def add(
        self,
        *,
        name: str,
        qualified_name: str,
        label: NodeLabel,
        kind: str,
        block: _Block,
        source: str = "",
        docstring: str = "",
        extra: dict[str, object] | None = None,
    ) -> str:
        uid = self._distinct_uid(f"{self.project_name}:{qualified_name}", name)
        self.entities.append(
            ParsedEntity(
                name=name,
                qualified_name=uid,
                label=label,
                kind=kind,
                line_start=block.line,
                line_end=block.last_line,
                file_path=self.file_path,
                visibility=Visibility.PUBLIC,
                source=source,
                docstring=docstring,
                extra_properties=extra or {},
            )
        )
        return uid

    def _distinct_uid(self, uid: str, name: str) -> str:
        """*uid*, made unique within this file.

        ``_qn_segment`` folds five different characters to ``_``, so distinct model objects
        can land on one uid: ``'Sales: YTD'`` and ``'Sales YTD'`` both become
        ``Sales_YTD``. That is not a missing entity but a merged one -- a single node with
        an arbitrary winner's DAX and the union of both edge sets -- and for a *table* it
        takes every measure and column with it.

        The suffix comes from the unfolded name rather than from a counter, so it is stable
        when the author reorders the file. Reordering a counter would change the uid and
        churn the whole subtree for a move that changed nothing.

        Within one file only. Two folding-equal names in two different files still collide,
        which matches what ``config.py`` accepts for the same fold; the realistic case --
        two measures on one table -- is the one closed here.
        """
        if uid not in self._uids:
            self._uids.add(uid)
            return uid
        distinct = f"{uid}~{hashlib.blake2s(name.encode(), digest_size=3).hexdigest()}"
        self._uids.add(distinct)
        return distinct

    def defines(self, parent_uid: str, child_uid: str) -> None:
        self.relationships.append(
            ParsedRelationship(from_qualified_name=parent_uid, rel_type=RelType.DEFINES, to_name=child_uid)
        )


def _describe(props: dict[str, str], extra_keys: tuple[str, ...]) -> dict[str, object]:
    """The property subset worth carrying, plus the lineage tag as a queryable non-identifier."""
    out: dict[str, object] = {k: props[k] for k in extra_keys if k in props}
    if tag := props.get("lineageTag"):
        out["lineageTag"] = tag
    return out


_TABLE_EXTRA = ("isHidden", "mode", "storageMode", "partitionMode")
_MEASURE_EXTRA = ("formatString", "displayFolder", "isHidden", "dataType")
_COLUMN_EXTRA = ("dataType", "isKey", "isHidden", "sourceColumn", "summarizeBy", "formatString", "sortByColumn")


# ---------------------------------------------------------------------------
# Handlers, one per TMDL keyword this parser recognises
# ---------------------------------------------------------------------------


def _handle_model(block: _Block, ctx: _Ctx, _lines: list[str]) -> None:
    """``model.tmdl`` -- the file root IS the model, rather than a wrapper around it."""
    ctx.recognised = True
    props = properties_of(block)
    header = _HEADER_RE.match(block.text)
    ctx.set_file_root(
        name=unquote((header.group("name") if header else "") or ctx.model),
        qualified_name=ctx.model_qn,
        kind=KIND_MODEL,
        docstring=block.description,
        extra=_describe(props, ("culture", "defaultPowerBIDataSourceVersion", "sourceQueryCulture")),
    )


def _handle_table(block: _Block, ctx: _Ctx, lines: list[str]) -> None:
    header = _HEADER_RE.match(block.text)
    table = unquote(header.group("name") or "") if header else ""
    if not table:
        return
    ctx.recognised = True
    props = properties_of(block)
    # A partition says how the table is loaded, and it is a property of the table rather
    # than a thing anyone searches for -- so it is folded on rather than given a node.
    # Its `source` is what ATL-169 reads to find the warehouse object behind the table.
    for child in block.children:
        child_header = _HEADER_RE.match(child.text)
        # Every partition, not the first: a table partitioned by year has one per year,
        # and reading only partition #1 reports a mode the table may not uniformly have.
        if (
            child_header
            and child_header.group("kw").lower() == "partition"
            and (mode := properties_of(child).get("mode", ""))
        ):
            props["mode"] = mode if props.get("mode", mode) == mode else "mixed"

    # Collected before anything is emitted: a measure may be declared above the column it
    # references, and `_emit_dax_edges` needs the whole set to tell a row-context column
    # reference from a measure call.
    own_columns = frozenset(
        unquote(child_header.group("name") or "")
        for child in block.children
        if (child_header := _HEADER_RE.match(child.text)) and child_header.group("kw").lower() == "column"
    )

    table_uid = ctx.add(
        name=table,
        qualified_name=f"{ctx.model_qn}.table.{_qn_segment(table)}",
        label=NodeLabel.TYPE_DEF,
        kind=KIND_TABLE,
        block=block,
        docstring=block.description,
        extra=_describe(props, _TABLE_EXTRA),
    )
    ctx.defines(ctx.root_uid, table_uid)
    # `defaultDetailRowsDefinition` -- DAX hanging off the table itself.
    _emit_expression_children(ctx, table_uid, block, lines, table=table, own_columns=own_columns)

    # Derived from the uid the table ACTUALLY got, not from the one requested: `ctx.add`
    # may have disambiguated it, and children built from the requested name would then
    # hang off a sibling table's namespace.
    table_qn = table_uid.split(":", 1)[1]

    for child in block.children:
        child_header = _HEADER_RE.match(child.text)
        if not child_header:
            continue
        keyword = child_header.group("kw").lower()
        kwargs = {"table": table, "table_uid": table_uid, "table_qn": table_qn, "own_columns": own_columns}
        if keyword == "measure":
            _handle_measure(child, ctx, lines, **kwargs)
        elif keyword == "column":
            _handle_column(child, ctx, lines, **kwargs)
        elif keyword == "calculationgroup":
            _handle_calculation_group(child, ctx, lines, **kwargs)


def _emit_expression_children(
    ctx: _Ctx,
    owner_uid: str,
    block: _Block,
    lines: list[str],
    *,
    table: str,
    own_columns: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Emit DAX edges for *block*'s ``key = <expr>`` children. Returns them for storage.

    These are auxiliary expressions -- a dynamic format string, a detail-rows definition,
    a KPI target -- and each is a dependency of the object that owns it, so the edges
    attribute to the owner rather than minting nodes nobody would search for. Same call
    the partition fold makes.

    Deliberately NOT appended to ``source``: that field holds the object's own expression,
    and a measure whose ``source`` silently also contained its format string would read
    wrong to anyone who fetched it. They go to ``extra_properties``, where Cypher reaches
    them, and their *edges* -- the half that makes blast_radius correct -- are emitted here.
    """
    aux = expression_children(block, lines)
    for expr in aux.values():
        _emit_dax_edges(ctx, owner_uid, expr, default_table=table, own_columns=own_columns)
    return aux


def _handle_measure(
    block: _Block,
    ctx: _Ctx,
    lines: list[str],
    *,
    table: str,
    table_uid: str,
    table_qn: str,
    own_columns: frozenset[str] = frozenset(),
) -> None:
    header = _HEADER_RE.match(block.text)
    name = unquote(header.group("name") or "") if header else ""
    if not name:
        return
    dax = _expression_of(block, lines)
    props = properties_of(block)
    # Read before the entity exists so it can be stored on it; the edges are emitted after,
    # once there is a uid for them to come from.
    aux = expression_children(block, lines)
    uid = ctx.add(
        name=name,
        qualified_name=f"{table_qn}.measure.{_qn_segment(name)}",
        # Callable, deliberately. Measures call each other, and every existing tool reads
        # CALLS as "executes" -- so a three-deep measure chain makes blast_radius answer
        # "what breaks if I change this" with machinery that already exists.
        label=NodeLabel.CALLABLE,
        kind=KIND_MEASURE,
        block=block,
        source=dax,
        docstring=block.description,
        extra=_describe(props, _MEASURE_EXTRA) | aux,
    )
    ctx.defines(table_uid, uid)
    _emit_dax_edges(ctx, uid, dax, default_table=table, own_columns=own_columns)
    _emit_expression_children(ctx, uid, block, lines, table=table, own_columns=own_columns)


def _handle_column(
    block: _Block,
    ctx: _Ctx,
    lines: list[str],
    *,
    table: str,
    table_uid: str,
    table_qn: str,
    own_columns: frozenset[str] = frozenset(),
) -> None:
    header = _HEADER_RE.match(block.text)
    name = unquote(header.group("name") or "") if header else ""
    if not name:
        return
    props = properties_of(block)
    # Present only on a calculated column; an ordinary one is sourced, not computed.
    # Keyed off the parsed default-property group rather than a bare `"=" in text`, which
    # a name like 'Margin = Price - Cost' would have satisfied.
    source = _expression_of(block, lines) if _has_default_property(block) else ""
    uid = ctx.add(
        name=name,
        qualified_name=f"{table_qn}.column.{_qn_segment(name)}",
        label=NodeLabel.VALUE,
        kind=KIND_COLUMN,
        block=block,
        source=source,
        docstring=block.description,
        extra=_describe(props, _COLUMN_EXTRA),
    )
    ctx.defines(table_uid, uid)
    # A calculated column depends on what its DAX reads exactly as a measure does. Without
    # this, blast_radius treats every calculated column as a leaf and the dependency is
    # recoverable only as text.
    _emit_dax_edges(ctx, uid, source, default_table=table, own_columns=own_columns)


def _handle_calculation_group(
    block: _Block,
    ctx: _Ctx,
    lines: list[str],
    *,
    table: str,
    table_uid: str,
    table_qn: str,
    own_columns: frozenset[str] = frozenset(),
) -> None:
    # `multipleOrEmptySelectionExpression` / `noSelectionExpression`: the group's default
    # behaviour, applied to every measure viewed through it, and typically the only place
    # it names the tables it reads.
    _emit_expression_children(ctx, table_uid, block, lines, table=table, own_columns=own_columns)
    for child in block.children:
        header = _HEADER_RE.match(child.text)
        if not header or header.group("kw").lower() != "calculationitem":
            continue
        name = unquote(header.group("name") or "")
        if not name:
            continue
        dax = _expression_of(child, lines)
        uid = ctx.add(
            name=name,
            qualified_name=f"{table_qn}.calcitem.{_qn_segment(name)}",
            label=NodeLabel.CALLABLE,
            kind=KIND_CALC_ITEM,
            block=child,
            source=dax,
            docstring=child.description,
            # `formatStringDefinition` is not in the property tuple, despite being one of
            # this object's documented properties: TMDL writes it with `=`, so it is an
            # expression child, not a `key: value` property, and `properties_of` cannot
            # see it by construction.
            extra=_describe(properties_of(child), ("ordinal",))
            | _emit_expression_children(ctx, table_uid, child, lines, table=table, own_columns=own_columns),
        )
        ctx.defines(table_uid, uid)
        _emit_dax_edges(ctx, uid, dax, default_table=table, own_columns=own_columns)


def _handle_function(block: _Block, ctx: _Ctx, lines: list[str]) -> None:
    """A DAX user-defined function, from ``functions.tmdl``.

    Top-level, not nested under a table, and called from measures, calculation items and
    role filters alike -- so a UDF is frequently the shared dependency several parts of a
    model have in common, and missing it leaves each of them looking independent.

    ``Callable`` for the same reason a measure is: something calls it, and the existing
    tooling reads CALLS as "executes".
    """
    header = _HEADER_RE.match(block.text)
    name = unquote(header.group("name") or "") if header else ""
    if not name:
        return
    ctx.recognised = True
    dax = _expression_of(block, lines)
    uid = ctx.add(
        name=name,
        qualified_name=f"{ctx.model_qn}.function.{_qn_segment(name)}",
        label=NodeLabel.CALLABLE,
        kind=KIND_FUNCTION,
        block=block,
        source=dax,
        # `/// @param` / `/// @returns` is the documented convention for UDF docs, and
        # descriptions are already collected for every object.
        docstring=block.description,
        extra=_describe(properties_of(block), ("returnType",)),
    )
    ctx.defines(ctx.root_uid, uid)
    _emit_dax_edges(ctx, uid, dax, default_table="")


def _handle_role(block: _Block, ctx: _Ctx, lines: list[str]) -> None:
    """A security role from ``roles/<Role>.tmdl``, and the tables its filters read.

    Row-level security is DAX naming real tables and columns, in a file nothing else in
    the model references -- so "what does this role depend on", and its mirror "what
    breaks if I rename this column", have no other witness in the graph.

    Three spellings occur in the wild and the documented one is the rare one: measured
    over public repositories, ``tablePermission <Table> = <DAX>`` (the filter as the
    object's default property, which is what Power BI Desktop writes) outnumbers an
    explicit ``filterExpression`` child roughly 28 to 1, and a colon-assigned
    ``filterExpression:`` appears in hand-written files. All three are read here.
    """
    header = _HEADER_RE.match(block.text)
    name = unquote(header.group("name") or "") if header else ""
    if not name:
        return
    ctx.recognised = True
    props = properties_of(block)
    role_uid = ctx.add(
        name=name,
        qualified_name=f"{ctx.model_qn}.role.{_qn_segment(name)}",
        label=NodeLabel.TYPE_DEF,
        kind=KIND_ROLE,
        block=block,
        docstring=block.description,
        extra=_describe(props, ("modelPermission",)),
    )
    ctx.defines(ctx.root_uid, role_uid)

    for child in block.children:
        child_header = _HEADER_RE.match(child.text)
        if not child_header or child_header.group("kw").lower() != "tablepermission":
            continue
        table = unquote(child_header.group("name") or "")
        if not table:
            continue
        # The permission names its table in the declaration itself, so the edge to it is
        # certain even when the filter expression is one this parser cannot read.
        ctx.relationships.append(
            ParsedRelationship(
                from_qualified_name=role_uid,
                rel_type=RelType.USES_TYPE,
                to_name=table,
                properties={"via": "rls"},
            )
        )
        # Spelling 1: the filter is the default property. Spellings 2 and 3: a
        # `filterExpression` child, assigned with `=` or with `:`.
        filters = [_expression_of(child, lines)] if _has_default_property(child) else []
        filters += list(expression_children(child, lines).values())
        if colon_form := properties_of(child).get("filterExpression"):
            filters.append(colon_form)
        for dax in filters:
            _emit_dax_edges(ctx, role_uid, dax, default_table=table)


def _handle_expression(block: _Block, ctx: _Ctx, lines: list[str]) -> None:
    """A shared M expression -- a named query every partition can draw from."""
    header = _HEADER_RE.match(block.text)
    name = unquote(header.group("name") or "") if header else ""
    if not name:
        return
    ctx.recognised = True
    props = properties_of(block)
    expression_uid = ctx.add(
        name=name,
        qualified_name=f"{ctx.model_qn}.expression.{_qn_segment(name)}",
        label=NodeLabel.CALLABLE,
        kind=KIND_EXPRESSION,
        block=block,
        source=_expression_of(block, lines),
        docstring=block.description,
        extra=_describe(props, ("queryGroup", "kind")),
    )
    ctx.defines(ctx.root_uid, expression_uid)


def _emit_dax_edges(
    ctx: _Ctx, from_uid: str, dax: str, *, default_table: str, own_columns: frozenset[str] = frozenset()
) -> None:
    """CALLS for every measure this expression reads, USES_TYPE for every table.

    Measure names are unique across a Power BI model, so a bare name resolves cleanly
    through ``resolve_calls``' project-wide rung -- and where two models in one project
    share a measure name, that rung grades the edge ``ambiguous`` rather than guessing,
    which is the right answer.

    The column reference lands on the **table**, not the column. ``USES_TYPE`` resolves
    against ``TypeDef`` (``resolve_type_refs``), and a column is a ``Value``, so a
    column-targeted edge would resolve to nothing at all. The column name rides along in
    the edge's properties, so nothing is lost and a column-precise edge is a later
    refinement rather than a re-parse.
    """
    if not dax:
        return
    measures, columns = dax_references(dax)
    for measure in sorted(measures):
        # A bare `[X]` is a measure reference EXCEPT in row context, where it is a column
        # of the table being iterated: `SUMX('Sales', [Quantity] * [Net Price])` names two
        # columns, not two measures. DAX makes the two syntactically identical, so without
        # a symbol table the only thing separating them is knowing this table's own
        # columns -- which the caller does.
        #
        # Getting it wrong is not a missed edge but a wrong one, and measures are very
        # often named after the column they aggregate, so a phantom CALLS would frequently
        # find a real measure to land on and read as structural fact. ADR-0014's premise
        # is that a confident wrong edge is worse than none.
        if measure in own_columns:
            columns.add((default_table, measure))
            continue
        ctx.relationships.append(
            ParsedRelationship(from_qualified_name=from_uid, rel_type=RelType.CALLS, to_name=measure)
        )
    for table, column in sorted(columns):
        target = table or default_table
        if not target:
            continue
        ctx.relationships.append(
            ParsedRelationship(
                from_qualified_name=from_uid,
                rel_type=RelType.USES_TYPE,
                to_name=target,
                properties={"column": column, "via": "dax"},
            )
        )


_ROOT_HANDLERS = {
    "model": _handle_model,
    "table": _handle_table,
    "expression": _handle_expression,
    "function": _handle_function,
    "role": _handle_role,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_tmdl(path: str, source: bytes, project_name: str) -> ParsedFile | None:
    """Parse one ``.tmdl`` file. ``None`` when it declares nothing this parser models.

    Declining is not failing: the framework turns it into an empty ``ParsedFile``, which
    is hashed and therefore skipped on every later pass. ``culture``, ``role``,
    ``perspective`` and ``queryGroup`` files all land here on purpose -- they carry
    formatting and permissions, not model structure.
    """
    # `utf-8-sig` strips a BOM when there is one and behaves as plain utf-8 when there is
    # not. Measured on a 322-file real-world sample: 3 files carry one. Left in place it
    # becomes a leading `\ufeff` on line 1, so the file's only root declaration fails to
    # match `_HEADER_RE` and the whole model file is silently dropped.
    text = source.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    ctx = _Ctx(project_name=project_name, file_path=path, model=model_name_for(path), line_count=len(lines))

    # The document node, before anything can want to hang off it. `model.tmdl` overwrites
    # it with the model itself; every other file keeps this one, named for the file.
    #
    # It is also what makes containment work at all. TMDL splits one model across many
    # files, and a `Module -DEFINES-> table` edge pointing across them cannot be written
    # at upsert time: uid-routed edges MATCH both endpoints, and nothing orders the files
    # within a batch. Anchoring each table to its own file's document node keeps every
    # structural edge inside the file that states it -- which is also the rule
    # `_recreate_file_relationships` deletes by.
    stem = PurePosixPath(path).stem
    ctx.set_file_root(
        name=stem,
        qualified_name=f"{ctx.model_qn}.definition.{_qn_segment(stem)}",
        kind=KIND_DEFINITION,
    )

    for block in parse_blocks(text):
        header = _HEADER_RE.match(block.text)
        # Lowercased: the TMDL API writes camelCase but reads case-insensitively, so a
        # hand-edited `Table Sales` is valid and must not silently drop the table.
        if header and (handler := _ROOT_HANDLERS.get(header.group("kw").lower())):
            handler(block, ctx, lines)

    # Nothing recognised -- a culture, role or perspective file. Declining is right: the
    # framework turns it into an empty ParsedFile, which is hashed and skipped on every
    # later pass. Tracked as a flag rather than an entity count, because a `model.tmdl`
    # declaring only the model has exactly one entity and that entity is the answer.
    if not ctx.recognised:
        return None
    return ParsedFile(
        file_path=path,
        language="tmdl",
        entities=ctx.entities,
        relationships=ctx.relationships,
    )


register_language(
    LanguageConfig(
        name="tmdl",
        extensions=frozenset({".tmdl"}),
        # No grammar: TMDL is indentation-scoped, tree-sitter needs an external scanner for
        # that, and Microsoft publishes none. See LanguageConfig.language.
        language=None,
        query=None,
        text_parse_func=parse_tmdl,
    )
)
