"""Graph schema definitions for Code Atlas.

Defines node labels, relationship types, kind discriminators, constraint/index
specs, and pure DDL generation functions.  Import-time validation ensures every
NodeLabel is covered by constraint and index registries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# Schema version — bump on every schema change that requires migration.
SCHEMA_VERSION: int = 14

# Sentinel ``project_name`` for nodes that are shared across every project.
#
# ``project_name`` carries an existence constraint on every entity label, so a
# genuinely global node cannot simply omit it (Memgraph reports the violation
# at COMMIT, not at statement time, so the offending write appears to succeed
# and the whole transaction dies later).  A reserved sentinel is the only way
# to express "belongs to no single project" — see NodeLabel.ENV_VAR.
#
# Anything that filters by project must treat this value as a member of every
# project, and anything that refuses to touch non-test data must allowlist it
# (see tests/integration/conftest.py::_assert_disposable_db).
GLOBAL_PROJECT: str = "_global"

# ---------------------------------------------------------------------------
# Node labels
# ---------------------------------------------------------------------------


class NodeLabel(StrEnum):
    # Containers
    PROJECT = "Project"
    PACKAGE = "Package"
    MODULE = "Module"
    # Code entities (discriminated by `kind`)
    TYPE_DEF = "TypeDef"
    CALLABLE = "Callable"
    VALUE = "Value"
    # Documentation
    DOC_FILE = "DocFile"
    DOC_SECTION = "DocSection"
    # Knowledge vault (zettelkasten notes — code-atlas + memory-dir dialects)
    NOTE = "Note"
    # External dependencies
    EXTERNAL_PACKAGE = "ExternalPackage"
    EXTERNAL_SYMBOL = "ExternalSymbol"
    # Referenced-but-not-defined runtime surface (see _EXTERNAL_LABELS)
    ENV_VAR = "EnvVar"
    RESOURCE_FILE = "ResourceFile"
    # Marker: stamped onto every _ENTITY_LABELS node alongside its primary
    # label (see _MARKER_LABELS) so a uid-only lookup that doesn't know which
    # of the 12 entity labels it's after can use one shared index instead of
    # an unindexed ScanAll over the whole graph.
    ENTITY = "Entity"
    # Meta
    SCHEMA_VERSION = "SchemaVersion"


# ---------------------------------------------------------------------------
# Relationship types
# ---------------------------------------------------------------------------


class RelType(StrEnum):
    # Structural
    CONTAINS = "CONTAINS"
    DEFINES = "DEFINES"
    # Type hierarchy
    INHERITS = "INHERITS"
    IMPLEMENTS = "IMPLEMENTS"
    # Call / data flow
    CALLS = "CALLS"
    # Names a callable as a VALUE without invoking it: passed as an argument, held in a
    # dispatch table, handed to a decorator. Deliberately not CALLS — blast_radius,
    # find_dead_code, the outline and Leiden all read CALLS as "executes", and a callback
    # that is registered today and invoked by a framework tomorrow did not execute here.
    REFERENCES = "REFERENCES"
    IMPORTS = "IMPORTS"
    USES_TYPE = "USES_TYPE"
    OVERRIDES = "OVERRIDES"
    # Runtime configuration surface (code -> EnvVar / ResourceFile)
    READS_ENV = "READS_ENV"
    REFERENCES_FILE = "REFERENCES_FILE"
    # Dependencies
    DEPENDS_ON = "DEPENDS_ON"
    # Documentation
    DOCUMENTS = "DOCUMENTS"
    # Similarity
    SIMILAR_TO = "SIMILAR_TO"
    EXPORTS = "EXPORTS"
    # Pattern-detected
    HANDLES_ROUTE = "HANDLES_ROUTE"
    HANDLES_EVENT = "HANDLES_EVENT"
    REGISTERED_BY = "REGISTERED_BY"
    INJECTED_INTO = "INJECTED_INTO"
    TESTS = "TESTS"
    HANDLES_COMMAND = "HANDLES_COMMAND"
    # Knowledge vault (Note <-> Note/DocSection)
    LINKS_TO = "LINKS_TO"
    DERIVED_FROM = "DERIVED_FROM"
    SUPERSEDES = "SUPERSEDES"
    # Git-derived signals (Module <-> Module, out-of-band — see git_signals.py)
    CO_CHANGES_WITH = "CO_CHANGES_WITH"


# ---------------------------------------------------------------------------
# Kind discriminators
# ---------------------------------------------------------------------------


class TypeDefKind(StrEnum):
    CLASS = "class"
    STRUCT = "struct"
    INTERFACE = "interface"
    TRAIT = "trait"
    ENUM = "enum"
    UNION = "union"
    TYPE_ALIAS = "type_alias"
    PROTOCOL = "protocol"
    RECORD = "record"
    DATA_TYPE = "data_type"
    TYPECLASS = "typeclass"
    ANNOTATION = "annotation"


class CallableKind(StrEnum):
    FUNCTION = "function"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    DESTRUCTOR = "destructor"
    STATIC_METHOD = "static_method"
    CLASS_METHOD = "class_method"
    PROPERTY = "property"
    CLOSURE = "closure"


class ValueKind(StrEnum):
    VARIABLE = "variable"
    CONSTANT = "constant"
    FIELD = "field"
    ENUM_MEMBER = "enum_member"


class NoteKind(StrEnum):
    """Note lifecycle stage — stored in the shared ``kind`` property, like CallableKind."""

    DRAFT = "draft"
    NOTE = "note"
    DECISION = "decision"


class Visibility(StrEnum):
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    INTERNAL = "internal"


# ---------------------------------------------------------------------------
# Label groupings
# ---------------------------------------------------------------------------

_CODE_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.PROJECT,
        NodeLabel.PACKAGE,
        NodeLabel.MODULE,
        NodeLabel.TYPE_DEF,
        NodeLabel.CALLABLE,
        NodeLabel.VALUE,
    }
)

_DOC_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.DOC_FILE,
        NodeLabel.DOC_SECTION,
        NodeLabel.NOTE,
    }
)

# "Exists only because something referenced it" — no source location of its
# own, MERGEd during post-batch resolution, never produced by a parser as a
# ParsedEntity.  EnvVar/ResourceFile share exactly that lifecycle with
# ExternalPackage/ExternalSymbol, which is why they live here rather than in
# _CODE_LABELS: joining this group is what gives them uid uniqueness, the
# uid+project_name existence constraints, and the property/composite indices.
_EXTERNAL_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.EXTERNAL_PACKAGE,
        NodeLabel.EXTERNAL_SYMBOL,
        NodeLabel.ENV_VAR,
        NodeLabel.RESOURCE_FILE,
    }
)

# Reference-counted labels: a node exists exactly as long as something points
# at it, so "zero incoming edges" means "unreferenced" and the node can be
# swept (see GraphBackend.gc_orphaned_reference_nodes).  ExternalPackage and
# ExternalSymbol are deliberately NOT here — ExternalPackage carries a
# per-project ``version`` and receives a structural CONTAINS edge from its
# package, so incoming-edge count is not a reference count for them.
#
# INVARIANT: never give these labels a structural incoming edge (a Project or
# Package CONTAINS, say).  It would make every node permanently referenced and
# silently disable the sweep.
_REFERENCE_COUNTED_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.ENV_VAR,
        NodeLabel.RESOURCE_FILE,
    }
)

_EMBEDDABLE_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.TYPE_DEF,
        NodeLabel.CALLABLE,
        NodeLabel.VALUE,
        NodeLabel.MODULE,
        NodeLabel.DOC_SECTION,
        NodeLabel.NOTE,
    }
)

# EnvVar/ResourceFile are text-searchable but NOT embeddable: an agent asking
# "where is DATABASE_URL read?" needs the keyword hit, and there is nothing to
# embed — the node is a bare identifier, and by the names-only invariant it
# never holds the variable's value or a default.
_TEXT_SEARCHABLE_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.TYPE_DEF,
        NodeLabel.CALLABLE,
        NodeLabel.VALUE,
        NodeLabel.MODULE,
        NodeLabel.DOC_SECTION,
        NodeLabel.NOTE,
        NodeLabel.ENV_VAR,
        NodeLabel.RESOURCE_FILE,
    }
)

# Stamped alongside a primary entity label on every node in this union, purely
# so a caller holding only a uid (not knowing which of the other 12 labels it
# belongs to) can MATCH (n:Entity {uid: ...}) against one shared index instead
# of an unindexed ScanAll over the whole graph. See NodeLabel.ENTITY.
#
# Deliberately NOT folded into _ENTITY_LABELS: that set drives the constraint and
# index registries, and a marker sits on *every node in the graph*, so each
# registry it joins is a cost every write pays. uid/project_name existence and
# uniqueness are already enforced by the primary label underneath it, and only
# two of the seven indexed properties are ever reached through the marker --
# see _MARKER_INDEX_PROPERTIES.
_MARKER_LABELS: frozenset[NodeLabel] = frozenset({NodeLabel.ENTITY})

# All non-meta labels (must have uid + project_name)
_ENTITY_LABELS: frozenset[NodeLabel] = _CODE_LABELS | _DOC_LABELS | _EXTERNAL_LABELS


# ---------------------------------------------------------------------------
# uid construction for reference-counted nodes
#
# Shared by both graph backends (and by the parsers that emit the references)
# so the two can never drift on the key that identifies the node.
# ---------------------------------------------------------------------------

ENV_VAR_PREFIX: str = "env/"
RESOURCE_FILE_PREFIX: str = "res/"


def env_var_uid(name: str) -> str:
    """uid for an environment variable — GLOBAL, deliberately unprefixed by project.

    Breaks the usual ``{project_name}:{qualified_name}`` uid format on purpose:
    an env var means the same thing in every repo, has no version dimension and
    carries no per-project attributes, so every callsite everywhere converges on
    one node.  That is what makes "who reads DATABASE_URL across all my repos" a
    single-node lookup instead of a name-join.  The node still needs a
    ``project_name`` for the existence constraint — it gets ``GLOBAL_PROJECT``.
    """
    return f"{ENV_VAR_PREFIX}{name}"


def resource_file_uid(project_name: str, path: str) -> str:
    """uid for a referenced (not indexed) file — PROJECT-SCOPED, unlike env vars.

    The asymmetry with :func:`env_var_uid` is intentional: a path is only
    meaningful relative to a project root, so ``data/fixtures.json`` in two
    repos is two different files and must not collapse into one node.
    """
    return f"{project_name}:{RESOURCE_FILE_PREFIX}{path}"


# ---------------------------------------------------------------------------
# Spec dataclasses (frozen, for generating DDL)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniqueConstraintSpec:
    label: NodeLabel
    property: str


@dataclass(frozen=True)
class ExistenceConstraintSpec:
    label: NodeLabel
    property: str


@dataclass(frozen=True)
class IndexSpec:
    label: NodeLabel
    property: str


@dataclass(frozen=True)
class CompositeIndexSpec:
    label: NodeLabel
    properties: tuple[str, ...]


@dataclass(frozen=True)
class VectorIndexSpec:
    name: str
    label: NodeLabel
    property: str
    dimension: int
    capacity: int
    metric: str = "cos"


@dataclass(frozen=True)
class TextIndexSpec:
    name: str
    label: NodeLabel


# ---------------------------------------------------------------------------
# Constraint and index registries
# ---------------------------------------------------------------------------

# uid uniqueness on all entity labels; version on SchemaVersion
UNIQUE_CONSTRAINTS: tuple[UniqueConstraintSpec, ...] = (
    *[UniqueConstraintSpec(label=lbl, property="uid") for lbl in sorted(_ENTITY_LABELS, key=lambda lbl: lbl.value)],
    UniqueConstraintSpec(label=NodeLabel.SCHEMA_VERSION, property="version"),
)

# uid + project_name existence on all entity labels; version on SchemaVersion
EXISTENCE_CONSTRAINTS: tuple[ExistenceConstraintSpec, ...] = (
    *[
        spec
        for lbl in sorted(_ENTITY_LABELS, key=lambda lbl: lbl.value)
        for spec in (
            ExistenceConstraintSpec(label=lbl, property="uid"),
            ExistenceConstraintSpec(label=lbl, property="project_name"),
        )
    ],
    ExistenceConstraintSpec(label=NodeLabel.SCHEMA_VERSION, property="version"),
)

# Property indices for fast lookups
_INDEX_PROPERTIES: tuple[str, ...] = (
    "uid",
    "qualified_name",
    "file_path",
    "name",
    "project_name",
    "kind",
    "content_hash",
)

# The only properties a marker label is looked up by: uid, for the uid-only
# MATCHes the marker exists to index, and (project_name, name) for cross-project
# import resolution. Everything else is reached through a primary label, which
# has its own index -- indexing it on the marker too would buy no query anything
# and cost every single node write.
_MARKER_INDEX_PROPERTIES: tuple[str, ...] = ("uid",)

LABEL_PROPERTY_INDICES: tuple[IndexSpec, ...] = (
    *(
        IndexSpec(label=lbl, property=prop)
        for lbl in sorted(_ENTITY_LABELS, key=lambda lbl: lbl.value)
        for prop in _INDEX_PROPERTIES
    ),
    *(
        IndexSpec(label=lbl, property=prop)
        for lbl in sorted(_MARKER_LABELS, key=lambda lbl: lbl.value)
        for prop in _MARKER_INDEX_PROPERTIES
    ),
)

# Composite property indices for multi-property lookups
_COMPOSITE_PROPERTIES: tuple[tuple[str, ...], ...] = (
    ("project_name", "file_path"),
    ("project_name", "name"),
)

_MARKER_COMPOSITE_PROPERTIES: tuple[tuple[str, ...], ...] = (("project_name", "name"),)

COMPOSITE_INDICES: tuple[CompositeIndexSpec, ...] = (
    *(
        CompositeIndexSpec(label=lbl, properties=props)
        for lbl in sorted(_ENTITY_LABELS, key=lambda lbl: lbl.value)
        for props in _COMPOSITE_PROPERTIES
    ),
    *(
        CompositeIndexSpec(label=lbl, properties=props)
        for lbl in sorted(_MARKER_LABELS, key=lambda lbl: lbl.value)
        for props in _MARKER_COMPOSITE_PROPERTIES
    ),
)

# Text (BM25) indices — one per searchable label
TEXT_INDICES: tuple[TextIndexSpec, ...] = tuple(
    TextIndexSpec(name=f"text_{lbl.value.lower()}", label=lbl)
    for lbl in sorted(_TEXT_SEARCHABLE_LABELS, key=lambda lbl: lbl.value)
)


def build_vector_index_specs(dimension: int, capacity: int = 50_000) -> tuple[VectorIndexSpec, ...]:
    """Build vector index specs from settings (dimension is runtime config)."""
    return tuple(
        VectorIndexSpec(
            name=f"vec_{lbl.value.lower()}",
            label=lbl,
            property="embedding",
            dimension=dimension,
            capacity=capacity,
        )
        for lbl in sorted(_EMBEDDABLE_LABELS, key=lambda lbl: lbl.value)
    )


# ---------------------------------------------------------------------------
# Cypher fragments
# ---------------------------------------------------------------------------


def primary_label_expr(var: str = "n") -> str:
    """Cypher expression for *var*'s own label, ignoring any marker label.

    Every entity node carries a marker (:Entity) alongside the label saying what
    it actually is, so ``labels(n)[0]`` is not reliably that label any more.
    Memgraph returns labels in write order, which makes "primary label first" a
    convention the write sites happen to keep, not something the database
    promises -- and a node written ``:Entity:Callable`` would silently report its
    type as ``Entity`` in every search result, diagram and impact row.

    Filtering markers out instead survives a marker being added, reordered, or
    missing altogether (a node written before the marker existed still yields its
    real label rather than null).
    """
    markers = ", ".join(f"'{lbl.value}'" for lbl in sorted(_MARKER_LABELS, key=lambda lbl: lbl.value))
    return f"[_lbl IN labels({var}) WHERE NOT _lbl IN [{markers}]][0]"


# ---------------------------------------------------------------------------
# DDL generation (pure functions, no I/O)
# ---------------------------------------------------------------------------


def generate_unique_constraint_ddl() -> list[str]:
    """Generate CREATE CONSTRAINT statements for unique properties."""
    return [
        f"CREATE CONSTRAINT ON (n:{spec.label.value}) ASSERT n.{spec.property} IS UNIQUE;"
        for spec in UNIQUE_CONSTRAINTS
    ]


def generate_existence_constraint_ddl() -> list[str]:
    """Generate CREATE CONSTRAINT statements for mandatory properties."""
    return [
        f"CREATE CONSTRAINT ON (n:{spec.label.value}) ASSERT EXISTS (n.{spec.property});"
        for spec in EXISTENCE_CONSTRAINTS
    ]


def generate_index_ddl() -> list[str]:
    """Generate CREATE INDEX statements for label-property pairs."""
    return [f"CREATE INDEX ON :{spec.label.value}({spec.property});" for spec in LABEL_PROPERTY_INDICES]


def generate_composite_index_ddl() -> list[str]:
    """Generate CREATE INDEX statements for composite (multi-property) indices."""
    return [f"CREATE INDEX ON :{spec.label.value}({', '.join(spec.properties)});" for spec in COMPOSITE_INDICES]


def generate_vector_index_ddl(dimension: int, capacity: int = 50_000) -> list[str]:
    """Generate CREATE VECTOR INDEX statements for embeddable labels (Memgraph 3.7+ DDL)."""
    specs = build_vector_index_specs(dimension, capacity)
    return [
        (
            f"CREATE VECTOR INDEX {spec.name} ON :{spec.label.value}({spec.property})"
            f' WITH CONFIG {{"dimension": {spec.dimension}, "capacity": {spec.capacity}, "metric": "{spec.metric}"}};'
        )
        for spec in specs
    ]


def generate_text_index_ddl() -> list[str]:
    """Generate CREATE TEXT INDEX statements for BM25 searchable labels (Memgraph 3.7+ DDL)."""
    return [f"CREATE TEXT INDEX {spec.name} ON :{spec.label.value};" for spec in TEXT_INDICES]


def generate_drop_vector_index_ddl() -> list[str]:
    """Generate DROP statements for all vector indices (Memgraph 3.7+ DDL)."""
    return [
        f"DROP VECTOR INDEX {spec.name};"
        for spec in build_vector_index_specs(0)  # dimension irrelevant for drops
    ]


def generate_drop_text_index_ddl() -> list[str]:
    """Generate DROP statements for all text indices (Memgraph 3.7+ DDL)."""
    return [f"DROP TEXT INDEX {spec.name};" for spec in TEXT_INDICES]


def generate_drop_redundant_marker_ddl() -> list[str]:
    """Generate DROP statements for marker-label indices and constraints nothing queries.

    Schema v13 introduced :Entity by folding _MARKER_LABELS into _ENTITY_LABELS --
    the set that drives every constraint and index registry -- so the marker picked
    up all seven property indices, both composites, and uid/project_name uniqueness
    and existence. A marker sits on *every node in the graph*, so that is nine index
    structures and three constraint checks paid by every single write, from a label
    whose entire purpose was to make writes cheaper.

    Only _MARKER_INDEX_PROPERTIES and _MARKER_COMPOSITE_PROPERTIES are ever reached
    through the marker. Everything else is reached through a primary label that has
    its own index, and uid/project_name are already unique and mandatory there.

    Schema application is additive -- _apply_full_schema/_migrate_indices only ever
    CREATE -- so a database that ran v13 keeps all nine until these statements run.
    """
    stmts: list[str] = []
    for lbl in sorted(_MARKER_LABELS, key=lambda lbl: lbl.value):
        stmts.extend(
            f"DROP INDEX ON :{lbl.value}({prop});"
            for prop in sorted(set(_INDEX_PROPERTIES) - set(_MARKER_INDEX_PROPERTIES))
        )
        stmts.extend(
            f"DROP INDEX ON :{lbl.value}({', '.join(props)});"
            for props in sorted(set(_COMPOSITE_PROPERTIES) - set(_MARKER_COMPOSITE_PROPERTIES))
        )
        stmts.append(f"DROP CONSTRAINT ON (n:{lbl.value}) ASSERT n.uid IS UNIQUE;")
        stmts.extend(
            f"DROP CONSTRAINT ON (n:{lbl.value}) ASSERT EXISTS (n.{prop});" for prop in ("project_name", "uid")
        )
    return stmts


# ---------------------------------------------------------------------------
# Import-time validation
# ---------------------------------------------------------------------------


def _validate_schema_completeness() -> None:
    """Ensure every NodeLabel is covered by constraint/index registries.

    Raises RuntimeError at import time if any label is missing — prevents
    the silent-drop bug (competitor insight P0).
    """
    all_labels = set(NodeLabel)
    # A marker never carries uid/project_name of its own -- it is stamped onto a
    # node that already has both, constrained by its primary label. Constraining
    # them again on the marker would re-check every write for nothing.
    constrained_labels = all_labels - _MARKER_LABELS

    # Unique constraints: every non-marker label must appear
    unique_labels = {spec.label for spec in UNIQUE_CONSTRAINTS}
    missing_unique = constrained_labels - unique_labels
    if missing_unique:
        raise RuntimeError(f"NodeLabels missing from UNIQUE_CONSTRAINTS: {missing_unique}")

    # Existence constraints: every non-marker label must appear
    existence_labels = {spec.label for spec in EXISTENCE_CONSTRAINTS}
    missing_existence = constrained_labels - existence_labels
    if missing_existence:
        raise RuntimeError(f"NodeLabels missing from EXISTENCE_CONSTRAINTS: {missing_existence}")

    # Label property indices: entity and marker labels must appear (SchemaVersion exempt).
    # Markers are here on purpose -- being indexed is the only reason one exists.
    index_labels = {spec.label for spec in LABEL_PROPERTY_INDICES}
    missing_index = (_ENTITY_LABELS | _MARKER_LABELS) - index_labels
    if missing_index:
        raise RuntimeError(f"Entity labels missing from LABEL_PROPERTY_INDICES: {missing_index}")

    # Label groupings must cover all non-meta labels
    grouped = _CODE_LABELS | _DOC_LABELS | _EXTERNAL_LABELS | _MARKER_LABELS | {NodeLabel.SCHEMA_VERSION}
    missing_group = all_labels - grouped
    if missing_group:
        raise RuntimeError(f"NodeLabels not in any label group: {missing_group}")


_validate_schema_completeness()
