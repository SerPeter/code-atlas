"""Graph schema definitions for Code Atlas.

Defines node labels, relationship types, kind discriminators, constraint/index
specs, and pure DDL generation functions.  Import-time validation ensures every
NodeLabel is covered by constraint and index registries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# Schema version — bump on every schema change that requires migration.
SCHEMA_VERSION: int = 1

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
    ADR = "ADR"
    # External dependencies
    EXTERNAL_PACKAGE = "ExternalPackage"
    EXTERNAL_SYMBOL = "ExternalSymbol"
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
    IMPORTS = "IMPORTS"
    USES_TYPE = "USES_TYPE"
    OVERRIDES = "OVERRIDES"
    # Dependencies
    DEPENDS_ON = "DEPENDS_ON"
    # Documentation
    DOCUMENTS = "DOCUMENTS"
    MOTIVATED_BY = "MOTIVATED_BY"
    # Similarity
    SIMILAR_TO = "SIMILAR_TO"
    # Pattern-detected
    HANDLES_ROUTE = "HANDLES_ROUTE"
    HANDLES_EVENT = "HANDLES_EVENT"
    REGISTERED_BY = "REGISTERED_BY"
    INJECTED_INTO = "INJECTED_INTO"
    TESTS = "TESTS"
    HANDLES_COMMAND = "HANDLES_COMMAND"


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
        NodeLabel.ADR,
    }
)

_EXTERNAL_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.EXTERNAL_PACKAGE,
        NodeLabel.EXTERNAL_SYMBOL,
    }
)

_EMBEDDABLE_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.TYPE_DEF,
        NodeLabel.CALLABLE,
        NodeLabel.VALUE,
        NodeLabel.MODULE,
        NodeLabel.DOC_SECTION,
    }
)

_TEXT_SEARCHABLE_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.TYPE_DEF,
        NodeLabel.CALLABLE,
        NodeLabel.VALUE,
        NodeLabel.MODULE,
        NodeLabel.DOC_SECTION,
    }
)

# All non-meta labels (must have uid + project_name)
_ENTITY_LABELS: frozenset[NodeLabel] = _CODE_LABELS | _DOC_LABELS | _EXTERNAL_LABELS

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
    "qualified_name",
    "file_path",
    "name",
    "project_name",
    "kind",
    "content_hash",
)

LABEL_PROPERTY_INDICES: tuple[IndexSpec, ...] = tuple(
    IndexSpec(label=lbl, property=prop)
    for lbl in sorted(_ENTITY_LABELS, key=lambda lbl: lbl.value)
    for prop in _INDEX_PROPERTIES
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
    """Generate CREATE TEXT INDEX statements for BM25 searchable labels."""
    return [f"CALL text_search.create_index('{spec.name}', '{spec.label.value}');" for spec in TEXT_INDICES]


def generate_drop_vector_index_ddl() -> list[str]:
    """Generate DROP statements for all vector indices (Memgraph 3.7+ DDL)."""
    return [
        f"DROP VECTOR INDEX {spec.name};"
        for spec in build_vector_index_specs(0)  # dimension irrelevant for drops
    ]


def generate_drop_text_index_ddl() -> list[str]:
    """Generate DROP statements for all text indices."""
    return [f"CALL text_search.drop_index('{spec.name}');" for spec in TEXT_INDICES]


# ---------------------------------------------------------------------------
# Import-time validation
# ---------------------------------------------------------------------------


def _validate_schema_completeness() -> None:
    """Ensure every NodeLabel is covered by constraint/index registries.

    Raises RuntimeError at import time if any label is missing — prevents
    the silent-drop bug (competitor insight P0).
    """
    all_labels = set(NodeLabel)

    # Unique constraints: every label must appear
    unique_labels = {spec.label for spec in UNIQUE_CONSTRAINTS}
    missing_unique = all_labels - unique_labels
    if missing_unique:
        raise RuntimeError(f"NodeLabels missing from UNIQUE_CONSTRAINTS: {missing_unique}")

    # Existence constraints: every label must appear
    existence_labels = {spec.label for spec in EXISTENCE_CONSTRAINTS}
    missing_existence = all_labels - existence_labels
    if missing_existence:
        raise RuntimeError(f"NodeLabels missing from EXISTENCE_CONSTRAINTS: {missing_existence}")

    # Label property indices: all entity labels must appear (SchemaVersion exempt)
    index_labels = {spec.label for spec in LABEL_PROPERTY_INDICES}
    missing_index = _ENTITY_LABELS - index_labels
    if missing_index:
        raise RuntimeError(f"Entity labels missing from LABEL_PROPERTY_INDICES: {missing_index}")

    # Label groupings must cover all non-meta labels
    grouped = _CODE_LABELS | _DOC_LABELS | _EXTERNAL_LABELS | {NodeLabel.SCHEMA_VERSION}
    missing_group = all_labels - grouped
    if missing_group:
        raise RuntimeError(f"NodeLabels not in any label group: {missing_group}")


_validate_schema_completeness()
