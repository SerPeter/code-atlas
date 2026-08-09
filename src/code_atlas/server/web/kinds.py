"""The twelve node kinds, and how the map draws them (v1.1 design).

Silhouette *and* colour both carry kind on the entity level. That is not redundancy:
every entity inside one module shares a community, so the community hue is unused there
and free to mean something else. Dashed means the entity sits outside the index.

The graph stores a `label` (Callable, TypeDef, Value, …) and a finer `kind`
(method, constructor, field, …). The design names twelve; the index emits more, so the
extras fold into the nearest drawn kind rather than being dropped — an entity missing
from the picture is worse than one drawn as its close relative.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Kind:
    """One drawable kind."""

    id: str
    label: str
    shape: str
    note: str = ""

    @property
    def color(self) -> str:
        return _KIND_COLOR[self.id]


# Order is the rail's order: containers, code, data, docs, then what is not ours.
KINDS: tuple[Kind, ...] = (
    Kind("module", "Module", "ring", "a file"),
    Kind("package", "Package", "ring", "a directory"),
    Kind("class", "Class", "square"),
    Kind("function", "Function", "circle"),
    Kind("method", "Method", "circle", "bound to a class"),
    Kind("constant", "Constant", "diamond"),
    Kind("env_var", "Environment variable", "diamond"),
    Kind("doc_file", "Doc file", "rect"),
    Kind("doc_section", "Doc section", "rect"),
    Kind("knowledge_note", "Knowledge note", "rect", "authored, not extracted"),
    Kind("external_package", "External package", "hollow", "outside the index"),
    Kind("external_symbol", "External symbol", "hollow", "outside the index"),
)

_KIND_COLOR = {
    "module": "var(--atlas-c8)",
    "package": "var(--atlas-c8)",
    "class": "var(--atlas-c5)",
    "function": "var(--atlas-c4)",
    "method": "var(--atlas-c3)",
    "constant": "var(--atlas-c1)",
    "env_var": "var(--atlas-c0)",
    "doc_file": "var(--atlas-c7)",
    "doc_section": "var(--atlas-c7)",
    "knowledge_note": "var(--atlas-c6)",
    "external_package": "var(--atlas-c2)",
    "external_symbol": "var(--atlas-c2)",
}

BY_ID = {k.id: k for k in KINDS}

# The index's `kind` values that the design does not name separately. Each folds into
# its nearest drawn kind: a constructor and a static method are methods, a field and a
# variable are data. Nothing is discarded.
_KIND_ALIASES = {
    "constructor": "method",
    "static_method": "method",
    "property": "method",
    "classmethod": "method",
    "field": "constant",
    "variable": "constant",
    "attribute": "constant",
    "enum_member": "constant",
    "type_alias": "class",
    "interface": "class",
    "struct": "class",
    "enum": "class",
    "trait": "class",
}

# Fallback by graph label, for a `kind` this does not recognise at all.
_LABEL_DEFAULT = {
    "Module": "module",
    "Package": "package",
    "TypeDef": "class",
    "Callable": "function",
    "Value": "constant",
    "DocFile": "doc_file",
    "DocSection": "doc_section",
    "Note": "knowledge_note",
    "ExternalPackage": "external_package",
    "ExternalSymbol": "external_symbol",
    "EnvVar": "env_var",
    "ResourceFile": "doc_file",
}


def classify(label: str, kind: str) -> str:
    """The drawn kind for a graph node.

    Falls back through: the design's own ids, the alias table, then the node's label.
    An unrecognised entity is drawn as a function rather than omitted — the map is a
    picture of what is there, and a silent gap is the one thing it must not have.
    """
    key = (kind or "").strip().lower()
    if key in BY_ID:
        return key
    if key in _KIND_ALIASES:
        return _KIND_ALIASES[key]
    return _LABEL_DEFAULT.get(label, "function")
