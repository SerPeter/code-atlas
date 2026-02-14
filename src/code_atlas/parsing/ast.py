"""Tree-sitter based parser for extracting code entities and relationships.

Parses source files using py-tree-sitter and extracts entities (classes,
functions, methods, imports, variables) and relationships (DEFINES, CALLS,
IMPORTS, INHERITS) for graph ingestion.

Language-specific parsers live in ``parsing.languages.*`` and register via
``register_language()`` at import time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from tree_sitter import Language, Parser, Query

from code_atlas.schema import NodeLabel, RelType, Visibility

if TYPE_CHECKING:
    from collections.abc import Callable

    from tree_sitter import Node

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedEntity:
    """A code entity extracted from a source file."""

    name: str
    qualified_name: str
    label: NodeLabel
    kind: str
    line_start: int
    line_end: int
    file_path: str
    docstring: str | None = None
    signature: str | None = None
    visibility: str = Visibility.PUBLIC
    tags: list[str] = field(default_factory=list)
    source: str | None = None
    header_path: str | None = None
    header_level: int | None = None
    content_hash: str = ""


@dataclass(frozen=True)
class ParsedRelationship:
    """A relationship between entities, extracted from source."""

    from_qualified_name: str
    rel_type: RelType
    to_name: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedFile:
    """Complete parse result for a single source file."""

    file_path: str
    language: str
    entities: list[ParsedEntity]
    relationships: list[ParsedRelationship]


# ---------------------------------------------------------------------------
# Language config registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LanguageConfig:
    """Configuration for a tree-sitter language."""

    name: str
    extensions: frozenset[str]
    language: Language
    query: Query
    parse_func: Callable[[str, bytes, Node, str], ParsedFile]


_LANGUAGES: dict[str, LanguageConfig] = {}
_EXTENSION_MAP: dict[str, str] = {}


def register_language(config: LanguageConfig) -> None:
    """Register a language configuration."""
    _LANGUAGES[config.name] = config
    for ext in config.extensions:
        _EXTENSION_MAP[ext] = config.name


def get_language_for_file(path: str) -> LanguageConfig | None:
    """Look up language config by file extension.

    Triggers plugin discovery on first call so that built-in and
    external languages are available.
    """
    from code_atlas.parsing.languages import discover_plugins  # noqa: PLC0415

    discover_plugins()

    suffix = PurePosixPath(path).suffix.lower()
    lang_name = _EXTENSION_MAP.get(suffix)
    if lang_name is None:
        return None
    return _LANGUAGES.get(lang_name)


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------


def _compute_content_hash(entity: ParsedEntity) -> str:
    """Compute a deterministic hash of an entity's semantic fields.

    Hashes name, kind, visibility, signature, docstring, and sorted tags.
    Excludes positional fields (line_start/line_end, file_path) so that
    moving code without changing it produces the same hash.
    """
    parts = [
        entity.name,
        entity.kind,
        entity.visibility,
        entity.signature or "",
        entity.docstring or "",
        ",".join(sorted(entity.tags)),
    ]
    data = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def node_text(node: Node) -> str:
    """Get the text content of a tree-sitter node as a string."""
    text = node.text
    if text is None:
        return ""
    return text.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Core parse function
# ---------------------------------------------------------------------------


def parse_file(path: str, source: bytes, project_name: str, *, max_source_chars: int = 2000) -> ParsedFile | None:
    """Parse a source file and extract entities + relationships.

    Returns ParsedFile with entities mapped to schema labels/kinds,
    qualified names built from file path + nesting. Returns None if
    the language is not supported.

    ``max_source_chars`` caps the ``source`` field on each entity.
    Set to 0 to disable source extraction entirely.
    """
    lang_config = get_language_for_file(path)
    if lang_config is None:
        return None

    parser = Parser(lang_config.language)
    tree = parser.parse(source)

    result = lang_config.parse_func(path, source, tree.root_node, project_name)

    # Post-parse pass: compute content hashes and truncate source
    def _finalize(e: ParsedEntity) -> ParsedEntity:
        updates: dict[str, Any] = {"content_hash": _compute_content_hash(e)}
        if max_source_chars > 0 and e.source:
            updates["source"] = e.source[:max_source_chars]
        elif max_source_chars <= 0:
            updates["source"] = None
        return replace(e, **updates)

    return ParsedFile(
        file_path=result.file_path,
        language=result.language,
        entities=[_finalize(e) for e in result.entities],
        relationships=result.relationships,
    )
