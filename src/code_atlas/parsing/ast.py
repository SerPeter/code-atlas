"""Tree-sitter based parser for extracting code entities and relationships.

Parses source files using py-tree-sitter and extracts entities (classes,
functions, methods, imports, variables) and relationships (DEFINES, CALLS,
IMPORTS, INHERITS) for graph ingestion.

Language-specific parsers live in ``parsing.languages.*`` and register via
``register_language()`` at import time.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from tree_sitter import Language, Parser, Query

from code_atlas.schema import NodeLabel, RelType, Visibility

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tree_sitter import Node

    from code_atlas.settings import RationaleSettings

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
    extra_properties: dict[str, Any] = field(default_factory=dict)
    rationale: str | None = None
    citations: list[str] = field(default_factory=list)


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
    comment_node_types: frozenset[str] = frozenset()
    """Tree-sitter node types that hold comments, e.g. ``frozenset({"comment"})``.

    Empty (the default) opts the language out of rationale extraction — see
    ``extract_rationale``. Languages opt in at registration rather than the
    framework guessing, because node-type naming differs per grammar.
    """


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

    Hashes name, kind, visibility, signature, docstring, sorted tags, and
    source (the full entity text, hashed before truncation — see _finalize).
    Excludes positional fields (line_start/line_end, file_path) so that
    moving code without changing it produces the same hash.

    ``extra_properties`` (frontmatter, currently Note-only) is folded in only
    when non-empty, so every pre-existing entity kind — which never
    populates it — keeps a byte-identical hash input and no spurious
    re-embed/re-diff is triggered by this field's addition.

    ``rationale``/``citations`` go one step further: they are *appended* only
    when set, rather than contributing an empty element. An extra ``""`` in the
    list would still add a ``\\0`` separator and change every existing hash, so
    entities with no intent-bearing comments must produce a parts list of
    exactly the eight elements above.
    """
    parts = [
        entity.name,
        entity.kind,
        entity.visibility,
        entity.signature or "",
        entity.docstring or "",
        ",".join(sorted(entity.tags)),
        entity.source or "",
        json.dumps(entity.extra_properties, sort_keys=True, default=str) if entity.extra_properties else "",
    ]
    if entity.rationale:
        parts.append(f"rationale:{entity.rationale}")
    if entity.citations:
        parts.append("citations:" + ",".join(sorted(entity.citations)))
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
# Rationale extraction (intent-bearing comments)
# ---------------------------------------------------------------------------

DEFAULT_RATIONALE_MARKERS: tuple[str, ...] = ("NOTE", "WHY", "HACK")
"""Intent-bearing comment markers extracted by default."""

DEFAULT_TASK_MARKERS: tuple[str, ...] = ("TODO", "FIXME")
"""Work-tracking markers. Off by default — far higher volume and much shorter
lived than rationale, so they get their own toggle."""

DEFAULT_CITATION_SCHEMES: tuple[str, ...] = ("ADR", "RFC")
"""Document-reference schemes recorded verbatim from comments. Recording only —
resolving ``ADR-0014`` to a wiki node is deliberately out of scope."""

MAX_RATIONALE_CHARS = 2000
"""Cap on the joined rationale text per entity, mirroring ``max_source_chars``.
A module entity in a comment-heavy file would otherwise accumulate unbounded."""

# Longest first: "///" and "//!" must win over "//".
_LINE_COMMENT_PREFIXES = ("///", "//!", "//", "#", "*")

# Uppercase marker, optional "(owner)", colon. Uppercase-only is deliberate:
# matching "Note:" would pull in ordinary prose.
_MARKER_RE = re.compile(r"^(?P<marker>[A-Z][A-Z0-9_]{1,15})(?:\([^)]*\))?[ \t]*:[ \t]*(?P<text>.*)$")

# Characters that may legitimately sit between a comment block and the
# declaration it annotates: decorators (@), attributes (#[ / [), more comments.
_GAP_PREFIXES = ("@", "#", "[", "/", "*")


@dataclass(frozen=True)
class _CommentBlock:
    """A run of adjacent, equally-indented comment nodes, with syntax stripped."""

    start_line: int
    end_line: int
    lines: tuple[str, ...]


def _clean_comment_lines(text: str) -> list[str]:
    """Strip comment syntax from a raw comment node, one entry per physical line."""
    body = text.strip()
    if body.startswith("/*"):
        body = body[2:]
        body = body.removesuffix("*/")
    cleaned: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        for prefix in _LINE_COMMENT_PREFIXES:
            if line.startswith(prefix):
                line = line[len(prefix) :]
                break
        cleaned.append(line.strip())
    return cleaned


def _collect_comment_blocks(source: bytes, root: Node, comment_types: frozenset[str]) -> list[_CommentBlock]:
    """Walk the tree for comment nodes and group adjacent ones into blocks.

    Two comments join the same block when the second starts on the line right
    after the first ends *and* at the same column, which is what a wrapped
    ``# NOTE: ...`` / ``# continued`` pair looks like. A trailing comment on a
    code line has a different column and stays its own block.
    """
    nodes: list[Node] = []
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        if node.type in comment_types:
            nodes.append(node)
            continue
        stack.extend(node.children)
    nodes.sort(key=lambda n: (n.start_point[0], n.start_point[1]))

    blocks: list[_CommentBlock] = []
    start_line = end_line = column = -1
    lines: list[str] = []
    for node in nodes:
        text = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        node_start = node.start_point[0] + 1
        node_end = node.end_point[0] + 1
        contiguous = lines and node_start == end_line + 1 and node.start_point[1] == column
        if not contiguous:
            if lines:
                blocks.append(_CommentBlock(start_line, end_line, tuple(lines)))
            lines = []
            start_line = node_start
            column = node.start_point[1]
        lines.extend(_clean_comment_lines(text))
        end_line = node_end
    if lines:
        blocks.append(_CommentBlock(start_line, end_line, tuple(lines)))
    return blocks


def _marker_entries(lines: Sequence[str], markers: frozenset[str]) -> list[str]:
    """Extract ``MARKER: text`` entries from a block, folding in wrapped lines.

    A blank line, or a differently-marked line, closes the current entry — so a
    ``TODO:`` right after a ``NOTE:`` never gets absorbed into the note when
    task markers are disabled.
    """
    entries: list[str] = []
    current: list[str] | None = None

    def flush() -> None:
        nonlocal current
        if current:
            entries.append(" ".join(current))
        current = None

    for line in lines:
        match = _MARKER_RE.match(line)
        if match is not None:
            flush()
            marker = match.group("marker")
            if marker in markers:
                current = [f"{marker}: {match.group('text').strip()}".rstrip()]
        elif not line:
            flush()
        elif current is not None:
            current.append(line)
    flush()
    return entries


def _find_citations(lines: Sequence[str], pattern: re.Pattern[str], schemes: dict[str, str]) -> list[str]:
    """Collect ``ADR-0014`` / ``RFC 7231`` style references from a comment block."""
    found: list[str] = []
    for line in lines:
        found.extend(f"{schemes[match.group(1).upper()]}-{match.group(2)}" for match in pattern.finditer(line))
    return found


def _citation_pattern(schemes: Sequence[str]) -> re.Pattern[str]:
    alternation = "|".join(re.escape(s) for s in schemes)
    return re.compile(rf"\b({alternation})[ \t\-_#]?(\d{{1,6}})\b", re.IGNORECASE)


def _gap_is_clear(text_lines: Sequence[str], after_line: int, before_line: int) -> bool:
    """True when nothing but blanks, decorators/attributes or comments separate two lines."""
    for line_no in range(after_line + 1, before_line):
        if line_no - 1 >= len(text_lines):
            break
        stripped = text_lines[line_no - 1].strip()
        if stripped and not stripped.startswith(_GAP_PREFIXES):
            return False
    return True


def _attribute_block(
    block: _CommentBlock,
    spans: Sequence[tuple[int, int, int]],
    text_lines: Sequence[str],
) -> int | None:
    """Pick the entity a comment block belongs to; returns an index into *spans*' entities.

    Preceding-declaration wins over enclosing-body: a ``# WHY:`` sitting above a
    method inside a class belongs to the method, not the class. The following
    declaration is only accepted when it is *nested inside* the innermost
    enclosing entity, which stops a trailing comment at the end of a function
    body from being claimed by the next top-level function.
    """
    containing = None
    inside = [s for s in spans if s[0] <= block.start_line <= s[1]]
    if inside:
        containing = min(inside, key=lambda s: (s[1] - s[0], -s[0], s[2]))

    limit = containing[1] if containing is not None else len(text_lines)
    after = [s for s in spans if block.end_line < s[0] <= limit]
    if after:
        following = min(after, key=lambda s: (s[0], s[1] - s[0], s[2]))
        if _gap_is_clear(text_lines, block.end_line, following[0]):
            return following[2]
    return containing[2] if containing is not None else None


def extract_rationale(
    source: bytes,
    root: Node,
    entities: list[ParsedEntity],
    *,
    comment_types: frozenset[str],
    markers: Sequence[str] = DEFAULT_RATIONALE_MARKERS,
    citation_schemes: Sequence[str] = DEFAULT_CITATION_SCHEMES,
) -> list[ParsedEntity]:
    """Attach intent-bearing comments to the smallest enclosing entity.

    Returns a new entity list; entities with no matching comment are returned
    unchanged (identity-equal), which is what keeps their ``content_hash``
    byte-identical to a pre-rationale index.
    """
    # Citation schemes match case-insensitively downstream (_citation_pattern uses
    # re.IGNORECASE), so this fast-path probe must too — a lowercase "see adr-0014"
    # would otherwise be dropped here before the matcher ever sees it. Markers stay
    # case-sensitive to match _marker_set's uppercase-only convention.
    lowered = source.lower()
    probes = [(m.encode("utf-8"), source) for m in markers]
    probes += [(s.lower().encode("utf-8"), lowered) for s in citation_schemes]
    if not any(probe in haystack for probe, haystack in probes):
        # Overwhelmingly the common case — skip the extra tree walk entirely.
        return entities

    blocks = _collect_comment_blocks(source, root, comment_types)
    if not blocks:
        return entities

    marker_set = frozenset(markers)
    pattern = _citation_pattern(citation_schemes) if citation_schemes else None
    scheme_map = {s.upper(): s for s in citation_schemes}
    text_lines = source.decode("utf-8", errors="replace").splitlines()
    spans = [(e.line_start, e.line_end, i) for i, e in enumerate(entities)]

    rationale_by_entity: dict[int, list[str]] = {}
    citations_by_entity: dict[int, list[str]] = {}
    for block in blocks:
        entries = _marker_entries(block.lines, marker_set) if marker_set else []
        refs = _find_citations(block.lines, pattern, scheme_map) if pattern is not None else []
        if not entries and not refs:
            continue
        target = _attribute_block(block, spans, text_lines)
        if target is None:
            continue
        if entries:
            rationale_by_entity.setdefault(target, []).extend(entries)
        if refs:
            citations_by_entity.setdefault(target, []).extend(refs)

    if not rationale_by_entity and not citations_by_entity:
        return entities

    updated = list(entities)
    for index in sorted(rationale_by_entity.keys() | citations_by_entity.keys()):
        text = "\n".join(rationale_by_entity.get(index, []))[:MAX_RATIONALE_CHARS]
        updated[index] = replace(
            updated[index],
            rationale=text or None,
            citations=sorted(set(citations_by_entity.get(index, []))),
        )
    return updated


def _resolve_rationale_markers(rationale: RationaleSettings | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Flatten rationale settings into ``(markers, citation_schemes)``."""
    if rationale is None:
        return DEFAULT_RATIONALE_MARKERS, DEFAULT_CITATION_SCHEMES
    if not rationale.enabled:
        return (), ()
    markers = tuple(rationale.markers)
    if rationale.tasks:
        markers += tuple(rationale.task_markers)
    return markers, tuple(rationale.citation_schemes) if rationale.citations else ()


# ---------------------------------------------------------------------------
# Core parse function
# ---------------------------------------------------------------------------


def parse_file(
    path: str,
    source: bytes,
    project_name: str,
    *,
    max_source_chars: int = 2000,
    rationale: RationaleSettings | None = None,
) -> ParsedFile | None:
    """Parse a source file and extract entities + relationships.

    Returns ParsedFile with entities mapped to schema labels/kinds,
    qualified names built from file path + nesting. Returns None if
    the language is not supported.

    ``max_source_chars`` caps the ``source`` field on each entity.
    Set to 0 to disable source extraction entirely.

    ``rationale`` configures intent-comment extraction; ``None`` uses the
    shipped defaults (NOTE/WHY/HACK plus ADR/RFC citations, no TODO/FIXME).
    """
    lang_config = get_language_for_file(path)
    if lang_config is None:
        return None

    parser = Parser(lang_config.language)
    tree = parser.parse(source)

    result = lang_config.parse_func(path, source, tree.root_node, project_name)

    entities = result.entities
    if lang_config.comment_node_types:
        markers, citation_schemes = _resolve_rationale_markers(rationale)
        if markers or citation_schemes:
            entities = extract_rationale(
                source,
                tree.root_node,
                entities,
                comment_types=lang_config.comment_node_types,
                markers=markers,
                citation_schemes=citation_schemes,
            )

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
        entities=[_finalize(e) for e in entities],
        relationships=result.relationships,
    )
