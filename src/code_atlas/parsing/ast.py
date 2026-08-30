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
import time
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from loguru import logger
from tree_sitter import Language, Parser, Query

from code_atlas.chunking import repair_fences, split_embed_text
from code_atlas.schema import NodeLabel, RelType, Visibility
from code_atlas.telemetry import get_metrics

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

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
    parse_func: Callable[[str, bytes, Node, str], ParsedFile | None]
    """Handler for a matched file.

    Returning ``None`` means "this handler declines the file" — the framework
    turns that into an *empty* ``ParsedFile`` rather than propagating ``None``.
    See ``parse_file`` for why declining must not look like an unsupported
    language.
    """
    filenames: frozenset[str] = frozenset()
    """Exact basenames claimed by this language, lowercased, e.g. ``{"dockerfile"}``.

    For formats identified by filename rather than extension. Checked before
    ``extensions`` (see ``get_language_for_file``), because such files have no
    usable suffix — ``PurePosixPath("Dockerfile").suffix`` is ``""``, and
    registering ``""`` as an extension would hijack every extensionless file in
    the repo (LICENSE, Makefile, .gitignore, ...).
    """
    comment_node_types: frozenset[str] = frozenset()
    """Tree-sitter node types that hold comments, e.g. ``frozenset({"comment"})``.

    Empty (the default) opts the language out of rationale extraction — see
    ``extract_rationale``. Languages opt in at registration rather than the
    framework guessing, because node-type naming differs per grammar.
    """
    ambiguous_extensions: frozenset[str] = frozenset()
    """Extensions this language is registered for but does not exclusively own.

    ``.h`` is the only one today: it is the standard C header extension and
    also what most C++ projects call their headers, and nothing in the name
    says which. Files with such a suffix have their language decided by
    ``resolve_dialect`` rather than by the suffix alone.
    """
    resolve_dialect: Callable[[bytes], str] | None = None
    """Given a file's bytes, return the registered language name to parse it as.

    Consulted only for suffixes listed in ``ambiguous_extensions``, so a
    language that leaves both fields empty pays nothing. Returning this
    language's own name keeps the registered default — which is what an
    undecidable file must get, because the registered default is the status quo
    and the status quo is the safe answer.
    """


_LANGUAGES: dict[str, LanguageConfig] = {}
_EXTENSION_MAP: dict[str, str] = {}
_FILENAME_MAP: dict[str, str] = {}
_AMBIGUOUS_EXTENSIONS: set[str] = set()
"""Suffixes whose language is decided by content — see ``resolve_dialect``.

Held here rather than read back off the matched config so that lookup for every
other extension is untouched, and costs one set membership test.
"""


def register_language(config: LanguageConfig) -> None:
    """Register a language configuration."""
    _LANGUAGES[config.name] = config
    for ext in config.extensions:
        _EXTENSION_MAP[ext] = config.name
    for filename in config.filenames:
        _FILENAME_MAP[filename] = config.name
    if config.resolve_dialect is not None:
        _AMBIGUOUS_EXTENSIONS.update(config.ambiguous_extensions)


def get_language_for_file(path: str, source: bytes | None = None) -> LanguageConfig | None:
    """Look up language config by exact basename, then by file extension.

    Basename wins so that extensionless formats (``Dockerfile``,
    ``Containerfile``) are reachable at all. It is a *whole-basename* match, so
    ``dockerfile.txt`` does not route to the container language — that file's
    basename is ``dockerfile.txt`` and its suffix is ``.txt``.

    For a suffix two languages share (``.h``), the winner is decided by the
    file's content — see ``LanguageConfig.resolve_dialect``. Pass *source* when
    you have it; callers that do not are read from disk, because the answer must
    not depend on who is asking. A caller that only needs "is this file
    indexable at all" can ignore this entirely: both candidates are non-None.

    Triggers plugin discovery on first call so that built-in and
    external languages are available.
    """
    from code_atlas.parsing.languages import discover_plugins  # noqa: PLC0415

    discover_plugins()

    posix_path = PurePosixPath(path)
    lang_name = _FILENAME_MAP.get(posix_path.name.lower())
    matched_by_name = lang_name is not None
    suffix = posix_path.suffix.lower()
    if lang_name is None:
        lang_name = _EXTENSION_MAP.get(suffix)
    if lang_name is None:
        return None
    config = _LANGUAGES.get(lang_name)
    if matched_by_name or suffix not in _AMBIGUOUS_EXTENSIONS or config is None:
        return config

    if source is None:
        source = _read_for_dialect(path)
        if source is None:
            # Undecidable — keep the registered default rather than guess.
            return config
    return _LANGUAGES.get(config.resolve_dialect(source), config) if config.resolve_dialect else config


def _read_for_dialect(path: str) -> bytes | None:
    """Read a file solely to disambiguate its language. None if unreadable.

    Only reached for a shared suffix whose caller passed no source. ``parse_file``
    always passes one, so this costs nothing on the indexing path.
    """
    try:
        with open(path, "rb") as fh:  # noqa: PTH123
            return fh.read()
    except OSError:
        return None


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


_TYPE_QUALIFIER_RE = re.compile(r"[\w$]+(?:\.|::)")


def normalize_type_text(text: str) -> str:
    """Normalize a type for an overload suffix: strip whitespace and package/namespace qualifiers.

    Shared by every language that lets two definitions share a name in one scope
    (ADR-0032), so ``java.util.List<String>`` and ``std::error_code&`` reduce the
    same way and the suffix reads as source rather than as a linker symbol.
    """
    return _TYPE_QUALIFIER_RE.sub("", "".join(text.split()))


def call_receiver_props(obj: Node | None, local_types: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Properties for a CALLS relationship whose callee was named on a receiver.

    Shared across languages because the distinction it records is not Python-specific:
    ``helper()`` must resolve in lexical scope, while ``client.scan()`` names a member of
    a type that may never have been indexed. Recording the receiver stops the resolver
    treating a project-wide name coincidence as identity (ADR-0022); recording its
    declared type says which implementation is actually called (ADR-0023).

    An empty dict for a bare call keeps the relationship byte-identical to what a
    receiver-less language emits, so nothing downstream has to special-case it.
    """
    if obj is None:
        return {}
    text = node_text(obj)
    if not text:
        return {}
    props: dict[str, Any] = {"receiver": text}
    declared = (local_types or {}).get(text)
    if declared:
        props["receiver_type"] = declared
    return props


def slice_without_comments(node: Node, source: bytes, end_byte: int, comment_types: frozenset[str]) -> str:
    """Source between ``node.start_byte`` and *end_byte*, with comments removed.

    A signature taken as a raw byte slice carries any comment inside it — a real
    ``async def f(  # noqa: PLR0912`` reached the rendered outline this way, putting a
    stray ``#`` in a format that gives ``#`` two other meanings.

    Cutting comments by regex cannot be made correct: ``#`` and ``//`` also occur inside
    string defaults. Excising the byte ranges the grammar itself labelled as comments is
    correct by construction — a hash inside a string literal belongs to a ``string`` node
    and survives, a real comment cannot — and it works for every language that declares
    its comment node types, with no per-language pattern.
    """
    spans: list[tuple[int, int]] = []

    def walk(n: Node) -> None:
        for child in n.children:
            if child.start_byte >= end_byte or child.end_byte <= node.start_byte:
                continue
            if child.type in comment_types:
                spans.append((child.start_byte, child.end_byte))
            else:
                walk(child)

    walk(node)
    if not spans:
        return source[node.start_byte : end_byte].decode("utf-8", errors="replace")

    kept: list[bytes] = []
    cursor = node.start_byte
    for start, stop in sorted(spans):
        kept.append(source[cursor:start])
        cursor = max(cursor, stop)
    kept.append(source[cursor:end_byte])
    return b"".join(kept).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Referenced-file heuristic (REFERENCES_FILE)
#
# Shared by every language that extracts file references, so they cannot drift
# on how permissive the test is. Deliberately strict: a false positive MINTS a
# ResourceFile node for a path that does not exist, and that node then shows up
# in search results and dependency views as if the repo really read it. A miss
# costs nothing but a missing edge, so the bar sits well above "contains a dot".
#
# This function inspects a STRING. It never touches the filesystem — no stat, no
# open, no resolve — which is what lets a reference to `.env` or `certs/key.pem`
# be recorded as a path without the parser ever reading the secret behind it.
# ---------------------------------------------------------------------------

MAX_RESOURCE_PATH_CHARS = 200
"""Upper bound on a referenced path. Anything longer is prose, not a path."""

# Whole-literal character allowlist. Excluding ':' rejects URLs ("s3://b/k.json")
# and Windows drive letters ("C:/x.json") in one rule; excluding whitespace,
# '{}', '%', '*' and '?' rejects prose, format templates and globs.
_RESOURCE_PATH_CHARS_RE = re.compile(r"[A-Za-z0-9_./\\-]+")

# A file extension on the final segment. Together with the separator rule below
# this is what keeps mode strings ("rb", "w+"), bare names and `Path(".")` out.
_RESOURCE_FILE_EXT_RE = re.compile(r"\.[A-Za-z0-9_]{1,10}\Z")


def looks_like_resource_path(literal: str) -> bool:
    """Is *literal* conservatively recognizable as a relative path to a file?

    Accepts ``data/x.json``, ``.env``, ``certs/server.pem``, ``../shared/a.yaml``
    and extensionless files below a directory (``.ssh/id_rsa``). Rejects absolute
    paths, ``~``-relative paths, URLs, globs, format templates, mode strings and
    bare extensionless names — see the module comment above for the bias.

    A directory separator counts as its own evidence, because the caller has
    already established that the literal is the first argument of an open/Path
    call and almost nothing else passed there contains a ``/``. An extensionless
    literal with no separator at all (``open("rb")``, ``Path("data")``) is the
    ambiguous case, and it is refused.
    """
    if not literal or len(literal) > MAX_RESOURCE_PATH_CHARS:
        return False
    if _RESOURCE_PATH_CHARS_RE.fullmatch(literal) is None:
        return False
    normalized = literal.replace("\\", "/")
    if normalized.startswith(("/", "-")) or "//" in normalized or normalized.endswith("/"):
        return False
    last_segment = normalized.rsplit("/", 1)[-1]
    if last_segment in (".", ".."):
        return False
    return "/" in normalized or _RESOURCE_FILE_EXT_RE.search(last_segment) is not None


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
# Pre-parse safety guard
#
# Some tree-sitter grammars die *natively* on pathological input — a C stack /
# scanner-buffer overflow inside ``Parser.parse()`` that takes the whole process
# down (Windows 0xC0000005, POSIX SIGSEGV). There is no Python exception to
# catch and no language handler involved: the kill happens before ``parse_func``
# is ever called, so the only place this can be stopped is here, on the raw
# bytes.
#
# Measured on this repo's pinned grammars (binary search, one subprocess per
# probe; "last ok" / "first crash" nesting depth):
#
#   NATIVE KILL (0xC0000005) inside Parser.parse() — no traceback, process dies:
#     yaml   block map, 1-space indent      253 / 256
#     yaml   block map, 2-space indent      253 / 256
#     yaml   block seq inline "- - - x"     253 / 256
#     yaml   block seq, newline+indent      250 / 253   <-- lowest observed
#     md     nested list (indented)         253 / 256
#     md     blockquote "> > > x"           253 / 256
#     md     blockquote ">>>x"              253 / 256
#     md     ordered list "1. 1. 1. x"      253 / 256
#     md     list inline "- - -" / "* * *"  253 / 256
#
#   RECOVERABLE RecursionError raised by the Python handler *after* parse()
#   returned fine — caught in parse_file, see the try block there:
#     yaml flow map/seq 490/493 · bash $( ) 496/500 · bash subshell + if 990/993
#     python indentation 496/500 · rust parens 990/993 · json 9590/9593 (object)
#     and 11321/11325 (array)
#
#   No failure up to depth 25600: xml, toml (array + inline table), hcl (list +
#   block), sql (parens + subqueries), python/ts/go/cpp/java/ruby/php parens,
#   markdown emphasis/link/inline-html.
#
# The uniform 256 for every yaml/markdown *block* construct is the external
# scanner's fixed serialization buffer overflowing, not C recursion — which is
# why *block* depth, not bracket depth or file size, is the thing to bound here.
# Bracket nesting never killed the process at any depth up to 25600: tree-sitter's
# core parser is iterative, so that whole family only ever surfaces as a
# RecursionError from a handler's own walk, and is handled by catching it in
# parse_file rather than by a byte heuristic (which would have to reject
# legitimate files — minified JS in site-packages measures bracket depth 270).
# ---------------------------------------------------------------------------

MAX_BLOCK_DEPTH = 64
"""Refuse input whose estimated indentation/marker block nesting reaches this.

The estimate counts *levels*, and one level can open two grammar blocks (a YAML
block sequence plus the mapping inside it), so the real tree depth at the
refusal point is at most ~128 — still half the 250 that kills the process.
Measured the other way: over 8017 real files in this repo plus its
site-packages, the deepest estimate is 5.

Deliberately not configurable — raising it re-arms a process kill.
"""

DEFAULT_MAX_PARSE_BYTES = 1_048_576

# Cap on the ``source`` field, applied in the post-parse pass. This is also the ceiling
# on what an embedding can ever see, because build_embed_text reads the stored (already
# truncated) value -- at the old 2000 a code entity could not exceed ~500 tokens, so the
# EmbedChunk overflow path (ADR-0040) was unreachable and no chunk existed in any graph.
# Mirrored by IndexSettings.max_source_chars, asserted equal by a unit test.
DEFAULT_MAX_SOURCE_CHARS = 48_000
"""Default ceiling on file size handed to tree-sitter. Mirrored by
``IndexSettings.max_parse_bytes`` — see that field for the timing curve."""

# Cheap C-speed pre-filter for _block_depth. A line's leading run of block
# markers — indentation columns, blockquote '>', list bullets "- "/"* "/"+ ",
# ordered-list "1. "/"1) " — must be at least MAX_BLOCK_DEPTH - 1 units long
# before the exact scan can possibly reach the limit: levels have strictly
# increasing indentation, so the deepest line carries one column per enclosing
# level, plus its own markers. Real files almost never trip it (9 of 8017), so
# the Python scan below effectively never runs.
_PREFIX_UNIT = r"(?:[ \t>]|[-*+][ \t]|[0-9]{1,9}[.)][ \t])"

# ``(?=\S)`` skips lines that are only padding whitespace — they open no block.
# "> " * 300 is still caught: the engine backtracks one unit and the final ">"
# satisfies the lookahead.
_DEEP_PREFIX_RE = re.compile(("(?m)^" + _PREFIX_UNIT + "{" + str(MAX_BLOCK_DEPTH - 1) + ",}(?=\\S)").encode())

_SPACE_TAB = (0x20, 0x09)
_BULLETS = (0x2D, 0x2A, 0x2B)  # - * +
_ORDERED_SEPS = (0x2E, 0x29)  # . )


def _prefix_shape(line: bytes) -> tuple[int, int]:
    """Measure a line's leading block-marker run: ``(columns, marker count)``.

    Stops at the first byte that cannot open a block, so ``"  - foo"`` is
    ``(4, 1)`` and ``"----------"`` (a horizontal rule, no space after the
    dash) is ``(0, 0)``.
    """
    i = 0
    markers = 0
    n = len(line)
    while i < n:
        char = line[i]
        if char in _SPACE_TAB:
            i += 1
        elif char == 0x3E:  # '>' blockquote
            i += 1
            markers += 1
        elif char in _BULLETS and i + 1 < n and line[i + 1] in _SPACE_TAB:
            i += 2
            markers += 1
        elif 0x30 <= char <= 0x39:  # ordered list "12. "
            end = i
            while end < n and 0x30 <= line[end] <= 0x39:
                end += 1
            if end + 1 < n and line[end] in _ORDERED_SEPS and line[end + 1] in _SPACE_TAB:
                i = end + 2
                markers += 1
            else:
                break
        else:
            break
    return i, markers


def _block_depth(source: bytes) -> int:
    """Estimate the deepest indentation/marker block nesting in *source*.

    Offside-rule accounting: a line's indentation closes every enclosing level
    at or beyond its own column and opens one of its own, and each marker on
    the line opens one more (``"> > > x"`` nests three blockquotes on a single
    line). Counting *levels* rather than columns is what keeps an aligned
    continuation line or a wide ASCII diagram — one deep line with no staircase
    under it — from reading as deep nesting.
    """
    deepest = 0
    levels: list[int] = []
    for line in source.splitlines():
        columns, markers = _prefix_shape(line)
        if markers == 0 and columns == len(line):
            # Whitespace-only: opens nothing. The `markers == 0` half is
            # load-bearing — `_prefix_shape` folds marker bytes into `columns`,
            # so a line that is ENTIRELY markers (`">" * 400`, `"- " * 400`)
            # also satisfies `columns == len(line)` while opening one block per
            # marker. Skipping those scored them 0 and handed the exact inputs
            # this guard exists to stop straight to a native process kill.
            continue
        while levels and columns <= levels[-1]:
            levels.pop()
        levels.append(columns)
        deepest = max(deepest, len(levels) + markers)
    return deepest


def _parse_hazard(source: bytes, max_parse_bytes: int) -> str | None:
    """Return why *source* must not be handed to tree-sitter, or ``None`` if it is safe."""
    if 0 < max_parse_bytes < len(source):
        return f"{len(source)} bytes exceeds max_parse_bytes={max_parse_bytes}"
    if _DEEP_PREFIX_RE.search(source) is not None:
        depth = _block_depth(source)
        if depth >= MAX_BLOCK_DEPTH:
            return f"block nesting depth {depth} reaches the limit of {MAX_BLOCK_DEPTH}"
    return None


# ---------------------------------------------------------------------------
# Core parse function
# ---------------------------------------------------------------------------


def elide_nested_entity_spans(entities: list[ParsedEntity], reference: Callable[[ParsedEntity], str]) -> None:
    """Replace, in each entity's ``source``, the span of any entity nested inside it.

    A nested definition is its own node with its own indexed text; carrying it whole in
    its parent's source indexes the same bytes twice. *reference* renders the line that
    stands in for it, so the structure an agent reads survives the removal.

    ONE implementation on purpose. The index arithmetic here is subtle enough to have
    shipped broken once: replacements are applied highest-line-first so an earlier one
    cannot shift a later index, and that argument holds **only for spans that do not
    overlap**. Eliding a grandchild as well as its parent breaks it -- the grandchild
    goes first, the line list shrinks, and the parent's now-stale slice eats the code
    below the nested definition. Hence ``_outermost``: a grandchild's text leaves anyway
    when its own parent's span is replaced.
    """
    position = {id(e): i for i, e in enumerate(entities)}
    for entity in list(entities):
        if not entity.source:
            continue
        lines = entity.source.splitlines()
        if not lines:
            continue
        spans = []
        for child in _outermost(_nested_within(entity, entities)):
            start = child.line_start - entity.line_start
            end = child.line_end - entity.line_start
            if 0 <= start <= end < len(lines):
                indent = " " * (len(lines[start]) - len(lines[start].lstrip()))
                spans.append((start, end, indent + reference(child)))
        if not spans:
            continue
        for start, end, replacement in sorted(spans, reverse=True):
            lines[start : end + 1] = [replacement]
        entities[position[id(entity)]] = replace(entity, source="\n".join(lines))


def _nested_within(entity: ParsedEntity, entities: list[ParsedEntity]) -> list[ParsedEntity]:
    """Entities inside *entity* that carry indexed text of their own.

    Containment is by qualified_name AND line range: the name alone would catch a
    sibling sharing a prefix, and the range alone would catch an unrelated entity in a
    file where two spans overlap.
    """
    prefix = entity.qualified_name + "."
    return [
        child
        for child in entities
        if child is not entity
        and child.qualified_name.startswith(prefix)
        and entity.line_start <= child.line_start
        and child.line_end <= entity.line_end
        and (child.docstring or child.source)
    ]


def _outermost(spans: list[ParsedEntity]) -> list[ParsedEntity]:
    """Drop any span contained in another. See elide_nested_entity_spans for why."""
    return [
        span
        for span in spans
        if not any(
            other is not span and other.line_start <= span.line_start and span.line_end <= other.line_end
            for other in spans
        )
    ]


# ---------------------------------------------------------------------------
# Oversized doc-section splitting
# ---------------------------------------------------------------------------

DEFAULT_MAX_DOC_SECTION_CHARS: int = 6000
"""Body size past which a DocSection becomes several nodes.

Headings are a markdown file's natural borders, and a file without them -- a
transcript, an export, a changelog dump -- produces one node holding the entire
document. Measured on one corpus: 51.5% of the DocSections from a YouTube-archive
directory exceeded the embedding model's input cap, against 0.8% of ordinary docs.

Splitting rather than multi-chunk embedding (which is what code entities get) is
deliberate, and it is the retrieval argument rather than the provider one: a
Callable's boundary is meaningful and worth keeping whole, while half a
header-less transcript is not a unit anyone wants returned. ``Note`` nodes are
excluded for the same reason read the other way -- a note's uid is an address that
``LINKS_TO`` edges point at, so splitting one would orphan the wikilink graph.

~6000 characters is roughly 1500-2000 tokens, under every current model's cap.
Set to 0 to disable splitting entirely.
"""

_MAX_DOC_PARTS: int = 200
"""Parts one section may become. Content past it is not indexed.

Only a pathology reaches this: at the default budget it takes a 1.2 MB section.
"""

_DOC_PART_SUFFIX: str = "#part"
"""Distinct from ``_dedupe_section_qn``'s ``#N``, which disambiguates duplicate
sibling headings — that one emits digits only, so the two can never collide."""


def split_oversized_doc_sections(
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    *,
    max_chars: int,
) -> tuple[list[ParsedEntity], list[ParsedRelationship]]:
    """Split DocSections whose body exceeds *max_chars* into consecutive parts.

    Part 1 keeps the original ``qualified_name``, so an unsplit section and the
    head of a split one address the same node as before and every relationship
    already pointing at it stays valid. Parts 2..N take a ``#partN`` suffix and get
    a copy of the CONTAINS edge that held part 1, so the DocFile still contains all
    of them.

    ``DOCUMENTS`` edges are left on part 1 rather than re-derived per part. They
    were extracted from the whole body, and re-running that extraction here would
    make the parser's reference detection depend on where an unrelated size budget
    happened to cut.

    Line numbers for parts 2..N are counted from the body's own newlines, so they
    inherit the offset between the heading line and where the body starts -- one or
    two lines low, not enough to matter for a citation and not worth re-deriving
    byte offsets through a strip() to fix.
    """
    if max_chars <= 0:
        return entities, relationships

    oversized = {
        id(e) for e in entities if e.label is NodeLabel.DOC_SECTION and e.docstring and len(e.docstring) > max_chars
    }
    if not oversized:
        return entities, relationships

    contains_by_target: dict[str, list[ParsedRelationship]] = {}
    for rel in relationships:
        if rel.rel_type is RelType.CONTAINS:
            contains_by_target.setdefault(rel.to_name, []).append(rel)

    out_entities: list[ParsedEntity] = []
    extra_rels: list[ParsedRelationship] = []
    for entity in entities:
        if id(entity) not in oversized:
            out_entities.append(entity)
            continue

        body = entity.docstring or ""
        split = split_embed_text(body, limit=max_chars, measure=len, max_chunks=_MAX_DOC_PARTS)
        # Fences second: the ladder cuts on blank lines, which occur inside a fenced
        # block, so a long example otherwise yields parts of unlabelled bare code.
        parts = repair_fences(split.chunks)
        if split.dropped:
            logger.warning(
                "Doc section {} is too large to index whole: {} of {} characters past part {} were dropped",
                entity.qualified_name,
                split.dropped,
                len(body),
                _MAX_DOC_PARTS,
            )
        if len(parts) <= 1:
            out_entities.append(entity)
            continue

        offset = 0
        for number, part in enumerate(parts, start=1):
            found = body.find(part, offset)
            start_off = found if found >= 0 else offset
            offset = start_off + len(part)
            line_start = entity.line_start + body.count("\n", 0, start_off)
            line_end = max(line_start, min(line_start + part.count("\n"), entity.line_end))
            if number == 1:
                name, qualified_name = entity.name, entity.qualified_name
            else:
                name = f"{entity.name} (part {number})"
                qualified_name = f"{entity.qualified_name}{_DOC_PART_SUFFIX}{number}"
                extra_rels.extend(
                    replace(rel, to_name=qualified_name) for rel in contains_by_target.get(entity.qualified_name, ())
                )
            out_entities.append(
                replace(
                    entity,
                    name=name,
                    qualified_name=qualified_name,
                    docstring=part,
                    line_start=line_start,
                    line_end=line_end,
                )
            )

        logger.debug(
            "Split oversized doc section {} ({} chars) into {} parts",
            entity.qualified_name,
            len(body),
            len(parts),
        )

    return out_entities, [*relationships, *extra_rels]


def parse_file(
    path: str,
    source: bytes,
    project_name: str,
    *,
    max_source_chars: int = DEFAULT_MAX_SOURCE_CHARS,
    max_parse_bytes: int = DEFAULT_MAX_PARSE_BYTES,
    max_doc_section_chars: int = DEFAULT_MAX_DOC_SECTION_CHARS,
    rationale: RationaleSettings | None = None,
) -> ParsedFile | None:
    """Parse a source file and extract entities + relationships.

    Returns ParsedFile with entities mapped to schema labels/kinds,
    qualified names built from file path + nesting. Returns None when the file
    cannot be parsed at all — either no language is registered for it, or the
    pre-parse guard refused it (see ``_parse_hazard``).

    ``max_source_chars`` caps the ``source`` field on each entity.
    Set to 0 to disable source extraction entirely.

    ``max_parse_bytes`` caps the file size handed to tree-sitter; 0 disables the
    ceiling. Wired from ``IndexSettings.max_parse_bytes``.

    ``max_doc_section_chars`` splits a doc section whose body exceeds it into
    consecutive nodes; 0 disables splitting. See
    :func:`split_oversized_doc_sections`.

    ``rationale`` configures intent-comment extraction; ``None`` uses the
    shipped defaults (NOTE/WHY/HACK plus ADR/RFC citations, no TODO/FIXME).
    """
    lang_config = get_language_for_file(path, source)
    if lang_config is None:
        return None

    hazard = _parse_hazard(source, max_parse_bytes)
    if hazard is not None:
        # None, not an empty ParsedFile: unlike a handler declining a dialect,
        # nothing here was parsed, so this file's existing graph entities must
        # not be diffed away. The re-parse-every-pass cost the empty-ParsedFile
        # path exists to avoid is nil in this case — the guard is a linear byte
        # scan over a source the hash gate has already read.
        logger.warning("parse: refusing {} — {}", path, hazard)
        return None

    # Measured, not spanned. This runs once per file -- a full index is ~60k calls --
    # so a span apiece would bury the batch-level trace it belongs to under its own
    # children. The batch gets one `ast.parse` span in the consumer; this level answers
    # the aggregate question instead: which language costs what, and is it the grammar
    # or our handler. Tree-sitter's error recovery is superlinear (an unparseable 4 MiB
    # T-SQL dump measured four minutes), so the grammar/handler split is not academic.
    parser = Parser(lang_config.language)
    _t0 = time.perf_counter()
    tree = parser.parse(source)
    _lang_attrs = {"language": lang_config.name}
    get_metrics().stage_seconds.record(time.perf_counter() - _t0, {"stage": "parse", "phase": "tree_sitter"})

    _t1 = time.perf_counter()
    try:
        result = lang_config.parse_func(path, source, tree.root_node, project_name)
    except RecursionError:
        # Handlers walk the tree recursively, so nesting the byte guard cannot
        # see — keyword blocks ("if/then/fi" 993 deep), bracket chains (490+) —
        # exhausts the interpreter stack instead of the scanner buffer. Left
        # uncaught this is a poison pill: the AST consumer's batch handler
        # catches it, logs "batch failed, will retry", and retries the same file
        # forever. Same clean skip as the pre-parse refusal.
        logger.warning("parse: refusing {} — handler recursion limit exceeded", path)
        return None
    finally:
        get_metrics().stage_seconds.record(time.perf_counter() - _t1, {"stage": "parse", "phase": "handler"})

    if result is None:
        # A handler declining a file (e.g. the config parser meeting a generic
        # YAML blob it has no dialect for) must NOT surface as "unsupported
        # language": the AST consumer only records a file hash for files that
        # produced a ParsedFile (consumers.py step 6), so returning None from
        # here would leave the file permanently outside the hash gate and force
        # a re-read + re-parse on every single indexing pass. An empty
        # ParsedFile creates no nodes and no embeddings, but does get hashed —
        # so the cost of a declined file amortises to zero after the first pass.
        result = ParsedFile(file_path=path, language=lang_config.name, entities=[], relationships=[])

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

    # Before hashing: a part is a node in its own right, so each needs its own
    # content_hash, and the CONTAINS edges the split adds must reach the returned list.
    relationships = result.relationships
    entities, relationships = split_oversized_doc_sections(entities, relationships, max_chars=max_doc_section_chars)

    # Post-parse pass: compute content hashes and truncate source
    def _finalize(e: ParsedEntity) -> ParsedEntity:
        updates: dict[str, Any] = {"content_hash": _compute_content_hash(e)}
        if max_source_chars > 0 and e.source:
            updates["source"] = e.source[:max_source_chars]
        elif max_source_chars <= 0:
            updates["source"] = None
        return replace(e, **updates)

    metrics = get_metrics()
    metrics.parse_seconds.record(time.perf_counter() - _t0, _lang_attrs)
    metrics.parse_bytes.record(len(source), _lang_attrs)

    return ParsedFile(
        file_path=result.file_path,
        language=result.language,
        entities=[_finalize(e) for e in entities],
        relationships=relationships,
    )
