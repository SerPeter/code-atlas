"""Markdown language support — tree-sitter parser for documentation files."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import tree_sitter_markdown as tsmarkdown
import yaml
from tree_sitter import Language, Query

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    register_language,
)
from code_atlas.schema import NodeLabel, NoteKind, RelType

if TYPE_CHECKING:
    from tree_sitter import Node

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language registration
# ---------------------------------------------------------------------------

_MD_LANGUAGE = Language(tsmarkdown.language())
_MD_QUERY = Query(_MD_LANGUAGE, "(document) @doc")

# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------

_ATX_LEVEL: dict[str, int] = {
    "atx_h1_marker": 1,
    "atx_h2_marker": 2,
    "atx_h3_marker": 3,
    "atx_h4_marker": 4,
    "atx_h5_marker": 5,
    "atx_h6_marker": 6,
}

_SETEXT_LEVEL: dict[str, int] = {
    "setext_h1_underline": 1,
    "setext_h2_underline": 2,
}

# ---------------------------------------------------------------------------
# Frontmatter + note-mode detection
# ---------------------------------------------------------------------------

# tree-sitter-markdown emits YAML frontmatter (--- ... ---) as a dedicated
# top-level node; TOML frontmatter (+++ ... +++) is intentionally unsupported
# — the vault convention is YAML only.
_FRONTMATTER_NODE_TYPE = "minus_metadata"

_NOTE_KIND_VALUES = frozenset(k.value for k in NoteKind)

_WIKILINK_RE = re.compile(r"\[\[([^\[\]|]+?)(?:\|([^\[\]]+?))?\]\]")
_INLINE_TAG_RE = re.compile(r"(?<!\w)#([a-zA-Z][\w-]*)")
_ATX_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}(?:\s|$)")
_ATX_H1_RE = re.compile(r"^#[ \t]+(.+?)\s*$", re.MULTILINE)

# Absolute path: POSIX (leading /) or Windows (drive letter + :/ or :\).
_ABS_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\])")
# Known file extensions (mirrors _FILE_REF_RE's set below) — a bare filename
# anchor (e.g. "foo.py") is path-like even without a "/"; a generic
# alphanumeric-suffix check would wrongly match a qualified name's last
# segment (e.g. "...foo.Bar").
_FILE_EXT_RE = re.compile(r"\.(?:py|pyi|js|ts|jsx|tsx|java|go|rs|rb|cpp|c|h|hpp|md)$")
# Short/ambiguous extensions that collide with a plausible dotted symbol's
# final segment (e.g. "models.Config.c", "physics.Model.h") — trusted as a
# path signal only when the anchor also contains a "/" (e.g. "src/foo.c").
_AMBIGUOUS_SHORT_EXTS = frozenset({"c", "h", "go", "rs", "rb"})
# Conventional extensionless filenames — no "/" and no recognized extension,
# so they'd otherwise fall through to uid-form classification.
_CONVENTIONAL_FILENAMES = frozenset(
    {
        "Dockerfile",
        "Makefile",
        "Rakefile",
        "Gemfile",
        "Procfile",
        "Vagrantfile",
        "Jenkinsfile",
        "LICENSE",
        "README",
        "CHANGELOG",
        "CONTRIBUTING",
    }
)


@dataclass(frozen=True)
class _Frontmatter:
    """Parsed YAML frontmatter plus the byte offset where the body begins."""

    raw: dict[str, Any]
    body_start: int


def _extract_frontmatter(root: Node, source: bytes) -> _Frontmatter | None:
    """Extract and parse a leading YAML frontmatter block, if present.

    Returns ``None`` for files with no frontmatter, non-YAML-mapping
    frontmatter, or malformed YAML — callers fall back to ordinary
    DocFile/DocSection parsing in every such case (no regression for
    existing docs, which never carry frontmatter today).
    """
    for child in root.children:
        if child.type != _FRONTMATTER_NODE_TYPE:
            continue
        raw_text = source[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
        lines = raw_text.splitlines()
        # Strip the delimiter lines (--- ... ---) themselves before parsing.
        yaml_lines = lines[1:-1] if len(lines) >= 2 and lines[-1].strip() == "---" else lines[1:]
        try:
            parsed = yaml.safe_load("\n".join(yaml_lines))
        except yaml.YAMLError:
            return None
        if not isinstance(parsed, dict):
            return None
        return _Frontmatter(raw=parsed, body_start=child.end_byte)
    return None


def _note_mode_kind(fm: dict[str, Any]) -> tuple[str, str] | None:
    """Return ``(note_kind, subtype)`` if frontmatter triggers note mode, else ``None``.

    Two dialects trigger note mode: the atlas vault (``id`` + ``kind`` in
    draft|note|decision) and the Claude Code harness memory format (``name``
    + ``description`` + ``metadata.type``).
    """
    kind = fm.get("kind")
    if isinstance(kind, str) and kind in _NOTE_KIND_VALUES and isinstance(fm.get("id"), str) and fm["id"].strip():
        return kind, ""

    if isinstance(fm.get("name"), str) and fm["name"].strip() and isinstance(fm.get("description"), str):
        metadata = fm.get("metadata")
        note_type = metadata.get("type") if isinstance(metadata, dict) else None
        if isinstance(note_type, str) and note_type.strip():
            return NoteKind.NOTE.value, note_type.strip()
    return None


def _note_slug(fm: dict[str, Any], path: str) -> str:
    """Derive the note's stable slug — frontmatter ``id``/``name``, else filename stem.

    The vault convention is filename == slug == frontmatter id, so this
    normally agrees with the file's own name; the fallback only matters for
    malformed frontmatter that still happened to trigger note mode.
    """
    explicit_id = fm.get("id")
    if isinstance(explicit_id, str) and explicit_id.strip():
        return explicit_id.strip()
    name = fm.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return PurePosixPath(path.replace("\\", "/")).stem


def _first_heading_title(body: str) -> str | None:
    """Return the first ATX H1 title in *body*, if any."""
    match = _ATX_H1_RE.search(body)
    return match.group(1).strip() if match else None


def _collect_inline_tags(body: str) -> list[str]:
    """Collect ``#tag`` mentions from *body*, skipping ATX heading lines."""
    tags: set[str] = set()
    for line in body.splitlines():
        if _ATX_HEADING_LINE_RE.match(line):
            continue
        tags.update(match.group(1) for match in _INLINE_TAG_RE.finditer(line))
    return sorted(tags)


def _resolve_note_ref(raw: str, project_name: str) -> str:
    """Compute the deterministic uid a note-vault reference points to.

    Bare slug -> same-project Note; ``project:slug`` -> cross-project Note
    (e.g. the global vault). Unresolved targets simply fail to MATCH
    downstream in ``_create_relationships`` — no phantom edges are created.
    """
    raw = raw.strip()
    if ":" in raw:
        proj, slug = raw.split(":", 1)
        return f"{proj}:note:{slug}"
    return f"{project_name}:note:{raw}"


def _extract_wikilinks(content: str, project_name: str, from_qn: str) -> list[ParsedRelationship]:
    """Extract ``[[target]]``/``[[target|alias]]`` links as LINKS_TO relationships.

    ``[[note#heading]]``/``[[note^block]]`` refs resolve to the target note
    only (v1) — the heading/block fragment is dropped.
    """
    rels: list[ParsedRelationship] = []
    for match in _WIKILINK_RE.finditer(content):
        target_raw = match.group(1).split("#", 1)[0].split("^", 1)[0].strip()
        if not target_raw:
            continue
        alias = (match.group(2) or target_raw).strip()
        rels.append(
            ParsedRelationship(
                from_qualified_name=from_qn,
                rel_type=RelType.LINKS_TO,
                to_name=_resolve_note_ref(target_raw, project_name),
                properties={"alias": alias},
            )
        )
    return rels


def _extract_frontmatter_refs(fm: dict[str, Any], project_name: str, from_qn: str) -> list[ParsedRelationship]:
    """Extract ``derived_from``/``supersedes`` frontmatter lists as relationships."""
    rels: list[ParsedRelationship] = []
    for key, rel_type in (("derived_from", RelType.DERIVED_FROM), ("supersedes", RelType.SUPERSEDES)):
        values = fm.get(key)
        if not isinstance(values, list):
            continue
        rels.extend(
            ParsedRelationship(
                from_qualified_name=from_qn,
                rel_type=rel_type,
                to_name=_resolve_note_ref(v, project_name),
            )
            for v in values
            if isinstance(v, str) and v.strip()
        )
    return rels


def _classify_anchor(raw: str) -> tuple[str, str | None, str, str | None]:
    """Classify a frontmatter ``anchors:`` entry into ``(form, project_hint, target, symbol)``.

    Forms — ``uid`` (canonical ``project:qualified.name``), ``path`` (bare
    relative, resolved within the note's own project), ``project_path``
    (``project:relative/path``), ``absolute_path``. Detection is
    deterministic (decision Q9): paths contain ``/`` or a file extension;
    uids are dotted qualified names. A short/ambiguous extension
    (``_AMBIGUOUS_SHORT_EXTS``) only counts as a path signal alongside a
    ``/``, since alone it's equally plausible as a dotted symbol's final
    segment (e.g. ``models.Config.c``); a curated allowlist of conventional
    extensionless filenames (``_CONVENTIONAL_FILENAMES``, e.g.
    ``Dockerfile``) is also treated as path-form. An optional ``#Symbol``
    suffix refines a path anchor to a specific entity within the resolved
    file. Resolution against the live graph happens in
    ``GraphClient.resolve_anchors`` — this only classifies the form.
    """
    main, _, symbol_part = raw.strip().partition("#")
    main = main.strip()
    symbol = symbol_part.strip() or None

    if _ABS_PATH_RE.match(main):
        return "absolute_path", None, main.replace("\\", "/"), symbol

    project_hint: str | None = None
    rest = main
    if ":" in main:
        prefix, _, tail = main.partition(":")
        if prefix and tail:
            project_hint = prefix
            rest = tail

    has_slash = "/" in rest
    ext_match = _FILE_EXT_RE.search(rest)
    # A short/ambiguous extension (see _AMBIGUOUS_SHORT_EXTS) only counts as a
    # path signal when paired with a "/" — on its own it's equally plausible
    # as a dotted symbol name's final segment.
    is_ambiguous_ext = ext_match is not None and ext_match.group(0)[1:] in _AMBIGUOUS_SHORT_EXTS
    is_path_like = has_slash or (ext_match is not None and not is_ambiguous_ext) or rest in _CONVENTIONAL_FILENAMES
    if is_path_like:
        rest = rest.replace("\\", "/")
        return ("project_path" if project_hint else "path"), project_hint, rest, symbol

    return "uid", None, main, symbol


def _extract_anchors(fm: dict[str, Any], from_qn: str) -> list[ParsedRelationship]:
    """Extract explicit ``anchors:`` frontmatter as DOCUMENTS relationships.

    Resolution (uid direct match, path/project_path/absolute_path matching,
    ``#Symbol`` refinement, never multi-linking) happens graph-side in
    ``GraphClient.resolve_anchors`` — parsing only classifies each anchor's
    form deterministically.
    """
    values = fm.get("anchors")
    if not isinstance(values, list):
        return []
    rels: list[ParsedRelationship] = []
    for v in values:
        if not isinstance(v, str) or not v.strip():
            continue
        raw = v.strip()
        form, project_hint, target, symbol = _classify_anchor(raw)
        if not target:
            _log.debug("Skipping malformed anchors: entry with empty target: %r", raw)
            continue
        props: dict[str, Any] = {
            "link_type": "anchor",
            "confidence": 1.0,
            "anchor_form": form,
            "anchor_raw": raw,
        }
        if project_hint:
            props["anchor_project"] = project_hint
        if symbol:
            props["anchor_symbol"] = symbol
        rels.append(
            ParsedRelationship(
                from_qualified_name=from_qn,
                rel_type=RelType.DOCUMENTS,
                to_name=target,
                properties=props,
            )
        )
    return rels


_HARNESS_DIALECT_EXTRA_KEYS = frozenset({"id", "kind", "tags", "derived_from", "supersedes", "anchors"})


def _parse_markdown_note(
    path: str,
    source: bytes,
    project_name: str,
    fm: dict[str, Any],
    body_start: int,
    note_kind: str,
    subtype: str,
) -> ParsedFile:
    """Emit ONE Note entity (atomic-note granularity) for a frontmatter-triggered file."""
    body = source[body_start:].decode("utf-8", errors="replace").strip()
    slug = _note_slug(fm, path)
    full_qn = f"{project_name}:note:{slug}"

    is_harness_dialect = "kind" not in fm
    if is_harness_dialect:
        title = str(fm["description"])
    else:
        title = _first_heading_title(body) or slug.replace("-", " ").replace("_", " ").strip().title()

    frontmatter_tags = [t.strip() for t in (fm.get("tags") or []) if isinstance(t, str) and t.strip()]
    tags = sorted(set(frontmatter_tags) | set(_collect_inline_tags(body)))

    extra_properties = {k: v for k, v in fm.items() if k not in _HARNESS_DIALECT_EXTRA_KEYS}
    if subtype:
        extra_properties["subtype"] = subtype

    total_lines = source.count(b"\n") + (1 if source and not source.endswith(b"\n") else 0)

    note_entity = ParsedEntity(
        name=title,
        qualified_name=full_qn,
        label=NodeLabel.NOTE,
        kind=note_kind,
        line_start=1,
        line_end=max(total_lines, 1),
        file_path=path,
        docstring=body or None,
        tags=tags,
        extra_properties=extra_properties,
    )

    relationships: list[ParsedRelationship] = [
        *_extract_wikilinks(body, project_name, full_qn),
        *_extract_frontmatter_refs(fm, project_name, full_qn),
        *_extract_anchors(fm, full_qn),
    ]
    relationships.extend(
        ParsedRelationship(
            from_qualified_name=full_qn,
            rel_type=RelType.DOCUMENTS,
            to_name=ref.target_name,
            properties={"link_type": ref.link_type, "confidence": ref.confidence, "is_file_ref": ref.is_file_ref},
        )
        for ref in _extract_doc_references("", body, level=0)
    )

    return ParsedFile(file_path=path, language="markdown", entities=[note_entity], relationships=relationships)


def _parse_markdown(
    path: str,
    source: bytes,
    root: Node,
    project_name: str,
) -> ParsedFile:
    """Extract DocFile/DocSection entities from a markdown parse tree.

    A file whose frontmatter triggers note mode (see ``_note_mode_kind``)
    instead emits a single Note entity via ``_parse_markdown_note`` — vault
    and harness-memory files skip DocFile/DocSection splitting entirely.
    """
    frontmatter = _extract_frontmatter(root, source)
    if frontmatter is not None:
        note_mode = _note_mode_kind(frontmatter.raw)
        if note_mode is not None:
            note_kind, subtype = note_mode
            return _parse_markdown_note(
                path, source, project_name, frontmatter.raw, frontmatter.body_start, note_kind, subtype
            )

    posix_path = path.replace("\\", "/")
    filename = PurePosixPath(posix_path).name
    file_qn = f"{project_name}:{posix_path}"
    total_lines = source.count(b"\n") + (1 if source and not source.endswith(b"\n") else 0)
    if not source:
        total_lines = 0

    entities: list[ParsedEntity] = []
    relationships: list[ParsedRelationship] = []

    # DocFile entity
    doc_file = ParsedEntity(
        name=filename,
        qualified_name=file_qn,
        label=NodeLabel.DOC_FILE,
        kind="doc_file",
        line_start=1,
        line_end=max(total_lines, 1),
        file_path=path,
    )
    entities.append(doc_file)

    # Walk top-level section nodes. seen_qns tracks every DocSection qualified_name
    # emitted so far in this file so duplicate sibling headings (same breadcrumb)
    # get disambiguated instead of silently colliding.
    heading_stack: list[tuple[int, str]] = []
    seen_qns: dict[str, int] = {}
    for child in root.children:
        if child.type == "section":
            _process_md_section(
                child, path, source, project_name, file_qn, heading_stack, entities, relationships, seen_qns
            )

    return ParsedFile(
        file_path=path,
        language="markdown",
        entities=entities,
        relationships=relationships,
    )


def _md_heading_info(node: Node, source: bytes) -> tuple[int, str] | None:
    """Extract (level, title) from an atx_heading or setext_heading node."""
    if node.type == "atx_heading":
        level = 0
        title = ""
        for child in node.children:
            if child.type in _ATX_LEVEL:
                level = _ATX_LEVEL[child.type]
            elif child.type == "inline":
                title = source[child.start_byte : child.end_byte].decode("utf-8", errors="replace").strip()
        return (level, title) if level else None

    if node.type == "setext_heading":
        level = 0
        title = ""
        for child in node.children:
            if child.type in _SETEXT_LEVEL:
                level = _SETEXT_LEVEL[child.type]
            elif child.type == "paragraph":
                for inner in child.children:
                    if inner.type == "inline":
                        title = source[inner.start_byte : inner.end_byte].decode("utf-8", errors="replace").strip()
                        break
        return (level, title) if level else None

    return None


def _collect_code_langs(nodes: list[Node], source: bytes) -> list[str]:
    """Collect fenced code block language annotations from a list of nodes."""
    langs: set[str] = set()
    for node in nodes:
        if node.type == "fenced_code_block":
            for child in node.children:
                if child.type == "info_string":
                    lang_text = source[child.start_byte : child.end_byte].decode("utf-8", errors="replace").strip()
                    if lang_text:
                        langs.add(lang_text)
    return sorted(f"lang:{lang}" for lang in langs)


# ---------------------------------------------------------------------------
# Doc-code reference extraction
# ---------------------------------------------------------------------------

_MIN_REF_NAME_LEN = 3

# Match `symbol_name` or `symbol_name()` or `module.Class.method` in backticks
_BACKTICK_SYMBOL_RE = re.compile(r"`([A-Za-z_][\w.]*(?:\(\))?)`")

# Match file paths like src/auth/service.py with at least one / and a known extension
_FILE_REF_RE = re.compile(r"(?<!\w)([\w./\\-]+\.(?:py|pyi|js|ts|jsx|tsx|java|go|rs|rb|cpp|c|h|hpp|md))(?!\w)")

# CamelCase or snake_case identifier (for headings)
_HEADING_IDENT_RE = re.compile(r"^(?:[A-Z][a-zA-Z0-9]*(?:[A-Z][a-z0-9]+)+|[a-z][a-z0-9]*(?:_[a-z0-9]+)+)$")


@dataclass(frozen=True)
class _DocRef:
    """A reference from a doc section to a code entity."""

    target_name: str
    link_type: str
    confidence: float
    is_file_ref: bool = False


def _extract_doc_references(title: str, content: str | None, level: int) -> list[_DocRef]:
    """Extract code references from a doc section's title and content.

    Returns a deduplicated list of _DocRef, keeping highest confidence per target_name.
    """
    refs: dict[str, _DocRef] = {}

    def _add(ref: _DocRef) -> None:
        existing = refs.get(ref.target_name)
        if existing is None or ref.confidence > existing.confidence:
            refs[ref.target_name] = ref

    # 1. Header-as-symbol: only H2+ (H1 is doc title, not a code ref)
    if level >= 2 and title and _HEADING_IDENT_RE.match(title) and len(title) >= _MIN_REF_NAME_LEN:
        _add(_DocRef(target_name=title, link_type="explicit", confidence=0.9))

    # 2. Backtick symbol mentions in content
    if content:
        for match in _BACKTICK_SYMBOL_RE.finditer(content):
            symbol = match.group(1)
            # Strip trailing ()
            symbol = symbol.rstrip("()")
            # Take last segment of dotted paths (auth.service.MyClass -> MyClass)
            if "." in symbol:
                symbol = symbol.rsplit(".", 1)[1]
            if len(symbol) >= _MIN_REF_NAME_LEN:
                _add(_DocRef(target_name=symbol, link_type="symbol_mention", confidence=0.8))

    # 3. File path references in content
    if content:
        for match in _FILE_REF_RE.finditer(content):
            path_str = match.group(1)
            # Normalize backslashes and require at least one /
            normalized = path_str.replace("\\", "/")
            if "/" in normalized and len(normalized) >= _MIN_REF_NAME_LEN:
                _add(_DocRef(target_name=normalized, link_type="file_ref", confidence=0.85, is_file_ref=True))

    return list(refs.values())


def _dedupe_section_qn(section_qn: str, seen_qns: dict[str, int]) -> str:
    """Disambiguate a DocSection qualified_name against duplicate sibling headings.

    Two sibling headings sharing the same title (and thus the same breadcrumb)
    would otherwise produce identical qualified_names, colliding on the same
    uid downstream. The first occurrence keeps the bare qn; every further
    occurrence gets a stable ``#N`` suffix so no section is silently dropped.
    """
    occurrence = seen_qns.get(section_qn, 0) + 1
    seen_qns[section_qn] = occurrence
    return section_qn if occurrence == 1 else f"{section_qn}#{occurrence}"


def _emit_md_section(
    *,
    path: str,
    source: bytes,
    project_name: str,
    file_qn: str,
    level: int,
    title: str,
    heading_stack: list[tuple[int, str]],
    heading_node: Node | None,
    content_nodes: list[Node],
    section_start_line: int,
    section_end_line: int,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen_qns: dict[str, int],
) -> None:
    """Create a DocSection entity and CONTAINS relationship."""
    # Update heading stack
    if level > 0:
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))

    header_path = " > ".join(t for _, t in heading_stack) if heading_stack else None
    tags = _collect_code_langs(content_nodes, source)

    # Extract content text
    if content_nodes:
        start_byte = content_nodes[0].start_byte
        end_byte = content_nodes[-1].end_byte
        content_text = source[start_byte:end_byte].decode("utf-8", errors="replace").strip() or None
    elif heading_node is not None:
        content_text = None
    else:
        content_text = None

    # Build name and qualified_name
    posix_path = path.replace("\\", "/")
    if level == 0:
        name = PurePosixPath(posix_path).name
        breadcrumb = name
    else:
        name = title
        breadcrumb = header_path or title

    section_qn = _dedupe_section_qn(f"{project_name}:{posix_path} > {breadcrumb}", seen_qns)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=section_qn,
            label=NodeLabel.DOC_SECTION,
            kind="section",
            line_start=section_start_line,
            line_end=max(section_end_line, section_start_line),
            file_path=path,
            docstring=content_text,
            header_path=header_path,
            header_level=level,
            tags=tags,
        )
    )

    relationships.append(
        ParsedRelationship(
            from_qualified_name=file_qn,
            rel_type=RelType.CONTAINS,
            to_name=section_qn,
        )
    )

    # Extract doc-code references and emit DOCUMENTS relationships
    relationships.extend(
        ParsedRelationship(
            from_qualified_name=section_qn,
            rel_type=RelType.DOCUMENTS,
            to_name=ref.target_name,
            properties={
                "link_type": ref.link_type,
                "confidence": ref.confidence,
                "is_file_ref": ref.is_file_ref,
            },
        )
        for ref in _extract_doc_references(title, content_text, level)
    )

    # Extract [[wikilinks]] — ordinary docs coexist with the note vault and
    # may link into it (or between themselves) using the same syntax.
    if content_text:
        relationships.extend(_extract_wikilinks(content_text, project_name, section_qn))


_HEADING_TYPES = frozenset({"atx_heading", "setext_heading"})


def _split_md_section_children(
    section_node: Node,
) -> list[tuple[str, Any]]:
    """Split a section node's children into ordered segments.

    Returns a list of ``(kind, payload)`` tuples:
    - ``("heading", (heading_node, [content_nodes]))``
    - ``("preamble", (None, [content_nodes]))``
    - ``("subsection", section_node)``
    """
    items: list[tuple[str, Any]] = []
    current_content: list[Node] = []
    prev_heading: Node | None = None

    for child in section_node.children:
        if child.type in _HEADING_TYPES:
            if prev_heading is not None:
                items.append(("heading", (prev_heading, current_content)))
                current_content = []
            elif current_content:
                items.append(("preamble", (None, current_content)))
                current_content = []
            prev_heading = child
        elif child.type == "section":
            if prev_heading is not None:
                items.append(("heading", (prev_heading, current_content)))
                current_content = []
                prev_heading = None
            elif current_content:
                items.append(("preamble", (None, current_content)))
                current_content = []
            items.append(("subsection", child))
        else:
            current_content.append(child)

    # Flush trailing segment
    if prev_heading is not None:
        items.append(("heading", (prev_heading, current_content)))
    elif current_content:
        items.append(("preamble", (None, current_content)))

    return items


def _node_end_line(node: Node) -> int:
    """Convert a node's ``end_point`` into a 1-based, inclusive end line number.

    Block nodes normally consume their trailing newline, so ``end_point`` lands
    at column 0 of the row *after* the last content line — using the 0-based
    row directly already yields the correct 1-based line number in that case.
    When the node ends mid-line (no trailing newline to consume, e.g. EOF),
    the row must be incremented to get the correct 1-based number.
    """
    row, col = node.end_point
    return row if col == 0 else row + 1


def _process_md_section(
    section_node: Node,
    path: str,
    source: bytes,
    project_name: str,
    file_qn: str,
    heading_stack: list[tuple[int, str]],
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen_qns: dict[str, int],
) -> None:
    """Process a section node, emitting DocSection entities.

    The tree-sitter-markdown grammar nests ATX headings into ``section`` nodes
    hierarchically, but setext headings at the same level appear as flat
    siblings.  This function splits children into segments and recurses.
    """
    for kind, payload in _split_md_section_children(section_node):
        if kind == "subsection":
            _process_md_section(
                payload, path, source, project_name, file_qn, heading_stack, entities, relationships, seen_qns
            )
            continue

        heading_node, content_nodes = payload
        if heading_node is not None:
            info = _md_heading_info(heading_node, source)
            level, title = info or (0, "")
            section_start = heading_node.start_point[0] + 1
        else:
            level, title = 0, ""
            section_start = content_nodes[0].start_point[0] + 1 if content_nodes else 1

        section_end = (
            _node_end_line(content_nodes[-1])
            if content_nodes
            else _node_end_line(heading_node)
            if heading_node is not None
            else section_start
        )

        _emit_md_section(
            path=path,
            source=source,
            project_name=project_name,
            file_qn=file_qn,
            level=level,
            title=title,
            heading_stack=heading_stack,
            heading_node=heading_node,
            content_nodes=content_nodes,
            section_start_line=section_start,
            section_end_line=section_end,
            entities=entities,
            relationships=relationships,
            seen_qns=seen_qns,
        )

    # Restore heading stack for sibling sections
    for child in section_node.children:
        if child.type in _HEADING_TYPES:
            info = _md_heading_info(child, source)
            if info:
                while heading_stack and heading_stack[-1][0] >= info[0]:
                    heading_stack.pop()
            break


# ---------------------------------------------------------------------------
# Language registration (after _parse_markdown is defined)
# ---------------------------------------------------------------------------

register_language(
    LanguageConfig(
        name="markdown",
        extensions=frozenset({".md"}),
        language=_MD_LANGUAGE,
        query=_MD_QUERY,
        parse_func=_parse_markdown,
    )
)
