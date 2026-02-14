"""Markdown language support — tree-sitter parser for documentation files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import tree_sitter_markdown as tsmarkdown
from tree_sitter import Language, Query

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    register_language,
)
from code_atlas.schema import NodeLabel, RelType

if TYPE_CHECKING:
    from tree_sitter import Node


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


def _parse_markdown(
    path: str,
    source: bytes,
    root: Node,
    project_name: str,
) -> ParsedFile:
    """Extract DocFile/DocSection entities from a markdown parse tree."""
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

    # Walk top-level section nodes
    heading_stack: list[tuple[int, str]] = []
    for child in root.children:
        if child.type == "section":
            _process_md_section(child, path, source, project_name, file_qn, heading_stack, entities, relationships)

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

    section_qn = f"{project_name}:{posix_path} > {breadcrumb}"

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


def _process_md_section(
    section_node: Node,
    path: str,
    source: bytes,
    project_name: str,
    file_qn: str,
    heading_stack: list[tuple[int, str]],
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a section node, emitting DocSection entities.

    The tree-sitter-markdown grammar nests ATX headings into ``section`` nodes
    hierarchically, but setext headings at the same level appear as flat
    siblings.  This function splits children into segments and recurses.
    """
    for kind, payload in _split_md_section_children(section_node):
        if kind == "subsection":
            _process_md_section(payload, path, source, project_name, file_qn, heading_stack, entities, relationships)
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
            content_nodes[-1].end_point[0]
            if content_nodes
            else heading_node.end_point[0]
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
