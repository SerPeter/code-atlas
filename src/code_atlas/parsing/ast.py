"""Tree-sitter based parser for extracting code entities and relationships.

Parses source files using py-tree-sitter and extracts entities (classes,
functions, methods, imports, variables) and relationships (DEFINES, CALLS,
IMPORTS, INHERITS) for graph ingestion.

Phase 1: Python only. Other languages follow via register_language().
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import tree_sitter_markdown as tsmarkdown
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Query

from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind, ValueKind, Visibility

if TYPE_CHECKING:
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


_LANGUAGES: dict[str, LanguageConfig] = {}
_EXTENSION_MAP: dict[str, str] = {}


def register_language(config: LanguageConfig) -> None:
    """Register a language configuration."""
    _LANGUAGES[config.name] = config
    for ext in config.extensions:
        _EXTENSION_MAP[ext] = config.name


def get_language_for_file(path: str) -> LanguageConfig | None:
    """Look up language config by file extension."""
    suffix = PurePosixPath(path).suffix.lower()
    lang_name = _EXTENSION_MAP.get(suffix)
    if lang_name is None:
        return None
    return _LANGUAGES.get(lang_name)


# ---------------------------------------------------------------------------
# Python tree-sitter query
# ---------------------------------------------------------------------------

# Extended query beyond the default tags.scm — captures classes, functions,
# imports, base classes, decorators, and module-level assignments.
_PYTHON_QUERY = """
; Class definitions
(class_definition
  name: (identifier) @class.name) @class.def

; Function/method definitions
(function_definition
  name: (identifier) @function.name) @function.def

; Decorated definitions (capture decorator name for tags)
(decorated_definition
  (decorator
    (identifier) @decorator.name)?
  (decorator
    (attribute
      attribute: (identifier) @decorator.attr))?
) @decorated.def

; Import statements
(import_statement
  name: (dotted_name) @import.name) @import.stmt

; Import-from statements
(import_from_statement
  module_name: (dotted_name)? @import_from.module
  name: (dotted_name)? @import_from.name) @import_from.stmt

; Import-from with aliased imports
(import_from_statement
  module_name: (dotted_name)? @import_from_alias.module
  name: (aliased_import
    name: (dotted_name) @import_from_alias.name)) @import_from_alias.stmt

; Base classes in class definitions
(class_definition
  name: (identifier) @base_class.class_name
  superclasses: (argument_list
    (identifier) @base_class.base)) @base_class.def

; Module-level assignments (variables/constants)
(module
  (expression_statement
    (assignment
      left: (identifier) @assign.name
      right: (_) @assign.value))) @assign.stmt

; Call expressions (for CALLS relationships)
(call
  function: (identifier) @call.name) @call.expr

(call
  function: (attribute
    attribute: (identifier) @call.attr)) @call.attr_expr
"""

# ---------------------------------------------------------------------------
# Python language registration
# ---------------------------------------------------------------------------

_PY_LANGUAGE = Language(tspython.language())
_PY_QUERY = Query(_PY_LANGUAGE, _PYTHON_QUERY)

register_language(
    LanguageConfig(
        name="python",
        extensions=frozenset({".py", ".pyi"}),
        language=_PY_LANGUAGE,
        query=_PY_QUERY,
    )
)

# ---------------------------------------------------------------------------
# Markdown language registration
# ---------------------------------------------------------------------------

_MD_LANGUAGE = Language(tsmarkdown.language())
_MD_QUERY = Query(_MD_LANGUAGE, "(document) @doc")

register_language(
    LanguageConfig(
        name="markdown",
        extensions=frozenset({".md"}),
        language=_MD_LANGUAGE,
        query=_MD_QUERY,
    )
)


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
# Internal helpers
# ---------------------------------------------------------------------------


def _node_text(node: Node) -> str:
    """Get the text content of a tree-sitter node as a string."""
    text = node.text
    if text is None:
        return ""
    return text.decode("utf-8", errors="replace")


def _module_qualified_name(file_path: str) -> str:
    """Convert file path to a Python module qualified name.

    ``src/code_atlas/parser.py`` → ``src.code_atlas.parser``
    ``src/code_atlas/__init__.py`` → ``src.code_atlas``
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    parts = list(p.parts)
    # Strip .py / .pyi extension
    if parts and parts[-1].endswith((".py", ".pyi")):
        filename = parts[-1]
        if filename in {"__init__.py", "__init__.pyi"}:
            parts = parts[:-1]
        else:
            parts[-1] = filename.rsplit(".", 1)[0]
    return ".".join(parts)


def _visibility_from_name(name: str) -> str:
    """Determine visibility from Python naming conventions."""
    if name.startswith("__") and name.endswith("__"):
        return Visibility.PUBLIC  # dunder methods are public
    if name.startswith("__"):
        return Visibility.PRIVATE  # name-mangled
    if name.startswith("_"):
        return Visibility.PRIVATE
    return Visibility.PUBLIC


def _extract_docstring(node: Node, source: bytes) -> str | None:
    """Extract docstring from the first statement of a function/class body."""
    body = node.child_by_field_name("body")
    if body is None:
        return None
    for child in body.children:
        if child.type == "expression_statement":
            for inner in child.children:
                if inner.type == "string":
                    raw = source[inner.start_byte : inner.end_byte].decode("utf-8", errors="replace")
                    # Strip triple quotes
                    for q in ('"""', "'''", '"', "'"):
                        if raw.startswith(q) and raw.endswith(q):
                            raw = raw[len(q) : -len(q)]
                            break
                    return raw.strip()
            break
        # Skip comments and pass statements
        if child.type not in ("comment", "pass_statement"):
            break
    return None


def _extract_signature(node: Node, source: bytes) -> str | None:
    """Extract function signature (def line without the body)."""
    if node.type != "function_definition":
        return None
    # Get everything from 'def' to the colon before the body
    params = node.child_by_field_name("parameters")
    name = node.child_by_field_name("name")
    ret = node.child_by_field_name("return_type")
    if name is None or params is None:
        return None
    end_byte = ret.end_byte if ret else params.end_byte
    sig_bytes = source[node.start_byte : end_byte]
    return sig_bytes.decode("utf-8", errors="replace")


def _is_inside_class(node: Node) -> str | None:
    """Check if a node is inside a class body. Returns class name or None."""
    parent = node.parent
    while parent is not None:
        if parent.type == "class_definition":
            name_node = parent.child_by_field_name("name")
            if name_node is not None:
                return _node_text(name_node)
        parent = parent.parent
    return None


def _is_async(node: Node) -> bool:
    """Check if a function_definition is async (has 'async' keyword prefix)."""
    if node.type != "function_definition":
        return False
    parent = node.parent
    if parent is not None and parent.type == "decorated_definition":
        # Check if decorated def's children include async
        for child in parent.children:
            if child.type == "function_definition":
                # Check preceding sibling
                prev = child.prev_sibling
                while prev is not None:
                    if prev.type == "async":
                        return True
                    prev = prev.prev_sibling
                break
    # Direct check: look for async keyword before the 'def' keyword
    prev = node.prev_sibling
    while prev is not None:
        if prev.type == "async":
            return True
        prev = prev.prev_sibling
    # Also check within the node text itself
    first_token = node.children[0] if node.children else None
    if first_token is not None and first_token.type == "async":
        return False  # tree-sitter doesn't nest this way, already checked
    return False


def _callable_kind_for_method(name: str, node: Node) -> str:
    """Determine the callable kind for a method inside a class."""
    # Check for decorators
    parent = node.parent
    if parent is not None and parent.type == "decorated_definition":
        for child in parent.children:
            if child.type == "decorator":
                dec_text = _node_text(child).strip()
                if "@staticmethod" in dec_text:
                    return CallableKind.STATIC_METHOD
                if "@classmethod" in dec_text:
                    return CallableKind.CLASS_METHOD
                if "@property" in dec_text:
                    return CallableKind.PROPERTY

    if name == "__init__":
        return CallableKind.CONSTRUCTOR
    if name == "__del__":
        return CallableKind.DESTRUCTOR
    return CallableKind.METHOD


def _get_decorators(node: Node) -> list[str]:
    """Extract decorator names from a decorated_definition parent.

    Preserves full decorator text including arguments so detectors can
    inspect route paths, event names, etc.  Multi-line decorators are
    collapsed to a single line with normalized whitespace.
    """
    tags: list[str] = []
    parent = node.parent
    if parent is not None and parent.type == "decorated_definition":
        for child in parent.children:
            if child.type == "decorator":
                dec_text = " ".join(_node_text(child).split()).lstrip("@").strip()
                tags.append(f"decorator:{dec_text}")
    return tags


# ---------------------------------------------------------------------------
# Core parse function
# ---------------------------------------------------------------------------


def parse_file(path: str, source: bytes, project_name: str) -> ParsedFile | None:
    """Parse a source file and extract entities + relationships.

    Returns ParsedFile with entities mapped to schema labels/kinds,
    qualified names built from file path + nesting. Returns None if
    the language is not supported.
    """
    lang_config = get_language_for_file(path)
    if lang_config is None:
        return None

    parser = Parser(lang_config.language)
    tree = parser.parse(source)

    if lang_config.name == "python":
        result = _parse_python(path, source, tree.root_node, project_name)
    elif lang_config.name == "markdown":
        result = _parse_markdown(path, source, tree.root_node, project_name)
    else:
        return None

    # Post-parse pass: compute content hashes for all entities
    return ParsedFile(
        file_path=result.file_path,
        language=result.language,
        entities=[replace(e, content_hash=_compute_content_hash(e)) for e in result.entities],
        relationships=result.relationships,
    )


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
            # Take last segment of dotted paths (auth.service.MyClass → MyClass)
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


def _parse_python(
    path: str,
    source: bytes,
    root: Node,
    project_name: str,
) -> ParsedFile:
    """Extract entities and relationships from a Python parse tree."""
    module_qn = _module_qualified_name(path)
    is_package = path.replace("\\", "/").endswith("__init__.py")

    entities: list[ParsedEntity] = []
    relationships: list[ParsedRelationship] = []

    # Track seen entities by (line_start, name) to dedup (competitor insight P1)
    seen: set[tuple[int, str]] = set()

    # Module/Package entity
    module_label = NodeLabel.PACKAGE if is_package else NodeLabel.MODULE
    entities.append(
        ParsedEntity(
            name=module_qn.rsplit(".", 1)[-1] if "." in module_qn else module_qn,
            qualified_name=f"{project_name}:{module_qn}",
            label=module_label,
            kind="package" if is_package else "module",
            line_start=1,
            line_end=root.end_point[0] + 1,
            file_path=path,
        )
    )

    # Walk the tree for classes, functions, imports, assignments
    _walk_python_node(root, path, source, project_name, module_qn, entities, relationships, seen)

    return ParsedFile(
        file_path=path,
        language="python",
        entities=entities,
        relationships=relationships,
    )


def _walk_python_node(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Recursively walk the parse tree to extract entities."""
    for child in node.children:
        # Handle decorated definitions — extract the inner def/class
        if child.type == "decorated_definition":
            for inner in child.children:
                if inner.type in ("function_definition", "class_definition"):
                    _process_definition(inner, path, source, project_name, module_qn, entities, relationships, seen)
            continue

        if child.type in ("function_definition", "class_definition"):
            _process_definition(child, path, source, project_name, module_qn, entities, relationships, seen)
            continue

        if child.type in ("import_statement", "import_from_statement"):
            _process_import(child, project_name, module_qn, relationships)
            continue

        if child.type == "expression_statement":
            _process_assignment(child, path, project_name, module_qn, node, entities, seen)
            continue

        # Recurse into blocks (if, for, try, with, etc.) but not into functions/classes
        if child.type not in ("function_definition", "class_definition"):
            _walk_python_node(child, path, source, project_name, module_qn, entities, relationships, seen)


def _process_definition(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process a class_definition or function_definition node."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = _node_text(name_node)
    line_start = node.start_point[0] + 1

    # Dedup by (line_start, name) — competitor insight P1
    key = (line_start, name)
    if key in seen:
        return
    seen.add(key)

    if node.type == "class_definition":
        _process_class(node, path, source, project_name, module_qn, name, entities, relationships, seen)
    elif node.type == "function_definition":
        _process_function(node, path, source, project_name, module_qn, name, entities, relationships)


def _process_class(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    name: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process a class_definition node."""
    class_name = _is_inside_class(node)
    docstring = _extract_docstring(node, source)
    tags = _get_decorators(node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    qn = f"{module_qn}.{name}" if class_name is None else f"{module_qn}.{class_name}.{name}"
    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=TypeDefKind.CLASS,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            visibility=_visibility_from_name(name),
            tags=tags,
        )
    )
    # DEFINES relationship from module → class
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )
    # Base classes → INHERITS
    superclasses = node.child_by_field_name("superclasses")
    if superclasses is not None:
        for base in superclasses.children:
            if base.type == "identifier":
                base_name = _node_text(base)
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=f"{project_name}:{qn}",
                        rel_type=RelType.INHERITS,
                        to_name=base_name,
                    )
                )
    # Recurse into class body for methods, nested classes, etc.
    body = node.child_by_field_name("body")
    if body is not None:
        _walk_python_node(body, path, source, project_name, module_qn, entities, relationships, seen)


def _process_function(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    name: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a function_definition node."""
    class_name = _is_inside_class(node)
    docstring = _extract_docstring(node, source)
    tags = _get_decorators(node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    is_method = class_name is not None
    if is_method:
        kind = _callable_kind_for_method(name, node)
        qn = f"{module_qn}.{class_name}.{name}"
    else:
        kind = CallableKind.FUNCTION
        qn = f"{module_qn}.{name}"

    if _is_async(node):
        tags = [*tags, "async"]

    signature = _extract_signature(node, source)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.CALLABLE,
            kind=kind,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            signature=signature,
            visibility=_visibility_from_name(name),
            tags=tags,
        )
    )

    # DEFINES relationship
    parent_qn = f"{module_qn}.{class_name}" if is_method else module_qn
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{parent_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # Walk function body for call sites
    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, source, f"{project_name}:{qn}", relationships)


def _process_import(
    node: Node,
    project_name: str,
    module_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Process import_statement or import_from_statement."""
    if node.type == "import_statement":
        for child in node.children:
            if child.type == "dotted_name":
                import_name = _node_text(child)
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=f"{project_name}:{module_qn}",
                        rel_type=RelType.IMPORTS,
                        to_name=import_name,
                    )
                )
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                if name_node is not None:
                    import_name = _node_text(name_node)
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=f"{project_name}:{module_qn}",
                            rel_type=RelType.IMPORTS,
                            to_name=import_name,
                        )
                    )
    elif node.type == "import_from_statement":
        module_node = node.child_by_field_name("module_name")
        module_name = _node_text(module_node) if module_node else ""
        # Collect imported names
        for child in node.children:
            if child.type == "dotted_name" and child != module_node:
                imported = _node_text(child)
                full_name = f"{module_name}.{imported}" if module_name else imported
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=f"{project_name}:{module_qn}",
                        rel_type=RelType.IMPORTS,
                        to_name=full_name,
                    )
                )
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                if name_node is not None:
                    imported = _node_text(name_node)
                    full_name = f"{module_name}.{imported}" if module_name else imported
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=f"{project_name}:{module_qn}",
                            rel_type=RelType.IMPORTS,
                            to_name=full_name,
                        )
                    )


def _process_assignment(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    parent: Node,
    entities: list[ParsedEntity],
    seen: set[tuple[int, str]],
) -> None:
    """Process module-level or class-level assignments as Value entities."""
    # Only process assignments at module or class body level
    if parent.type not in ("module", "block"):
        return

    for child in node.children:
        if child.type != "assignment":
            continue
        left = child.child_by_field_name("left")
        if left is None or left.type != "identifier":
            continue
        name = _node_text(left)
        line_start = child.start_point[0] + 1

        key = (line_start, name)
        if key in seen:
            continue
        seen.add(key)

        class_name = _is_inside_class(node)
        if class_name is not None:
            qn = f"{module_qn}.{class_name}.{name}"
            kind = ValueKind.FIELD
        else:
            qn = f"{module_qn}.{name}"
            kind = ValueKind.CONSTANT if name.isupper() else ValueKind.VARIABLE

        entities.append(
            ParsedEntity(
                name=name,
                qualified_name=f"{project_name}:{qn}",
                label=NodeLabel.VALUE,
                kind=kind,
                line_start=line_start,
                line_end=child.end_point[0] + 1,
                file_path=path,
                visibility=_visibility_from_name(name),
            )
        )


def _extract_calls(
    node: Node,
    source: bytes,
    from_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Recursively extract call expressions from a function body."""
    for child in node.children:
        if child.type == "call":
            func = child.child_by_field_name("function")
            if func is not None:
                if func.type == "identifier":
                    call_name = _node_text(func)
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=from_qn,
                            rel_type=RelType.CALLS,
                            to_name=call_name,
                        )
                    )
                elif func.type == "attribute":
                    attr = func.child_by_field_name("attribute")
                    if attr is not None:
                        call_name = _node_text(attr)
                        relationships.append(
                            ParsedRelationship(
                                from_qualified_name=from_qn,
                                rel_type=RelType.CALLS,
                                to_name=call_name,
                            )
                        )
        # Recurse but don't descend into nested function/class definitions
        if child.type not in ("function_definition", "class_definition", "decorated_definition"):
            _extract_calls(child, source, from_qn, relationships)
