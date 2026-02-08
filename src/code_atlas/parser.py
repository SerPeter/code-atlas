"""Tree-sitter based parser for extracting code entities and relationships.

Parses source files using py-tree-sitter and extracts entities (classes,
functions, methods, imports, variables) and relationships (DEFINES, CALLS,
IMPORTS, INHERITS) for graph ingestion.

Phase 1: Python only. Other languages follow via register_language().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

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


@dataclass(frozen=True)
class ParsedRelationship:
    """A relationship between entities, extracted from source."""

    from_qualified_name: str
    rel_type: RelType
    to_name: str


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
    """Extract decorator names from a decorated_definition parent."""
    tags: list[str] = []
    parent = node.parent
    if parent is not None and parent.type == "decorated_definition":
        for child in parent.children:
            if child.type == "decorator":
                dec_text = _node_text(child).strip().lstrip("@")
                # Take just the name (before any parentheses)
                paren = dec_text.find("(")
                if paren >= 0:
                    dec_text = dec_text[:paren]
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
        return _parse_python(path, source, tree.root_node, project_name)

    # Future languages
    return None


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
