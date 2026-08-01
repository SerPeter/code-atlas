"""Python language support — tree-sitter parser and pattern detectors."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import tree_sitter_python as tspython
from tree_sitter import Language, Query

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    looks_like_resource_path,
    node_text,
    register_language,
    slice_without_comments,
)
from code_atlas.parsing.detectors import (
    DetectorResult,
    PropertyEnrichment,
    register_detector,
)
from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind, ValueKind, Visibility

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tree_sitter import Node

    from code_atlas.graph.client import GraphClient


# ---------------------------------------------------------------------------
# Tree-sitter query
# ---------------------------------------------------------------------------

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
# Language registration
# ---------------------------------------------------------------------------

_PY_LANGUAGE = Language(tspython.language())
_PY_QUERY = Query(_PY_LANGUAGE, _PYTHON_QUERY)


# ---------------------------------------------------------------------------
# Python helpers
# ---------------------------------------------------------------------------


# Conventional source-root directories that are not import packages (src-layout).
_SOURCE_ROOTS: frozenset[str] = frozenset({"src"})


def module_qualified_name(file_path: str) -> str:
    """Convert file path to a Python module qualified name.

    ``src/code_atlas/parser.py`` -> ``code_atlas.parser`` (source-root dirs
    like ``src/`` are stripped so names match the import system)
    ``code_atlas/__init__.py`` -> ``code_atlas``
    ``src/__init__.py`` -> ``src`` (a source root that is itself a package is kept)
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
    if len(parts) > 1 and parts[0] in _SOURCE_ROOTS:
        parts = parts[1:]
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


# Matches a Python string literal prefix (r/f/b/u, any case, up to 2 chars,
# e.g. r/f/b/u/rb/br/rf/fr) when immediately followed by a quote character.
_STRING_PREFIX_RE = re.compile(r"^[A-Za-z]{1,2}(?=['\"])")


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
                    # Strip string prefix (r"""/f"""/b""" etc.) before quote matching
                    prefix_match = _STRING_PREFIX_RE.match(raw)
                    if prefix_match:
                        raw = raw[prefix_match.end() :]
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


_PY_COMMENT_TYPES = frozenset({"comment"})


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
    # Not a raw slice: a multi-line signature can carry a lint-suppression comment, and
    # rendering one into the outline puts a stray hash in a format that already gives
    # the hash character two other meanings.
    return slice_without_comments(node, source, end_byte, _PY_COMMENT_TYPES)


def _is_inside_class(node: Node) -> str | None:
    """Check if a node is inside a class body.

    Returns the dotted path of all enclosing class names (outermost first,
    e.g. ``"Outer.Middle"`` for a member of ``Outer.Middle.Inner``), or None
    if not nested in a class.
    """
    names: list[str] = []
    parent = node.parent
    while parent is not None:
        if parent.type == "class_definition":
            name_node = parent.child_by_field_name("name")
            if name_node is not None:
                names.append(node_text(name_node))
        parent = parent.parent
    if not names:
        return None
    return ".".join(reversed(names))


def _is_async(node: Node) -> bool:
    """Check if a function_definition is async ('async' is its first child).

    This holds regardless of whether the function is decorated — the
    ``async`` keyword is always the first child of the ``function_definition``
    node itself in tree-sitter-python's grammar.
    """
    if node.type != "function_definition":
        return False
    first_child = node.children[0] if node.children else None
    return first_child is not None and first_child.type == "async"


def _is_type_checking_condition(condition: Node) -> bool:
    """Check if an `if` condition node refers to TYPE_CHECKING.

    Matches both the bare-identifier form (``if TYPE_CHECKING:``) and the
    attribute form (``if typing.TYPE_CHECKING:``).
    """
    if condition.type == "identifier":
        return node_text(condition) == "TYPE_CHECKING"
    if condition.type == "attribute":
        attr = condition.child_by_field_name("attribute")
        return attr is not None and node_text(attr) == "TYPE_CHECKING"
    return False


def _callable_kind_for_method(name: str, node: Node) -> str:
    """Determine the callable kind for a method inside a class."""
    # Check for decorators
    parent = node.parent
    if parent is not None and parent.type == "decorated_definition":
        for child in parent.children:
            if child.type == "decorator":
                dec_text = node_text(child).strip()
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
                dec_text = " ".join(node_text(child).split()).lstrip("@").strip()
                tags.append(f"decorator:{dec_text}")
    return tags


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _tag_conditional_definitions(entities: list[ParsedEntity]) -> None:
    """Tag duplicate qualified_name entries as conditional (e.g. platform guards).

    The first occurrence is left unchanged; subsequent duplicates get a
    ``"conditional"`` tag added.  Modifies the list in place.
    """
    seen_qns: set[str] = set()
    for idx, entity in enumerate(entities):
        if entity.qualified_name in seen_qns:
            entities[idx] = replace(entity, tags=[*entity.tags, "conditional"])
        else:
            seen_qns.add(entity.qualified_name)


# ---------------------------------------------------------------------------
# Python parse entry point
# ---------------------------------------------------------------------------


def _parse_python(
    path: str,
    source: bytes,
    root: Node,
    project_name: str,
) -> ParsedFile:
    """Extract entities and relationships from a Python parse tree."""
    module_qn = module_qualified_name(path)
    is_package = path.replace("\\", "/").endswith(("__init__.py", "__init__.pyi"))
    # __package__ equivalent: the package itself for __init__, else the parent
    package_qn = module_qn if is_package else (module_qn.rsplit(".", 1)[0] if "." in module_qn else "")

    entities: list[ParsedEntity] = []
    relationships: list[ParsedRelationship] = []

    # Track seen entities by (line_start, name) to dedup (competitor insight P1)
    seen: set[tuple[int, str]] = set()
    # Track class names that are Enum subclasses (for enum_member detection)
    enum_classes: set[str] = set()

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
    _walk_python_node(
        root, path, source, project_name, module_qn, package_qn, entities, relationships, seen, enum_classes
    )

    # Runtime config surface. Runs after the main walk because it needs both the
    # finished entity list (to attribute each reference to its enclosing entity)
    # and the IMPORTS relationships (to know whether a bare `getenv`/`environ`
    # actually came from `os`).
    _extract_config_refs(root, source, entities, relationships)

    # Post-processing: tag conditional (duplicate) definitions
    _tag_conditional_definitions(entities)

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
    package_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    enum_classes: set[str],
    *,
    in_type_checking: bool = False,
) -> None:
    """Recursively walk the parse tree to extract entities."""
    for child in node.children:
        # Detect `if TYPE_CHECKING:` blocks and recurse with type_only flag
        if child.type == "if_statement":
            condition = child.child_by_field_name("condition")
            if condition is not None and _is_type_checking_condition(condition):
                body = child.child_by_field_name("consequence")
                if body is not None:
                    _walk_python_node(
                        body,
                        path,
                        source,
                        project_name,
                        module_qn,
                        package_qn,
                        entities,
                        relationships,
                        seen,
                        enum_classes,
                        in_type_checking=True,
                    )
                # Also walk the else/elif branch (runtime code)
                alternative = child.child_by_field_name("alternative")
                if alternative is not None:
                    _walk_python_node(
                        alternative,
                        path,
                        source,
                        project_name,
                        module_qn,
                        package_qn,
                        entities,
                        relationships,
                        seen,
                        enum_classes,
                    )
                continue

        # Handle decorated definitions — extract the inner def/class
        if child.type == "decorated_definition":
            for inner in child.children:
                if inner.type in ("function_definition", "class_definition"):
                    _process_definition(
                        inner,
                        path,
                        source,
                        project_name,
                        module_qn,
                        package_qn,
                        entities,
                        relationships,
                        seen,
                        enum_classes,
                    )
            continue

        if child.type in ("function_definition", "class_definition"):
            _process_definition(
                child, path, source, project_name, module_qn, package_qn, entities, relationships, seen, enum_classes
            )
            continue

        if child.type in ("import_statement", "import_from_statement"):
            _process_import(child, project_name, module_qn, package_qn, relationships, type_only=in_type_checking)
            continue

        if child.type == "expression_statement":
            _process_assignment(child, path, project_name, module_qn, node, entities, seen, enum_classes)
            continue

        # Recurse into blocks (if, for, try, with, etc.) but not into functions/classes
        if child.type not in ("function_definition", "class_definition"):
            _walk_python_node(
                child,
                path,
                source,
                project_name,
                module_qn,
                package_qn,
                entities,
                relationships,
                seen,
                enum_classes,
                in_type_checking=in_type_checking,
            )


def _process_definition(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    package_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    enum_classes: set[str],
) -> None:
    """Process a class_definition or function_definition node."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1

    # Dedup by (line_start, name) — competitor insight P1
    key = (line_start, name)
    if key in seen:
        return
    seen.add(key)

    if node.type == "class_definition":
        _process_class(
            node, path, source, project_name, module_qn, package_qn, name, entities, relationships, seen, enum_classes
        )
    elif node.type == "function_definition":
        _process_function(node, path, source, project_name, module_qn, name, entities, relationships)


_ENUM_BASES: frozenset[str] = frozenset({"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"})


def _process_class(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    package_qn: str,
    name: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    enum_classes: set[str],
) -> None:
    """Process a class_definition node."""
    class_name = _is_inside_class(node)
    docstring = _extract_docstring(node, source)
    tags = _get_decorators(node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    qn = f"{module_qn}.{name}" if class_name is None else f"{module_qn}.{class_name}.{name}"

    # Detect Enum subclasses from superclass list
    is_enum = False
    superclasses = node.child_by_field_name("superclasses")
    if superclasses is not None:
        for base in superclasses.children:
            if base.type == "identifier" and node_text(base) in _ENUM_BASES:
                is_enum = True
                break

    kind = TypeDefKind.ENUM if is_enum else TypeDefKind.CLASS
    if is_enum:
        enum_classes.add(name)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=kind,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            visibility=_visibility_from_name(name),
            tags=tags,
        )
    )
    # DEFINES relationship from module -> class
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )
    # Base classes -> INHERITS
    if superclasses is not None:
        for base in superclasses.children:
            if base.type == "identifier":
                base_name = node_text(base)
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
        _walk_python_node(
            body, path, source, project_name, module_qn, package_qn, entities, relationships, seen, enum_classes
        )


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
            source=node_text(node),
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

    # Extract USES_TYPE from parameter/return type annotations
    _extract_type_refs(node, f"{project_name}:{qn}", relationships)

    # Walk function body for call sites
    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, source, f"{project_name}:{qn}", relationships)


def _resolve_relative_import(package_qn: str, relative_text: str) -> str | None:
    """Resolve a relative import ('.', '.mod', '..pkg.sub') against the
    importing module's package. Returns an absolute dotted module path, or
    ``None`` when the dots escape the top-level package.
    """
    dots = len(relative_text) - len(relative_text.lstrip("."))
    suffix = relative_text[dots:]
    parts = package_qn.split(".") if package_qn else []
    if dots > len(parts):
        return None  # relative import beyond top-level package
    base_parts = parts[: len(parts) - (dots - 1)] if dots > 1 else parts
    base = ".".join(base_parts)
    return f"{base}.{suffix}" if suffix else base


def _process_import(  # noqa: PLR0912
    node: Node,
    project_name: str,
    module_qn: str,
    package_qn: str,
    relationships: list[ParsedRelationship],
    *,
    type_only: bool = False,
) -> None:
    """Process import_statement or import_from_statement."""
    props: dict[str, Any] = {"type_only": True} if type_only else {}
    if node.type == "import_statement":
        for child in node.children:
            if child.type == "dotted_name":
                import_name = node_text(child)
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=f"{project_name}:{module_qn}",
                        rel_type=RelType.IMPORTS,
                        to_name=import_name,
                        properties=props,
                    )
                )
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                if name_node is not None:
                    import_name = node_text(name_node)
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=f"{project_name}:{module_qn}",
                            rel_type=RelType.IMPORTS,
                            to_name=import_name,
                            properties=props,
                        )
                    )
    elif node.type == "import_from_statement":
        module_node = node.child_by_field_name("module_name")
        if module_node is not None and module_node.type == "relative_import":
            resolved = _resolve_relative_import(package_qn, node_text(module_node))
            if resolved is None:
                return  # beyond top-level package — nothing to link
            module_name = resolved
        else:
            module_name = node_text(module_node) if module_node else ""
        # Collect imported names
        for child in node.children:
            if child.type == "dotted_name" and child != module_node:
                imported = node_text(child)
                full_name = f"{module_name}.{imported}" if module_name else imported
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=f"{project_name}:{module_qn}",
                        rel_type=RelType.IMPORTS,
                        to_name=full_name,
                        properties=props,
                    )
                )
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                if name_node is not None:
                    imported = node_text(name_node)
                    full_name = f"{module_name}.{imported}" if module_name else imported
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=f"{project_name}:{module_qn}",
                            rel_type=RelType.IMPORTS,
                            to_name=full_name,
                            properties=props,
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
    enum_classes: set[str],
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
        name = node_text(left)
        line_start = child.start_point[0] + 1

        key = (line_start, name)
        if key in seen:
            continue
        seen.add(key)

        class_name = _is_inside_class(node)
        if class_name is not None:
            qn = f"{module_qn}.{class_name}.{name}"
            # enum_classes stores bare (undotted) class names — compare against
            # the nearest enclosing class, not the full dotted chain.
            nearest_class_name = class_name.rsplit(".", 1)[-1]
            kind = ValueKind.ENUM_MEMBER if nearest_class_name in enum_classes else ValueKind.FIELD
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
                source=node_text(child),
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
                    call_name = node_text(func)
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
                        call_name = node_text(attr)
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


# ---------------------------------------------------------------------------
# Runtime config surface (READS_ENV / REFERENCES_FILE)
#
# SECURITY INVARIANT — capture the env var NAME, never its value or default.
#
# ``os.getenv("API_KEY", "sk-live-abc123")`` carries a live secret in its SECOND
# argument. Everything below reads the FIRST positional argument and stops:
# ``_first_positional_argument`` returns on the first non-``(`` child of the
# argument list, so no later argument node is ever even looked at, and the
# emitted ``ParsedRelationship`` has no ``properties`` — there is no channel a
# default could ride out on. graph/client.py's ``_plan_config_refs`` enforces the
# same rule again on the write side.
#
# SECOND INVARIANT — a referenced file is a PATH, never contents.
#
# Recording that code reads ``.env`` or ``certs/server.pem`` is the whole point
# of REFERENCES_FILE, so those paths are emitted like any other. Nothing here
# opens, stats or resolves them: the only inputs are tree-sitter nodes and the
# already-in-memory source bytes, and the only string operations are the pure
# predicates in ``looks_like_resource_path``.
# ---------------------------------------------------------------------------

# Byte probes for the cheap pre-filter — a file mentioning none of these cannot
# produce a config reference, so it skips the extra tree walk entirely. The
# openers carry their "(" because bare "open"/"Path" match nearly every module
# (a `Path` annotation, the word "open" in a prose docstring); with the paren
# only 28% of modules reach the walk. A space before the paren is not valid
# formatting anywhere this runs.
_CONFIG_REF_PROBES: tuple[bytes, ...] = (b"getenv", b"environ", b"open(", b"Path(")

# String prefixes (f/b/r/u) are rejected wholesale: an f-string is not a literal,
# and bytes/raw forms are rare enough here that decoding them is not worth the
# escaping subtleties.
_PLAIN_STRING_QUOTES: frozenset[str] = frozenset({'"', "'", '"""', "'''"})

# Env var names are matched against the near-universal shell-identifier shape.
# Anything else is far more likely to be a misparse than a real variable.
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")

# Callables whose first string argument names a file. Kept tiny on purpose:
# every addition is a new way to mint a ResourceFile node for something that is
# not a path. ``Path(...)`` also covers the ``Path("x").read_text()`` form,
# because the literal sits in the inner ``Path`` call either way.
_FILE_OPENER_NAMES: frozenset[str] = frozenset({"open", "Path"})


def _plain_string_value(node: Node | None) -> str | None:
    """Return the text of a plain string literal, or ``None`` if it is not one.

    Rejects f-strings (an ``interpolation`` child), prefixed strings, strings
    containing escape sequences (``escape_sequence`` hangs off ``string_content``
    and would otherwise be emitted un-unescaped), and implicit concatenation
    (which parses as ``concatenated_string``, not ``string``).
    """
    if node is None or node.type != "string":
        return None
    parts: list[str] = []
    for child in node.children:
        if child.type == "string_start":
            if node_text(child) not in _PLAIN_STRING_QUOTES:
                return None
        elif child.type == "string_content":
            if child.children:  # escape_sequence and friends
                return None
            parts.append(node_text(child))
        elif child.type != "string_end":
            return None
    return "".join(parts)


def _first_positional_argument(call: Node) -> Node | None:
    """The first positional argument node of a call, or ``None``.

    Returns as soon as the first argument is seen, so arguments after it — the
    secret in ``os.getenv("API_KEY", "sk-live-abc123")`` — are never visited.
    """
    args = call.child_by_field_name("arguments")
    if args is None or args.type != "argument_list":
        return None
    for child in args.children:
        if child.type == "(":
            continue
        if child.type in ("keyword_argument", "list_splat", "dictionary_splat", ")"):
            return None
        return child
    return None


def _first_string_argument(call: Node) -> str | None:
    """The first positional argument of a call, when it is a plain string literal."""
    return _plain_string_value(_first_positional_argument(call))


def _identifier_named(node: Node | None, name: str) -> bool:
    return node is not None and node.type == "identifier" and node_text(node) == name


def _is_os_environ(node: Node | None) -> bool:
    """True for the ``os.environ`` attribute node."""
    if node is None or node.type != "attribute":
        return False
    return _identifier_named(node.child_by_field_name("attribute"), "environ") and _identifier_named(
        node.child_by_field_name("object"), "os"
    )


def _is_environ_ref(node: Node | None, direct_imports: frozenset[str]) -> bool:
    """True for ``os.environ`` or a bare ``environ`` that came from ``from os import environ``.

    The ``direct_imports`` membership is tested BEFORE ``_identifier_named``: the
    set is empty for almost every file, and that ordering skips a text decode on
    the object of every ``something.get(...)`` in the codebase.
    """
    return _is_os_environ(node) or ("environ" in direct_imports and _identifier_named(node, "environ"))


def _env_name_from_call(call: Node, callee: str, func: Node, direct_imports: frozenset[str]) -> str | None:
    """``os.getenv(...)`` / ``getenv(...)`` / ``os.environ.get(...)`` / ``environ.get(...)``.

    *callee* is the already-decoded terminal name of *func* — see
    ``_config_ref_from_call`` for why it is passed in rather than re-derived.
    """
    if func.type == "identifier":
        if not (callee == "getenv" and "getenv" in direct_imports):
            return None
    else:
        obj = func.child_by_field_name("object")
        if callee == "getenv":
            if not _identifier_named(obj, "os"):
                return None
        elif not _is_environ_ref(obj, direct_imports):
            return None
    name = _first_string_argument(call)
    return name if name is not None and _ENV_NAME_RE.fullmatch(name) else None


def _env_name_from_subscript(node: Node, direct_imports: frozenset[str]) -> str | None:
    """``os.environ["NAME"]`` / ``environ["NAME"]``."""
    if not _is_environ_ref(node.child_by_field_name("value"), direct_imports):
        return None
    name = _plain_string_value(node.child_by_field_name("subscript"))
    return name if name is not None and _ENV_NAME_RE.fullmatch(name) else None


def _file_path_from_call(call: Node, callee: str, func: Node) -> str | None:
    """``open("data/x.json")`` / ``Path("config/y.yaml")`` / ``pathlib.Path(...)``."""
    # Only the attribute form of ``Path`` (``pathlib.Path(...)``) is honored.
    # ``x.open(...)`` is deliberately excluded — it matches archive members and
    # mock objects as readily as real files.
    if func.type != "identifier" and callee != "Path":
        return None
    literal = _first_string_argument(call)
    if literal is None or not looks_like_resource_path(literal):
        return None
    return literal


# Terminal callee names that can possibly produce a config reference. One set
# lookup against one decoded name rejects the overwhelming majority of call
# nodes before any further node access — this walk sees every call in the file,
# so that first check is the whole cost of the pass on most files.
_CONFIG_CALL_NAMES: frozenset[str] = frozenset({"getenv", "get"}) | _FILE_OPENER_NAMES


def _config_ref_from_call(call: Node, direct_imports: frozenset[str]) -> tuple[RelType, str] | None:
    """Classify one call node as an env read, a file reference, or neither."""
    func = call.child_by_field_name("function")
    if func is None:
        return None
    if func.type == "identifier":
        callee = node_text(func)
    elif func.type == "attribute":
        attr = func.child_by_field_name("attribute")
        if attr is None:
            return None
        callee = node_text(attr)
    else:
        return None
    if callee not in _CONFIG_CALL_NAMES:
        return None

    if callee in _FILE_OPENER_NAMES:
        path = _file_path_from_call(call, callee, func)
        return (RelType.REFERENCES_FILE, path) if path is not None else None
    env_name = _env_name_from_call(call, callee, func, direct_imports)
    return (RelType.READS_ENV, env_name) if env_name is not None else None


def _direct_os_imports(relationships: list[ParsedRelationship]) -> frozenset[str]:
    """Names pulled straight out of ``os`` (``from os import getenv, environ``).

    Gating the bare forms on a real import is what stops a project's own local
    ``getenv()`` helper or an unrelated ``environ`` dict from minting EnvVar nodes.
    """
    return frozenset(
        name
        for name in ("getenv", "environ")
        if any(r.rel_type == RelType.IMPORTS and r.to_name == f"os.{name}" for r in relationships)
    )


def _innermost_owner(spans: list[tuple[int, int, int]], line: int) -> int:
    """Index of the smallest entity span covering *line*; 0 (the module) if none.

    Attributing to the innermost entity means a module-level
    ``DATABASE_URL = os.getenv("DATABASE_URL")`` hangs off the Value it defines
    rather than off the whole module, which is the more useful answer to "what
    reads DATABASE_URL".
    """
    covering = [(end - start, -start, index) for start, end, index in spans if start <= line <= end]
    return min(covering)[2] if covering else 0


def _walk_all_nodes(root: Node) -> Iterator[Node]:
    """Yield every node under *root* in pre-order, via a tree-sitter cursor.

    Config references can sit anywhere — module level, a class body, a nested
    function — so this is the one pass that does not stop at definition
    boundaries the way ``_walk_python_node`` and ``_extract_calls`` do. The
    cursor rather than a Python stack over ``node.children``: the latter
    materializes a fresh list at every node and measured ~2x slower over this
    repo's dependency tree.
    """
    cursor = root.walk()
    while True:
        yield cursor.node
        if cursor.goto_first_child():
            continue
        while not cursor.goto_next_sibling():
            if not cursor.goto_parent():
                return


def _extract_config_refs(
    root: Node,
    source: bytes,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Emit READS_ENV / REFERENCES_FILE from the enclosing entity of each reference."""
    if not entities or not any(probe in source for probe in _CONFIG_REF_PROBES):
        return

    direct_imports = _direct_os_imports(relationships)
    spans = [(e.line_start, e.line_end, i) for i, e in enumerate(entities)]
    seen: set[tuple[str, RelType, str]] = set()
    found: list[tuple[int, RelType, str]] = []  # (line, rel_type, target name)

    for node in _walk_all_nodes(root):
        if node.type == "call":
            ref = _config_ref_from_call(node, direct_imports)
        elif node.type == "subscript":
            env_name = _env_name_from_subscript(node, direct_imports)
            ref = (RelType.READS_ENV, env_name) if env_name is not None else None
        else:
            continue
        if ref is not None:
            found.append((node.start_point[0] + 1, *ref))

    for line, rel_type, target in found:
        from_qn = entities[_innermost_owner(spans, line)].qualified_name
        key = (from_qn, rel_type, target)
        if key in seen:
            continue
        seen.add(key)
        # No `properties`: the reference is a name, and there is nothing else
        # about it that may be persisted (see the security invariant above).
        relationships.append(ParsedRelationship(from_qualified_name=from_qn, rel_type=rel_type, to_name=target))


# ---------------------------------------------------------------------------
# USES_TYPE extraction
# ---------------------------------------------------------------------------

_PYTHON_BUILTIN_TYPES: frozenset[str] = frozenset(
    {
        "int",
        "str",
        "float",
        "bool",
        "bytes",
        "None",
        "list",
        "dict",
        "set",
        "tuple",
        "type",
        "object",
        "complex",
        "frozenset",
        "bytearray",
        "memoryview",
        "Any",
    }
)

# Container types whose subscript arguments should be inspected for non-builtin refs
_PYTHON_CONTAINER_TYPES: frozenset[str] = frozenset(
    {
        "list",
        "dict",
        "set",
        "tuple",
        "frozenset",
        "List",
        "Dict",
        "Set",
        "Tuple",
        "FrozenSet",  # typing module aliases
        "Optional",
        "Union",
        "Sequence",
        "Mapping",
        "Iterable",
        "Iterator",
        "Callable",
        "ClassVar",
        "Final",
        "Literal",
        "Annotated",
        "Type",
    }
)


def _collect_type_names_from_annotation(node: Node) -> list[str]:
    """Extract non-builtin type names from a type annotation AST node.

    Handles simple identifiers, attribute access (a.B → B), and subscript
    types like Optional[Foo], list[Bar], Union[A, B].
    """
    names: list[str] = []
    _walk_type_node(node, names)
    return names


def _walk_type_node(node: Node, names: list[str]) -> None:  # noqa: PLR0912
    """Recursively walk a type annotation node to collect type names."""
    if node.type == "identifier":
        name = node_text(node)
        if name not in _PYTHON_BUILTIN_TYPES and name not in _PYTHON_CONTAINER_TYPES:
            names.append(name)
    elif node.type == "attribute":
        # e.g., module.ClassName — take the last attribute
        attr = node.child_by_field_name("attribute")
        if attr is not None:
            name = node_text(attr)
            if name not in _PYTHON_BUILTIN_TYPES and name not in _PYTHON_CONTAINER_TYPES:
                names.append(name)
    elif node.type == "subscript":
        _walk_subscript_type(node, names)
    elif node.type in ("binary_operator", "union_type"):
        # X | Y syntax (Python 3.10+) or Union members
        for child in node.children:
            if child.type not in ("|", ","):
                _walk_type_node(child, names)
    elif node.type == "tuple":
        # Multiple subscript args: dict[str, int], Union[A, B]
        for child in node.children:
            if child.type != ",":
                _walk_type_node(child, names)
    elif node.type == "string":
        # Forward reference: "ClassName" — extract the string content
        text = node_text(node).strip("'\"")
        if text and text.isidentifier() and text not in _PYTHON_BUILTIN_TYPES:
            names.append(text)
    else:
        for child in node.children:
            _walk_type_node(child, names)


def _walk_subscript_type(node: Node, names: list[str]) -> None:
    """Handle subscript type nodes like Optional[Foo] or list[int]."""
    value = node.child_by_field_name("value")
    if value is not None:
        # Resolve the terminal name for both identifiers and attributes (e.g. typing.Optional)
        if value.type == "identifier":
            value_name = node_text(value)
        elif value.type == "attribute":
            attr = value.child_by_field_name("attribute")
            value_name = node_text(attr) if attr is not None else ""
        else:
            value_name = ""
        if value_name in _PYTHON_BUILTIN_TYPES or value_name in _PYTHON_CONTAINER_TYPES:
            # Builtin/container — descend into ALL type arguments.
            # tree-sitter only tags the first arg as the "subscript" field;
            # additional args (e.g. Dict[str, Result]) are unnamed siblings.
            _syntax = frozenset({"[", "]", ","})
            for child in node.children:
                if child.id != value.id and child.type not in _syntax:
                    _walk_type_node(child, names)
        else:
            # Non-builtin subscript (e.g., MyGeneric[T]) — emit the outer type
            _walk_type_node(value, names)


def _extract_type_refs(
    node: Node,
    from_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Extract USES_TYPE relationships from function parameter and return type annotations."""
    seen_types: set[str] = set()

    # Parameter type annotations
    params = node.child_by_field_name("parameters")
    if params is not None:
        for param in params.children:
            type_node = param.child_by_field_name("type")
            if type_node is not None:
                for name in _collect_type_names_from_annotation(type_node):
                    if name not in seen_types:
                        seen_types.add(name)
                        relationships.append(
                            ParsedRelationship(
                                from_qualified_name=from_qn,
                                rel_type=RelType.USES_TYPE,
                                to_name=name,
                            )
                        )

    # Return type annotation
    return_type = node.child_by_field_name("return_type")
    if return_type is not None:
        for name in _collect_type_names_from_annotation(return_type):
            if name not in seen_types:
                seen_types.add(name)
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=from_qn,
                        rel_type=RelType.USES_TYPE,
                        to_name=name,
                    )
                )


# ---------------------------------------------------------------------------
# Pattern detector helpers
# ---------------------------------------------------------------------------

_STRING_RE = re.compile(r"""(['"])((?:(?!\1).)*)\1""")
_DEPENDS_RE = re.compile(r"Depends\(\s*([A-Za-z_]\w*)\s*\)")


def _parse_decorator_tag(tag: str) -> tuple[str, str]:
    """Split a decorator tag into (name, args_text).

    >>> _parse_decorator_tag("decorator:app.get('/users')")
    ("app.get", "'/users'")
    >>> _parse_decorator_tag("decorator:staticmethod")
    ("staticmethod", "")
    >>> _parse_decorator_tag("not_a_decorator")
    ("", "")
    """
    if not tag.startswith("decorator:"):
        return ("", "")
    body = tag[len("decorator:") :]
    paren = body.find("(")
    if paren < 0:
        return (body, "")
    name = body[:paren]
    args = body[paren + 1 :].rstrip(")")
    return (name, args)


def _extract_first_string_arg(text: str) -> str | None:
    """Extract the first string literal value from argument text.

    >>> _extract_first_string_arg("'/users/{id}', response_model=User")
    '/users/{id}'
    """
    match = _STRING_RE.search(text)
    return match.group(2) if match else None


def _extract_depends_names(text: str) -> list[str]:
    """Find all ``Depends(name)`` references in text (e.g. a signature).

    >>> _extract_depends_names("def f(db=Depends(get_db), cache=Depends(get_cache))")
    ['get_db', 'get_cache']
    """
    return _DEPENDS_RE.findall(text)


# ---------------------------------------------------------------------------
# Concrete detector implementations
# ---------------------------------------------------------------------------

# HTTP method suffixes recognized on route decorators
_ROUTE_SUFFIXES: frozenset[str] = frozenset(
    {".get", ".post", ".put", ".delete", ".patch", ".head", ".options", ".route", ".api_route"}
)

# Map decorator suffix to HTTP method
_SUFFIX_TO_METHOD: dict[str, str] = {
    ".get": "GET",
    ".post": "POST",
    ".put": "PUT",
    ".delete": "DELETE",
    ".patch": "PATCH",
    ".head": "HEAD",
    ".options": "OPTIONS",
    ".route": "ANY",
    ".api_route": "ANY",
}

# Known event-handler decorator names (suffix or full name)
_EVENT_PATTERNS: dict[str, str] = {
    "app.task": "celery",
    "shared_task": "celery",
    "celery.task": "celery",
    "receiver": "django",
    "dramatiq.actor": "dramatiq",
    "event_handler": "generic",
    "on_event": "generic",
}


class DecoratorRoutingDetector:
    """Detect HTTP route handlers from framework decorators."""

    @property
    def name(self) -> str:
        return "decorator_routing"

    async def detect(
        self,
        parsed: ParsedFile,
        project_name: str,  # noqa: ARG002
        graph: GraphClient,  # noqa: ARG002
    ) -> DetectorResult:
        enrichments: list[PropertyEnrichment] = []
        for entity in parsed.entities:
            for tag in entity.tags:
                dec_name, args_text = _parse_decorator_tag(tag)
                if not dec_name:
                    continue
                # Check if decorator ends with a route suffix
                for suffix, method in _SUFFIX_TO_METHOD.items():
                    if dec_name.endswith(suffix):
                        route_path = _extract_first_string_arg(args_text) if args_text else None
                        if route_path is None:
                            break
                        enrichments.append(
                            PropertyEnrichment(
                                qualified_name=entity.qualified_name,
                                properties={"route_path": route_path, "http_method": method},
                            )
                        )
                        break
        return DetectorResult(enrichments=enrichments)


class EventHandlerDetector:
    """Detect event/task handlers from framework decorators."""

    @property
    def name(self) -> str:
        return "event_handlers"

    async def detect(
        self,
        parsed: ParsedFile,
        project_name: str,  # noqa: ARG002
        graph: GraphClient,  # noqa: ARG002
    ) -> DetectorResult:
        enrichments: list[PropertyEnrichment] = []
        for entity in parsed.entities:
            for tag in entity.tags:
                dec_name, args_text = _parse_decorator_tag(tag)
                if not dec_name:
                    continue
                framework = _EVENT_PATTERNS.get(dec_name)
                if framework is None:
                    continue
                event_name = _extract_first_string_arg(args_text) if args_text else None
                if event_name is None:
                    # Celery tasks use the function name as the task name
                    event_name = entity.name
                enrichments.append(
                    PropertyEnrichment(
                        qualified_name=entity.qualified_name,
                        properties={"event_name": event_name, "event_framework": framework},
                    )
                )
        return DetectorResult(enrichments=enrichments)


class TestMappingDetector:
    """Map test classes/functions to their subjects via naming conventions."""

    @property
    def name(self) -> str:
        return "test_mapping"

    async def detect(self, parsed: ParsedFile, project_name: str, graph: GraphClient) -> DetectorResult:
        relationships: list[ParsedRelationship] = []
        for entity in parsed.entities:
            target_name = self._extract_target_name(entity)
            if target_name is None:
                continue
            # Look up target in graph
            target_uid = await self._find_target(graph, project_name, entity, target_name)
            if target_uid:
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=entity.qualified_name,
                        rel_type=RelType.TESTS,
                        to_name=target_uid,
                    )
                )
        return DetectorResult(relationships=relationships)

    @staticmethod
    def _extract_target_name(entity: ParsedEntity) -> str | None:
        """Derive the subject name from a test entity name."""
        if entity.label == NodeLabel.TYPE_DEF and entity.name.startswith("Test"):
            return entity.name[4:] or None
        if entity.label == NodeLabel.CALLABLE and entity.name.startswith("test_"):
            return entity.name[5:] or None
        return None

    @staticmethod
    async def _find_target(graph: GraphClient, project_name: str, source: ParsedEntity, target_name: str) -> str | None:
        if graph is None:
            return None
        # TypeDef test -> look for TypeDef; Callable test -> look for Callable
        label = "TypeDef" if source.label == NodeLabel.TYPE_DEF else "Callable"
        return await graph.find_entity_uid(project_name, label, target_name)


class ClassOverridesDetector:
    """Detect method overrides by checking parent classes for same-name methods."""

    @property
    def name(self) -> str:
        return "class_overrides"

    async def detect(
        self,
        parsed: ParsedFile,
        project_name: str,
        graph: GraphClient,
    ) -> DetectorResult:
        if graph is None:
            return DetectorResult()

        # Build class_qn -> [base_names] map from INHERITS relationships
        class_bases: dict[str, list[str]] = {}
        for rel in parsed.relationships:
            if rel.rel_type == RelType.INHERITS:
                class_bases.setdefault(rel.from_qualified_name, []).append(rel.to_name)

        if not class_bases:
            return DetectorResult()

        relationships: list[ParsedRelationship] = []
        for entity in parsed.entities:
            if entity.kind not in (
                CallableKind.METHOD,
                CallableKind.CONSTRUCTOR,
                CallableKind.DESTRUCTOR,
                CallableKind.STATIC_METHOD,
                CallableKind.CLASS_METHOD,
            ):
                continue
            # Derive class qualified_name: strip ".method_name" from entity qn
            dot_pos = entity.qualified_name.rfind(".")
            if dot_pos < 0:
                continue
            class_qn = entity.qualified_name[:dot_pos]
            bases = class_bases.get(class_qn, [])
            if not bases:
                continue
            # Query graph for parent method (include tags for abstractmethod detection)
            found = await graph.find_overridden_method(project_name, bases, entity.name)
            if found is not None:
                parent_uid, parent_tags = found
                is_abstract = any(t == "decorator:abstractmethod" for t in parent_tags)
                rel_type = RelType.IMPLEMENTS if is_abstract else RelType.OVERRIDES
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=entity.qualified_name,
                        rel_type=rel_type,
                        to_name=parent_uid,
                    )
                )
        return DetectorResult(relationships=relationships)


class DIInjectionDetector:
    """Detect FastAPI Depends() injection patterns."""

    @property
    def name(self) -> str:
        return "di_injection"

    async def detect(self, parsed: ParsedFile, project_name: str, graph: GraphClient) -> DetectorResult:
        enrichments: list[PropertyEnrichment] = []
        relationships: list[ParsedRelationship] = []
        for entity in parsed.entities:
            if not entity.signature:
                continue
            dep_names = _extract_depends_names(entity.signature)
            if not dep_names:
                continue
            enrichments.append(
                PropertyEnrichment(
                    qualified_name=entity.qualified_name,
                    properties={"di_framework": "fastapi", "dependencies": dep_names},
                )
            )
            # Try to resolve provider UIDs in graph
            if graph is None:
                continue
            for dep_name in dep_names:
                dep_uid = await graph.find_entity_uid(project_name, "Callable", dep_name)
                if dep_uid:
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=dep_uid,
                            rel_type=RelType.INJECTED_INTO,
                            to_name=entity.qualified_name,
                        )
                    )
        return DetectorResult(relationships=relationships, enrichments=enrichments)


class CLICommandDetector:
    """Detect CLI command handlers from click/typer decorators."""

    @property
    def name(self) -> str:
        return "cli_commands"

    async def detect(
        self,
        parsed: ParsedFile,
        project_name: str,  # noqa: ARG002
        graph: GraphClient,  # noqa: ARG002
    ) -> DetectorResult:
        # Check relationships for typer imports (e.g. `import typer`, `from typer import ...`)
        has_typer_import = any(
            rel.rel_type == RelType.IMPORTS and rel.to_name.startswith("typer") for rel in parsed.relationships
        )

        enrichments: list[PropertyEnrichment] = []
        for entity in parsed.entities:
            for tag in entity.tags:
                dec_name, args_text = _parse_decorator_tag(tag)
                if not dec_name:
                    continue
                if not dec_name.endswith(".command"):
                    continue
                command_name = _extract_first_string_arg(args_text) if args_text else None
                if command_name is None:
                    command_name = entity.name
                framework = "typer" if has_typer_import or "typer" in dec_name.lower() else "click"
                enrichments.append(
                    PropertyEnrichment(
                        qualified_name=entity.qualified_name,
                        properties={
                            "command_name": command_name,
                            "cli_framework": framework,
                        },
                    )
                )
                break  # One command decorator per entity is enough
        return DetectorResult(enrichments=enrichments)


_DATACLASS_TAGS: dict[str, tuple[str, list[str]]] = {
    "dataclass": ("dataclasses", ["__init__", "__repr__", "__eq__"]),
    "dataclasses.dataclass": ("dataclasses", ["__init__", "__repr__", "__eq__"]),
    "attr.s": ("attrs", ["__init__", "__repr__", "__eq__", "__hash__"]),
    "attr.define": ("attrs", ["__init__", "__repr__", "__eq__"]),
    "attr.attrs": ("attrs", ["__init__", "__repr__", "__eq__", "__hash__"]),
}

_PYDANTIC_BASES: frozenset[str] = frozenset({"BaseModel"})


class DataclassSynthesisDetector:
    """Detect synthesized methods from @dataclass / attrs / Pydantic classes."""

    @property
    def name(self) -> str:
        return "dataclass_synthesis"

    async def detect(
        self,
        parsed: ParsedFile,
        project_name: str,  # noqa: ARG002
        graph: GraphClient,  # noqa: ARG002
    ) -> DetectorResult:
        # Build class_qn -> set[base_names] for INHERITS relationships
        class_bases: dict[str, set[str]] = {}
        for rel in parsed.relationships:
            if rel.rel_type == RelType.INHERITS:
                class_bases.setdefault(rel.from_qualified_name, set()).add(rel.to_name)

        enrichments: list[PropertyEnrichment] = []
        for entity in parsed.entities:
            if entity.label != NodeLabel.TYPE_DEF:
                continue
            # Check decorator tags
            for tag in entity.tags:
                dec_name, _ = _parse_decorator_tag(tag)
                if dec_name in _DATACLASS_TAGS:
                    framework, methods = _DATACLASS_TAGS[dec_name]
                    enrichments.append(
                        PropertyEnrichment(
                            qualified_name=entity.qualified_name,
                            properties={"synthesis_framework": framework, "synthesized_methods": methods},
                        )
                    )
                    break
            else:
                # Check Pydantic via inheritance
                bases = class_bases.get(entity.qualified_name, set())
                if bases & _PYDANTIC_BASES:
                    enrichments.append(
                        PropertyEnrichment(
                            qualified_name=entity.qualified_name,
                            properties={
                                "synthesis_framework": "pydantic",
                                "synthesized_methods": ["__init__", "__repr__", "__eq__", "__hash__"],
                            },
                        )
                    )
        return DetectorResult(enrichments=enrichments)


class ModuleExportsDetector:
    """Detect ``__all__`` exports and emit EXPORTS relationships."""

    _ALL_NAMES_RE = re.compile(r"""['"](\w+)['"]""")

    @property
    def name(self) -> str:
        return "module_exports"

    async def detect(
        self,
        parsed: ParsedFile,
        project_name: str,  # noqa: ARG002
        graph: GraphClient,  # noqa: ARG002
    ) -> DetectorResult:
        # Find __all__ Value entity
        all_entity = None
        for entity in parsed.entities:
            if entity.label == NodeLabel.VALUE and entity.name == "__all__":
                all_entity = entity
                break
        if all_entity is None or not all_entity.source:
            return DetectorResult()

        # Extract exported names
        exported_names = self._ALL_NAMES_RE.findall(all_entity.source)
        if not exported_names:
            return DetectorResult()

        # Find the module/package entity for this file
        module_entity = None
        for entity in parsed.entities:
            if entity.label in (NodeLabel.MODULE, NodeLabel.PACKAGE):
                module_entity = entity
                break
        if module_entity is None:
            return DetectorResult()

        # Build name -> qualified_name lookup for entities in this file
        # (excluding the module/package entity itself and the __all__ value entity)
        name_to_qn: dict[str, str] = {}
        for entity in parsed.entities:
            if entity.label in (NodeLabel.MODULE, NodeLabel.PACKAGE) or entity.name == "__all__":
                continue
            name_to_qn.setdefault(entity.name, entity.qualified_name)

        enrichments = [
            PropertyEnrichment(
                qualified_name=module_entity.qualified_name,
                properties={"public_api": exported_names},
            )
        ]

        relationships: list[ParsedRelationship] = []
        for exp_name in exported_names:
            target_qn = name_to_qn.get(exp_name)
            if target_qn:
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=module_entity.qualified_name,
                        rel_type=RelType.EXPORTS,
                        to_name=target_qn,
                    )
                )

        return DetectorResult(enrichments=enrichments, relationships=relationships)


# ---------------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------------

register_detector(DecoratorRoutingDetector())
register_detector(EventHandlerDetector())
register_detector(TestMappingDetector())
register_detector(ClassOverridesDetector())
register_detector(DIInjectionDetector())
register_detector(CLICommandDetector())
register_detector(DataclassSynthesisDetector())
register_detector(ModuleExportsDetector())


# ---------------------------------------------------------------------------
# Language registration (after _parse_python is defined)
# ---------------------------------------------------------------------------

register_language(
    LanguageConfig(
        name="python",
        extensions=frozenset({".py", ".pyi"}),
        language=_PY_LANGUAGE,
        query=_PY_QUERY,
        parse_func=_parse_python,
        comment_node_types=frozenset({"comment"}),
    )
)
