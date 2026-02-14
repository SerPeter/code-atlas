"""Python language support — tree-sitter parser and pattern detectors."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import tree_sitter_python as tspython
from tree_sitter import Language, Query

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    node_text,
    register_language,
)
from code_atlas.parsing.detectors import (
    DetectorResult,
    PropertyEnrichment,
    register_detector,
)
from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind, ValueKind, Visibility

if TYPE_CHECKING:
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


def _module_qualified_name(file_path: str) -> str:
    """Convert file path to a Python module qualified name.

    ``src/code_atlas/parser.py`` -> ``src.code_atlas.parser``
    ``src/code_atlas/__init__.py`` -> ``src.code_atlas``
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
                return node_text(name_node)
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
# Python parse entry point
# ---------------------------------------------------------------------------


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
    name = node_text(name_node)
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
    # DEFINES relationship from module -> class
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )
    # Base classes -> INHERITS
    superclasses = node.child_by_field_name("superclasses")
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
                import_name = node_text(child)
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
                    import_name = node_text(name_node)
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=f"{project_name}:{module_qn}",
                            rel_type=RelType.IMPORTS,
                            to_name=import_name,
                        )
                    )
    elif node.type == "import_from_statement":
        module_node = node.child_by_field_name("module_name")
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
        name = node_text(left)
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
        records = await graph.execute(
            f"MATCH (n:{label} {{project_name: $p, name: $n}}) RETURN n.uid AS uid LIMIT 1",
            {"p": project_name, "n": target_name},
        )
        return records[0]["uid"] if records else None


class ClassOverridesDetector:
    """Detect method overrides by checking parent classes for same-name methods."""

    @property
    def name(self) -> str:
        return "class_overrides"

    async def detect(
        self,
        parsed: ParsedFile,
        project_name: str,  # noqa: ARG002
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
            # Query graph for parent method
            records = await graph.execute(
                "MATCH (base:TypeDef)-[:DEFINES]->(m:Callable)"
                " WHERE base.name IN $bases AND m.name = $method"
                " RETURN m.uid AS uid LIMIT 1",
                {"bases": bases, "method": entity.name},
            )
            if records:
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=entity.qualified_name,
                        rel_type=RelType.OVERRIDES,
                        to_name=records[0]["uid"],
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
                records = await graph.execute(
                    "MATCH (n:Callable {project_name: $p, name: $n}) RETURN n.uid AS uid LIMIT 1",
                    {"p": project_name, "n": dep_name},
                )
                if records:
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=records[0]["uid"],
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


# ---------------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------------

register_detector(DecoratorRoutingDetector())
register_detector(EventHandlerDetector())
register_detector(TestMappingDetector())
register_detector(ClassOverridesDetector())
register_detector(DIInjectionDetector())
register_detector(CLICommandDetector())


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
    )
)
