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

# Bases whose presence makes a class a declaration rather than an implementation.
_ABSTRACT_BASES = frozenset({"Protocol", "ABC", "ABCMeta"})

# Receiver types that are definitively not project classes. Recorded so the resolver can
# decline rather than guess — `set.add`, `dict.get` and `list.append` collide with common
# project method names.
_BUILTIN_CONTAINERS = frozenset({"set", "dict", "list", "tuple", "frozenset", "str", "bytes", "bytearray", "deque"})
_CONTAINER_LITERALS = {"list": "list", "dictionary": "dict", "set": "set", "tuple": "tuple", "string": "str"}


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


def _is_stub_body(node: Node, tags: list[str]) -> bool:
    """Whether a function can never do anything at runtime.

    This is a per-METHOD question and was previously answered per-class, via the class's
    bases. That conflated two different things: a ``Protocol`` really is all stubs, but
    ``ABC`` is the standard base for a class with ONE abstractmethod and a dozen concrete
    ones — TierConsumer in this repo has 1 of 16. Treating its real methods as stubs
    deleted the true callee from candidate sets and left a same-named sibling to be
    promoted to a resolved edge, which mis-resolved even `await super().run()`.

    A body qualifies only when it is literally ``...``, or the method is decorated
    ``@abstractmethod``. Deliberately NOT ``pass`` or a docstring alone: those are real
    no-op implementations that run and can legitimately be the callee — TierConsumer's
    ``_pre_run``/``_post_run`` hooks are exactly that shape.
    """
    if any("abstractmethod" in t for t in tags):
        return True
    body = node.child_by_field_name("body")
    if body is None:
        return False
    statements = [c for c in body.children if c.type not in (":", "comment")]
    return (
        len(statements) == 1
        and statements[0].type == "expression_statement"
        and any(c.type == "ellipsis" for c in statements[0].children)
    )


def _emit_registrations(node: Node, from_qn: str, relationships: list[ParsedRelationship]) -> None:
    """Link a decorated definition to the decorator that registers it.

    `@register("greet")` is how a handler joins a registry, and the graph held only the
    decorator's source text as a tag — a string, not an edge, so "what does register
    register?" had no answer and the handler looked unreachable.

    Every decorator is emitted and resolution does the filtering: it links only to a
    Callable in this project, so `@property`, `@staticmethod` and `@dataclass` drop out on
    their own without a hand-maintained blocklist that would need a new entry per library.
    """
    parent = node.parent
    if parent is None or parent.type != "decorated_definition":
        return
    for child in parent.children:
        if child.type != "decorator":
            continue
        expr = next((c for c in child.children if c.type in ("identifier", "attribute", "call")), None)
        if expr is None:
            continue
        if expr.type == "call":
            expr = expr.child_by_field_name("function")
            if expr is None:
                continue
        if expr.type == "attribute":
            # `@mcp.tool()` — the registry is an object, so the attribute name is the best
            # available handle. It resolves only if the project also defines a callable of
            # that name; otherwise it drops, which is the honest outcome.
            expr = expr.child_by_field_name("attribute")
            if expr is None:
                continue
        if expr.type != "identifier":
            continue
        relationships.append(
            ParsedRelationship(
                from_qualified_name=from_qn,
                rel_type=RelType.REGISTERED_BY,
                to_name=node_text(expr),
            )
        )


def _decorator_surface(node: Node) -> dict[str, str]:
    """The registration surface a decorator declares, without knowing the framework.

    `@app.get("/users/{id}")`, `@app.command("mine-git-history")`, `@celery.task(
    name="send.email")` and `@register("greet")` are one shape: a decorator expression
    plus a string that names the thing being registered. Three separate detectors used to
    extract exactly this and write it under three different property names, each gated on
    a hard-coded framework list that goes stale the moment a new framework appears.

    The parser records the shape and never decides what it MEANS. `app.get` versus
    `app.command` versus `celery.task` is a classification the caller makes at query time,
    where adding a framework is a filter you write rather than a detector someone has to
    ship. That also disposes of the false-positive problem: `@pytest.mark.parametrize(
    "a,b", ...)` is recorded honestly as "decorated by parametrize with 'a,b'" instead of
    needing a blocklist entry to stop it being called a route.

    A decorator that is CALLED is always recorded, string argument or not: `@app.command()`
    with bare parens is a registration too, and its surface key is the function's own name.
    Supplying that default is the caller's job — "an unnamed Typer command takes the
    function name" is framework knowledge, and keeping it out here is the point. Dropping
    the no-argument form outright would have lost 8 of this repo's 11 CLI commands.

    First decorator carrying a string literal wins, falling back to the first called
    decorator — a handler has one registration decorator and any number of plain modifiers.
    """
    parent = node.parent
    if parent is None or parent.type != "decorated_definition":
        return {}
    fallback = ""
    for child in parent.children:
        if child.type != "decorator":
            continue
        call = next((c for c in child.children if c.type == "call"), None)
        if call is None:
            continue
        func = call.child_by_field_name("function")
        if func is None:
            continue
        fallback = fallback or node_text(func)
        args = call.child_by_field_name("arguments")
        if args is None:
            continue
        for arg in args.children:
            literal = arg
            if arg.type == "keyword_argument":
                value = arg.child_by_field_name("value")
                if value is None:
                    continue
                literal = value
            if literal.type != "string":
                continue
            text = _plain_string_value(literal)
            if text:
                return {"decorator_name": node_text(func), "decorator_arg": text}
    return {"decorator_name": fallback} if fallback else {}


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

    # Callables handed to a module-level registration call. This is where the registry
    # pattern actually lives — `register_language(parse_func=_parse_python)` and
    # `register_detector(...)` sit at module scope, below every def they name — and module
    # scope is not a function body, so _extract_calls never sees it. Without this, every
    # language's own entry point has no inbound edge: 15 of the 30 dead-code hits in
    # parsing/languages were exactly that.
    _extract_module_level_references(root, f"{project_name}:{module_qn}", relationships)

    # Runtime config surface. Runs after the main walk because it needs both the
    # finished entity list (to attribute each reference to its enclosing entity)
    # and the IMPORTS relationships (to know whether a bare `getenv`/`environ`
    # actually came from `os`).
    _extract_config_refs(root, source, entities, relationships)

    # Module-level constants a body reads. Runs late for the same reason as the
    # config pass: only the finished entity list says which names are module-level
    # Values, and only then is a bare identifier's match lexically grounded.
    _extract_constant_reads(root, entities, relationships)

    # Most identifier arguments are ordinary values — `path`, `node`, `entities`. Emitting
    # a reference for every one produced 379 from this file alone, so they are filtered
    # here rather than at the call site: only now is the entity list complete enough to
    # say which names are actually callables. Same reason _extract_config_refs runs late.
    _filter_value_references(entities, relationships)

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
            _process_assignment(child, path, project_name, module_qn, node, entities, relationships, seen, enum_classes)
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

    # Detect Enum and abstract bases from the superclass list.
    is_enum = False
    is_abstract = False
    superclasses = node.child_by_field_name("superclasses")
    if superclasses is not None:
        for base in superclasses.children:
            # `identifier` alone misses the dotted forms `typing.Protocol` / `abc.ABC`,
            # which are an `attribute` node — the last component is the base name.
            if base.type == "identifier":
                base_name = node_text(base)
            elif base.type == "attribute":
                attr = base.child_by_field_name("attribute")
                base_name = node_text(attr) if attr is not None else ""
            else:
                continue
            if base_name in _ENUM_BASES:
                is_enum = True
            if base_name in _ABSTRACT_BASES:
                is_abstract = True

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
            # A Protocol/ABC declaration's methods are `...` stubs that can never run, so
            # a call resolved to one is resolved to nothing. The parser already knew this
            # and threw it away: it emits INHERITS -> Protocol, and both write paths drop
            # that edge because `Protocol` is not an in-project TypeDef.
            extra_properties={"is_abstract": True} if is_abstract else {},
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
    enclosing_qn: str | None = None,
) -> None:
    """Process a function_definition node.

    *enclosing_qn* is set when this def is nested inside another function, and makes it
    own its parent's qualified name the way a method owns its class's. Without it, a
    nested def inherited `_is_inside_class`'s answer and would be named as a method of a
    class it is only lexically inside.
    """
    class_name = None if enclosing_qn else _is_inside_class(node)
    docstring = _extract_docstring(node, source)
    tags = _get_decorators(node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    is_method = class_name is not None
    if enclosing_qn:
        kind = CallableKind.FUNCTION
        qn = f"{enclosing_qn}.{name}"
    elif is_method:
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
            # Per-method, not per-class: an ABC's concrete methods are real code and
            # must stay resolvable. See _is_stub_body.
            extra_properties=(({"is_stub": True} if _is_stub_body(node, tags) else {}) | _decorator_surface(node)),
        )
    )

    # DEFINES relationship
    parent_qn = enclosing_qn or (f"{module_qn}.{class_name}" if is_method else module_qn)
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{parent_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    _emit_registrations(node, f"{project_name}:{qn}", relationships)

    # Extract USES_TYPE from parameter/return type annotations
    _extract_type_refs(node, f"{project_name}:{qn}", relationships)

    # Walk function body for call sites
    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, source, f"{project_name}:{qn}", relationships, _local_declared_types(node, body))
        # ...then the defs that body encloses. _extract_calls deliberately stops at a
        # nested definition, so without this its calls belong to nobody: measured on this
        # repo, 0 of 64 nested functions existed and 458 call expressions in server/mcp.py
        # alone were dropped, taking all 23 @mcp.tool handlers with them.
        _process_nested_functions(body, path, source, project_name, module_qn, qn, entities, relationships)
        if is_method and name == "__init__" and class_name is not None:
            _process_self_attributes(
                body,
                path,
                project_name,
                f"{module_qn}.{class_name}",
                _local_declared_types(node, body),
                entities,
                relationships,
            )


def _process_self_attributes(
    body: Node,
    path: str,
    project_name: str,
    class_qn: str,
    local_types: dict[str, str],
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Turn `self.x = <injected>` in a constructor into a typed field of the class.

    Constructor injection is how most of this codebase is wired — `self.graph = graph`,
    `self.bus = bus` — and none of it existed: ASTConsumer.graph was not a node at all, so
    "what does ASTConsumer depend on?" could not be asked even though the parameter it
    comes from is annotated and USES_TYPE already resolves that same annotation for the
    method.

    Only `__init__`, and only a plain `self.name = ...`. The type comes from the parameter
    annotation the value was handed from, or from a one-step `self.x = Foo()` — the same
    two sources ADR-0023 measured at 90.7%. An unannotated assignment stays untyped rather
    than guessed.
    """
    seen_attrs: set[str] = set()

    def walk(n: Node) -> None:
        for child in n.children:
            if child.type == "expression_statement":
                for inner in child.children:
                    if inner.type != "assignment":
                        continue
                    left = inner.child_by_field_name("left")
                    if left is None or left.type != "attribute":
                        continue
                    obj = left.child_by_field_name("object")
                    attr = left.child_by_field_name("attribute")
                    if obj is None or attr is None or node_text(obj) != "self":
                        continue
                    name = node_text(attr)
                    if name in seen_attrs:
                        continue
                    seen_attrs.add(name)
                    qn = f"{class_qn}.{name}"
                    entities.append(
                        ParsedEntity(
                            name=name,
                            qualified_name=f"{project_name}:{qn}",
                            label=NodeLabel.VALUE,
                            kind=ValueKind.FIELD,
                            line_start=inner.start_point[0] + 1,
                            line_end=inner.end_point[0] + 1,
                            file_path=path,
                            source=node_text(inner),
                            visibility=_visibility_from_name(name),
                        )
                    )
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=f"{project_name}:{class_qn}",
                            rel_type=RelType.DEFINES,
                            to_name=f"{project_name}:{qn}",
                        )
                    )
                    right = inner.child_by_field_name("right")
                    type_name = ""
                    if right is not None and right.type == "identifier":
                        type_name = local_types.get(node_text(right), "")
                    elif right is not None and right.type == "call":
                        fn = right.child_by_field_name("function")
                        if fn is not None and fn.type == "identifier" and node_text(fn)[:1].isupper():
                            type_name = node_text(fn)
                    if type_name:
                        relationships.append(
                            ParsedRelationship(
                                from_qualified_name=f"{project_name}:{qn}",
                                rel_type=RelType.USES_TYPE,
                                to_name=type_name,
                                properties={"on": "value"},
                            )
                        )
            if child.type not in ("function_definition", "class_definition", "decorated_definition"):
                walk(child)

    walk(body)


def _process_nested_functions(
    body: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    enclosing_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Index `def`s written inside another function, at any depth.

    A decorated nested def arrives wrapped in a `decorated_definition`, which is how the
    registrar pattern (`@mcp.tool()` inside `_register_x`) appears — unwrap it, or the
    handlers stay invisible. Lambdas are deliberately not entities: their calls already
    attribute to the enclosing function, which is where a reader would look for them.
    """
    for child in body.children:
        # Descend through compound statements first. A `def` inside `with`/`if`/`try`/`for`
        # is not a direct child of the body, and walking only direct children missed
        # EmbedClient._embed_call and _build_kwargs — the product's only litellm call path.
        if child.type not in ("function_definition", "class_definition", "decorated_definition"):
            _process_nested_functions(
                child, path, source, project_name, module_qn, enclosing_qn, entities, relationships
            )

        target = child
        if child.type == "decorated_definition":
            inner = child.child_by_field_name("definition")
            if inner is None or inner.type != "function_definition":
                continue
            target = inner
        elif child.type != "function_definition":
            continue

        name_node = target.child_by_field_name("name")
        if name_node is None:
            continue
        _process_function(
            target,
            path,
            source,
            project_name,
            module_qn,
            node_text(name_node),
            entities,
            relationships,
            enclosing_qn=enclosing_qn,
        )


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
    relationships: list[ParsedRelationship],
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

        if class_name is not None:
            # A class never claimed its own fields. That is the other half of why 433 of
            # 450 field nodes had zero edges of ANY kind — not merely no type edge, no
            # owner either — so a field was unreachable from the class that declares it
            # and the class looked like it had nothing but methods.
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=f"{project_name}:{module_qn}.{class_name}",
                    rel_type=RelType.DEFINES,
                    to_name=f"{project_name}:{qn}",
                )
            )

        # A field's declared type is what a class is BUILT FROM, and it was being
        # discarded: 433 of 450 field nodes in this repo had no edge of any kind, so
        # AtlasSettings' only outgoing edge was DEFINES to one method despite fields typed
        # MemgraphSettings/RedisSettings/EmbeddingSettings. Marked `on: field` because it
        # resolves in the field's own scope rather than through the Callable lookup that
        # signature-derived USES_TYPE uses — a Value is not in that lookup at all.
        annotation = child.child_by_field_name("type")
        if annotation is not None:
            relationships.extend(
                ParsedRelationship(
                    from_qualified_name=f"{project_name}:{qn}",
                    rel_type=RelType.USES_TYPE,
                    to_name=type_name,
                    properties={"on": "value"},
                )
                for type_name in _collect_type_names_from_annotation(annotation)
            )


def _receiver_props(obj: Node | None, local_types: dict[str, str] | None) -> dict[str, Any]:
    """Receiver expression, plus its declared class when one is known.

    The expression alone says "do not trust a project-wide name match" (ADR-0022). The
    type says which implementation is actually called — measured, it sends 772 of 915
    fanned-out sites to exactly one concrete class and only 24 to the Protocol.
    """
    if obj is None:
        return {}
    text = node_text(obj)
    props: dict[str, Any] = {"receiver": text}
    declared = (local_types or {}).get(text)
    if declared:
        props["receiver_type"] = declared
    return props


# Annotations that name no concrete class. Left in, each one silently DELETED every call on
# that receiver: the external-receiver guard drops a call whose declared type matches no
# project class, and `Any` matches none anywhere. An `Any` annotation was strictly worse
# than no annotation, which is the opposite of what an annotation should ever do.
_OPAQUE_TYPE_NAMES: frozenset[str] = frozenset({"Any", "object", "Self", "type", "None", "Optional"})


def _plain_type_name(annotation: str) -> str:
    """A bare class name from an annotation, or "" when it is not one.

    Deliberately conservative. `GraphClient` resolves; `GraphClient | None`,
    `list[Store]` and `"GraphClient"` do not. A wrong type would send a call to the
    wrong implementation with full confidence, which is the failure this whole line of
    work exists to remove — declining to guess costs only a fallback to today's
    behaviour.
    """
    text = annotation.strip()
    if text in _OPAQUE_TYPE_NAMES:
        return ""
    if text.isidentifier():
        return text
    # `set[str]` / `dict[str, int]` -> the base. The subscript does not change which
    # class the receiver is, and this form is how container annotations are actually
    # written, so rejecting it left the builtin-collision edges in place.
    base, bracket, rest = text.partition("[")
    if bracket and rest.endswith("]") and base.strip().isidentifier():
        return base.strip()
    # Unions are deliberately NOT unwrapped. `Store | None` really can reach Store, so
    # returning either half would be wrong: one loses a real edge, the other invents one.
    return ""


def _local_declared_types(func_node: Node, body: Node) -> dict[str, str]:
    """Best-effort ``{local name: class name}`` for the receivers in one function.

    Two sources, measured to cover 90.7% of the call sites that a name-only resolver
    fans out across every implementation: parameter annotations (59.8%) and one-step
    local construction ``x = Foo(...)`` (a further 30.9%). Anything else is left unknown
    rather than inferred, so resolution degrades to the status quo instead of to a guess.
    """
    types: dict[str, str] = {}

    params = func_node.child_by_field_name("parameters")
    if params is not None:
        for child in params.children:
            if child.type not in ("typed_parameter", "typed_default_parameter"):
                continue
            name_node = child.child_by_field_name("name") or (child.children[0] if child.children else None)
            ann = child.child_by_field_name("type")
            if name_node is None or ann is None:
                continue
            base = _plain_type_name(node_text(ann))
            if base:
                types[node_text(name_node)] = base

    def walk(n: Node) -> None:
        for child in n.children:
            if child.type == "expression_statement":
                for inner in child.children:
                    if inner.type != "assignment":
                        continue
                    left = inner.child_by_field_name("left")
                    right = inner.child_by_field_name("right")
                    if left is None or right is None or left.type != "identifier":
                        continue
                    if right.type == "call":
                        fn = right.child_by_field_name("function")
                        # Only `Foo(...)`, never `mod.Foo(...)`: the dotted form's class
                        # name is not necessarily the attribute's own name.
                        if fn is not None and fn.type == "identifier":
                            name = node_text(fn)
                            # Builtins are recorded deliberately, not skipped. Knowing a
                            # receiver is a `set` is what lets the resolver refuse to
                            # invent an edge: `seen.add(x)` was resolving to project
                            # methods named `add`, 156 times.
                            if name[:1].isupper() or name in _BUILTIN_CONTAINERS:
                                types.setdefault(node_text(left), name)
                    elif right.type in _CONTAINER_LITERALS:
                        types.setdefault(node_text(left), _CONTAINER_LITERALS[right.type])
            # Do not descend into nested defs — their locals are a different scope.
            if child.type not in ("function_definition", "class_definition", "decorated_definition"):
                walk(child)

    walk(body)
    return types


def _extract_constant_reads(  # noqa: PLR0915  # one lexical pass: scope walk, shadow set, load test
    root: Node, entities: list[ParsedEntity], relationships: list[ParsedRelationship]
) -> None:
    """REFERENCES from a body to the module-level constants it reads.

    `_match_brace` reading `_OPEN_BRACE` was invisible: CALLS covers callables and
    the value-reference pass covers callables-as-values, so a Value's only inbound
    edge was its DEFINES — one REFERENCES edge onto a Value in the whole graph, and
    every constant looked unused.

    The same lexical ground rules as ADR-0022: only names that are module-level
    Values of THIS file, only where no local binding shadows them. A bare name not
    bound locally resolves to module scope — that is Python's own rule, not a guess.
    """
    module_qn = entities[0].qualified_name
    values = {e.name for e in entities if e.label == NodeLabel.VALUE and e.qualified_name == f"{module_qn}.{e.name}"}
    if not values:
        return
    callables = [(e.line_start, e.line_end, e.qualified_name) for e in entities if e.label == NodeLabel.CALLABLE]

    def _qn_for(def_node: Node) -> str:
        line = def_node.start_point[0] + 1
        best: tuple[int, int, str] | None = None
        for start, end, qn in callables:
            if start <= line <= end and (best is None or start >= best[0]):
                best = (start, end, qn)
        return best[2] if best else module_qn

    emitted: set[tuple[str, str]] = set()

    def _is_load(node: Node) -> bool:  # noqa: PLR0911  # each store context is one early exit
        parent = node.parent
        if parent is None:
            return True
        if parent.type == "assignment" and parent.child_by_field_name("left") == node:
            return False
        if parent.type == "augmented_assignment" and parent.child_by_field_name("left") == node:
            return False
        if parent.type == "call" and parent.child_by_field_name("function") == node:
            return False
        if parent.type == "attribute" and parent.child_by_field_name("attribute") == node:
            return False
        if parent.type == "keyword_argument" and parent.child_by_field_name("name") == node:
            return False
        return not (
            parent.type in ("function_definition", "class_definition") and parent.child_by_field_name("name") == node
        )

    def _bindings(def_node: Node) -> frozenset[str]:
        """Names the function binds locally — they shadow the module constant.

        `global NAME` un-shadows: it declares the name module-scoped, so reading or
        writing it really does touch the Value.
        """
        bound: set[str] = set()
        unshadowed: set[str] = set()

        def collect(node: Node) -> None:
            for child in node.children:
                t = child.type
                if t in ("function_definition", "class_definition"):
                    name = child.child_by_field_name("name")
                    if name is not None:
                        bound.add(node_text(name))
                    continue  # a nested scope's internals are its own pass
                if t in ("global_statement", "nonlocal_statement"):
                    for ident in child.children:
                        if ident.type == "identifier":
                            unshadowed.add(node_text(ident))
                    continue
                if t in ("assignment", "augmented_assignment", "named_expression"):
                    left = child.child_by_field_name("left") or child.child_by_field_name("name")
                    if left is not None:
                        _collect_target_names(left, bound)
                if t in ("for_statement", "for_in_clause"):
                    left = child.child_by_field_name("left")
                    if left is not None:
                        _collect_target_names(left, bound)
                if t == "as_pattern_target":
                    _collect_target_names(child, bound)
                collect(child)

        params = def_node.child_by_field_name("parameters")
        if params is not None:
            for ident in params.children:
                _collect_target_names(ident, bound)
        body = def_node.child_by_field_name("body")
        if body is not None:
            collect(body)
        return frozenset(bound - unshadowed)

    def walk(node: Node, owner_qn: str, shadow: frozenset[str]) -> None:
        for child in node.children:
            t = child.type
            if t == "function_definition":
                walk(child, _qn_for(child), shadow | _bindings(child))
                continue
            if t in ("import_statement", "import_from_statement", "parameters", "decorator"):
                continue
            if t == "identifier":
                name = node_text(child)
                if name in values and name not in shadow and _is_load(child):
                    key = (owner_qn, name)
                    if key not in emitted:
                        emitted.add(key)
                        relationships.append(
                            ParsedRelationship(
                                from_qualified_name=owner_qn,
                                rel_type=RelType.REFERENCES,
                                to_name=name,
                                properties={"via": "const", "line": child.start_point[0] + 1},
                            )
                        )
                continue
            walk(child, owner_qn, shadow)

    walk(root, module_qn, frozenset())


def _collect_target_names(node: Node, into: set[str]) -> None:
    """Every identifier a binding target introduces, tuples and stars included."""
    if node.type == "identifier":
        into.add(node_text(node))
        return
    for child in node.children:
        _collect_target_names(child, into)


def _filter_value_references(entities: list[ParsedEntity], relationships: list[ParsedRelationship]) -> None:
    """Drop REFERENCES whose name is not a callable this file can actually see.

    Keeps a name defined here as a Callable, or imported by this module. A project-wide
    match is deliberately not attempted: `foo(bar)` where `bar` is a local that happens to
    share a name with some distant function is precisely the false edge ADR-0022 removed
    from call resolution, and a wrong REFERENCES is worse than none now that
    find_dead_code reads it as proof of life.
    """
    # Functions only, never methods. A method is unreachable by bare identifier — it needs
    # a receiver — so matching one means a local variable merely shares its name. Measured:
    # the `@property def name` on every detector class made every `foo(name)` in the file
    # look like a reference to it.
    local_callables = {e.name for e in entities if e.label == NodeLabel.CALLABLE and e.kind == CallableKind.FUNCTION}
    imported = {r.to_name.rsplit(".", 1)[-1] for r in relationships if r.rel_type == RelType.IMPORTS}
    known = local_callables | imported
    # A dispatch table is a Value, not a Callable, so it would never survive the callable
    # filter. It is kept on a different ground: the name came from a subscript CALL, which
    # is unambiguous about being a lookup rather than an incidental identifier.
    known_tables = {e.name for e in entities if e.label == NodeLabel.VALUE}
    # `self.x` survives the methods-only exclusion above for the reason that exclusion
    # exists: it was there because a BARE name matching a method is coincidence. An
    # explicit `self.` receiver is not a coincidence, so a method is exactly what it
    # should match — but only one declared in this same file.
    local_methods = {e.name for e in entities if e.label == NodeLabel.CALLABLE}

    def _keep(r: ParsedRelationship) -> bool:
        if r.rel_type != RelType.REFERENCES:
            return True
        via = r.properties.get("via")
        if via == "table":
            return r.to_name in known_tables
        # A constant read was emitted against this file's own module-level Values,
        # so the membership test here is belt-and-braces, not the decision.
        if via == "const":
            return r.to_name in known_tables
        if via == "self":
            return r.to_name in local_methods
        return r.to_name in known

    relationships[:] = [r for r in relationships if _keep(r)]


def _extract_table_references(node: Node, from_qn: str, relationships: list[ParsedRelationship]) -> None:
    """Record the callables a dict literal holds — the other registry shape.

    `TABLE = {"greet": handle_greet}` and `parse_func=_parse_python` are the same idea
    written two ways, and only the second was captured. A handler reachable solely through
    a table had no inbound edge at all, so find_dead_code called it dead and blast_radius
    reported nothing downstream of the dispatcher.

    Values only, never keys: `{"greet": ...}` names a string, not a callable.
    """
    for child in node.children:
        if child.type != "pair":
            continue
        value = child.child_by_field_name("value")
        if value is not None and value.type == "identifier":
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=from_qn,
                    rel_type=RelType.REFERENCES,
                    to_name=node_text(value),
                )
            )


def _extract_module_level_references(node: Node, module_uid: str, relationships: list[ParsedRelationship]) -> None:
    """Scan everything OUTSIDE a function body for callables passed as values.

    Attributed to the module, which is the scope that actually runs the call. Function
    bodies are skipped because _extract_calls already covers them and would double up;
    class bodies are included, since `field(default_factory=_ready_event)` executes at
    class-creation time and the name it hands over is a real reference.

    A `decorated_definition` is only skipped when it wraps a FUNCTION. Skipping it wholesale
    also skipped every decorated class, so an undecorated `class Plain` emitted
    `REFERENCES -> _ready_event` and the identical `@dataclass class AppContext` emitted
    nothing — and `@dataclass` is how this codebase writes most of its classes.
    """
    for child in node.children:
        if child.type == "call":
            _extract_value_references(child, module_uid, relationships)
            _extract_module_level_call(child, module_uid, relationships)
        elif child.type == "dictionary":
            _extract_table_references(child, module_uid, relationships)
        if not _is_decorated_function(child) and child.type != "function_definition":
            _extract_module_level_references(child, module_uid, relationships)


def _extract_module_level_call(call_node: Node, module_uid: str, relationships: list[ParsedRelationship]) -> None:
    """A call that runs at import time, attributed to the module that runs it.

    Bare identifiers ONLY. `_validate_schema_completeness()` at the foot of a module is
    lexically grounded exactly as `helper()` is inside a function, so the same reasoning
    that lets ADR-0022 trust one lets it trust the other. `re.compile(...)`,
    `app.add_typer(...)` and the other 109 dotted call sites are the case that ADR
    refuses to guess at — the receiver is a module-level name with no declared type, so
    the attribute says nothing about which class is being called.

    Without this, `_extract_calls` never saw module scope at all: every one of the 9,023
    CALLS edges had a Callable source and not one had a Module source, so a function
    called only at import time looked unreachable.

    Emits BOTH a call and a type use, because the name alone cannot say which it is:
    `_validate_schema_completeness()` invokes a function, `OutputMode()` constructs a
    class, and both parse identically. Each resolver constrains its own target — CALLS
    must land on a Callable, USES_TYPE on a TypeDef — so at most one of the pair can
    resolve, and a builtin like `frozenset()` matches neither and resolves to nothing.
    Guessing from capitalisation instead would miss every lowercase factory class.
    """
    func = call_node.child_by_field_name("function")
    if func is None or func.type != "identifier":
        return
    name = node_text(func)
    relationships.append(
        ParsedRelationship(
            from_qualified_name=module_uid,
            rel_type=RelType.CALLS,
            to_name=name,
            properties={"line": call_node.start_point[0] + 1},
        )
    )
    relationships.append(
        ParsedRelationship(
            from_qualified_name=module_uid,
            rel_type=RelType.USES_TYPE,
            to_name=name,
            # Same routing as an annotated Value: resolved in module scope by
            # resolve_value_references, not through the Callable lookup.
            properties={"on": "value"},
        )
    )


def _is_decorated_function(node: Node) -> bool:
    """A `decorated_definition` wrapping a function rather than a class."""
    if node.type != "decorated_definition":
        return False
    target = node.child_by_field_name("definition")
    return target is None or target.type == "function_definition"


def _extract_value_references(call_node: Node, from_qn: str, relationships: list[ParsedRelationship]) -> None:
    """Record a callable named as a VALUE in a call's arguments, not invoked.

    `run_with(on_complete)`, `field(default_factory=_ready_event)`, `register_language(
    parse_func=_parse_python)` — the callee is handed over, never called here. The parser
    emitted nothing for any of them, so the graph reported the handler dead: 15 of the 30
    dead-code hits in parsing/languages were every language's own entry point, reachable
    only through `parse_func=`.

    A bare identifier, or `self.<name>` / `cls.<name>`. Any other attribute is skipped:
    `mod.handler`'s name is not necessarily the callable's own, which is the same trap
    ADR-0022 recorded for calls. `self` is not that case — it pins the name to the
    enclosing class, so `asyncio.to_thread(self._walk_dir, d)` names exactly one method.
    """
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return
    for arg in args.children:
        node = arg.child_by_field_name("value") if arg.type == "keyword_argument" else arg
        if node is None:
            continue
        if node.type == "identifier":
            target, via = node, ""
        elif node.type == "attribute" and _is_self_attribute(node):
            target, via = node.child_by_field_name("attribute"), "self"
        else:
            continue
        if target is None:
            continue
        relationships.append(
            ParsedRelationship(
                from_qualified_name=from_qn,
                rel_type=RelType.REFERENCES,
                to_name=node_text(target),
                properties={"via": via} if via else {},
            )
        )


def _is_self_attribute(node: Node) -> bool:
    """``self.x`` / ``cls.x`` — an attribute whose receiver is the enclosing instance."""
    obj = node.child_by_field_name("object")
    return obj is not None and obj.type == "identifier" and node_text(obj) in _SELF_RECEIVER_NAMES


_SELF_RECEIVER_NAMES: frozenset[str] = frozenset({"self", "cls"})


def _extract_calls(
    node: Node,
    source: bytes,
    from_qn: str,
    relationships: list[ParsedRelationship],
    local_types: dict[str, str] | None = None,
) -> None:
    """Recursively extract call expressions from a function body."""
    for child in node.children:
        if child.type == "dictionary":
            _extract_table_references(child, from_qn, relationships)
        if child.type == "call":
            _extract_value_references(child, from_qn, relationships)
            func = child.child_by_field_name("function")
            if func is not None:
                if func.type == "identifier":
                    call_name = node_text(func)
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=from_qn,
                            rel_type=RelType.CALLS,
                            to_name=call_name,
                            properties={"line": child.start_point[0] + 1},
                        )
                    )
                elif func.type == "subscript":
                    # `_HANDLERS[name](payload)` — the callee is a lookup, so there is no
                    # name to resolve against. Link to the TABLE and let the table's own
                    # references reach the members: fanning out to all of them would hand
                    # one call site as many full-confidence edges as the table has
                    # entries, which is ADR-0022's failure rebuilt from the other side.
                    base = func.child_by_field_name("value")
                    if base is not None and base.type == "identifier":
                        relationships.append(
                            ParsedRelationship(
                                from_qualified_name=from_qn,
                                rel_type=RelType.REFERENCES,
                                to_name=node_text(base),
                                properties={"via": "table"},
                            )
                        )
                elif func.type == "attribute":
                    attr = func.child_by_field_name("attribute")
                    obj = func.child_by_field_name("object")
                    if attr is not None:
                        call_name = node_text(attr)
                        relationships.append(
                            ParsedRelationship(
                                from_qualified_name=from_qn,
                                rel_type=RelType.CALLS,
                                to_name=call_name,
                                # The receiver is what separates a name the resolver may
                                # trust from one it may not: `helper()` must resolve in
                                # lexical scope, but `client.scan()` names a member of a
                                # type that may never have been indexed. Both arms used
                                # to emit an identical relationship, discarding the one
                                # fact that distinguishes them.
                                properties=_receiver_props(obj, local_types) | {"line": child.start_point[0] + 1},
                            )
                        )
        # Recurse but don't descend into nested function/class definitions
        if child.type not in ("function_definition", "class_definition", "decorated_definition"):
            _extract_calls(child, source, from_qn, relationships, local_types)


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
        yield cursor.node  # ty: ignore[invalid-yield]  # cursor.node is Optional in the stubs, never None mid-walk
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

        # A name defined here gets its uid straight away. A RE-EXPORT — which is what
        # `__all__` in an `__init__.py` almost always is — is defined in a submodule, so
        # the bare name goes out and resolution links it post-batch against what this
        # module imports. Before this, 5 of the 6 `__all__` declarations in this repo
        # produced no edge at all, because name_to_qn only ever held local definitions.
        relationships: list[ParsedRelationship] = [
            ParsedRelationship(
                from_qualified_name=module_entity.qualified_name,
                rel_type=RelType.EXPORTS,
                to_name=name_to_qn.get(exp_name) or exp_name,
                # A local definition already carries its uid; a re-export leaves a bare
                # name for post-batch resolution against this module's imports.
                properties={} if exp_name in name_to_qn else {"by_name": True},
            )
            for exp_name in exported_names
        ]

        return DetectorResult(enrichments=enrichments, relationships=relationships)


# ---------------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------------

register_detector(TestMappingDetector())
register_detector(ClassOverridesDetector())
register_detector(DIInjectionDetector())
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
