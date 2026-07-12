"""Ruby language support — tree-sitter parser for Ruby source files."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    node_text,
    register_language,
)
from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind, ValueKind, Visibility

if TYPE_CHECKING:
    from tree_sitter import Node

try:
    import tree_sitter_ruby as ts_ruby
    from tree_sitter import Language, Query

    _RUBY_LANGUAGE = Language(ts_ruby.language())
    # Minimal query — we walk the tree manually like the Python parser.
    _RUBY_QUERY = Query(_RUBY_LANGUAGE, "(program) @root")
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UPPER_RE_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789")


def _is_constant_name(name: str) -> bool:
    """Return True if *name* looks like a Ruby constant (ALL_CAPS)."""
    return len(name) > 0 and name[0].isupper() and all(ch in _UPPER_RE_CHARS for ch in name)


def _module_qualified_name(file_path: str) -> str:
    """Convert file path to a dot-separated module name.

    ``lib/models/user.rb`` -> ``lib.models.user``
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    parts = list(p.parts)
    if parts and parts[-1].endswith((".rb", ".rake", ".gemspec")):
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return ".".join(parts)


def _resolve_constant_name(node: Node) -> str:
    """Resolve a constant or scope_resolution node to a dotted name."""
    if node.type == "constant":
        return node_text(node)
    if node.type == "scope_resolution":
        scope = node.child_by_field_name("scope")
        name = node.child_by_field_name("name")
        scope_str = _resolve_constant_name(scope) if scope is not None else ""
        name_str = node_text(name) if name is not None else ""
        if scope_str:
            return f"{scope_str}.{name_str}"
        return name_str
    return node_text(node)


def _extract_ruby_docstring(node: Node, source: bytes) -> str | None:
    """Extract YARD/RDoc-style doc comment immediately preceding *node*.

    Ruby doc comments are contiguous ``#`` comment lines directly above a
    definition.
    """
    prev = node.prev_sibling
    # When inside a body_statement, the prev sibling may be within the same body.
    # Walk up to the parent's children perspective if needed.
    comment_lines: list[str] = []
    while prev is not None and prev.type == "comment":
        raw = source[prev.start_byte : prev.end_byte].decode("utf-8", errors="replace")
        # Strip leading '# ' or '#'
        stripped = raw.lstrip("#").strip()
        comment_lines.append(stripped)
        prev = prev.prev_sibling

    if not comment_lines:
        return None
    comment_lines.reverse()
    return "\n".join(comment_lines).strip() or None


def _extract_method_signature(node: Node, source: bytes) -> str | None:
    """Extract method signature from a ``method`` or ``singleton_method`` node."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    params = node.child_by_field_name("parameters")
    end_byte = params.end_byte if params is not None else name_node.end_byte
    sig_bytes = source[node.start_byte : end_byte]
    return sig_bytes.decode("utf-8", errors="replace").strip()


def _visibility_from_name(name: str) -> str:
    """Determine visibility from Ruby naming conventions.

    Ruby doesn't use name-based visibility (it uses explicit private/protected),
    but names starting with _ are conventionally private.
    """
    if name.startswith("_"):
        return Visibility.PRIVATE
    return Visibility.PUBLIC


# ---------------------------------------------------------------------------
# Tree walkers
# ---------------------------------------------------------------------------


def _extract_calls(
    node: Node,
    from_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Recursively extract call expressions from a method body.

    In Ruby, bare identifiers at statement level (``validate``) are implicit
    method calls without parentheses.  Explicit calls (``puts "hello"``)
    appear as ``call`` nodes.
    """
    for child in node.children:
        if child.type == "call":
            method_node = child.child_by_field_name("method")
            if method_node is not None:
                call_name = node_text(method_node)
                if call_name:
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=from_qn,
                            rel_type=RelType.CALLS,
                            to_name=call_name,
                        )
                    )
        elif child.type == "identifier" and node.type == "body_statement":
            # Bare identifier at statement level — implicit method call
            call_name = node_text(child)
            if call_name:
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=from_qn,
                        rel_type=RelType.CALLS,
                        to_name=call_name,
                    )
                )
        # Recurse but don't descend into nested method/class/module definitions
        if child.type not in ("method", "singleton_method", "class", "module"):
            _extract_calls(child, from_qn, relationships)


def _walk_ruby_body(
    body_node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    parent_qn: str,
    parent_type: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    scope_stack: list[str],
) -> None:
    """Walk a Ruby body_statement (class/module body) extracting entities.

    *parent_type* is ``"class"``, ``"module"``, or ``"top"`` to control
    visibility tracking and method kind assignment.
    """
    current_visibility = Visibility.PUBLIC

    for child in body_node.children:
        # Visibility modifier calls: private, protected, public
        if child.type == "call" and _is_visibility_modifier(child):
            new_vis = _handle_visibility_call(child)
            if new_vis is not None:
                current_visibility = new_vis
            else:
                # Inline form: `private def foo` — the definition is wrapped in
                # the call's arguments. Extract it with the modifier's visibility;
                # the modifier applies to that method only, not subsequent ones.
                _process_inline_visibility_method(
                    child,
                    path,
                    source,
                    project_name,
                    module_qn,
                    parent_qn,
                    parent_type,
                    entities,
                    relationships,
                    scope_stack,
                )
            continue

        # Also handle bare identifiers: `private`, `protected`, `public` on their own line
        if child.type == "identifier" and node_text(child) in ("private", "protected", "public"):
            vis_name = node_text(child)
            current_visibility = {"private": Visibility.PRIVATE, "protected": Visibility.PROTECTED}.get(
                vis_name, Visibility.PUBLIC
            )
            continue

        if child.type == "class":
            _process_ruby_class(
                child, path, source, project_name, module_qn, parent_qn, entities, relationships, scope_stack
            )
            continue

        if child.type == "module":
            _process_ruby_module(
                child, path, source, project_name, module_qn, parent_qn, entities, relationships, scope_stack
            )
            continue

        if child.type == "method":
            _process_ruby_method(
                child,
                path,
                source,
                project_name,
                module_qn,
                parent_qn,
                parent_type,
                current_visibility,
                entities,
                relationships,
                scope_stack,
            )
            continue

        if child.type == "singleton_method":
            _process_ruby_singleton_method(
                child,
                path,
                source,
                project_name,
                module_qn,
                parent_qn,
                current_visibility,
                entities,
                relationships,
                scope_stack,
            )
            continue

        if child.type == "assignment":
            _process_ruby_assignment(
                child, path, project_name, module_qn, parent_qn, parent_type, entities, scope_stack
            )
            continue

        # Handle attr_reader, attr_writer, attr_accessor as tags
        if child.type == "call":
            _process_ruby_call_directive(
                child, path, project_name, module_qn, parent_qn, entities, relationships, scope_stack
            )
            continue


def _is_visibility_modifier(call_node: Node) -> bool:
    """Check if a call node is a visibility modifier (private/protected/public)."""
    method_node = call_node.child_by_field_name("method")
    if method_node is None:
        return False
    name = node_text(method_node)
    return name in ("private", "protected", "public")


def _visibility_modifier_value(call_node: Node) -> str:
    """Map a visibility modifier call node to its Visibility value."""
    method_node = call_node.child_by_field_name("method")
    name = node_text(method_node) if method_node is not None else ""
    return {"private": Visibility.PRIVATE, "protected": Visibility.PROTECTED}.get(name, Visibility.PUBLIC)


def _process_inline_visibility_method(
    call_node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    parent_qn: str,
    parent_type: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    scope_stack: list[str],
) -> None:
    """Process ``private def foo`` / ``public def foo`` / ``protected def foo``.

    The method definition is wrapped in the visibility call's arguments; the
    modifier's visibility applies to that method only. Symbol arguments
    (``private :foo``) contain no definition nodes and are skipped.
    """
    inline_vis = _visibility_modifier_value(call_node)
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return
    for arg in args.children:
        if arg.type == "method":
            _process_ruby_method(
                arg,
                path,
                source,
                project_name,
                module_qn,
                parent_qn,
                parent_type,
                inline_vis,
                entities,
                relationships,
                scope_stack,
            )
        elif arg.type == "singleton_method":
            _process_ruby_singleton_method(
                arg,
                path,
                source,
                project_name,
                module_qn,
                parent_qn,
                inline_vis,
                entities,
                relationships,
                scope_stack,
            )


def _handle_visibility_call(call_node: Node) -> str | None:
    """Handle a visibility modifier call. Returns new visibility if no args (block modifier)."""
    method_node = call_node.child_by_field_name("method")
    if method_node is None:
        return None
    name = node_text(method_node)
    args = call_node.child_by_field_name("arguments")
    if args is None:
        # No arguments — this changes visibility for all subsequent methods
        return {"private": Visibility.PRIVATE, "protected": Visibility.PROTECTED}.get(name, Visibility.PUBLIC)
    # Has arguments (e.g. `private :method_name`) — skip, this is per-method
    return None


def _process_ruby_class(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    parent_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    scope_stack: list[str],
) -> None:
    """Process a ``class`` node."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    resolved_name = _resolve_constant_name(name_node)
    # Store the bare (last-segment) name so compact paths (`class Admin::User`)
    # match the same entity `name` as the equivalent nested form
    # (`module Admin; class User`), keeping name-based INHERITS resolution
    # consistent regardless of declaration style.
    name = resolved_name.rsplit(".", 1)[-1]
    docstring = _extract_ruby_docstring(node, source)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    new_scope = [*scope_stack, resolved_name]
    dotted = ".".join(new_scope)
    qn = f"{module_qn}.{dotted}"

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
            visibility=Visibility.PUBLIC,
        )
    )

    # DEFINES from parent
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{parent_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # Superclass -> INHERITS
    superclass_node = node.child_by_field_name("superclass")
    if superclass_node is not None:
        # superclass node wraps the actual constant/scope_resolution
        for sc_child in superclass_node.children:
            if sc_child.type in ("constant", "scope_resolution"):
                base_name = _resolve_constant_name(sc_child)
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=f"{project_name}:{qn}",
                        rel_type=RelType.INHERITS,
                        to_name=base_name,
                    )
                )
                break

    # Walk class body
    body = node.child_by_field_name("body")
    if body is not None:
        _walk_ruby_body(body, path, source, project_name, module_qn, qn, "class", entities, relationships, new_scope)


def _process_ruby_module(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    parent_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    scope_stack: list[str],
) -> None:
    """Process a ``module`` node (Ruby modules are mixins/namespaces)."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    resolved_name = _resolve_constant_name(name_node)
    # See _process_ruby_class: keep entity `name` bare regardless of
    # compact (`module Admin::Helpers`) vs nested declaration style.
    name = resolved_name.rsplit(".", 1)[-1]
    docstring = _extract_ruby_docstring(node, source)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    new_scope = [*scope_stack, resolved_name]
    dotted = ".".join(new_scope)
    qn = f"{module_qn}.{dotted}"

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=TypeDefKind.PROTOCOL,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            visibility=Visibility.PUBLIC,
        )
    )

    # DEFINES from parent
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{parent_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # Walk module body
    body = node.child_by_field_name("body")
    if body is not None:
        _walk_ruby_body(body, path, source, project_name, module_qn, qn, "module", entities, relationships, new_scope)


def _process_ruby_method(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    parent_qn: str,
    parent_type: str,
    visibility: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    scope_stack: list[str],
) -> None:
    """Process a ``method`` (instance method) node."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    docstring = _extract_ruby_docstring(node, source)
    signature = _extract_method_signature(node, source)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    if parent_type in ("class", "module"):
        kind = CallableKind.CONSTRUCTOR if name == "initialize" else CallableKind.METHOD
    else:
        kind = CallableKind.FUNCTION

    new_scope = [*scope_stack, name]
    dotted = ".".join(new_scope)
    qn = f"{module_qn}.{dotted}"

    # Merge name-based visibility with tracked visibility
    effective_vis = visibility if visibility != Visibility.PUBLIC else _visibility_from_name(name)

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
            visibility=effective_vis,
        )
    )

    # DEFINES from parent
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{parent_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # CALLS from method body
    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, f"{project_name}:{qn}", relationships)


def _process_ruby_singleton_method(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    parent_qn: str,
    visibility: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    scope_stack: list[str],
) -> None:
    """Process a ``singleton_method`` (``def self.foo``) node."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    docstring = _extract_ruby_docstring(node, source)
    signature = _extract_method_signature(node, source)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    new_scope = [*scope_stack, name]
    dotted = ".".join(new_scope)
    qn = f"{module_qn}.{dotted}"

    effective_vis = visibility if visibility != Visibility.PUBLIC else _visibility_from_name(name)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.CALLABLE,
            kind=CallableKind.STATIC_METHOD,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            signature=signature,
            source=node_text(node),
            visibility=effective_vis,
        )
    )

    # DEFINES from parent
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{parent_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # CALLS from method body
    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, f"{project_name}:{qn}", relationships)


def _process_ruby_assignment(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    parent_qn: str,  # noqa: ARG001
    parent_type: str,
    entities: list[ParsedEntity],
    scope_stack: list[str],
) -> None:
    """Process an assignment node to extract constants/variables."""
    left = node.child_by_field_name("left")
    if left is None:
        return

    if left.type in ("constant", "identifier"):
        name = node_text(left)
    else:
        return

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    # Determine kind
    if left.type == "constant" or _is_constant_name(name):
        kind = ValueKind.CONSTANT
    elif parent_type in ("class", "module"):
        kind = ValueKind.FIELD
    else:
        kind = ValueKind.VARIABLE

    new_scope = [*scope_stack, name]
    dotted = ".".join(new_scope)
    qn = f"{module_qn}.{dotted}"

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.VALUE,
            kind=kind,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            source=node_text(node),
            visibility=_visibility_from_name(name),
        )
    )


def _process_require(node: Node, project_name: str, parent_qn: str, relationships: list[ParsedRelationship]) -> None:
    """Extract IMPORTS relationships from ``require`` / ``require_relative`` calls."""
    args = node.child_by_field_name("arguments")
    if args is None:
        return
    for arg_child in args.children:
        if arg_child.type == "string":
            import_name = node_text(arg_child).strip("\"'")
            if import_name:
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=f"{project_name}:{parent_qn}",
                        rel_type=RelType.IMPORTS,
                        to_name=import_name,
                    )
                )


def _process_mixin(node: Node, project_name: str, parent_qn: str, relationships: list[ParsedRelationship]) -> None:
    """Extract INHERITS relationships from ``include`` / ``extend`` / ``prepend`` calls."""
    args = node.child_by_field_name("arguments")
    if args is None:
        return
    for arg_child in args.children:
        if arg_child.type in ("constant", "scope_resolution"):
            mixin_name = _resolve_constant_name(arg_child)
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=f"{project_name}:{parent_qn}",
                    rel_type=RelType.INHERITS,
                    to_name=mixin_name,
                )
            )


def _process_attr_directive(
    node: Node,
    method_name: str,
    path: str,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    scope_stack: list[str],
) -> None:
    """Extract Value entities from ``attr_reader`` / ``attr_writer`` / ``attr_accessor``."""
    args = node.child_by_field_name("arguments")
    if args is None:
        return
    for arg_child in args.children:
        if arg_child.type in ("simple_symbol", "symbol"):
            sym_name = node_text(arg_child).lstrip(":")
            qn = f"{module_qn}.{'.'.join([*scope_stack, sym_name])}"
            entities.append(
                ParsedEntity(
                    name=sym_name,
                    qualified_name=f"{project_name}:{qn}",
                    label=NodeLabel.VALUE,
                    kind=ValueKind.FIELD,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    file_path=path,
                    visibility=Visibility.PUBLIC,
                    tags=[f"synthesized:{method_name}"],
                )
            )


def _process_ruby_call_directive(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    parent_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    scope_stack: list[str],
) -> None:
    """Process call nodes that are directives: require, include, extend, prepend, attr_*."""
    method_node = node.child_by_field_name("method")
    if method_node is None:
        return
    method_name = node_text(method_node)

    if method_name in ("require", "require_relative"):
        _process_require(node, project_name, parent_qn, relationships)
    elif method_name in ("include", "extend", "prepend"):
        _process_mixin(node, project_name, parent_qn, relationships)
    elif method_name in ("attr_reader", "attr_writer", "attr_accessor"):
        _process_attr_directive(node, method_name, path, project_name, module_qn, entities, scope_stack)


# ---------------------------------------------------------------------------
# Top-level parse entry point
# ---------------------------------------------------------------------------


def _parse_ruby(
    path: str,
    source: bytes,
    root: Node,
    project_name: str,
) -> ParsedFile:
    """Extract entities and relationships from a Ruby parse tree."""
    module_qn = _module_qualified_name(path)

    entities: list[ParsedEntity] = []
    relationships: list[ParsedRelationship] = []

    # Module entity (file-level)
    module_name = module_qn.rsplit(".", 1)[-1] if "." in module_qn else module_qn
    entities.append(
        ParsedEntity(
            name=module_name,
            qualified_name=f"{project_name}:{module_qn}",
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=root.end_point[0] + 1,
            file_path=path,
        )
    )

    # Walk top-level children
    _walk_ruby_body(root, path, source, project_name, module_qn, module_qn, "top", entities, relationships, [])

    return ParsedFile(
        file_path=path,
        language="ruby",
        entities=entities,
        relationships=relationships,
    )


# ---------------------------------------------------------------------------
# Language registration
# ---------------------------------------------------------------------------

if _AVAILABLE:
    register_language(
        LanguageConfig(
            name="ruby",
            extensions=frozenset({".rb", ".rake", ".gemspec"}),
            language=_RUBY_LANGUAGE,
            query=_RUBY_QUERY,
            parse_func=_parse_ruby,
        )
    )
