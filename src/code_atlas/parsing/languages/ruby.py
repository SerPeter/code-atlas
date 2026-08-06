"""Ruby language support — tree-sitter parser for Ruby source files."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    call_receiver_props,
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
# Node-type vocabulary
# ---------------------------------------------------------------------------

#: Forms that open a new naming scope. The walker handles each itself and never
#: falls through to the transparent recursion below.
_SCOPE_NODES = frozenset({"method", "singleton_method", "class", "module", "singleton_class"})

#: Anonymous callable forms. No entity by design (ADR-0031) — a `do ... end` has
#: no name a developer could refer to — but their bodies are walked, so the calls
#: inside them attribute to the enclosing method, or to the module.
_BLOCK_NODES = frozenset({"do_block", "block", "lambda"})

#: Nodes whose children are statements. A bare identifier is only an implicit
#: method call in one of these; anywhere else it is an operand.
_STATEMENT_CONTAINERS = frozenset({"program", "body_statement", "block_body", "then", "else", "do", "begin", "ensure"})

_VISIBILITY_NAMES = frozenset({"private", "protected", "public"})
_VISIBILITY_BY_NAME = {
    "private": Visibility.PRIVATE,
    "protected": Visibility.PROTECTED,
    "public": Visibility.PUBLIC,
}

_PARAM_LISTS = frozenset({"method_parameters", "block_parameters", "lambda_parameters"})

_MIXIN_DIRECTIVES = frozenset({"include", "extend", "prepend"})
_REQUIRE_DIRECTIVES = frozenset({"require", "require_relative"})
_ATTR_DIRECTIVES = frozenset({"attr_reader", "attr_writer", "attr_accessor"})


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
# Local-variable bindings
# ---------------------------------------------------------------------------


def _collect_identifiers(node: Node, out: set[str]) -> None:
    """Add every ``identifier`` in *node*'s subtree to *out*."""
    if node.type == "identifier":
        out.add(node_text(node))
        return
    for child in node.children:
        _collect_identifiers(child, out)


def _collect_params(params: Node, out: set[str]) -> None:
    """Add the names bound by a parameter list, not their default expressions.

    ``def f(a = helper)`` binds ``a``; ``helper`` is a call, and adding it would
    silence a real edge.
    """
    for child in params.children:
        if child.type == "identifier":
            out.add(node_text(child))
        elif child.type == "destructured_parameter":
            _collect_params(child, out)
        else:
            name = child.child_by_field_name("name")
            if name is not None:
                _collect_identifiers(name, out)


def _local_names(node: Node) -> frozenset[str]:
    """Names bound as local variables in *node*'s own scope.

    Ruby scopes locals to the enclosing ``def``/``class``/``module``, so the walk
    stops at those. A block is not a scope, so its parameters and assignments
    belong to whatever encloses it and are collected here too.

    This is what keeps a bare identifier statement from being read as an implicit
    method call. Both shapes are common in real Ruby: measured on sinatra, 146
    identifiers stand alone as statements and roughly half of them — ``result``,
    ``value``, ``output`` — are local reads, while ``content_type``, ``pass`` and
    ``not_found`` are genuine receiver-less calls. Only the binding set separates
    them.
    """
    out: set[str] = set()
    stack = list(node.children)
    while stack:
        current = stack.pop()
        if current.type in _SCOPE_NODES:
            continue
        if current.type in ("assignment", "operator_assignment"):
            left = current.child_by_field_name("left")
            if left is not None:
                _collect_identifiers(left, out)
        elif current.type in _PARAM_LISTS:
            _collect_params(current, out)
        elif current.type == "exception_variable":
            _collect_identifiers(current, out)
        elif current.type == "for":
            pattern = current.child_by_field_name("pattern")
            if pattern is not None:
                _collect_identifiers(pattern, out)
        stack.extend(current.children)
    return frozenset(out)


# ---------------------------------------------------------------------------
# Walk context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ctx:
    """Where the walker is, and what that means for the next node it meets."""

    path: str
    source: bytes
    project: str
    module_qn: str
    entities: list[ParsedEntity]
    relationships: list[ParsedRelationship]
    parent_qn: str
    """Nearest enclosing class/module (or the file module) — owns DEFINES,
    IMPORTS and INHERITS."""
    parent_type: str
    """``"class"``, ``"module"``, ``"singleton"`` (a ``class << self`` body) or
    ``"top"``. Decides what kind a ``def`` produces."""
    scope: tuple[str, ...] = ()
    anonymous: bool = False
    """True once a block sits between here and the nearest named owner.

    A ``def`` here gets no entity: ``superclass.class_eval do def call ... end``
    attaches the method to the receiver, not to the lexical class, so no scope
    path names it and every such ``def`` in the file would claim the same uid
    (ADR-0032). The body is still walked and its calls attribute upward.

    A ``class`` or ``module`` clears it, because Ruby resolves the constant it
    declares against the lexical nesting and a block is transparent to that — a
    ``class TestApp`` inside a ``describe`` block really is scoped by whatever
    encloses the block. A ``class << self`` does not clear it: its ``self`` is
    the block's receiver, which is exactly what cannot be named."""
    callable_qn: str | None = None
    """Nearest enclosing named callable. ``None`` inside a class or module body,
    including inside a DSL block in one — a block opens no named scope."""
    declarative: bool = True
    """True in a class/module/top-level body, where an assignment declares a
    constant or field. False inside a callable or a block, where it binds a
    local."""
    locals: frozenset[str] = field(default_factory=frozenset)

    @property
    def call_from(self) -> str:
        """Source uid for a CALLS edge.

        The nearest enclosing named callable, or the module. A block never
        appears here: its calls belong to whoever defines it (ADR-0031).
        """
        return f"{self.project}:{self.callable_qn or self.module_qn}"

    @property
    def defines_from(self) -> str:
        """Source uid for a DEFINES edge — an enclosing ``def`` outranks the class."""
        return f"{self.project}:{self.callable_qn or self.parent_qn}"


# ---------------------------------------------------------------------------
# Tree walkers
# ---------------------------------------------------------------------------


def _walk(node: Node, ctx: _Ctx, vis: list[str]) -> None:
    """Dispatch every child of *node*.

    *vis* is a one-element cell holding the visibility a bare ``private`` has
    switched on, shared across the siblings of one class body.
    """
    for child in node.children:
        _dispatch(child, ctx, vis)


def _dispatch(child: Node, ctx: _Ctx, vis: list[str]) -> None:
    """Route one node.

    Everything that is not a named scope is walked straight through with the same
    context — an ``if``, a ``begin``, a ``rescue``, a string interpolation, and
    above all a ``do ... end``. That last one is the whole point: sinatra is
    3,947 blocks against 896 methods, so a walker that stops at a block sees
    almost none of the code.
    """
    kind = child.type
    if kind == "class":
        _process_ruby_class(child, ctx)
    elif kind == "module":
        _process_ruby_module(child, ctx)
    elif kind == "singleton_class":
        _process_singleton_class(child, ctx)
    elif kind == "method":
        _process_ruby_method(child, ctx, vis[0])
    elif kind == "singleton_method":
        _process_ruby_singleton_method(child, ctx, vis[0])
    elif kind == "call":
        _dispatch_call(child, ctx, vis)
    elif kind == "identifier":
        _dispatch_identifier(child, ctx, vis)
    elif kind in _BLOCK_NODES:
        # No entity and no new call scope. Value extraction stops, because a
        # block's assignments bind locals rather than declaring fields, and
        # anything defined below here loses its claim to a qualified name.
        _walk(child, replace(ctx, declarative=False, anonymous=True), vis)
    elif kind == "assignment":
        if ctx.declarative:
            _process_ruby_assignment(child, ctx)
        _walk(child, ctx, vis)
    else:
        _walk(child, ctx, vis)


def _dispatch_call(node: Node, ctx: _Ctx, vis: list[str]) -> None:
    """Emit the CALLS edge for a ``call`` node, then handle it as a directive."""
    method_node = node.child_by_field_name("method")
    name = node_text(method_node) if method_node is not None else ""
    receiver = node.child_by_field_name("receiver")

    if name:
        ctx.relationships.append(
            ParsedRelationship(
                from_qualified_name=ctx.call_from,
                rel_type=RelType.CALLS,
                to_name=name,
                properties=call_receiver_props(receiver),
            )
        )

    # A directive is only a directive when it is called on nothing. `include Foo`
    # declares a mixin; `base.include(Foo)` and `obj.extend(M)` name a receiver
    # this walker cannot resolve, and inventing an INHERITS from them would be a
    # guess.
    if receiver is None:
        if name in _VISIBILITY_NAMES and _apply_visibility_call(node, name, ctx, vis):
            # The wrapped definition has been processed already; walking on would
            # emit it a second time.
            return
        _process_call_directive(node, name, ctx)

    _walk(node, ctx, vis)


def _dispatch_identifier(node: Node, ctx: _Ctx, vis: list[str]) -> None:
    """A bare identifier standing alone as a statement — an implicit call, or a local read."""
    parent = node.parent
    if parent is None or parent.type not in _STATEMENT_CONTAINERS:
        return
    name = node_text(node)
    if not name:
        return
    if name in _VISIBILITY_NAMES:
        if ctx.declarative:
            vis[0] = _VISIBILITY_BY_NAME[name]
        return
    if name in ctx.locals:
        return
    ctx.relationships.append(
        ParsedRelationship(
            from_qualified_name=ctx.call_from,
            rel_type=RelType.CALLS,
            to_name=name,
        )
    )


def _apply_visibility_call(node: Node, name: str, ctx: _Ctx, vis: list[str]) -> bool:
    """Apply ``private`` / ``protected`` / ``public``.

    Returns True when the call was fully consumed — either it took no arguments
    and switched the section, or it wrapped a definition that has now been
    extracted. ``private :foo`` consumes nothing and returns False, so the
    ordinary walk continues over its arguments.
    """
    value = _VISIBILITY_BY_NAME[name]
    args = node.child_by_field_name("arguments")
    if args is None:
        # No arguments — this changes visibility for all subsequent methods.
        if ctx.declarative:
            vis[0] = value
        return True

    # Inline form: `private def foo`. The modifier applies to that method only,
    # not to the ones after it.
    consumed = False
    for arg in args.children:
        if arg.type == "method":
            _process_ruby_method(arg, ctx, value)
            consumed = True
        elif arg.type == "singleton_method":
            _process_ruby_singleton_method(arg, ctx, value)
            consumed = True
    return consumed


def _process_call_directive(node: Node, name: str, ctx: _Ctx) -> None:
    """Handle require / include / extend / prepend / attr_* on a receiver-less call."""
    if name in _REQUIRE_DIRECTIVES:
        _process_require(node, ctx)
    elif name in _MIXIN_DIRECTIVES:
        _process_mixin(node, ctx)
    elif name in _ATTR_DIRECTIVES and ctx.callable_qn is None:
        _process_attr_directive(node, name, ctx)


# ---------------------------------------------------------------------------
# Definition processors
# ---------------------------------------------------------------------------


def _process_ruby_class(node: Node, ctx: _Ctx) -> None:
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
    docstring = _extract_ruby_docstring(node, ctx.source)

    new_scope = (*ctx.scope, resolved_name)
    qn = f"{ctx.module_qn}.{'.'.join(new_scope)}"

    ctx.entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{ctx.project}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=TypeDefKind.CLASS,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=ctx.path,
            docstring=docstring,
            visibility=Visibility.PUBLIC,
        )
    )

    ctx.relationships.append(
        ParsedRelationship(
            from_qualified_name=ctx.defines_from,
            rel_type=RelType.DEFINES,
            to_name=f"{ctx.project}:{qn}",
        )
    )

    # Superclass -> INHERITS
    superclass_node = node.child_by_field_name("superclass")
    if superclass_node is not None:
        # superclass node wraps the actual constant/scope_resolution
        for sc_child in superclass_node.children:
            if sc_child.type in ("constant", "scope_resolution"):
                ctx.relationships.append(
                    ParsedRelationship(
                        from_qualified_name=f"{ctx.project}:{qn}",
                        rel_type=RelType.INHERITS,
                        to_name=_resolve_constant_name(sc_child),
                    )
                )
                break

    _walk(node, _body_ctx(ctx, node, parent_qn=qn, parent_type="class", scope=new_scope), [Visibility.PUBLIC])


def _process_ruby_module(node: Node, ctx: _Ctx) -> None:
    """Process a ``module`` node (Ruby modules are mixins/namespaces)."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    resolved_name = _resolve_constant_name(name_node)
    # See _process_ruby_class: keep entity `name` bare regardless of
    # compact (`module Admin::Helpers`) vs nested declaration style.
    name = resolved_name.rsplit(".", 1)[-1]
    docstring = _extract_ruby_docstring(node, ctx.source)

    new_scope = (*ctx.scope, resolved_name)
    qn = f"{ctx.module_qn}.{'.'.join(new_scope)}"

    ctx.entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{ctx.project}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=TypeDefKind.PROTOCOL,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=ctx.path,
            docstring=docstring,
            visibility=Visibility.PUBLIC,
        )
    )

    ctx.relationships.append(
        ParsedRelationship(
            from_qualified_name=ctx.defines_from,
            rel_type=RelType.DEFINES,
            to_name=f"{ctx.project}:{qn}",
        )
    )

    _walk(node, _body_ctx(ctx, node, parent_qn=qn, parent_type="module", scope=new_scope), [Visibility.PUBLIC])


def _process_singleton_class(node: Node, ctx: _Ctx) -> None:
    """Process a ``class << self`` body.

    It declares no entity of its own — it reopens the enclosing class — so the
    scope and the DEFINES parent are inherited unchanged, and only the kind a
    ``def`` produces changes. sinatra writes 75 of its class methods this way,
    and every one of them was invisible.
    """
    _walk(
        node,
        # `anonymous` carries through: `class << self` inside a block reopens the
        # singleton of the block's receiver, which is the thing we cannot name.
        _body_ctx(
            ctx,
            node,
            parent_qn=ctx.parent_qn,
            parent_type="singleton",
            scope=ctx.scope,
            anonymous=ctx.anonymous,
        ),
        [Visibility.PUBLIC],
    )


def _body_ctx(
    ctx: _Ctx,
    node: Node,
    *,
    parent_qn: str,
    parent_type: str,
    scope: tuple[str, ...],
    anonymous: bool = False,
) -> _Ctx:
    """Context for the body of a class, module or ``class << self``.

    ``callable_qn`` resets: a class body is not executed by whatever ``def`` may
    lexically surround it, so its calls belong to the module.
    """
    body = node.child_by_field_name("body")
    return replace(
        ctx,
        parent_qn=parent_qn,
        parent_type=parent_type,
        scope=scope,
        anonymous=anonymous,
        callable_qn=None,
        declarative=True,
        locals=_local_names(body) if body is not None else frozenset(),
    )


def _def_scope(ctx: _Ctx, name: str, *, singleton: bool) -> tuple[str, ...]:
    """Scope path for a ``def``.

    A singleton method takes an extra ``self`` segment, so ``def self.settings``
    and ``def settings`` in one class stop sharing a uid (ADR-0032). The instance
    method keeps the shorter path because it is the commoner of the two, which
    bounds the churn; and ``self`` is a Ruby keyword, so the segment can never
    collide with a class or module a developer could actually declare.
    """
    if singleton:
        return (*ctx.scope, "self", name)
    return (*ctx.scope, name)


def _has_self_receiver(node: Node) -> bool:
    """True for ``def self.foo``, false for ``def enc.foo`` or ``def @@x.foo``.

    Only ``self`` names the enclosing class. An arbitrary receiver is a runtime
    object this walker cannot resolve, so it earns no segment.
    """
    obj = node.child_by_field_name("object")
    return obj is not None and obj.type == "self"


def _declined_ctx(ctx: _Ctx, node: Node) -> _Ctx:
    """Context for the body of a ``def`` that got no entity.

    ``callable_qn`` and ``scope`` stay put, so the calls inside land on the
    nearest named enclosing scope rather than on nobody (ADR-0031). Only the
    local bindings change: they are the declined ``def``'s own, since Ruby does
    not close a ``def`` over the locals around it.
    """
    return replace(ctx, declarative=False, locals=_local_names(node))


def _callable_kind(name: str, ctx: _Ctx) -> str:
    """Kind for a ``def``, given what encloses it."""
    if ctx.callable_qn is not None:
        # A def inside a def is a helper of that def, not a member of the class.
        return CallableKind.FUNCTION
    if ctx.parent_type == "singleton":
        return CallableKind.STATIC_METHOD
    if ctx.parent_type in ("class", "module"):
        return CallableKind.CONSTRUCTOR if name == "initialize" else CallableKind.METHOD
    return CallableKind.FUNCTION


def _process_ruby_method(node: Node, ctx: _Ctx, visibility: str) -> None:
    """Process a ``method`` (instance method) node."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)

    if ctx.anonymous:
        _walk(node, _declined_ctx(ctx, node), [Visibility.PUBLIC])
        return

    # `def foo` inside `class << self` is a singleton method wearing the instance
    # spelling, and takes the same `self` segment as `def self.foo`.
    new_scope = _def_scope(ctx, name, singleton=ctx.parent_type == "singleton")
    qn = f"{ctx.module_qn}.{'.'.join(new_scope)}"

    # Merge name-based visibility with tracked visibility
    effective_vis = visibility if visibility != Visibility.PUBLIC else _visibility_from_name(name)

    ctx.entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{ctx.project}:{qn}",
            label=NodeLabel.CALLABLE,
            kind=_callable_kind(name, ctx),
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=ctx.path,
            docstring=_extract_ruby_docstring(node, ctx.source),
            signature=_extract_method_signature(node, ctx.source),
            source=node_text(node),
            visibility=effective_vis,
        )
    )

    ctx.relationships.append(
        ParsedRelationship(
            from_qualified_name=ctx.defines_from,
            rel_type=RelType.DEFINES,
            to_name=f"{ctx.project}:{qn}",
        )
    )

    _walk(node, _callable_ctx(ctx, node, qn, new_scope), [Visibility.PUBLIC])


def _process_ruby_singleton_method(node: Node, ctx: _Ctx, visibility: str) -> None:
    """Process a ``singleton_method`` (``def self.foo``) node."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)

    if ctx.anonymous:
        _walk(node, _declined_ctx(ctx, node), [Visibility.PUBLIC])
        return

    new_scope = _def_scope(ctx, name, singleton=_has_self_receiver(node))
    qn = f"{ctx.module_qn}.{'.'.join(new_scope)}"

    effective_vis = visibility if visibility != Visibility.PUBLIC else _visibility_from_name(name)

    ctx.entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{ctx.project}:{qn}",
            label=NodeLabel.CALLABLE,
            kind=CallableKind.STATIC_METHOD,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=ctx.path,
            docstring=_extract_ruby_docstring(node, ctx.source),
            signature=_extract_method_signature(node, ctx.source),
            source=node_text(node),
            visibility=effective_vis,
        )
    )

    ctx.relationships.append(
        ParsedRelationship(
            from_qualified_name=ctx.defines_from,
            rel_type=RelType.DEFINES,
            to_name=f"{ctx.project}:{qn}",
        )
    )

    _walk(node, _callable_ctx(ctx, node, qn, new_scope), [Visibility.PUBLIC])


def _callable_ctx(ctx: _Ctx, node: Node, qn: str, scope: tuple[str, ...]) -> _Ctx:
    """Context for a method body.

    Locals are collected from the whole ``def``, parameters included, so the
    body's bare identifiers can be told apart from implicit calls.
    """
    return replace(ctx, callable_qn=qn, scope=scope, declarative=False, locals=_local_names(node))


def _process_ruby_assignment(node: Node, ctx: _Ctx) -> None:
    """Process an assignment node to extract constants/variables."""
    left = node.child_by_field_name("left")
    if left is None or left.type not in ("constant", "identifier"):
        return
    name = node_text(left)

    # Determine kind
    if left.type == "constant" or _is_constant_name(name):
        kind = ValueKind.CONSTANT
    elif ctx.parent_type in ("class", "module", "singleton"):
        kind = ValueKind.FIELD
    else:
        kind = ValueKind.VARIABLE

    qn = f"{ctx.module_qn}.{'.'.join((*ctx.scope, name))}"

    ctx.entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{ctx.project}:{qn}",
            label=NodeLabel.VALUE,
            kind=kind,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=ctx.path,
            source=node_text(node),
            visibility=_visibility_from_name(name),
        )
    )


def _process_require(node: Node, ctx: _Ctx) -> None:
    """Extract IMPORTS relationships from ``require`` / ``require_relative`` calls."""
    args = node.child_by_field_name("arguments")
    if args is None:
        return
    for arg_child in args.children:
        if arg_child.type == "string":
            import_name = node_text(arg_child).strip("\"'")
            if import_name:
                ctx.relationships.append(
                    ParsedRelationship(
                        from_qualified_name=f"{ctx.project}:{ctx.parent_qn}",
                        rel_type=RelType.IMPORTS,
                        to_name=import_name,
                    )
                )


def _process_mixin(node: Node, ctx: _Ctx) -> None:
    """Extract INHERITS relationships from ``include`` / ``extend`` / ``prepend`` calls."""
    args = node.child_by_field_name("arguments")
    if args is None:
        return
    for arg_child in args.children:
        if arg_child.type in ("constant", "scope_resolution"):
            ctx.relationships.append(
                ParsedRelationship(
                    from_qualified_name=f"{ctx.project}:{ctx.parent_qn}",
                    rel_type=RelType.INHERITS,
                    to_name=_resolve_constant_name(arg_child),
                )
            )


def _process_attr_directive(node: Node, method_name: str, ctx: _Ctx) -> None:
    """Extract Value entities from ``attr_reader`` / ``attr_writer`` / ``attr_accessor``."""
    args = node.child_by_field_name("arguments")
    if args is None:
        return
    for arg_child in args.children:
        if arg_child.type in ("simple_symbol", "symbol"):
            sym_name = node_text(arg_child).lstrip(":")
            qn = f"{ctx.module_qn}.{'.'.join((*ctx.scope, sym_name))}"
            ctx.entities.append(
                ParsedEntity(
                    name=sym_name,
                    qualified_name=f"{ctx.project}:{qn}",
                    label=NodeLabel.VALUE,
                    kind=ValueKind.FIELD,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    file_path=ctx.path,
                    visibility=Visibility.PUBLIC,
                    tags=[f"synthesized:{method_name}"],
                )
            )


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

    ctx = _Ctx(
        path=path,
        source=source,
        project=project_name,
        module_qn=module_qn,
        entities=entities,
        relationships=relationships,
        parent_qn=module_qn,
        parent_type="top",
        locals=_local_names(root),
    )
    _walk(root, ctx, [Visibility.PUBLIC])

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
