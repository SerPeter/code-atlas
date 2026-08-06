"""TypeScript and JavaScript language support — tree-sitter parser."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_qualified_name(file_path: str) -> str:
    """Convert file path to a module qualified name.

    ``src/components/Button.tsx`` -> ``src.components.Button``
    ``src/components/index.ts`` -> ``src.components``  (like Python's ``__init__.py``)
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    parts = list(p.parts)
    if parts:
        filename = parts[-1]
        # Strip extension
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename
        if stem == "index":
            parts = parts[:-1]
        else:
            parts[-1] = stem
    return ".".join(parts)


def _extract_jsdoc(node: Node, source: bytes) -> str | None:
    """Extract JSDoc comment (``/** ... */``) immediately before a declaration node.

    Looks at the previous sibling in the parent's children list.
    """
    prev = node.prev_sibling
    if prev is None:
        return None
    # Also check for export_statement wrapping
    if prev is None and node.parent is not None and node.parent.type == "export_statement":
        prev = node.parent.prev_sibling
    if prev is None or prev.type != "comment":
        return None
    text = source[prev.start_byte : prev.end_byte].decode("utf-8", errors="replace")
    if not text.startswith("/**"):
        return None
    # Strip /** and */ delimiters
    text = text[3:].removesuffix("*/")
    # Clean up leading * on each line
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("* "):
            stripped = stripped[2:]
        elif stripped.startswith("*"):
            stripped = stripped[1:]
        cleaned.append(stripped)
    return "\n".join(cleaned).strip() or None


def _extract_jsdoc_from_export(node: Node, source: bytes) -> str | None:
    """Extract JSDoc from an export_statement's previous sibling, for wrapped decls."""
    parent = node.parent
    if parent is not None and parent.type == "export_statement":
        return _extract_jsdoc(parent, source)
    return _extract_jsdoc(node, source)


def _extract_signature(node: Node, source: bytes) -> str | None:
    """Extract function/method signature (declaration line without the body).

    Works for function_declaration, method_definition, and arrow_function.
    """
    body = node.child_by_field_name("body")
    if body is not None:
        sig_bytes = source[node.start_byte : body.start_byte].rstrip()
        return sig_bytes.decode("utf-8", errors="replace").rstrip("{").rstrip()
    # No body — use full node text (e.g. abstract method signatures)
    return node_text(node)


def _get_visibility(node: Node) -> str:
    """Determine visibility from access modifier keywords on class members."""
    for child in node.children:
        if child.type == "accessibility_modifier":
            modifier = node_text(child).strip()
            if modifier == "private":
                return Visibility.PRIVATE
            if modifier == "protected":
                return Visibility.PROTECTED
            if modifier == "public":
                return Visibility.PUBLIC
    # Check for #private syntax
    name_node = node.child_by_field_name("name")
    if name_node is not None and name_node.type == "private_property_identifier":
        return Visibility.PRIVATE
    return Visibility.PUBLIC


def _get_string_content(string_node: Node) -> str:
    """Extract the text content of a string node (strip quotes)."""
    for child in string_node.children:
        if child.type == "string_fragment":
            return node_text(child)
    # Fallback: strip surrounding quotes
    text = node_text(string_node)
    if len(text) >= 2 and text[0] in ('"', "'", "`") and text[-1] in ('"', "'", "`"):
        return text[1:-1]
    return text


# ---------------------------------------------------------------------------
# Salesforce Lightning Web Components
# ---------------------------------------------------------------------------

_SALESFORCE_APEX_PREFIX = "@salesforce/apex/"
_SALESFORCE_SCHEMA_PREFIX = "@salesforce/schema/"


def _salesforce_import_target(specifier: str) -> str | None:
    """Rewrite an LWC ``@salesforce/*`` module specifier into a graph import target.

    An LWC calls server-side code through pseudo-modules whose *specifier* names
    the target exactly — the cleanest cross-tier edge source Salesforce offers::

        @salesforce/apex/AccountService.getAccounts  ->  apex.AccountService.getAccounts
        @salesforce/schema/Account.Name              ->  sobject.Account

    ``apex.<Class>.<method>`` is the qualified name ``parsing/languages/apex.py``
    stores for Apex members, so ``GraphClient.resolve_imports`` matches it as an
    *internal* import and wires the LWC module straight to the real ``Callable``
    (which then lets CALLS resolution's import strategy resolve the call site
    exactly, instead of guessing by bare name project-wide).

    ``sobject.<Object>`` is the same target the Apex parser emits for SOQL and
    DML, so both tiers meet on one ``ext/sobject.<Object>`` node.  The field half
    of ``Account.Name`` is dropped: object-level is the granularity the Apex side
    can supply, and a half-populated field graph is worse than none.

    Returns ``None`` for every other specifier, including other ``@salesforce/*``
    pseudo-modules (labels, static resources, user context), which stay ordinary
    external imports.
    """
    if specifier.startswith(_SALESFORCE_APEX_PREFIX):
        member = specifier.removeprefix(_SALESFORCE_APEX_PREFIX).strip("/")
        return f"apex.{member}" if member else None
    if specifier.startswith(_SALESFORCE_SCHEMA_PREFIX):
        reference = specifier.removeprefix(_SALESFORCE_SCHEMA_PREFIX).strip("/")
        sobject = reference.split(".")[0]
        return f"sobject.{sobject}" if sobject else None
    return None


def _is_exported(node: Node) -> bool:
    """Check if a declaration node is wrapped in an export_statement."""
    parent = node.parent
    return parent is not None and parent.type == "export_statement"


def _get_decorator_tags(node: Node) -> list[str]:
    """Extract decorator tags from a class or method declaration.

    TypeScript decorators normally appear as children of the declaration node
    itself, but for exported classes (``@Injectable()\nexport class X {}``) the
    grammar attaches them to the wrapping export_statement instead.
    """
    parent = node.parent
    decorator_source = parent if parent is not None and parent.type == "export_statement" else node
    tags: list[str] = []
    for child in decorator_source.children:
        if child.type == "decorator":
            dec_text = node_text(child).lstrip("@").strip()
            tags.append(f"decorator:{dec_text}")
    return tags


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------


_TRANSPARENT_CALLEE_WRAPPERS = frozenset({"await_expression", "non_null_expression"})
"""Nodes that sit between a call and the name it calls without renaming it."""

_NAME_BOUND_FUNCTION_TYPES = frozenset({"arrow_function", "function_expression", "generator_function"})
"""Anonymous function forms that become entities when bound to a name (ADR-0031)."""


def _binding_path(node: Node) -> list[str] | None:
    """The names a developer would chain to reach *node*, or ``None`` if there are none.

    ``const handlers = {…}`` gives ``["handlers"]`` and ``const cfg = {hooks: {…}}``
    gives ``["cfg", "hooks"]``. An object in argument position, in an array, or
    returned inline has no such chain and yields ``None`` — nothing names it, so
    nothing inside it is nameable through it either.
    """
    parts: list[str] = []
    cur = node
    while True:
        parent = cur.parent
        if parent is None:
            return None
        if parent.type in ("variable_declarator", "public_field_definition"):
            name_node = parent.child_by_field_name("name")
            # A destructuring pattern binds no single name — and parses as
            # object_pattern, so it is never the node we walked up from.
            if name_node is None or name_node.type not in ("identifier", "property_identifier"):
                return None
            parts.append(node_text(name_node))
            parts.reverse()
            return parts
        if parent.type != "pair":
            return None
        key = parent.child_by_field_name("key")
        if key is None or key.type not in ("property_identifier", "string"):
            return None
        parts.append(node_text(key) if key.type == "property_identifier" else _get_string_content(key))
        cur = parent.parent
        if cur is None or cur.type != "object":
            return None


def _object_method_prefix(node: Node, owner_qn: str) -> str | None:
    """Qualified-name prefix for a ``method_definition`` outside a class body.

    ADR-0031's test is whether a developer has a name for the thing, and the
    grammar node's spelling is not that test: ``foo({async fetch() {…}})``
    produces a ``method_definition``, but the method hangs off an anonymous
    inline object and nothing can refer to it — it is a callback, exactly like
    the arrow it could have been written as. Only when the enclosing object
    literal (or anonymous ``class`` expression) is itself bound to a name does
    the method inherit one, and then it is named through that binding:
    ``handlers.fetch``.

    ``None`` means unbound, which means no entity.
    """
    container = node.parent
    if container is not None and container.type == "class_body":
        # A *named* class expression goes through _process_class and never
        # arrives here; an anonymous one is nameable only via its container.
        container = container.parent
    if container is None or container.type not in ("object", "class"):
        return None
    path = _binding_path(container)
    return ".".join([owner_qn, *path]) if path is not None else None


def _callee(node: Node | None) -> Node | None:
    """Strip the wrappers a grammar puts between a call and its callee.

    ``x!()`` wraps the callee in ``non_null_expression``, and with explicit type
    arguments the TS grammar puts ``await`` *inside* the call —
    ``await ky.get(u).json<T>()`` parses as
    ``call_expression(function: await_expression(member_expression))``. Both wrap
    a name that is still there; ``(fn())()`` and ``obj[key]()`` do not name
    anything statically and are left unresolved on purpose.
    """
    while node is not None and node.type in _TRANSPARENT_CALLEE_WRAPPERS:
        named = [c for c in node.children if c.is_named]
        node = named[-1] if named else None
    return node


def _emit_call(node: Node, from_qn: str, relationships: list[ParsedRelationship]) -> None:
    """Emit the CALLS relationship for one ``call_expression`` or ``new_expression``.

    ``new Foo()`` is a call to ``Foo``, the same edge ``jvm.py`` emits for
    ``object_creation_expression`` — without it 8% of TypeScript's call nodes
    have no edge at all.
    """
    field = "constructor" if node.type == "new_expression" else "function"
    target = _callee(node.child_by_field_name(field))
    if target is None:
        return
    if target.type == "identifier":
        relationships.append(
            ParsedRelationship(
                from_qualified_name=from_qn,
                rel_type=RelType.CALLS,
                to_name=node_text(target),
            )
        )
    elif target.type == "member_expression":
        prop = target.child_by_field_name("property")
        if prop is not None:
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=from_qn,
                    rel_type=RelType.CALLS,
                    to_name=node_text(prop),
                    properties=call_receiver_props(target.child_by_field_name("object")),
                )
            )


def _visit(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    owner_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    *,
    qualifiable: bool = True,
) -> None:
    """Handle one node inside *owner_qn*'s scope, then descend.

    A form that carries a name of its own opens a new scope and is handed to its
    processor; everything else — an arrow passed as a callback, an object
    literal, an ``if`` block — is descended into with the scope unchanged, so its
    calls attribute to the nearest enclosing *named* scope (ADR-0031).

    ``qualifiable`` is False once an anonymous scope stands between here and
    *owner_qn*, which is what ADR-0032 keys on: the binding a category-2 callable
    owes its name to is then unreachable from *owner_qn*, so the qualified name
    it would take names the enclosing scope rather than the definition.
    """
    kind = node.type
    if kind in _SCOPE_BOUNDARY_TYPES:
        _process_node(
            node, path, source, project_name, owner_qn, entities, relationships, seen, qualifiable=qualifiable
        )
        return
    if kind == "class" and node.child_by_field_name("name") is not None:
        # A named class expression: `globalThis.Headers = class Headers {...}`.
        _process_class(node, path, source, project_name, owner_qn, entities, relationships, seen)
        return
    if kind == "method_definition":
        # An object-literal method — `{async fetch() {...}}`, `{get duplex() {...}}`.
        # Class bodies are consumed by _process_class_body and never reach here,
        # except for an anonymous `class {...}` expression.
        prefix = _object_method_prefix(node, owner_qn) if qualifiable else None
        if prefix is not None:
            _process_method(node, path, source, project_name, owner_qn, entities, relationships, seen, qn_prefix=prefix)
            return
        # Unbound, or bound only inside an anonymous scope: a callback with a
        # method's spelling. No entity, and its calls belong to the scope that
        # wrote it — the same treatment an arrow gets. The body is itself an
        # anonymous scope, so nothing declared inside it is qualifiable either.
        _walk_scope(node, path, source, project_name, owner_qn, entities, relationships, seen, qualifiable=False)
        return
    if kind in ("call_expression", "new_expression"):
        _emit_call(node, f"{project_name}:{owner_qn}", relationships)
    if kind in _NAME_BOUND_FUNCTION_TYPES:
        # Reached transparently, so no declarator claimed it: this is a callback,
        # and everything lexically inside it is one anonymous link away from
        # *owner_qn*.
        _walk_scope(node, path, source, project_name, owner_qn, entities, relationships, seen, qualifiable=False)
        return
    _walk_scope(node, path, source, project_name, owner_qn, entities, relationships, seen, qualifiable=qualifiable)


def _walk_scope(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    owner_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    *,
    qualifiable: bool = True,
) -> None:
    """Visit every child of *node* in *owner_qn*'s scope."""
    for child in node.children:
        _visit(child, path, source, project_name, owner_qn, entities, relationships, seen, qualifiable=qualifiable)


def _process_import(
    node: Node,
    project_name: str,
    module_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Process an import_statement, emitting IMPORTS relationships."""
    source_node = node.child_by_field_name("source")
    if source_node is None:
        # Find the string node among children
        for child in node.children:
            if child.type == "string":
                source_node = child
                break
    if source_node is None:
        return

    # Detect `import type` syntax — tree-sitter-typescript has a "type" keyword child
    is_type_import = any(child.type == "type" for child in node.children)
    props: dict[str, Any] = {"type_only": True} if is_type_import else {}

    import_source = _get_string_content(source_node)
    # LWC's @salesforce/* pseudo-modules are rewritten, not duplicated: emitting
    # the raw specifier too would leave a second, unjoinable ext/ stub next to the
    # resolved target.
    salesforce_target = _salesforce_import_target(import_source)
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.IMPORTS,
            to_name=salesforce_target or import_source,
            properties=props,
        )
    )


def _heritage_type_name(node: Node) -> str | None:
    """Resolve a heritage clause child to a bare type/interface name.

    Handles plain identifiers, qualified names (``ns.Base``, ``ns.deep.IFace``),
    and generic instantiations (``IRepo<User>``) by taking the outermost or
    last-segment name — matching the bare-name IMPLEMENTS/INHERITS contract.
    """
    if node.type in ("identifier", "type_identifier"):
        return node_text(node)
    if node.type == "member_expression":
        prop = node.child_by_field_name("property")
        return node_text(prop) if prop is not None else None
    if node.type == "nested_type_identifier":
        for child in node.children:
            if child.type == "type_identifier":
                return node_text(child)
        return None
    if node.type == "generic_type":
        name_node = node.child_by_field_name("name")
        return _heritage_type_name(name_node) if name_node is not None else None
    return None


def _extract_heritage(node: Node, from_qn: str, relationships: list[ParsedRelationship]) -> None:
    """Extract extends/implements relationships from a class_heritage child."""
    for child in node.children:
        if child.type != "class_heritage":
            continue
        for clause in child.children:
            if clause.type == "extends_clause":
                relationships.extend(
                    ParsedRelationship(from_qualified_name=from_qn, rel_type=RelType.INHERITS, to_name=name)
                    for base in clause.children
                    if (name := _heritage_type_name(base)) is not None
                )
            elif clause.type == "implements_clause":
                relationships.extend(
                    ParsedRelationship(from_qualified_name=from_qn, rel_type=RelType.IMPLEMENTS, to_name=name)
                    for iface in clause.children
                    if (name := _heritage_type_name(iface)) is not None
                )


def _extract_interface_heritage(node: Node, from_qn: str, relationships: list[ParsedRelationship]) -> None:
    """Extract INHERITS relationships from an interface's extends_type_clause."""
    for child in node.children:
        if child.type != "extends_type_clause":
            continue
        relationships.extend(
            ParsedRelationship(from_qualified_name=from_qn, rel_type=RelType.INHERITS, to_name=name)
            for base in child.children
            if (name := _heritage_type_name(base)) is not None
        )


def _process_class(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    owner_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process a class_declaration, abstract_class_declaration, or named class expression."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    key = (line_start, name)
    if key in seen:
        return
    seen.add(key)

    qn = f"{owner_qn}.{name}"
    docstring = _extract_jsdoc_from_export(node, source)
    tags = _get_decorator_tags(node)

    is_abstract = node.type == "abstract_class_declaration"
    if is_abstract:
        tags = [*tags, "abstract"]

    if _is_exported(node):
        tags = [*tags, "exported"]

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
            tags=tags,
        )
    )

    # DEFINES relationship from the enclosing scope -> class
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{owner_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    _extract_heritage(node, f"{project_name}:{qn}", relationships)

    # Process class body
    body = node.child_by_field_name("body")
    if body is not None:
        _process_class_body(body, path, source, project_name, qn, entities, relationships, seen)


def _process_class_body(
    body: Node,
    path: str,
    source: bytes,
    project_name: str,
    class_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process members of a class body."""
    for child in body.children:
        if child.type == "method_definition":
            _process_method(child, path, source, project_name, class_qn, entities, relationships, seen)
        elif child.type == "abstract_method_signature":
            _process_abstract_method(child, path, project_name, class_qn, entities, relationships)
        elif child.type == "public_field_definition":
            _process_class_field(child, path, source, project_name, class_qn, entities, relationships, seen)


def _process_method(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    owner_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    *,
    qn_prefix: str | None = None,
) -> None:
    """Process a method_definition — a class member, or a bound object-literal method.

    ``qn_prefix`` names the method when the thing that owns it is not the thing
    that declares it: an object literal held by ``const handlers`` is named
    ``handlers.fetch``, but it is the enclosing *scope* that defines it, since
    the binding itself may be a local with no node of its own.
    """
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    key = (line_start, name)
    if key in seen:
        return
    seen.add(key)

    kind = CallableKind.CONSTRUCTOR if name == "constructor" else CallableKind.METHOD

    # Check for static
    is_static = False
    for child in node.children:
        if child.type == "static":
            is_static = True
            break
    if is_static:
        kind = CallableKind.STATIC_METHOD

    visibility = _get_visibility(node)
    tags: list[str] = _get_decorator_tags(node)

    # Check for async
    for child in node.children:
        if child.type == "async":
            tags = [*tags, "async"]
            break

    docstring = _extract_jsdoc(node, source)
    signature = _extract_signature(node, source)
    qn = f"{qn_prefix or owner_qn}.{name}"

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
            visibility=visibility,
            tags=tags,
        )
    )

    # DEFINES relationship from the owning class (or enclosing scope) -> method
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{owner_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # Extract USES_TYPE from parameter/return type annotations
    _extract_type_refs_ts(node, f"{project_name}:{qn}", relationships)

    # CALLS from the whole method, so a default parameter value counts too
    _walk_scope(node, path, source, project_name, qn, entities, relationships, seen)


def _process_abstract_method(
    node: Node,
    path: str,
    project_name: str,
    class_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process an abstract_method_signature in a class body."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    visibility = _get_visibility(node)
    signature = node_text(node)
    qn = f"{class_qn}.{name}"

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.CALLABLE,
            kind=CallableKind.METHOD,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            signature=signature,
            source=node_text(node),
            visibility=visibility,
            tags=["abstract"],
        )
    )

    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{class_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )


def _process_class_field(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    class_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process a public_field_definition in a class body."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    key = (line_start, name)
    if key in seen:
        return
    seen.add(key)

    visibility = _get_visibility(node)
    qn = f"{class_qn}.{name}"

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.VALUE,
            kind=ValueKind.FIELD,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            source=node_text(node),
            visibility=visibility,
        )
    )

    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{class_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # A field initialiser runs when the class is constructed, so its calls — and
    # any callback it installs — belong to the class.
    value_node = node.child_by_field_name("value")
    if value_node is None:
        return
    if value_node.type in _NAME_BOUND_FUNCTION_TYPES:
        # `handler = () => {…}` is named through the field, so the arrow is not
        # an anonymous link: walk its body rather than routing it back through
        # _visit, which would read it as an unclaimed callback.
        _walk_scope(value_node, path, source, project_name, class_qn, entities, relationships, seen)
        return
    _visit(value_node, path, source, project_name, class_qn, entities, relationships, seen)


def _process_interface(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    owner_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process an interface_declaration node."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    key = (line_start, name)
    if key in seen:
        return
    seen.add(key)

    qn = f"{owner_qn}.{name}"
    docstring = _extract_jsdoc_from_export(node, source)
    tags: list[str] = []
    if _is_exported(node):
        tags.append("exported")

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=TypeDefKind.INTERFACE,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            visibility=Visibility.PUBLIC,
            tags=tags,
        )
    )

    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{owner_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    _extract_interface_heritage(node, f"{project_name}:{qn}", relationships)


def _process_enum(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    owner_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process an enum_declaration node."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    key = (line_start, name)
    if key in seen:
        return
    seen.add(key)

    qn = f"{owner_qn}.{name}"
    docstring = _extract_jsdoc_from_export(node, source)
    tags: list[str] = []
    if _is_exported(node):
        tags.append("exported")

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=TypeDefKind.ENUM,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            visibility=Visibility.PUBLIC,
            tags=tags,
        )
    )

    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{owner_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # Enum members
    body = node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            member_name: str | None = None
            if child.type == "property_identifier":
                member_name = node_text(child)
            elif child.type == "enum_assignment":
                member_name_node = child.child_by_field_name("name")
                if member_name_node is not None:
                    member_name = node_text(member_name_node)
            if member_name is not None:
                member_qn = f"{qn}.{member_name}"
                member_line = child.start_point[0] + 1
                entities.append(
                    ParsedEntity(
                        name=member_name,
                        qualified_name=f"{project_name}:{member_qn}",
                        label=NodeLabel.VALUE,
                        kind=ValueKind.ENUM_MEMBER,
                        line_start=member_line,
                        line_end=child.end_point[0] + 1,
                        file_path=path,
                        source=node_text(child),
                        visibility=Visibility.PUBLIC,
                    )
                )
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=f"{project_name}:{qn}",
                        rel_type=RelType.DEFINES,
                        to_name=f"{project_name}:{member_qn}",
                    )
                )


def _process_type_alias(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    owner_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process a type_alias_declaration node."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    key = (line_start, name)
    if key in seen:
        return
    seen.add(key)

    qn = f"{owner_qn}.{name}"
    docstring = _extract_jsdoc_from_export(node, source)
    tags: list[str] = []
    if _is_exported(node):
        tags.append("exported")

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=TypeDefKind.TYPE_ALIAS,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            source=node_text(node),
            visibility=Visibility.PUBLIC,
            tags=tags,
        )
    )

    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{owner_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )


def _process_function(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    owner_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process a function_declaration or generator_function_declaration node.

    Nested declarations reach here too — ``owner_qn`` is then the enclosing
    function rather than the module, so ``function abortHandler()`` declared
    inside ``delay`` is named ``…delay.abortHandler``.
    """
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    key = (line_start, name)
    if key in seen:
        return
    seen.add(key)

    qn = f"{owner_qn}.{name}"
    docstring = _extract_jsdoc_from_export(node, source)
    signature = _extract_signature(node, source)
    tags: list[str] = []

    # Check for async
    for child in node.children:
        if child.type == "async":
            tags.append("async")
            break

    if _is_exported(node):
        tags.append("exported")

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.CALLABLE,
            kind=CallableKind.FUNCTION,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            signature=signature,
            source=node_text(node),
            visibility=Visibility.PUBLIC,
            tags=tags,
        )
    )

    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{owner_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # Extract USES_TYPE from parameter/return type annotations
    _extract_type_refs_ts(node, f"{project_name}:{qn}", relationships)

    # CALLS from the whole declaration, so a default parameter value counts too
    _walk_scope(node, path, source, project_name, qn, entities, relationships, seen)


def _process_lexical_declaration(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    owner_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    *,
    is_exported: bool = False,
    qualifiable: bool = True,
) -> None:
    """Process a lexical_declaration (const/let) node."""
    # Determine const vs let
    is_const = False
    for child in node.children:
        if child.type == "const":
            is_const = True
            break

    for child in node.children:
        if child.type == "variable_declarator":
            _process_variable_declarator(
                child,
                node,
                path,
                source,
                project_name,
                owner_qn,
                entities,
                relationships,
                seen,
                is_const=is_const,
                is_exported=is_exported,
                qualifiable=qualifiable,
            )


def _process_variable_declaration(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    owner_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    *,
    is_exported: bool = False,
    qualifiable: bool = True,
) -> None:
    """Process a variable_declaration (var) node."""
    for child in node.children:
        if child.type == "variable_declarator":
            _process_variable_declarator(
                child,
                node,
                path,
                source,
                project_name,
                owner_qn,
                entities,
                relationships,
                seen,
                is_const=False,
                is_exported=is_exported,
                qualifiable=qualifiable,
            )


def _process_variable_declarator(
    node: Node,
    parent_decl: Node,
    path: str,
    source: bytes,
    project_name: str,
    owner_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    *,
    is_const: bool,
    is_exported: bool,
    qualifiable: bool = True,
) -> None:
    """Process a single variable_declarator within a lexical/variable declaration.

    Four outcomes, in the order the checks run:

    * bound to a function form under a chain of named scopes — a Callable. The
      grammar calls the value anonymous; the codebase calls it by the binding
      (ADR-0031 category 2).
    * bound to a function form inside an anonymous scope — no entity (ADR-0032),
      see below.
    * bound to anything else at module scope — a Value, as before.
    * anything else — no entity. A local ``const`` is not worth a graph node,
      but its initialiser still runs, so the tail below walks it either way.
    """
    name_node = node.child_by_field_name("name")
    value_node = node.child_by_field_name("value")
    name = node_text(name_node) if name_node is not None and name_node.type == "identifier" else None

    line_start = parent_decl.start_point[0] + 1
    line_end = parent_decl.end_point[0] + 1

    if name is None or (line_start, name) in seen:
        # A destructuring pattern names no single thing, and a re-declaration is
        # already recorded — but the initialiser's calls are still made.
        _walk_value(node, path, source, project_name, owner_qn, entities, relationships, seen, qualifiable=qualifiable)
        return
    seen.add((line_start, name))

    if value_node is not None and value_node.type in _NAME_BOUND_FUNCTION_TYPES:
        if not qualifiable:
            # ADR-0032: the binding is real, but every scope between it and
            # *owner_qn* must be named for the binding to name a definition, and
            # one of them is a callback. `test('a', async t => {const customFetch
            # = …})` written eight times in a file yields eight bodies and one
            # uid, which upsert into a single node holding an arbitrary winner's
            # source and the union of every edge set — a confident wrong answer,
            # worse than the silence of no entity. ADR-0031 forbids the escape of
            # a positional name, so the entity is declined instead.
            #
            # The body is still walked, so its calls reach the graph attributed
            # to the nearest named scope; only the node and its USES_TYPE edges
            # go, which is the same treatment an unbound object method gets.
            _walk_scope(
                value_node,
                path,
                source,
                project_name,
                owner_qn,
                entities,
                relationships,
                seen,
                qualifiable=False,
            )
            return
        qn = f"{owner_qn}.{name}"
        docstring = _extract_jsdoc_from_export(parent_decl, source)
        signature = _extract_signature(value_node, source)
        # Prepend the name to the signature for readability
        if signature:
            prefix = "const " if is_const else "let "
            signature = f"{prefix}{name} = {signature}"
        tags: list[str] = []

        for child in value_node.children:
            if child.type == "async":
                tags.append("async")
                break

        if is_exported:
            tags.append("exported")

        entities.append(
            ParsedEntity(
                name=name,
                qualified_name=f"{project_name}:{qn}",
                label=NodeLabel.CALLABLE,
                kind=CallableKind.FUNCTION,
                line_start=line_start,
                line_end=line_end,
                file_path=path,
                docstring=docstring,
                signature=signature,
                source=node_text(parent_decl),
                visibility=Visibility.PUBLIC,
                tags=tags,
            )
        )

        relationships.append(
            ParsedRelationship(
                from_qualified_name=f"{project_name}:{owner_qn}",
                rel_type=RelType.DEFINES,
                to_name=f"{project_name}:{qn}",
            )
        )

        # Extract USES_TYPE from the function's type annotations
        _extract_type_refs_ts(value_node, f"{project_name}:{qn}", relationships)

        # CALLS from the function itself, which now owns everything inside it
        _walk_scope(value_node, path, source, project_name, qn, entities, relationships, seen)
        return

    if _at_module_scope(parent_decl):
        qn = f"{owner_qn}.{name}"
        kind = ValueKind.CONSTANT if is_const else ValueKind.VARIABLE
        tags_val: list[str] = []
        if is_exported:
            tags_val.append("exported")

        entities.append(
            ParsedEntity(
                name=name,
                qualified_name=f"{project_name}:{qn}",
                label=NodeLabel.VALUE,
                kind=kind,
                line_start=line_start,
                line_end=line_end,
                file_path=path,
                source=node_text(parent_decl),
                visibility=Visibility.PUBLIC,
                tags=tags_val,
            )
        )

        relationships.append(
            ParsedRelationship(
                from_qualified_name=f"{project_name}:{owner_qn}",
                rel_type=RelType.DEFINES,
                to_name=f"{project_name}:{qn}",
            )
        )

    _walk_value(node, path, source, project_name, owner_qn, entities, relationships, seen, qualifiable=qualifiable)


def _walk_value(
    declarator: Node,
    path: str,
    source: bytes,
    project_name: str,
    owner_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    *,
    qualifiable: bool = True,
) -> None:
    """Walk a declarator's initialiser, attributing its calls to *owner_qn*.

    An initialiser that is not itself a function runs in the scope that declares
    it — ``const x = compute()`` at module scope is a call the module makes, and
    ``export const supports = (() => {…})()`` is a whole IIFE of them.
    """
    value_node = declarator.child_by_field_name("value")
    if value_node is not None:
        _visit(value_node, path, source, project_name, owner_qn, entities, relationships, seen, qualifiable=qualifiable)


# ---------------------------------------------------------------------------
# USES_TYPE extraction
# ---------------------------------------------------------------------------

_TS_BUILTIN_TYPES: frozenset[str] = frozenset(
    {
        "string",
        "number",
        "boolean",
        "void",
        "null",
        "undefined",
        "any",
        "never",
        "unknown",
        "object",
        "symbol",
        "bigint",
    }
)


def _collect_type_names_ts(node: Node) -> list[str]:
    """Extract non-builtin type names from a TypeScript type annotation node."""
    names: list[str] = []
    _walk_type_node_ts(node, names)
    return names


def _walk_type_node_ts(node: Node, names: list[str]) -> None:
    """Recursively walk a TS type annotation to collect type identifiers."""
    if node.type in ("type_identifier", "identifier"):
        name = node_text(node)
        if name not in _TS_BUILTIN_TYPES:
            names.append(name)
    elif node.type == "nested_type_identifier":
        # e.g., Namespace.Type — take the last part
        for child in node.children:
            if child.type == "type_identifier":
                name = node_text(child)
                if name not in _TS_BUILTIN_TYPES:
                    names.append(name)
    else:
        for child in node.children:
            _walk_type_node_ts(child, names)


def _extract_type_refs_ts(
    node: Node,
    from_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Extract USES_TYPE relationships from TS function parameter and return type annotations."""
    seen_types: set[str] = set()

    # Parameter type annotations
    params = node.child_by_field_name("parameters")
    if params is not None:
        for param in params.children:
            # TypeScript parameters have type_annotation children
            for child in param.children:
                if child.type == "type_annotation":
                    for name in _collect_type_names_ts(child):
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
        for name in _collect_type_names_ts(return_type):
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
# Top-level node dispatcher
# ---------------------------------------------------------------------------

# Node types that represent declarations we process directly
_DECLARATION_TYPES = frozenset(
    {
        "class_declaration",
        "abstract_class_declaration",
        "interface_declaration",
        "enum_declaration",
        "type_alias_declaration",
        "function_declaration",
        "generator_function_declaration",
        "lexical_declaration",
        "variable_declaration",
        "import_statement",
    }
)

# What _visit hands to _process_node instead of walking through. Every one of
# these carries its own name, so it opens a scope; anything else is transparent
# and its calls belong to whatever encloses it.
_SCOPE_BOUNDARY_TYPES = _DECLARATION_TYPES | {"export_statement"}


def _at_module_scope(node: Node) -> bool:
    """Is *node* a statement of the program itself, rather than of some body?

    Only these declare something the module owns. A ``const`` inside a function
    (or inside a top-level ``if``) is a local: walked for its calls, but not
    worth a graph node of its own.
    """
    parent = node.parent
    if parent is not None and parent.type == "export_statement":
        parent = parent.parent
    return parent is not None and parent.type == "program"


def _process_export_statement(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    owner_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    *,
    qualifiable: bool = True,
) -> None:
    """Unwrap an export_statement and process the inner declaration."""
    decl = node.child_by_field_name("declaration")
    if decl is not None:
        _process_node(
            decl,
            path,
            source,
            project_name,
            owner_qn,
            entities,
            relationships,
            seen,
            is_exported=True,
            qualifiable=qualifiable,
        )
        return

    # Re-export form: `export { x } from './mod'`, `export * from './mod'`,
    # `export * as ns from './mod'` — no local declaration, just a source module.
    source_node = node.child_by_field_name("source")
    if source_node is not None:
        relationships.append(
            ParsedRelationship(
                from_qualified_name=f"{project_name}:{owner_qn}",
                rel_type=RelType.IMPORTS,
                to_name=_get_string_content(source_node),
            )
        )
        return

    for child in node.children:
        if child.type in _DECLARATION_TYPES:
            _process_node(
                child,
                path,
                source,
                project_name,
                owner_qn,
                entities,
                relationships,
                seen,
                is_exported=True,
                qualifiable=qualifiable,
            )
        else:
            # `export default <expression>` declares nothing, but the expression
            # is real code — an arrow, a call, an object full of methods.
            _visit(child, path, source, project_name, owner_qn, entities, relationships, seen, qualifiable=qualifiable)


def _process_node(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    owner_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    *,
    is_exported: bool = False,
    qualifiable: bool = True,
) -> None:
    """Dispatch processing for one declaration inside *owner_qn*'s scope.

    ``qualifiable`` reaches only the binding-named forms. A ``function``,
    ``class``, ``interface``, ``enum`` or ``type`` declares its own name rather
    than borrowing a variable's, so ADR-0031 category 1 keeps its entity wherever
    it sits — ``delay``'s ``function abortHandler()`` lives inside a callback and
    is still ``delay.abortHandler`` — and it reopens a named scope for whatever
    it contains.
    """
    node_type = node.type
    args = (node, path, source, project_name, owner_qn, entities, relationships, seen)

    if node_type == "export_statement":
        _process_export_statement(*args, qualifiable=qualifiable)
    elif node_type in ("class_declaration", "abstract_class_declaration"):
        _process_class(*args)
    elif node_type == "interface_declaration":
        _process_interface(*args)
    elif node_type == "enum_declaration":
        _process_enum(*args)
    elif node_type == "type_alias_declaration":
        _process_type_alias(*args)
    elif node_type in ("function_declaration", "generator_function_declaration"):
        _process_function(*args)
    elif node_type == "lexical_declaration":
        _process_lexical_declaration(*args, is_exported=is_exported, qualifiable=qualifiable)
    elif node_type == "variable_declaration":
        _process_variable_declaration(*args, is_exported=is_exported, qualifiable=qualifiable)
    elif node_type == "import_statement":
        # An import statement only ever appears at the top level, so the scope
        # that owns it is the module.
        _process_import(node, project_name, owner_qn, relationships)


# ---------------------------------------------------------------------------
# Main parse entry point
# ---------------------------------------------------------------------------


def _parse_typescript(
    path: str,
    source: bytes,
    root: Node,
    project_name: str,
) -> ParsedFile:
    """Extract entities and relationships from a TypeScript/JavaScript parse tree."""
    module_qn = _module_qualified_name(path)

    entities: list[ParsedEntity] = []
    relationships: list[ParsedRelationship] = []
    seen: set[tuple[int, str]] = set()

    # Determine language name from file extension
    posix_path = path.replace("\\", "/")
    suffix = PurePosixPath(posix_path).suffix.lower()
    language = "typescript" if suffix in (".ts", ".tsx") else "javascript"

    # Module entity
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

    # Walk the whole tree. The module owns every call that no named callable
    # encloses — import-time work, test-file setup, an IIFE's insides — which on
    # a real codebase is 9% of all call sites (ADR-0031).
    _walk_scope(root, path, source, project_name, module_qn, entities, relationships, seen)

    return ParsedFile(
        file_path=path,
        language=language,
        entities=entities,
        relationships=relationships,
    )


# ---------------------------------------------------------------------------
# Language registration
# ---------------------------------------------------------------------------

try:
    import tree_sitter_typescript as _ts_ts
    from tree_sitter import Language, Query

    _TS_LANGUAGE = Language(_ts_ts.language_typescript())
    _TS_QUERY = Query(_TS_LANGUAGE, "(program) @root")

    register_language(
        LanguageConfig(
            name="typescript",
            extensions=frozenset({".ts"}),
            language=_TS_LANGUAGE,
            query=_TS_QUERY,
            parse_func=_parse_typescript,
            comment_node_types=frozenset({"comment"}),
        )
    )

    # .tsx needs the separate TSX grammar — the plain typescript grammar has no
    # JSX productions (and the tsx grammar conflicts with old-style <T>expr
    # type assertions, so .ts stays on language_typescript).
    _TSX_LANGUAGE = Language(_ts_ts.language_tsx())
    _TSX_QUERY = Query(_TSX_LANGUAGE, "(program) @root")

    register_language(
        LanguageConfig(
            name="tsx",
            extensions=frozenset({".tsx"}),
            language=_TSX_LANGUAGE,
            query=_TSX_QUERY,
            parse_func=_parse_typescript,
            comment_node_types=frozenset({"comment"}),
        )
    )
except ImportError:
    pass

try:
    import tree_sitter_javascript as _ts_js
    from tree_sitter import Language as _Language
    from tree_sitter import Query as _Query

    _JS_LANGUAGE = _Language(_ts_js.language())
    _JS_QUERY = _Query(_JS_LANGUAGE, "(program) @root")

    register_language(
        LanguageConfig(
            name="javascript",
            extensions=frozenset({".js", ".jsx", ".mjs", ".cjs"}),
            language=_JS_LANGUAGE,
            query=_JS_QUERY,
            parse_func=_parse_typescript,
            comment_node_types=frozenset({"comment"}),
        )
    )
except ImportError:
    pass
