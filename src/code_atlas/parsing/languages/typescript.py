"""TypeScript and JavaScript language support — tree-sitter parser."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

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


def _extract_calls(
    node: Node,
    source: bytes,
    from_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Recursively extract call expressions from a function body."""
    for child in node.children:
        if child.type == "call_expression":
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
                elif func.type == "member_expression":
                    prop = func.child_by_field_name("property")
                    if prop is not None:
                        call_name = node_text(prop)
                        relationships.append(
                            ParsedRelationship(
                                from_qualified_name=from_qn,
                                rel_type=RelType.CALLS,
                                to_name=call_name,
                            )
                        )
        # Recurse but don't descend into nested function/class definitions —
        # arrow_function IS recursed into: unlike named function/class
        # declarations, arrow functions passed as callback arguments are never
        # processed as their own entities, so their calls must attribute to
        # the enclosing entity or they are silently dropped.
        if child.type not in (
            "function_declaration",
            "class_declaration",
            "abstract_class_declaration",
        ):
            _extract_calls(child, source, from_qn, relationships)


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
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.IMPORTS,
            to_name=import_source,
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
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process a class_declaration or abstract_class_declaration node."""
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

    qn = f"{module_qn}.{name}"
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

    # DEFINES relationship from module -> class
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    _extract_heritage(node, f"{project_name}:{qn}", relationships)

    # Process class body
    body = node.child_by_field_name("body")
    if body is not None:
        _process_class_body(body, path, source, project_name, module_qn, qn, entities, relationships, seen)


def _process_class_body(
    body: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    class_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process members of a class body."""
    for child in body.children:
        if child.type == "method_definition":
            _process_method(child, path, source, project_name, module_qn, class_qn, entities, relationships, seen)
        elif child.type == "abstract_method_signature":
            _process_abstract_method(child, path, source, project_name, module_qn, class_qn, entities, relationships)
        elif child.type == "public_field_definition":
            _process_class_field(child, path, project_name, module_qn, class_qn, entities, relationships, seen)


def _process_method(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,  # noqa: ARG001
    class_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process a method_definition in a class body."""
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
    qn = f"{class_qn}.{name}"

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

    # DEFINES relationship from class -> method
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{class_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # Extract USES_TYPE from parameter/return type annotations
    _extract_type_refs_ts(node, f"{project_name}:{qn}", relationships)

    # Extract CALLS from method body
    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, source, f"{project_name}:{qn}", relationships)


def _process_abstract_method(
    node: Node,
    path: str,
    source: bytes,  # noqa: ARG001
    project_name: str,
    module_qn: str,  # noqa: ARG001
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
    project_name: str,
    module_qn: str,  # noqa: ARG001
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


def _process_interface(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
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

    qn = f"{module_qn}.{name}"
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
            from_qualified_name=f"{project_name}:{module_qn}",
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
    module_qn: str,
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

    qn = f"{module_qn}.{name}"
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
            from_qualified_name=f"{project_name}:{module_qn}",
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
    module_qn: str,
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

    qn = f"{module_qn}.{name}"
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
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )


def _process_function(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process a function_declaration node."""
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

    qn = f"{module_qn}.{name}"
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
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # Extract USES_TYPE from parameter/return type annotations
    _extract_type_refs_ts(node, f"{project_name}:{qn}", relationships)

    # CALLS from function body
    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, source, f"{project_name}:{qn}", relationships)


def _process_lexical_declaration(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    *,
    is_exported: bool = False,
) -> None:
    """Process a lexical_declaration (const/let) node at module level."""
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
                module_qn,
                entities,
                relationships,
                seen,
                is_const=is_const,
                is_exported=is_exported,
            )


def _process_variable_declaration(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    *,
    is_exported: bool = False,
) -> None:
    """Process a variable_declaration (var) node at module level."""
    for child in node.children:
        if child.type == "variable_declarator":
            _process_variable_declarator(
                child,
                node,
                path,
                source,
                project_name,
                module_qn,
                entities,
                relationships,
                seen,
                is_const=False,
                is_exported=is_exported,
            )


def _process_variable_declarator(
    node: Node,
    parent_decl: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    *,
    is_const: bool,
    is_exported: bool,
) -> None:
    """Process a single variable_declarator within a lexical/variable declaration."""
    name_node = node.child_by_field_name("name")
    if name_node is None or name_node.type != "identifier":
        return
    name = node_text(name_node)
    value_node = node.child_by_field_name("value")

    line_start = parent_decl.start_point[0] + 1
    line_end = parent_decl.end_point[0] + 1

    key = (line_start, name)
    if key in seen:
        return
    seen.add(key)

    # Check if value is an arrow function → treat as function
    if value_node is not None and value_node.type == "arrow_function":
        qn = f"{module_qn}.{name}"
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
                from_qualified_name=f"{project_name}:{module_qn}",
                rel_type=RelType.DEFINES,
                to_name=f"{project_name}:{qn}",
            )
        )

        # Extract USES_TYPE from arrow function type annotations
        _extract_type_refs_ts(value_node, f"{project_name}:{qn}", relationships)

        # Extract CALLS from arrow function body
        body = value_node.child_by_field_name("body")
        if body is not None:
            _extract_calls(body, source, f"{project_name}:{qn}", relationships)
        return

    # Regular value (const/let/var)
    qn = f"{module_qn}.{name}"
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
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )


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
        "lexical_declaration",
        "variable_declaration",
        "import_statement",
    }
)


def _process_export_statement(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Unwrap an export_statement and process the inner declaration."""
    decl = node.child_by_field_name("declaration")
    if decl is not None:
        _process_node(
            decl,
            path,
            source,
            project_name,
            module_qn,
            entities,
            relationships,
            seen,
            is_exported=True,
        )
        return

    # Re-export form: `export { x } from './mod'`, `export * from './mod'`,
    # `export * as ns from './mod'` — no local declaration, just a source module.
    source_node = node.child_by_field_name("source")
    if source_node is not None:
        relationships.append(
            ParsedRelationship(
                from_qualified_name=f"{project_name}:{module_qn}",
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
                module_qn,
                entities,
                relationships,
                seen,
                is_exported=True,
            )


def _process_node(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
    *,
    is_exported: bool = False,
) -> None:
    """Dispatch processing for a single top-level node."""
    node_type = node.type
    args = (node, path, source, project_name, module_qn, entities, relationships, seen)

    if node_type == "export_statement":
        _process_export_statement(*args)
    elif node_type in ("class_declaration", "abstract_class_declaration"):
        _process_class(*args)
    elif node_type == "interface_declaration":
        _process_interface(*args)
    elif node_type == "enum_declaration":
        _process_enum(*args)
    elif node_type == "type_alias_declaration":
        _process_type_alias(*args)
    elif node_type == "function_declaration":
        _process_function(*args)
    elif node_type == "lexical_declaration":
        _process_lexical_declaration(*args, is_exported=is_exported)
    elif node_type == "variable_declaration":
        _process_variable_declaration(*args, is_exported=is_exported)
    elif node_type == "import_statement":
        _process_import(node, project_name, module_qn, relationships)


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

    # Walk top-level children
    for child in root.children:
        _process_node(child, path, source, project_name, module_qn, entities, relationships, seen)

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
        )
    )
except ImportError:
    pass
