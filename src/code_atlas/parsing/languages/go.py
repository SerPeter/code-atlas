"""Go language support -- tree-sitter parser for Go source files."""

from __future__ import annotations

import logging
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

_log = logging.getLogger(__name__)

try:
    import tree_sitter_go as ts_go  # type: ignore[unresolved-import]
    from tree_sitter import Language, Query

    _GO_LANGUAGE = Language(ts_go.language())
    _GO_QUERY = Query(_GO_LANGUAGE, "(source_file) @root")
    _HAS_GO = True
except ImportError:
    _HAS_GO = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_qualified_name(file_path: str) -> str:
    """Convert a Go file path to a module qualified name.

    ``src/server.go`` -> ``src.server``
    ``cmd/main_test.go`` -> ``cmd.main_test``
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    parts = list(p.parts)
    if parts and parts[-1].endswith(".go"):
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def _visibility_from_name(name: str) -> str:
    """Determine visibility from Go capitalization conventions.

    Uppercase first letter -> PUBLIC, lowercase -> INTERNAL.
    """
    if not name:
        return Visibility.INTERNAL
    if name[0].isupper():
        return Visibility.PUBLIC
    return Visibility.INTERNAL


def _extract_doc_comment(node: Node, source: bytes) -> str | None:
    """Extract Go doc comments (contiguous // lines immediately before a declaration).

    Go doc comments are contiguous single-line ``//`` comments directly
    preceding a declaration, with no blank lines in between.
    """
    lines: list[str] = []
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment":
        text = source[prev.start_byte : prev.end_byte].decode("utf-8", errors="replace")
        # Only collect // comments (not /* */ block comments)
        if text.startswith("//"):
            comment_body = text[2:].strip()
            lines.append(comment_body)
        else:
            break
        prev = prev.prev_sibling
    if not lines:
        return None
    lines.reverse()
    return "\n".join(lines)


def _extract_function_signature(node: Node, source: bytes) -> str | None:
    """Extract the function/method signature line without the body."""
    body = node.child_by_field_name("body")
    if body is not None:
        sig_bytes = source[node.start_byte : body.start_byte].rstrip()
        return sig_bytes.decode("utf-8", errors="replace")
    # No body (e.g. interface method spec or extern func)
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _extract_receiver_type(node: Node) -> str | None:
    """Extract the receiver type name from a method_declaration.

    ``func (s *Server) Handle(...)`` -> ``Server``
    ``func (s Server) Handle(...)`` -> ``Server``
    """
    if node.type != "method_declaration":
        return None
    # The first parameter_list child is the receiver
    for child in node.children:
        if child.type == "parameter_list":
            # Look inside for the type
            for param in child.children:
                if param.type == "parameter_declaration":
                    type_node = param.child_by_field_name("type")
                    if type_node is not None:
                        return _type_name_from_node(type_node)
            break
    return None


def _type_name_from_node(type_node: Node) -> str:
    """Extract the simple type name from a type node, stripping pointers."""
    if type_node.type == "pointer_type":
        # *Server -> Server
        for child in type_node.children:
            if child.type == "type_identifier":
                return node_text(child)
        # Fallback: recurse into nested pointers
        for child in type_node.children:
            if child.type == "pointer_type":
                return _type_name_from_node(child)
    if type_node.type == "type_identifier":
        return node_text(type_node)
    return node_text(type_node)


def _detect_build_tags(root: Node, source: bytes) -> list[str]:
    """Detect //go:build and //go:generate directives from top-level comments."""
    tags: list[str] = []
    for child in root.children:
        if child.type != "comment":
            # Only scan leading comments
            if child.type != "package_clause":
                continue
            continue
        text = source[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
        if text.startswith("//go:build "):
            constraint = text[len("//go:build ") :].strip()
            tags.append(f"build_tag:{constraint}")
        elif text.startswith("//go:generate "):
            tags.append("go_generate")
    return tags


def _is_embedding(field_node: Node) -> str | None:
    """Check if a struct field_declaration is an embedding (anonymous field).

    An embedded field has no ``name`` field -- just a type.
    Returns the embedded type name or None.

    Patterns:
    - ``field_declaration`` with only a type child (type_identifier or qualified_type)
    - For ``qualified_type`` (e.g. ``fmt.Stringer``), return the type part
    """
    # If the field has named children with field_identifier, it's not embedding
    for child in field_node.children:
        if child.type == "field_identifier":
            return None

    # Look for type_identifier or qualified_type without a name
    for child in field_node.children:
        if child.type == "type_identifier":
            return node_text(child)
        if child.type == "qualified_type":
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                return node_text(name_node)
            return node_text(child)
        if child.type == "pointer_type":
            return _type_name_from_node(child)
    return None


# ---------------------------------------------------------------------------
# Extract calls
# ---------------------------------------------------------------------------


def _extract_calls(
    node: Node,
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
                elif func.type == "selector_expression":
                    # e.g. fmt.Println -> extract "Println"
                    field_node = func.child_by_field_name("field")
                    if field_node is not None:
                        call_name = node_text(field_node)
                        relationships.append(
                            ParsedRelationship(
                                from_qualified_name=from_qn,
                                rel_type=RelType.CALLS,
                                to_name=call_name,
                            )
                        )
        # Recurse but don't descend into nested function literals
        if child.type != "func_literal":
            _extract_calls(child, from_qn, relationships)


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------


def _parse_go(
    path: str,
    source: bytes,
    root: Node,
    project_name: str,
) -> ParsedFile:
    """Extract entities and relationships from a Go parse tree."""
    module_qn = _module_qualified_name(path)
    entities: list[ParsedEntity] = []
    relationships: list[ParsedRelationship] = []

    # Detect file-level build tags from comments before package clause
    file_tags = _detect_build_tags(root, source)

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
            tags=file_tags,
        )
    )

    # Walk top-level declarations
    for child in root.children:
        if child.type == "import_declaration":
            _process_import(child, project_name, module_qn, relationships)
        elif child.type == "type_declaration":
            _process_type_declaration(child, path, source, project_name, module_qn, entities, relationships)
        elif child.type == "function_declaration":
            _process_function(child, path, source, project_name, module_qn, entities, relationships)
        elif child.type == "method_declaration":
            _process_method(child, path, source, project_name, module_qn, entities, relationships)
        elif child.type == "const_declaration":
            _process_const_var(child, path, project_name, module_qn, entities, relationships, is_const=True)
        elif child.type == "var_declaration":
            _process_const_var(child, path, project_name, module_qn, entities, relationships, is_const=False)

    return ParsedFile(
        file_path=path,
        language="go",
        entities=entities,
        relationships=relationships,
    )


# ---------------------------------------------------------------------------
# Declaration processors
# ---------------------------------------------------------------------------


def _process_import(
    node: Node,
    project_name: str,
    module_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Process import_declaration (single or grouped)."""
    for child in node.children:
        if child.type == "import_spec":
            _process_import_spec(child, project_name, module_qn, relationships)
        elif child.type == "import_spec_list":
            for spec in child.children:
                if spec.type == "import_spec":
                    _process_import_spec(spec, project_name, module_qn, relationships)


def _process_import_spec(
    spec: Node,
    project_name: str,
    module_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Process a single import_spec node."""
    # The path is an interpreted_string_literal child
    for child in spec.children:
        if child.type == "interpreted_string_literal":
            import_path = node_text(child).strip('"')
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=f"{project_name}:{module_qn}",
                    rel_type=RelType.IMPORTS,
                    to_name=import_path,
                )
            )
            return


def _process_type_declaration(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process type_declaration containing type_spec or type_alias."""
    for child in node.children:
        if child.type == "type_spec":
            _process_type_spec(child, node, path, source, project_name, module_qn, entities, relationships)
        elif child.type == "type_alias":
            _process_type_alias(child, node, path, source, project_name, module_qn, entities, relationships)


def _process_type_spec(
    spec: Node,
    decl: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a type_spec (struct, interface, or other named type)."""
    name_node = spec.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    type_node = spec.child_by_field_name("type")

    docstring = _extract_doc_comment(decl, source)
    qn = f"{module_qn}.{name}"
    line_start = decl.start_point[0] + 1
    line_end = decl.end_point[0] + 1

    if type_node is not None and type_node.type == "struct_type":
        kind = TypeDefKind.STRUCT
    elif type_node is not None and type_node.type == "interface_type":
        kind = TypeDefKind.INTERFACE
    else:
        kind = TypeDefKind.TYPE_ALIAS

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
        )
    )

    # DEFINES: module -> type
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # Process struct fields
    if type_node is not None and type_node.type == "struct_type":
        _process_struct_fields(type_node, path, source, project_name, module_qn, name, entities, relationships)

    # Process interface method specs
    if type_node is not None and type_node.type == "interface_type":
        _process_interface_methods(type_node, path, source, project_name, module_qn, name, entities, relationships)


def _process_type_alias(
    alias_node: Node,
    decl: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a type_alias (type Alias = int)."""
    name_node = alias_node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    docstring = _extract_doc_comment(decl, source)
    qn = f"{module_qn}.{name}"

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=TypeDefKind.TYPE_ALIAS,
            line_start=decl.start_point[0] + 1,
            line_end=decl.end_point[0] + 1,
            file_path=path,
            docstring=docstring,
            visibility=_visibility_from_name(name),
        )
    )

    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )


def _process_struct_fields(
    struct_node: Node,
    path: str,
    source: bytes,  # noqa: ARG001
    project_name: str,
    module_qn: str,
    struct_name: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process struct fields, including embeddings."""
    struct_qn = f"{module_qn}.{struct_name}"
    # Find the field_declaration_list
    for child in struct_node.children:
        if child.type == "field_declaration_list":
            for field_node in child.children:
                if field_node.type != "field_declaration":
                    continue

                # Check for embedding first
                embedded_type = _is_embedding(field_node)
                if embedded_type is not None:
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=f"{project_name}:{struct_qn}",
                            rel_type=RelType.INHERITS,
                            to_name=embedded_type,
                        )
                    )
                    continue

                # Regular field: extract name(s)
                for fc in field_node.children:
                    if fc.type == "field_identifier":
                        field_name = node_text(fc)
                        field_qn = f"{struct_qn}.{field_name}"
                        entities.append(
                            ParsedEntity(
                                name=field_name,
                                qualified_name=f"{project_name}:{field_qn}",
                                label=NodeLabel.VALUE,
                                kind=ValueKind.FIELD,
                                line_start=field_node.start_point[0] + 1,
                                line_end=field_node.end_point[0] + 1,
                                file_path=path,
                                visibility=_visibility_from_name(field_name),
                                source=node_text(field_node),
                            )
                        )
                        relationships.append(
                            ParsedRelationship(
                                from_qualified_name=f"{project_name}:{struct_qn}",
                                rel_type=RelType.DEFINES,
                                to_name=f"{project_name}:{field_qn}",
                            )
                        )


def _process_interface_methods(
    iface_node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    iface_name: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process interface method specifications and embeddings.

    tree-sitter-go v0.25 uses ``method_elem`` for method specs and
    ``type_elem`` for embedded interfaces, both as direct children of
    ``interface_type``.
    """
    iface_qn = f"{module_qn}.{iface_name}"
    for child in iface_node.children:
        if child.type == "method_elem":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            method_name = node_text(name_node)
            method_qn = f"{iface_qn}.{method_name}"
            sig = source[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
            entities.append(
                ParsedEntity(
                    name=method_name,
                    qualified_name=f"{project_name}:{method_qn}",
                    label=NodeLabel.CALLABLE,
                    kind=CallableKind.METHOD,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    file_path=path,
                    signature=sig,
                    visibility=_visibility_from_name(method_name),
                )
            )
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=f"{project_name}:{iface_qn}",
                    rel_type=RelType.DEFINES,
                    to_name=f"{project_name}:{method_qn}",
                )
            )
        elif child.type == "type_elem":
            # Interface embedding (e.g. Reader, io.Writer inside another interface)
            for inner in child.children:
                if inner.type in ("type_identifier", "qualified_type"):
                    embedded = node_text(inner)
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=f"{project_name}:{iface_qn}",
                            rel_type=RelType.INHERITS,
                            to_name=embedded,
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
) -> None:
    """Process a function_declaration."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    qn = f"{module_qn}.{name}"
    docstring = _extract_doc_comment(node, source)
    signature = _extract_function_signature(node, source)

    tags: list[str] = []
    if name == "init":
        tags.append("init_func")
    elif name == "main":
        tags.append("entry_point")
    elif name.startswith("Test"):
        tags.append("test")
    elif name.startswith("Benchmark"):
        tags.append("benchmark")
    elif name.startswith("Example"):
        tags.append("example")

    # Check for build directives in preceding comments
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment":
        text = source[prev.start_byte : prev.end_byte].decode("utf-8", errors="replace")
        if text.startswith("//go:build "):
            constraint = text[len("//go:build ") :].strip()
            tags.append(f"build_tag:{constraint}")
        elif text.startswith("//go:generate "):
            tags.append("go_generate")
        prev = prev.prev_sibling

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.CALLABLE,
            kind=CallableKind.FUNCTION,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=path,
            docstring=docstring,
            signature=signature,
            source=node_text(node),
            visibility=_visibility_from_name(name),
            tags=tags,
        )
    )

    # DEFINES: module -> function
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # Extract call sites from body
    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, f"{project_name}:{qn}", relationships)


def _process_method(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a method_declaration (function with receiver)."""
    # Method name is a field_identifier child (not a field on the node)
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    receiver_type = _extract_receiver_type(node)

    if receiver_type:
        qn = f"{module_qn}.{receiver_type}.{name}"
        parent_qn = f"{module_qn}.{receiver_type}"
    else:
        qn = f"{module_qn}.{name}"
        parent_qn = module_qn

    docstring = _extract_doc_comment(node, source)
    signature = _extract_function_signature(node, source)

    tags: list[str] = []
    if name.startswith("Test"):
        tags.append("test")

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.CALLABLE,
            kind=CallableKind.METHOD,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=path,
            docstring=docstring,
            signature=signature,
            source=node_text(node),
            visibility=_visibility_from_name(name),
            tags=tags,
        )
    )

    # DEFINES: struct -> method
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{parent_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # Extract call sites from body
    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, f"{project_name}:{qn}", relationships)


def _process_const_var(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    *,
    is_const: bool,
) -> None:
    """Process const_declaration or var_declaration (single or grouped).

    Grouped declarations use ``const_spec_list`` / ``var_spec_list`` wrappers.
    """
    kind = ValueKind.CONSTANT if is_const else ValueKind.VARIABLE
    for child in node.children:
        if child.type in ("const_spec", "var_spec"):
            _process_const_var_spec(child, path, project_name, module_qn, kind, entities, relationships)
        elif child.type in ("const_spec_list", "var_spec_list"):
            for spec in child.children:
                if spec.type in ("const_spec", "var_spec"):
                    _process_const_var_spec(spec, path, project_name, module_qn, kind, entities, relationships)


def _process_const_var_spec(
    spec: Node,
    path: str,
    project_name: str,
    module_qn: str,
    kind: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a single const_spec or var_spec."""
    name_node = spec.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    qn = f"{module_qn}.{name}"

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.VALUE,
            kind=kind,
            line_start=spec.start_point[0] + 1,
            line_end=spec.end_point[0] + 1,
            file_path=path,
            source=node_text(spec),
            visibility=_visibility_from_name(name),
        )
    )

    # DEFINES: module -> const/var
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )


# ---------------------------------------------------------------------------
# Language registration
# ---------------------------------------------------------------------------

if _HAS_GO:
    register_language(
        LanguageConfig(
            name="go",
            extensions=frozenset({".go"}),
            language=_GO_LANGUAGE,
            query=_GO_QUERY,
            parse_func=_parse_go,
        )
    )
