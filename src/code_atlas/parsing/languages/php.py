"""PHP language support — tree-sitter parser for PHP source files."""

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
    import tree_sitter_php as ts_php  # type: ignore[unresolved-import]
    from tree_sitter import Language, Query

    _PHP_LANGUAGE = Language(ts_php.language_php())
    _PHP_QUERY = Query(_PHP_LANGUAGE, "(program) @root")
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_qualified_name(file_path: str) -> str:
    """Convert a file path to a dot-separated module qualified name.

    ``src/Models/User.php`` -> ``src.Models.User``
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    parts = list(p.parts)
    if parts and parts[-1].endswith(".php"):
        parts[-1] = parts[-1][: -len(".php")]
    return ".".join(parts)


def _get_visibility(node: Node) -> str:
    """Extract visibility from modifier children of a declaration node."""
    for child in node.children:
        if child.type == "visibility_modifier":
            text = node_text(child).strip()
            if text == "private":
                return Visibility.PRIVATE
            if text == "protected":
                return Visibility.PROTECTED
            if text == "public":
                return Visibility.PUBLIC
    return Visibility.PUBLIC


def _get_modifier_tags(node: Node) -> list[str]:
    """Extract modifier tags (abstract, static, final, readonly) from a node."""
    tags: list[str] = []
    for child in node.children:
        if child.type == "abstract_modifier":
            tags.append("abstract")
        elif child.type == "static_modifier":
            tags.append("static")
        elif child.type == "final_modifier":
            tags.append("final")
        elif child.type == "readonly_modifier":
            tags.append("readonly")
    return tags


def _get_attribute_tags(node: Node) -> list[str]:
    """Extract PHP 8 attribute tags from a node's attribute_list field or children."""
    tags: list[str] = []
    attr_list = node.child_by_field_name("attributes")
    if attr_list is not None:
        _collect_attribute_tags(attr_list, tags)
    for child in node.children:
        if child.type == "attribute_list" and child != attr_list:
            _collect_attribute_tags(child, tags)
    return tags


def _collect_attribute_tags(attr_list_node: Node, tags: list[str]) -> None:
    """Walk attribute_list/attribute_group nodes and collect attribute tags."""
    if attr_list_node.type == "attribute_list":
        for group in attr_list_node.children:
            if group.type == "attribute_group":
                for attr in group.children:
                    if attr.type == "attribute":
                        attr_text = node_text(attr).strip()
                        tags.append(f"attribute:{attr_text}")
    elif attr_list_node.type == "attribute_group":
        for attr in attr_list_node.children:
            if attr.type == "attribute":
                attr_text = node_text(attr).strip()
                tags.append(f"attribute:{attr_text}")


def _extract_phpdoc(node: Node, parent: Node) -> str | None:
    """Extract PHPDoc comment preceding a declaration node.

    PHPDoc comments (``/** ... */``) appear as sibling ``comment`` nodes
    immediately before the declaration in the parent's children list.
    """
    prev: Node | None = None
    for child in parent.children:
        if child == node:
            break
        prev = child

    if prev is not None and prev.type == "comment":
        text = node_text(prev)
        if text.startswith("/**"):
            text = text[3:].removesuffix("*/")
            lines = text.split("\n")
            cleaned: list[str] = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("*"):
                    stripped = stripped[1:].strip()
                cleaned.append(stripped)
            result = "\n".join(cleaned).strip()
            return result or None
    return None


def _extract_signature(node: Node, source: bytes) -> str | None:
    """Extract function/method signature (declaration line without body)."""
    if node.type not in ("function_definition", "method_declaration"):
        return None
    body = node.child_by_field_name("body")
    if body is not None:
        sig_bytes = source[node.start_byte : body.start_byte].rstrip()
        return sig_bytes.decode("utf-8", errors="replace").strip()
    sig_text = node_text(node).rstrip(";").strip()
    return sig_text or None


def _extract_string_content(node: Node) -> str | None:
    """Extract string content from a string/encapsed_string node."""
    for child in node.children:
        if child.type == "string_content":
            return node_text(child)
    raw = node_text(node).strip("'\"")
    return raw or None


# ---------------------------------------------------------------------------
# Require/include helpers
# ---------------------------------------------------------------------------

_REQUIRE_INCLUDE_TYPES = frozenset(
    {
        "require_expression",
        "require_once_expression",
        "include_expression",
        "include_once_expression",
    }
)


# ---------------------------------------------------------------------------
# Parse entry point
# ---------------------------------------------------------------------------


def _parse_php(
    path: str,
    source: bytes,
    root: Node,
    project_name: str,
) -> ParsedFile:
    """Extract entities and relationships from a PHP parse tree."""
    module_qn = _module_qualified_name(path)

    entities: list[ParsedEntity] = []
    relationships: list[ParsedRelationship] = []
    seen: set[tuple[int, str]] = set()

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

    # Walk program children
    _walk_php_node(root, path, source, project_name, module_qn, entities, relationships, seen)

    return ParsedFile(
        file_path=path,
        language="php",
        entities=entities,
        relationships=relationships,
    )


def _walk_php_node(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Walk the parse tree and extract entities and relationships."""
    for child in node.children:
        if child.type in (
            "class_declaration",
            "interface_declaration",
            "trait_declaration",
            "enum_declaration",
        ):
            _process_type_def(
                child,
                node,
                path,
                source,
                project_name,
                module_qn,
                entities,
                relationships,
                seen,
            )
        elif child.type == "function_definition":
            _process_function(
                child,
                node,
                path,
                source,
                project_name,
                module_qn,
                entities,
                relationships,
                seen,
            )
        elif child.type == "namespace_use_declaration":
            _process_namespace_use(child, project_name, module_qn, relationships)
        elif child.type == "expression_statement":
            _process_expression_statement(child, project_name, module_qn, relationships)
        elif child.type in ("if_statement", "else_clause", "else_if_clause"):
            _walk_php_node(child, path, source, project_name, module_qn, entities, relationships, seen)


# ---------------------------------------------------------------------------
# Type definitions (class, interface, trait, enum)
# ---------------------------------------------------------------------------


def _collect_clause_rels(
    node: Node,
    clause_type: str,
    project_name: str,
    from_qn: str,
    rel_type: RelType,
    relationships: list[ParsedRelationship],
) -> None:
    """Collect INHERITS/IMPLEMENTS relationships from base_clause or class_interface_clause."""
    clause = None
    for child in node.children:
        if child.type == clause_type:
            clause = child
            break
    if clause is None:
        return
    relationships.extend(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{from_qn}",
            rel_type=rel_type,
            to_name=node_text(child),
        )
        for child in clause.children
        if child.type in ("name", "qualified_name")
    )


_TYPE_KIND_MAP: dict[str, str] = {
    "class_declaration": TypeDefKind.CLASS,
    "interface_declaration": TypeDefKind.INTERFACE,
    "trait_declaration": TypeDefKind.TRAIT,
    "enum_declaration": TypeDefKind.ENUM,
}


def _process_type_def(
    node: Node,
    parent: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process a class/interface/trait/enum declaration."""
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

    kind = _TYPE_KIND_MAP.get(node.type, TypeDefKind.CLASS)
    qn = f"{module_qn}.{name}"
    docstring = _extract_phpdoc(node, parent)
    tags = _get_modifier_tags(node) + _get_attribute_tags(node)

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
            visibility=Visibility.PUBLIC,
            tags=tags,
        )
    )

    # DEFINES from module -> type
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # Base class -> INHERITS (extends)
    _collect_clause_rels(node, "base_clause", project_name, qn, RelType.INHERITS, relationships)

    # Interfaces -> IMPLEMENTS
    _collect_clause_rels(node, "class_interface_clause", project_name, qn, RelType.IMPLEMENTS, relationships)

    # Walk class body for members
    body = node.child_by_field_name("body")
    if body is not None:
        _walk_class_body(body, path, source, project_name, module_qn, name, entities, relationships, seen)


def _walk_class_body(
    body: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    class_name: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Walk a class/interface/trait/enum body to extract members."""
    class_qn = f"{module_qn}.{class_name}"

    for child in body.children:
        if child.type == "method_declaration":
            _process_method(
                child,
                body,
                path,
                source,
                project_name,
                module_qn,
                class_name,
                entities,
                relationships,
                seen,
            )
        elif child.type == "property_declaration":
            _process_property(child, path, project_name, module_qn, class_name, entities, relationships, seen)
        elif child.type == "const_declaration":
            _process_const(child, path, project_name, module_qn, class_name, entities, relationships, seen)
        elif child.type == "enum_case":
            _process_enum_case(child, path, project_name, module_qn, class_name, entities, relationships, seen)
        elif child.type == "use_declaration":
            # Trait usage: `use TraitName;`
            relationships.extend(
                ParsedRelationship(
                    from_qualified_name=f"{project_name}:{class_qn}",
                    rel_type=RelType.INHERITS,
                    to_name=node_text(use_child),
                )
                for use_child in child.children
                if use_child.type in ("name", "qualified_name")
            )


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def _process_method(
    node: Node,
    parent: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    class_name: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process a method_declaration inside a class/interface/trait."""
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

    # Determine callable kind
    if name == "__construct":
        kind = CallableKind.CONSTRUCTOR
    elif name == "__destruct":
        kind = CallableKind.DESTRUCTOR
    else:
        has_static = any(c.type == "static_modifier" for c in node.children)
        kind = CallableKind.STATIC_METHOD if has_static else CallableKind.METHOD

    qn = f"{module_qn}.{class_name}.{name}"
    visibility = _get_visibility(node)
    docstring = _extract_phpdoc(node, parent)
    signature = _extract_signature(node, source)
    tags = _get_modifier_tags(node) + _get_attribute_tags(node)

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

    # DEFINES: class -> method
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}.{class_name}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # CALLS from method body
    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, source, f"{project_name}:{qn}", relationships)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _process_function(
    node: Node,
    parent: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process a top-level function_definition."""
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
    docstring = _extract_phpdoc(node, parent)
    signature = _extract_signature(node, source)
    tags = _get_attribute_tags(node)

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

    # DEFINES: module -> function
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # CALLS from function body
    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, source, f"{project_name}:{qn}", relationships)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def _process_property(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    class_name: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process a property_declaration (class field)."""
    visibility = _get_visibility(node)
    tags = _get_modifier_tags(node)

    for child in node.children:
        if child.type == "property_element":
            var_name_node = child.child_by_field_name("name")
            if var_name_node is None:
                continue
            name = _extract_property_name(var_name_node)
            if not name:
                continue

            line_start = child.start_point[0] + 1
            key = (line_start, name)
            if key in seen:
                continue
            seen.add(key)

            qn = f"{module_qn}.{class_name}.{name}"
            entities.append(
                ParsedEntity(
                    name=name,
                    qualified_name=f"{project_name}:{qn}",
                    label=NodeLabel.VALUE,
                    kind=ValueKind.FIELD,
                    line_start=line_start,
                    line_end=child.end_point[0] + 1,
                    file_path=path,
                    source=node_text(node),
                    visibility=visibility,
                    tags=tags,
                )
            )
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=f"{project_name}:{module_qn}.{class_name}",
                    rel_type=RelType.DEFINES,
                    to_name=f"{project_name}:{qn}",
                )
            )


def _extract_property_name(var_name_node: Node) -> str:
    """Extract the property name from a variable_name node (strip $)."""
    for child in var_name_node.children:
        if child.type == "name":
            return node_text(child)
    return node_text(var_name_node).lstrip("$")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _process_const(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    class_name: str | None,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process a const_declaration."""
    visibility = _get_visibility(node) if class_name else Visibility.PUBLIC
    parent_qn = f"{module_qn}.{class_name}" if class_name else module_qn

    for child in node.children:
        if child.type == "const_element":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                for c in child.children:
                    if c.type == "name":
                        name_node = c
                        break
            if name_node is None:
                continue
            name = node_text(name_node)

            line_start = child.start_point[0] + 1
            key = (line_start, name)
            if key in seen:
                continue
            seen.add(key)

            qn = f"{parent_qn}.{name}"
            entities.append(
                ParsedEntity(
                    name=name,
                    qualified_name=f"{project_name}:{qn}",
                    label=NodeLabel.VALUE,
                    kind=ValueKind.CONSTANT,
                    line_start=line_start,
                    line_end=child.end_point[0] + 1,
                    file_path=path,
                    source=node_text(node),
                    visibility=visibility,
                )
            )
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=f"{project_name}:{parent_qn}",
                    rel_type=RelType.DEFINES,
                    to_name=f"{project_name}:{qn}",
                )
            )


# ---------------------------------------------------------------------------
# Enum cases
# ---------------------------------------------------------------------------


def _process_enum_case(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    class_name: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    seen: set[tuple[int, str]],
) -> None:
    """Process an enum_case declaration."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1

    key = (line_start, name)
    if key in seen:
        return
    seen.add(key)

    qn = f"{module_qn}.{class_name}.{name}"
    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.VALUE,
            kind=ValueKind.ENUM_MEMBER,
            line_start=line_start,
            line_end=node.end_point[0] + 1,
            file_path=path,
            source=node_text(node),
            visibility=Visibility.PUBLIC,
        )
    )
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}.{class_name}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )


# ---------------------------------------------------------------------------
# Imports (namespace use declarations + require/include)
# ---------------------------------------------------------------------------


def _process_namespace_use(
    node: Node,
    project_name: str,
    module_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    r"""Process a namespace_use_declaration (``use App\Models\User;``)."""
    for child in node.children:
        if child.type == "namespace_use_clause":
            import_name = node_text(child)
            if " as " in import_name:
                import_name = import_name.split(" as ")[0].strip()
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=f"{project_name}:{module_qn}",
                    rel_type=RelType.IMPORTS,
                    to_name=import_name,
                )
            )
        elif child.type == "namespace_use_group":
            prefix = ""
            for sibling in node.children:
                if sibling.type == "namespace_name":
                    prefix = node_text(sibling)
                    break
            for group_child in child.children:
                if group_child.type == "namespace_use_clause":
                    clause_name = node_text(group_child)
                    full_name = f"{prefix}\\{clause_name}" if prefix else clause_name
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=f"{project_name}:{module_qn}",
                            rel_type=RelType.IMPORTS,
                            to_name=full_name,
                        )
                    )


def _process_expression_statement(
    node: Node,
    project_name: str,
    module_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Check expression_statement children for require/include expressions."""
    for child in node.children:
        if child.type in _REQUIRE_INCLUDE_TYPES:
            for arg in child.children:
                if arg.type in ("encapsed_string", "string"):
                    content = _extract_string_content(arg)
                    if content:
                        relationships.append(
                            ParsedRelationship(
                                from_qualified_name=f"{project_name}:{module_qn}",
                                rel_type=RelType.IMPORTS,
                                to_name=content,
                            )
                        )
                    break


# ---------------------------------------------------------------------------
# Call extraction
# ---------------------------------------------------------------------------


def _extract_calls(
    node: Node,
    source: bytes,
    from_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Recursively extract call expressions from a function/method body."""
    for child in node.children:
        if child.type == "function_call_expression":
            func = child.child_by_field_name("function")
            if func is not None and func.type == "name":
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=from_qn,
                        rel_type=RelType.CALLS,
                        to_name=node_text(func),
                    )
                )
        elif child.type in ("member_call_expression", "scoped_call_expression"):
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=from_qn,
                        rel_type=RelType.CALLS,
                        to_name=node_text(name_node),
                    )
                )

        # Recurse but don't descend into nested function definitions or class bodies
        if child.type not in ("function_definition", "class_declaration", "anonymous_function_creation_expression"):
            _extract_calls(child, source, from_qn, relationships)


# ---------------------------------------------------------------------------
# Language registration
# ---------------------------------------------------------------------------

if _AVAILABLE:
    register_language(
        LanguageConfig(
            name="php",
            extensions=frozenset({".php"}),
            language=_PHP_LANGUAGE,
            query=_PHP_QUERY,
            parse_func=_parse_php,
        )
    )
