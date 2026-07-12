"""Rust language support — tree-sitter parser for Rust source files."""

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

# ---------------------------------------------------------------------------
# Grammar import (optional dependency)
# ---------------------------------------------------------------------------

try:
    import tree_sitter_rust as ts_rust
    from tree_sitter import Language, Query

    _RUST_LANGUAGE = Language(ts_rust.language())
    _RUST_QUERY = Query(_RUST_LANGUAGE, "(source_file) @root")
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_qualified_name(file_path: str) -> str:
    """Convert a Rust file path to a module-style qualified name.

    ``src/server.rs``       -> ``src.server``
    ``src/server/mod.rs``   -> ``src.server``
    ``lib.rs``              -> ``lib``
    ``main.rs``             -> ``main``
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    parts = list(p.parts)
    if parts and parts[-1].endswith(".rs"):
        filename = parts[-1]
        if filename == "mod.rs":
            parts = parts[:-1]
        else:
            parts[-1] = filename[:-3]  # strip .rs
    return ".".join(parts)


def _visibility_from_node(node: Node) -> str:
    """Determine visibility from a Rust item's visibility_modifier child."""
    for child in node.children:
        if child.type == "visibility_modifier":
            text = node_text(child)
            if "pub(crate)" in text:
                return Visibility.INTERNAL
            if "pub(super)" in text:
                return Visibility.PROTECTED
            if text.startswith("pub"):
                return Visibility.PUBLIC
    return Visibility.PRIVATE


def _extract_doc_comments(node: Node) -> str | None:
    """Extract ``///`` or ``//!`` doc comments immediately preceding a node.

    Walks backward through siblings collecting consecutive line_comment nodes
    that start with ``///`` or ``//!``.
    """
    lines: list[str] = []
    prev = node.prev_sibling
    while prev is not None:
        if prev.type == "line_comment":
            text = node_text(prev)
            if text.startswith(("///", "//!")):
                line = text[3:].strip()
                lines.append(line)
            else:
                break
        elif prev.type == "attribute_item":
            # Skip attributes between doc comments and item
            prev = prev.prev_sibling
            continue
        else:
            break
        prev = prev.prev_sibling
    if not lines:
        return None
    lines.reverse()
    return "\n".join(lines)


def _extract_inner_doc_comments(root: Node) -> str | None:
    """Extract ``//!`` inner doc comments from the top of a file."""
    lines: list[str] = []
    for child in root.children:
        if child.type == "line_comment":
            text = node_text(child)
            if text.startswith("//!"):
                lines.append(text[3:].strip())
            else:
                break
        else:
            break
    return "\n".join(lines) if lines else None


def _extract_attributes(node: Node) -> list[str]:
    """Extract ``#[...]`` attribute tags from preceding siblings."""
    tags: list[str] = []
    prev = node.prev_sibling
    while prev is not None:
        if prev.type == "attribute_item":
            text = node_text(prev).strip()
            if text.startswith("#[") and text.endswith("]"):
                inner = text[2:-1]
                tags.append(f"attribute:{inner}")
        elif prev.type == "line_comment":
            prev = prev.prev_sibling
            continue
        else:
            break
        prev = prev.prev_sibling
    tags.reverse()
    return tags


def _extract_function_signature(node: Node, source: bytes) -> str | None:
    """Extract the function signature (everything up to the body block)."""
    body = node.child_by_field_name("body")
    if body is not None:
        sig_bytes = source[node.start_byte : body.start_byte].rstrip()
        return sig_bytes.decode("utf-8", errors="replace")
    # No body (e.g. trait method declaration) — use the full text
    text = node_text(node).strip().rstrip(";")
    return text or None


def _has_self_param(node: Node) -> bool:
    """Check if a function_item has a self/&self/&mut self parameter."""
    params = node.child_by_field_name("parameters")
    if params is None:
        return False
    for child in params.children:
        if child.type in ("self_parameter", "self"):
            return True
        # Some grammars nest the self parameter differently
        text = node_text(child).strip()
        if text in ("self", "&self", "&mut self"):
            return True
    return False


def _is_async_fn(node: Node) -> bool:
    """Check if a function_item has the ``async`` keyword."""
    for child in node.children:
        if child.type == "async" or node_text(child) == "async":
            return True
        if child.type == "function_modifiers":
            for mod in child.children:
                if mod.type == "async" or node_text(mod) == "async":
                    return True
    return False


def _is_unsafe_fn(node: Node) -> bool:
    """Check if a function_item has the ``unsafe`` keyword."""
    for child in node.children:
        if child.type == "unsafe" or node_text(child) == "unsafe":
            return True
        if child.type == "function_modifiers":
            for mod in child.children:
                if mod.type == "unsafe" or node_text(mod) == "unsafe":
                    return True
    return False


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------


def _parse_rust(
    path: str,
    source: bytes,
    root: Node,
    project_name: str,
) -> ParsedFile:
    """Extract entities and relationships from a Rust parse tree."""
    module_qn = _module_qualified_name(path)

    entities: list[ParsedEntity] = []
    relationships: list[ParsedRelationship] = []

    # Module entity
    inner_doc = _extract_inner_doc_comments(root)
    entities.append(
        ParsedEntity(
            name=module_qn.rsplit(".", 1)[-1] if "." in module_qn else module_qn,
            qualified_name=f"{project_name}:{module_qn}",
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=root.end_point[0] + 1,
            file_path=path,
            docstring=inner_doc,
        )
    )

    # Walk top-level items
    _walk_rust_items(root, path, source, project_name, module_qn, None, entities, relationships)

    return ParsedFile(
        file_path=path,
        language="rust",
        entities=entities,
        relationships=relationships,
    )


def _walk_rust_items(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    owner_name: str | None,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Walk the children of a node and extract Rust entities."""
    for child in node.children:
        if child.type == "struct_item":
            _process_struct(child, path, project_name, module_qn, entities, relationships)
        elif child.type == "enum_item":
            _process_enum(child, path, project_name, module_qn, entities, relationships)
        elif child.type == "trait_item":
            _process_trait(child, path, source, project_name, module_qn, entities, relationships)
        elif child.type == "union_item":
            _process_union(child, path, project_name, module_qn, entities, relationships)
        elif child.type == "type_item":
            _process_type_alias(child, path, project_name, module_qn, entities, relationships)
        elif child.type == "function_item":
            _process_function(child, path, source, project_name, module_qn, None, entities, relationships)
        elif child.type == "impl_item":
            _process_impl(child, path, source, project_name, module_qn, entities, relationships)
        elif child.type == "use_declaration":
            _process_use(child, project_name, module_qn, relationships)
        elif child.type in ("const_item", "static_item"):
            _process_const_static(child, path, project_name, module_qn, owner_name, entities, relationships)
        elif child.type == "mod_item":
            _process_mod(child, path, source, project_name, module_qn, entities, relationships)


# ---------------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------------


def _process_struct(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    qn = f"{module_qn}.{name}"
    vis = _visibility_from_node(node)
    doc = _extract_doc_comments(node)
    tags = _extract_attributes(node)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=TypeDefKind.STRUCT,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=path,
            docstring=doc,
            visibility=vis,
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

    # Extract struct fields
    body = node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            if child.type == "field_declaration":
                _process_field(child, path, project_name, module_qn, name, entities, relationships)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


def _process_enum(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    qn = f"{module_qn}.{name}"
    vis = _visibility_from_node(node)
    doc = _extract_doc_comments(node)
    tags = _extract_attributes(node)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=TypeDefKind.ENUM,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=path,
            docstring=doc,
            visibility=vis,
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

    # Extract enum variants
    body = node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            if child.type == "enum_variant":
                _process_enum_variant(child, path, project_name, module_qn, name, entities, relationships)


# ---------------------------------------------------------------------------
# Trait
# ---------------------------------------------------------------------------


def _process_trait(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    qn = f"{module_qn}.{name}"
    vis = _visibility_from_node(node)
    doc = _extract_doc_comments(node)
    tags = _extract_attributes(node)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=TypeDefKind.TRAIT,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=path,
            docstring=doc,
            visibility=vis,
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

    # Supertraits (trait bounds after colon)
    bounds = node.child_by_field_name("bounds")
    if bounds is not None:
        _extract_trait_bounds(bounds, f"{project_name}:{qn}", relationships)

    # Trait body can contain function signatures and default methods
    body = node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            if child.type == "function_item":
                _process_function(child, path, source, project_name, module_qn, name, entities, relationships)
            elif child.type == "function_signature_item":
                _process_function_signature(child, path, project_name, module_qn, name, entities, relationships)


def _extract_trait_bounds(bounds_node: Node, from_qn: str, relationships: list[ParsedRelationship]) -> None:
    """Extract supertrait names from trait_bound nodes."""
    for child in bounds_node.children:
        if child.type == "type_identifier":
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=from_qn,
                    rel_type=RelType.INHERITS,
                    to_name=node_text(child),
                )
            )
        elif child.type == "generic_type":
            # e.g. Iterator<Item=u32> — extract the base name
            type_node = child.child_by_field_name("type")
            if type_node is not None:
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=from_qn,
                        rel_type=RelType.INHERITS,
                        to_name=node_text(type_node),
                    )
                )
        elif child.type == "scoped_type_identifier":
            # e.g. std::fmt::Display — extract the last segment
            text = node_text(child)
            name = text.rsplit("::", 1)[-1]
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=from_qn,
                    rel_type=RelType.INHERITS,
                    to_name=name,
                )
            )
        # Recurse into nested trait_bound containers
        elif child.type == "trait_bound":
            _extract_trait_bounds(child, from_qn, relationships)


# ---------------------------------------------------------------------------
# Union
# ---------------------------------------------------------------------------


def _process_union(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    qn = f"{module_qn}.{name}"
    vis = _visibility_from_node(node)
    doc = _extract_doc_comments(node)
    tags = _extract_attributes(node)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=TypeDefKind.UNION,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=path,
            docstring=doc,
            visibility=vis,
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


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------


def _process_type_alias(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    *,
    owner_name: str | None = None,
) -> None:
    """Process a ``type_item`` (type alias or impl-block associated type).

    ``owner_name`` scopes an associated type declared inside an ``impl`` block
    to its implementing type instead of hoisting it to module scope, which
    would otherwise collide with a module-level type alias of the same name
    (or with another impl's associated type of the same name).  Like impl
    methods (see ``_process_function``'s ``from_impl`` branch), the owner
    type may be defined in another file, so the DEFINES rel carries
    ``parent_type_name`` for post-batch resolution (S5:
    ``GraphClient.resolve_member_defines``), with this file's module as the
    fallback parent.
    """
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    qn = f"{module_qn}.{owner_name}.{name}" if owner_name is not None else f"{module_qn}.{name}"
    vis = _visibility_from_node(node)
    doc = _extract_doc_comments(node)
    tags = _extract_attributes(node)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=TypeDefKind.TYPE_ALIAS,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=path,
            docstring=doc,
            visibility=vis,
            tags=tags,
        )
    )
    properties = {"parent_type_name": owner_name} if owner_name is not None else {}
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
            properties=properties,
        )
    )


# ---------------------------------------------------------------------------
# Function (top-level or inside impl/trait)
# ---------------------------------------------------------------------------


def _process_function(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    owner_name: str | None,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    *,
    from_impl: bool = False,
) -> None:
    """Process a ``function_item`` node.

    ``owner_name`` is the type/trait name when the function is inside an
    impl or trait block, None for top-level functions.  ``from_impl`` marks
    impl-block members, whose owner type may be defined in another file.
    """
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    vis = _visibility_from_node(node)
    doc = _extract_doc_comments(node)
    tags = _extract_attributes(node)
    sig = _extract_function_signature(node, source)

    if _is_async_fn(node):
        tags = [*tags, "async"]
    if _is_unsafe_fn(node):
        tags = [*tags, "unsafe"]

    if owner_name is not None:
        qn = f"{module_qn}.{owner_name}.{name}"
        if _has_self_param(node):
            kind: str = CallableKind.METHOD
        else:
            kind = CallableKind.STATIC_METHOD
    else:
        qn = f"{module_qn}.{name}"
        kind = CallableKind.FUNCTION

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.CALLABLE,
            kind=kind,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=path,
            docstring=doc,
            signature=sig,
            source=node_text(node),
            visibility=vis,
            tags=tags,
        )
    )
    if from_impl and owner_name is not None:
        # DEFINES: impl'd type -> method.  The type may be defined in another
        # file, so emit its NAME for post-batch resolution via
        # GraphClient.resolve_member_defines, with this file's module as the
        # fallback parent in from_qualified_name.
        relationships.append(
            ParsedRelationship(
                from_qualified_name=f"{project_name}:{module_qn}",
                rel_type=RelType.DEFINES,
                to_name=f"{project_name}:{qn}",
                properties={"parent_type_name": owner_name},
            )
        )
    else:
        # DEFINES: module or same-file trait -> function
        parent_qn = f"{module_qn}.{owner_name}" if owner_name is not None else module_qn
        relationships.append(
            ParsedRelationship(
                from_qualified_name=f"{project_name}:{parent_qn}",
                rel_type=RelType.DEFINES,
                to_name=f"{project_name}:{qn}",
            )
        )

    # Extract CALLS from the function body
    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, f"{project_name}:{qn}", relationships)


def _process_function_signature(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    owner_name: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a ``function_signature_item`` in a trait body (no body block)."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    vis = _visibility_from_node(node)
    doc = _extract_doc_comments(node)
    tags = _extract_attributes(node)

    # Signature: the full text minus trailing semicolon
    sig = node_text(node).strip().rstrip(";").strip()

    if _is_async_fn(node):
        tags = [*tags, "async"]
    if _is_unsafe_fn(node):
        tags = [*tags, "unsafe"]

    qn = f"{module_qn}.{owner_name}.{name}"
    if _has_self_param(node):
        kind: str = CallableKind.METHOD
    else:
        kind = CallableKind.STATIC_METHOD

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.CALLABLE,
            kind=kind,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=path,
            docstring=doc,
            signature=sig,
            source=node_text(node),
            visibility=vis,
            tags=tags,
        )
    )
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}.{owner_name}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )


# ---------------------------------------------------------------------------
# Impl blocks
# ---------------------------------------------------------------------------


def _process_impl(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process an ``impl_item`` node.

    Handles both inherent impls (``impl Foo { ... }``) and trait impls
    (``impl Trait for Foo { ... }``).
    """
    # Determine the type being implemented
    type_name = _impl_type_name(node)
    if type_name is None:
        return

    # Detect trait impl: ``impl Trait for Type``
    trait_name = _impl_trait_name(node)
    if trait_name is not None:
        relationships.append(
            ParsedRelationship(
                from_qualified_name=f"{project_name}:{module_qn}.{type_name}",
                rel_type=RelType.IMPLEMENTS,
                to_name=trait_name,
            )
        )

    # Walk impl body for methods, associated functions, consts
    body = node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            if child.type == "function_item":
                _process_function(
                    child, path, source, project_name, module_qn, type_name, entities, relationships, from_impl=True
                )
            elif child.type == "const_item":
                _process_const_static(child, path, project_name, module_qn, type_name, entities, relationships)
            elif child.type == "type_item":
                _process_type_alias(child, path, project_name, module_qn, entities, relationships, owner_name=type_name)


def _impl_type_name(node: Node) -> str | None:
    """Extract the type name from an impl_item.

    For ``impl Foo``, returns ``Foo``.
    For ``impl Trait for Foo``, returns ``Foo``.
    For ``impl<T> Foo<T>``, returns ``Foo``.
    """
    # tree-sitter-rust uses "type" field for the implementing type
    type_node = node.child_by_field_name("type")
    if type_node is not None:
        return _type_node_name(type_node)

    # Fallback: look for type_identifier children
    # For `impl Trait for Type`, the 'type' field points to the implementing type.
    # For inherent impls `impl Type`, the 'type' field also works.
    return None


def _impl_trait_name(node: Node) -> str | None:
    """Extract the trait name from a trait impl (``impl Trait for Type``).

    Returns None for inherent impls.
    """
    trait_node = node.child_by_field_name("trait")
    if trait_node is not None:
        return _type_node_name(trait_node)
    return None


def _type_node_name(type_node: Node) -> str | None:  # noqa: PLR0911
    """Extract a simple type name from a type node.

    Handles ``type_identifier``, ``generic_type``, and ``scoped_type_identifier``,
    and unwraps ``reference_type``/``pointer_type`` (``&Foo``, ``*const Foo``) to
    the underlying type name. Composite types with no single owning type (e.g.
    ``tuple_type``) return None rather than a garbage name built from raw node
    text.
    """
    if type_node.type == "type_identifier":
        return node_text(type_node)
    if type_node.type == "generic_type":
        inner = type_node.child_by_field_name("type")
        if inner is not None:
            return node_text(inner)
    if type_node.type == "scoped_type_identifier":
        # Take the last segment: ``std::collections::HashMap`` -> ``HashMap``
        name_node = type_node.child_by_field_name("name")
        if name_node is not None:
            return node_text(name_node)
        text = node_text(type_node)
        return text.rsplit("::", 1)[-1]
    if type_node.type in ("reference_type", "pointer_type"):
        # &Foo, &mut Foo, *const Foo, *mut Foo -> Foo
        inner = type_node.child_by_field_name("type")
        return _type_node_name(inner) if inner is not None else None
    if type_node.type == "tuple_type":
        # No single owning type, e.g. `impl Trait for (Foo, Bar)`.
        return None
    return node_text(type_node) if type_node.text else None


# ---------------------------------------------------------------------------
# Use declarations (imports)
# ---------------------------------------------------------------------------


def _process_use(
    node: Node,
    project_name: str,
    module_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Process a ``use_declaration`` and emit IMPORTS relationships."""
    arg = node.child_by_field_name("argument")
    if arg is None:
        # Fallback: look for the first non-keyword child
        for child in node.children:
            if child.type not in ("use", ";", "visibility_modifier"):
                arg = child
                break
    if arg is None:
        return
    paths = _collect_use_paths(arg, "")
    relationships.extend(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.IMPORTS,
            to_name=use_path,
        )
        for use_path in paths
    )


def _collect_use_paths(node: Node, prefix: str) -> list[str]:  # noqa: PLR0911, PLR0912
    """Recursively collect fully-qualified use paths."""
    if node.type == "use_as_clause":
        # `use foo::bar as baz;` — import the original path
        path_node = node.child_by_field_name("path")
        if path_node is not None:
            return _collect_use_paths(path_node, prefix)
        return [prefix] if prefix else []

    if node.type == "scoped_identifier":
        # e.g. std::collections::HashMap
        path = node_text(node).replace("::", ".")
        full = f"{prefix}.{path}" if prefix else path
        return [full]

    if node.type == "scoped_use_list":
        # e.g. std::collections::{HashMap, BTreeMap}
        path_node = node.child_by_field_name("path")
        new_prefix = ""
        if path_node is not None:
            path_text = node_text(path_node).replace("::", ".")
            new_prefix = f"{prefix}.{path_text}" if prefix else path_text
        else:
            new_prefix = prefix
        list_node = node.child_by_field_name("list")
        if list_node is not None:
            return _collect_use_paths(list_node, new_prefix)
        return []

    if node.type == "use_list":
        # e.g. {HashMap, BTreeMap}
        paths: list[str] = []
        for child in node.children:
            if child.type in ("{", "}", ","):
                continue
            paths.extend(_collect_use_paths(child, prefix))
        return paths

    if node.type == "use_wildcard":
        # e.g. std::collections::*
        path_node = node.child_by_field_name("path") or node.child_by_field_name("name")
        if path_node is not None:
            full = node_text(path_node).replace("::", ".")
            if prefix:
                full = f"{prefix}.{full}"
            return [f"{full}.*"]
        return [f"{prefix}.*"] if prefix else ["*"]

    if node.type in ("identifier", "self"):
        text = node_text(node)
        return [f"{prefix}.{text}" if prefix else text]

    # Default: try to use the node text directly
    text = node_text(node).strip()
    if text:
        path = text.replace("::", ".")
        full = f"{prefix}.{path}" if prefix else path
        return [full]
    return []


# ---------------------------------------------------------------------------
# Const / Static
# ---------------------------------------------------------------------------


def _process_const_static(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    owner_name: str | None,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    vis = _visibility_from_node(node)
    doc = _extract_doc_comments(node)
    tags = _extract_attributes(node)

    if owner_name is not None:
        qn = f"{module_qn}.{owner_name}.{name}"
        parent_qn = f"{module_qn}.{owner_name}"
    else:
        qn = f"{module_qn}.{name}"
        parent_qn = module_qn

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.VALUE,
            kind=ValueKind.CONSTANT,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=path,
            docstring=doc,
            source=node_text(node),
            visibility=vis,
            tags=tags,
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
# Struct fields
# ---------------------------------------------------------------------------


def _process_field(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    owner_name: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    vis = _visibility_from_node(node)
    doc = _extract_doc_comments(node)
    qn = f"{module_qn}.{owner_name}.{name}"

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.VALUE,
            kind=ValueKind.FIELD,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=path,
            docstring=doc,
            visibility=vis,
        )
    )
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}.{owner_name}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )


# ---------------------------------------------------------------------------
# Enum variants
# ---------------------------------------------------------------------------


def _process_enum_variant(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    enum_name: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    doc = _extract_doc_comments(node)
    qn = f"{module_qn}.{enum_name}.{name}"

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.VALUE,
            kind=ValueKind.ENUM_MEMBER,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=path,
            docstring=doc,
            visibility=Visibility.PUBLIC,
        )
    )
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}.{enum_name}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )


# ---------------------------------------------------------------------------
# Mod declarations
# ---------------------------------------------------------------------------


def _process_mod(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process ``mod`` items.

    ``mod foo;`` declarations reference another file and emit IMPORTS.
    Inline modules (``mod foo { ... }``) define a nested module here: emit a
    Module entity and walk the body with the extended qualified name.
    """
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    body = node.child_by_field_name("body")
    if body is None:
        relationships.append(
            ParsedRelationship(
                from_qualified_name=f"{project_name}:{module_qn}",
                rel_type=RelType.IMPORTS,
                to_name=name,
            )
        )
        return

    inner_qn = f"{module_qn}.{name}"
    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{inner_qn}",
            label=NodeLabel.MODULE,
            kind="module",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            file_path=path,
            docstring=_extract_doc_comments(node),
            visibility=_visibility_from_node(node),
            tags=_extract_attributes(node),
        )
    )
    relationships.append(
        ParsedRelationship(
            from_qualified_name=f"{project_name}:{module_qn}",
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{inner_qn}",
        )
    )
    _walk_rust_items(body, path, source, project_name, inner_qn, None, entities, relationships)


# ---------------------------------------------------------------------------
# Call extraction
# ---------------------------------------------------------------------------


def _extract_calls(
    node: Node,
    from_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Recursively extract call expressions from a block."""
    for child in node.children:
        if child.type == "call_expression":
            func = child.child_by_field_name("function")
            if func is not None:
                call_name = _call_target_name(func)
                if call_name:
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=from_qn,
                            rel_type=RelType.CALLS,
                            to_name=call_name,
                        )
                    )
        # Recurse but don't descend into nested function definitions or closures
        if child.type not in ("function_item", "closure_expression"):
            _extract_calls(child, from_qn, relationships)


def _call_target_name(node: Node) -> str | None:
    """Extract the function name from a call_expression's function child."""
    if node.type == "identifier":
        return node_text(node)
    if node.type == "field_expression":
        # e.g. self.foo() or obj.method() — extract method name
        field = node.child_by_field_name("field")
        if field is not None:
            return node_text(field)
    if node.type == "scoped_identifier":
        # e.g. Vec::new() — extract the last segment
        name = node.child_by_field_name("name")
        if name is not None:
            return node_text(name)
    if node.type == "generic_function":
        # e.g. foo::<T>() — extract the function name
        func = node.child_by_field_name("function")
        if func is not None:
            return _call_target_name(func)
    return None


# ---------------------------------------------------------------------------
# Language registration
# ---------------------------------------------------------------------------

if _AVAILABLE:
    register_language(
        LanguageConfig(
            name="rust",
            extensions=frozenset({".rs"}),
            language=_RUST_LANGUAGE,
            query=_RUST_QUERY,
            parse_func=_parse_rust,
        )
    )
