"""C and C++ language support — tree-sitter parser for C/C++ source files."""

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
# Grammar imports (optional — may not be installed)
# ---------------------------------------------------------------------------

_C_AVAILABLE = False
_CPP_AVAILABLE = False

try:
    import tree_sitter_c as ts_c
    from tree_sitter import Language, Query

    _C_LANGUAGE = Language(ts_c.language())
    _C_QUERY = Query(_C_LANGUAGE, "(translation_unit) @root")
    _C_AVAILABLE = True
except ImportError:
    _log.debug("tree-sitter-c not installed — C language support disabled")

try:
    import tree_sitter_cpp as ts_cpp
    from tree_sitter import Language, Query

    _CPP_LANGUAGE = Language(ts_cpp.language())
    _CPP_QUERY = Query(_CPP_LANGUAGE, "(translation_unit) @root")
    _CPP_AVAILABLE = True
except ImportError:
    _log.debug("tree-sitter-cpp not installed — C++ language support disabled")


# ---------------------------------------------------------------------------
# C/C++ file extensions
# ---------------------------------------------------------------------------

_C_EXTENSIONS = frozenset({".c", ".h"})
_CPP_EXTENSIONS = frozenset({".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".hh"})

# Node types that represent type definitions with a body
_TYPE_DEF_NODES = frozenset({"struct_specifier", "enum_specifier", "union_specifier", "class_specifier"})

# Tags derived from storage class / qualifier specifiers
_TAG_KEYWORDS = frozenset({"virtual", "override", "static", "const", "inline", "extern"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_qualified_name(file_path: str) -> str:
    """Convert file path to a module-style qualified name.

    ``src/server.c`` -> ``src.server``
    ``include/utils.h`` -> ``include.utils``
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    parts = list(p.parts)
    if parts:
        filename = parts[-1]
        # Strip all C/C++ extensions
        dot = filename.rfind(".")
        if dot > 0:
            parts[-1] = filename[:dot]
    return ".".join(parts)


def _is_cpp_file(path: str) -> bool:
    """Return True if path has a C++ extension."""
    suffix = PurePosixPath(path.replace("\\", "/")).suffix.lower()
    return suffix in _CPP_EXTENSIONS


def _extract_doxygen_comment(node: Node, source: bytes) -> str | None:
    """Extract Doxygen-style doc comment immediately before a declaration.

    Recognizes ``///`` line comments and ``/** ... */`` block comments.
    """
    prev = node.prev_named_sibling
    if prev is None or prev.type != "comment":
        return None
    # Check the comment is immediately before this node (no gap > 1 line)
    if node.start_point[0] - prev.end_point[0] > 1:
        return None
    text = source[prev.start_byte : prev.end_byte].decode("utf-8", errors="replace")

    # Collect consecutive comment lines above (for multi-line /// style)
    comment_lines = [text]
    cursor = prev.prev_named_sibling
    while cursor is not None and cursor.type == "comment":
        if prev.start_point[0] - cursor.end_point[0] <= 1:
            line_text = source[cursor.start_byte : cursor.end_byte].decode("utf-8", errors="replace")
            comment_lines.insert(0, line_text)
            prev = cursor
            cursor = cursor.prev_named_sibling
        else:
            break

    return _clean_doc_comment("\n".join(comment_lines))


def _clean_doc_comment(text: str) -> str | None:
    """Strip comment delimiters from a doc comment string."""
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # /** ... */ block comment
        if stripped.startswith("/**"):
            stripped = stripped[3:]
        elif stripped.startswith("/*"):
            stripped = stripped[2:]
        stripped = stripped.removesuffix("*/")
        # /// line comment
        if stripped.startswith("///"):
            stripped = stripped[3:]
        elif stripped.startswith("//"):
            # Not a doxygen comment, skip
            continue
        # Leading * in block comment lines
        stripped = stripped.lstrip("*").lstrip()
        if stripped:
            cleaned.append(stripped)
    result = " ".join(cleaned).strip()
    return result or None


def _extract_tags(node: Node) -> list[str]:
    """Extract tag keywords (virtual, static, const, etc.) from a declaration node.

    Checks both direct children of the node and children of the declarator,
    since ``override`` appears as a ``virtual_specifier`` inside ``function_declarator``.
    """
    tags: list[str] = []
    _collect_tag_keywords(node, tags)
    # Also check inside the declarator (override/final are virtual_specifier children)
    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        _collect_tag_keywords(declarator, tags)
    return tags


def _collect_tag_keywords(node: Node, tags: list[str]) -> None:
    """Collect tag keywords from direct children of a node."""
    for child in node.children:
        if child.type in (
            "storage_class_specifier",
            "type_qualifier",
            "virtual_function_specifier",
            "virtual",
            "override",
            "virtual_specifier",
            "function_specifier",
        ):
            kw = node_text(child).strip()
            if kw in _TAG_KEYWORDS:
                tags.append(kw)
            else:
                # virtual_specifier may contain override/final as inner nodes
                for inner in child.children:
                    inner_kw = node_text(inner).strip()
                    if inner_kw in _TAG_KEYWORDS:
                        tags.append(inner_kw)


def _has_storage_class(node: Node, keyword: str) -> bool:
    """Check if a declaration has a given storage class specifier (e.g. 'static', 'extern')."""
    for child in node.children:
        if child.type == "storage_class_specifier" and node_text(child).strip() == keyword:
            return True
    return False


def _extract_function_signature(node: Node, source: bytes) -> str | None:
    """Extract function/method signature — the declaration line without the body."""
    body = node.child_by_field_name("body")
    if body is None:
        # No body (forward declaration) — use entire node
        sig = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace").strip()
        return sig or None
    # Everything up to the body
    sig_bytes = source[node.start_byte : body.start_byte]
    sig = sig_bytes.decode("utf-8", errors="replace").strip()
    # Remove trailing opening brace if present
    if sig.endswith("{"):
        sig = sig[:-1].strip()
    return sig or None


_LEAF_DECLARATOR_TYPES = frozenset(
    {
        "identifier",
        "type_identifier",
        "field_identifier",
        "primitive_type",  # tree-sitter-c treats some typedef names (e.g. size_t) as primitive_type
        "destructor_name",  # ~ClassName
        "operator_name",  # operator+, operator==, etc. — no 'declarator' field, no named children
    }
)


def _get_declarator_name(declarator: Node) -> str | None:
    """Recursively extract the identifier name from a declarator tree.

    Handles function_declarator, pointer_declarator, array_declarator, etc.
    """
    if declarator.type in _LEAF_DECLARATOR_TYPES:
        return node_text(declarator)

    # function_declarator, pointer_declarator, etc.: has a `declarator` child
    inner = declarator.child_by_field_name("declarator")
    if inner is not None:
        return _get_declarator_name(inner)

    # Fallback: search named children
    for child in declarator.named_children:
        if child.type in _LEAF_DECLARATOR_TYPES:
            return node_text(child)
        result = _get_declarator_name(child)
        if result is not None:
            return result
    return None


def _template_wrapper(node: Node) -> Node | None:
    """Return the enclosing template_declaration wrapper, if any."""
    parent = node.parent
    return parent if parent is not None and parent.type == "template_declaration" else None


def _prototype_declarator(declarator: Node) -> Node | None:
    """Return the function_declarator of a function prototype, unwrapping pointer/reference returns.

    Returns None for non-function declarators and for function-pointer
    declarators like ``int (*cb)(int)`` whose inner declarator is parenthesized.
    """
    node = declarator
    while node.type in ("pointer_declarator", "reference_declarator"):
        # reference_declarator carries its inner declarator as an unnamed field
        inner = node.child_by_field_name("declarator") or next(
            (c for c in node.named_children if c.type.endswith("declarator")), None
        )
        if inner is None:
            return None
        node = inner
    if node.type != "function_declarator":
        return None
    inner = node.child_by_field_name("declarator")
    if inner is None or inner.type == "parenthesized_declarator":
        return None
    return node


def _get_qualified_declarator_name(declarator: Node) -> tuple[list[str], str | None]:
    """Extract name from declarator, returning (scope_parts, name) for qualified names.

    For ``Outer::Inner::method_name``, returns ``(["Outer", "Inner"], "method_name")``
    — tree-sitter-cpp parses this as ``qualified_identifier(scope: 'Outer', name:
    qualified_identifier('Inner::method_name'))``, so nested scopes are unwrapped
    recursively rather than taking the inner qualified_identifier's text verbatim
    (which would otherwise leak a '::'-joined name like 'Inner::method_name').
    For ``ClassName::method_name``, returns ``(["ClassName"], "method_name")``.
    For ``plain_func``, returns ``([], "plain_func")``.
    """
    if declarator.type == "qualified_identifier":
        scope_node = declarator.child_by_field_name("scope")
        name_node = declarator.child_by_field_name("name")
        if scope_node is not None and scope_node.type == "template_type":
            # Box<T>::method — strip template arguments from the scope
            scope_node = scope_node.child_by_field_name("name") or scope_node
        scope = node_text(scope_node) if scope_node is not None else None
        scope_parts = [scope] if scope else []
        if name_node is not None and name_node.type == "qualified_identifier":
            inner_scope_parts, name = _get_qualified_declarator_name(name_node)
            return (scope_parts + inner_scope_parts, name)
        name = node_text(name_node) if name_node is not None else None
        return (scope_parts, name)

    # function_declarator wrapping a qualified_identifier
    inner = declarator.child_by_field_name("declarator")
    if inner is not None:
        if inner.type == "qualified_identifier":
            return _get_qualified_declarator_name(inner)
        return ([], _get_declarator_name(inner))

    return ([], _get_declarator_name(declarator))


def _is_destructor_name(name: str) -> bool:
    """Check if a name looks like a destructor (starts with ~)."""
    return name.startswith("~")


def _is_constructor(name: str, class_name: str | None) -> bool:
    """Check if a function name matches the current class name (constructor)."""
    if class_name is None:
        return False
    return name == class_name


def _method_callable_kind(name: str, class_name: str, tags: list[str]) -> str:
    """Determine the callable kind for a method inside a class."""
    if _is_destructor_name(name):
        return CallableKind.DESTRUCTOR
    if _is_constructor(name, class_name):
        return CallableKind.CONSTRUCTOR
    if "static" in tags:
        return CallableKind.STATIC_METHOD
    return CallableKind.METHOD


# ---------------------------------------------------------------------------
# Core parse function
# ---------------------------------------------------------------------------


def _parse_cpp(
    path: str,
    source: bytes,
    root: Node,
    project_name: str,
) -> ParsedFile:
    """Extract entities and relationships from a C or C++ parse tree."""
    is_cpp = _is_cpp_file(path)
    module_qn = _module_qualified_name(path)
    lang = "cpp" if is_cpp else "c"

    entities: list[ParsedEntity] = []
    relationships: list[ParsedRelationship] = []

    # Module entity
    entities.append(
        ParsedEntity(
            name=module_qn.rsplit(".", 1)[-1] if "." in module_qn else module_qn,
            qualified_name=f"{project_name}:{module_qn}",
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=root.end_point[0] + 1,
            file_path=path,
        )
    )

    # Walk the top-level translation_unit
    _walk_translation_unit(
        root,
        path=path,
        source=source,
        project_name=project_name,
        module_qn=module_qn,
        is_cpp=is_cpp,
        namespace_parts=[],
        class_stack=[],
        current_visibility=Visibility.PUBLIC,
        entities=entities,
        relationships=relationships,
    )

    return ParsedFile(
        file_path=path,
        language=lang,
        entities=entities,
        relationships=relationships,
    )


def _build_qn(module_qn: str, namespace_parts: list[str], class_stack: list[str], name: str) -> str:
    """Build a qualified name from namespace, class stack, and entity name."""
    parts = [module_qn, *namespace_parts, *class_stack, name]
    return ".".join(parts)


def _parent_qn(project_name: str, module_qn: str, namespace_parts: list[str], class_stack: list[str]) -> str:
    """Build the parent qualified name (module + namespaces + classes)."""
    parts = [module_qn, *namespace_parts, *class_stack]
    return f"{project_name}:{'.'.join(parts)}"


def _walk_translation_unit(  # noqa: PLR0912
    node: Node,
    *,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    is_cpp: bool,
    namespace_parts: list[str],
    class_stack: list[str],
    current_visibility: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Recursively walk the AST and extract entities/relationships."""
    for child in node.children:
        # ----- #include -----
        if child.type == "preproc_include":
            _process_include(child, project_name, module_qn, namespace_parts, class_stack, relationships)
            continue

        # ----- namespace (C++ only) -----
        if is_cpp and child.type == "namespace_definition":
            _process_namespace(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                entities=entities,
                relationships=relationships,
            )
            continue

        # ----- template wrapper (C++ only) -----
        if is_cpp and child.type == "template_declaration":
            # The declared class/function is a child of the wrapper — recurse so
            # the existing dispatch branches process it.
            _walk_translation_unit(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=current_visibility,
                entities=entities,
                relationships=relationships,
            )
            continue

        # ----- Access specifiers (C++ only) -----
        if is_cpp and child.type == "access_specifier":
            spec_text = node_text(child).strip().rstrip(":")
            if spec_text == "public":
                current_visibility = Visibility.PUBLIC
            elif spec_text == "private":
                current_visibility = Visibility.PRIVATE
            elif spec_text == "protected":
                current_visibility = Visibility.PROTECTED
            continue

        # ----- struct / class / enum / union -----
        if child.type in _TYPE_DEF_NODES:
            _process_type_def(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=current_visibility,
                entities=entities,
                relationships=relationships,
            )
            continue

        # ----- type_definition (typedef) -----
        if child.type == "type_definition":
            _process_typedef(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=current_visibility,
                entities=entities,
                relationships=relationships,
            )
            continue

        # ----- function_definition -----
        if child.type == "function_definition":
            _process_function(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=current_visibility,
                entities=entities,
                relationships=relationships,
            )
            continue

        # ----- declaration (global variables, forward declarations) -----
        if child.type == "declaration":
            _process_declaration(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=current_visibility,
                entities=entities,
                relationships=relationships,
            )
            continue

        # ----- field_declaration (struct/class fields) -----
        if child.type == "field_declaration":
            _process_field_declaration(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=current_visibility,
                entities=entities,
                relationships=relationships,
            )
            continue


def _include_path_text(path_node: Node) -> str:
    """Strip delimiters from a #include path node's text.

    ``system_lib_string`` (``<vector>``) and ``string_literal`` (``"util.h"``)
    both include their delimiters in ``node.text`` — strip them so the emitted
    IMPORTS name is a bare path (``vector``, ``util.h``) instead of garbage
    like ``<vector>`` or ``"util.h"`` (which corrupts ExternalPackage naming
    downstream in ``resolve_imports``).
    """
    text = node_text(path_node)
    if len(text) >= 2 and text[0] in '<"' and text[-1] in '>"':
        return text[1:-1]
    return text


def _process_include(
    node: Node,
    project_name: str,
    module_qn: str,
    namespace_parts: list[str],
    class_stack: list[str],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a #include directive and emit an IMPORTS relationship."""
    path_node = node.child_by_field_name("path")
    if path_node is None:
        return
    include_path = _include_path_text(path_node)
    if not include_path:
        return

    from_qn = _parent_qn(project_name, module_qn, namespace_parts, class_stack)
    relationships.append(
        ParsedRelationship(
            from_qualified_name=from_qn,
            rel_type=RelType.IMPORTS,
            to_name=include_path,
        )
    )


def _process_namespace(
    node: Node,
    *,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    is_cpp: bool,
    namespace_parts: list[str],
    class_stack: list[str],
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a namespace_definition and recurse into its body."""
    name_node = node.child_by_field_name("name")
    ns_name = node_text(name_node) if name_node is not None else ""
    if not ns_name:
        # Anonymous namespace — treat body as same scope
        body = node.child_by_field_name("body")
        if body is not None:
            _walk_translation_unit(
                body,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=Visibility.PRIVATE,
                entities=entities,
                relationships=relationships,
            )
        return

    new_ns = [*namespace_parts, ns_name]
    body = node.child_by_field_name("body")
    if body is not None:
        _walk_translation_unit(
            body,
            path=path,
            source=source,
            project_name=project_name,
            module_qn=module_qn,
            is_cpp=is_cpp,
            namespace_parts=new_ns,
            class_stack=class_stack,
            current_visibility=Visibility.PUBLIC,
            entities=entities,
            relationships=relationships,
        )


def _process_type_def(
    node: Node,
    *,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    is_cpp: bool,
    namespace_parts: list[str],
    class_stack: list[str],
    current_visibility: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process struct_specifier, class_specifier, enum_specifier, or union_specifier."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        # Anonymous struct/enum/union — skip
        return
    name = node_text(name_node)
    if not name:
        return

    # Determine kind
    kind_map = {
        "struct_specifier": TypeDefKind.STRUCT,
        "class_specifier": TypeDefKind.CLASS,
        "enum_specifier": TypeDefKind.ENUM,
        "union_specifier": TypeDefKind.UNION,
    }
    kind = kind_map.get(node.type, TypeDefKind.STRUCT)

    template_parent = _template_wrapper(node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    # Doc comments for templates sit above the template_declaration wrapper
    docstring = _extract_doxygen_comment(template_parent or node, source)
    qn = _build_qn(module_qn, namespace_parts, class_stack, name)

    # Visibility: at file scope (no class stack), use current_visibility or PUBLIC
    # If inside class, use current_visibility (set by access specifiers)
    visibility = current_visibility
    if not class_stack and not _has_storage_class(node, "static"):
        visibility = Visibility.PUBLIC

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
            visibility=visibility,
            tags=["template"] if template_parent is not None else [],
        )
    )

    # DEFINES from parent
    parent_full_qn = _parent_qn(project_name, module_qn, namespace_parts, class_stack)
    relationships.append(
        ParsedRelationship(
            from_qualified_name=parent_full_qn,
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # C++ class/struct inheritance: `: public Base, private Other`
    if is_cpp:
        _extract_base_classes(node, project_name, qn, relationships)

    # Process body contents (fields, methods, enum values)
    body = node.child_by_field_name("body")
    if body is not None:
        # Default visibility for class body
        body_visibility = Visibility.PRIVATE if node.type == "class_specifier" else Visibility.PUBLIC

        new_class_stack = [*class_stack, name]

        if node.type == "enum_specifier":
            # Enum body contains enumerator nodes
            _process_enum_body(
                body,
                path=path,
                project_name=project_name,
                module_qn=module_qn,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                enum_name=name,
                entities=entities,
                relationships=relationships,
            )
        else:
            _walk_translation_unit(
                body,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=new_class_stack,
                current_visibility=body_visibility,
                entities=entities,
                relationships=relationships,
            )


def _extract_base_classes(
    node: Node,
    project_name: str,
    class_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Extract base class specifiers and emit INHERITS relationships."""
    for child in node.children:
        if child.type == "base_class_clause":
            for base in child.children:
                if base.type in ("type_identifier", "qualified_identifier"):
                    base_name = node_text(base)
                    if base_name:
                        relationships.append(
                            ParsedRelationship(
                                from_qualified_name=f"{project_name}:{class_qn}",
                                rel_type=RelType.INHERITS,
                                to_name=base_name,
                            )
                        )
                elif base.type == "base_class_specifier":
                    # Some tree-sitter-cpp versions wrap bases in specifier nodes
                    for inner in base.children:
                        if inner.type in ("type_identifier", "qualified_identifier"):
                            base_name = node_text(inner)
                            if base_name:
                                relationships.append(
                                    ParsedRelationship(
                                        from_qualified_name=f"{project_name}:{class_qn}",
                                        rel_type=RelType.INHERITS,
                                        to_name=base_name,
                                    )
                                )


def _process_enum_body(
    body: Node,
    *,
    path: str,
    project_name: str,
    module_qn: str,
    namespace_parts: list[str],
    class_stack: list[str],
    enum_name: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process enum body (enumerator_list) to extract enum members."""
    for child in body.children:
        if child.type == "enumerator":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            name = node_text(name_node)
            if not name:
                continue

            enum_stack = [*class_stack, enum_name]
            qn = _build_qn(module_qn, namespace_parts, enum_stack, name)
            line_start = child.start_point[0] + 1
            line_end = child.end_point[0] + 1

            entities.append(
                ParsedEntity(
                    name=name,
                    qualified_name=f"{project_name}:{qn}",
                    label=NodeLabel.VALUE,
                    kind=ValueKind.ENUM_MEMBER,
                    line_start=line_start,
                    line_end=line_end,
                    file_path=path,
                    visibility=Visibility.PUBLIC,
                )
            )

            # DEFINES from enum -> member
            parent_qn = _build_qn(module_qn, namespace_parts, class_stack, enum_name)
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=f"{project_name}:{parent_qn}",
                    rel_type=RelType.DEFINES,
                    to_name=f"{project_name}:{qn}",
                )
            )


def _process_typedef(
    node: Node,
    *,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    namespace_parts: list[str],
    class_stack: list[str],
    current_visibility: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a type_definition (typedef) node."""
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return
    name = _get_declarator_name(declarator)
    if not name:
        return

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    docstring = _extract_doxygen_comment(node, source)
    qn = _build_qn(module_qn, namespace_parts, class_stack, name)

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
            visibility=current_visibility,
        )
    )

    parent_full_qn = _parent_qn(project_name, module_qn, namespace_parts, class_stack)
    relationships.append(
        ParsedRelationship(
            from_qualified_name=parent_full_qn,
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )


def _process_function(
    node: Node,
    *,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    is_cpp: bool,
    namespace_parts: list[str],
    class_stack: list[str],
    current_visibility: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a function_definition node."""
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return

    tags = _extract_tags(node)
    template_parent = _template_wrapper(node)
    if template_parent is not None:
        tags.append("template")
    scope_parts, name = _get_qualified_declarator_name(declarator)

    if name is None:
        return

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    # Doc comments for templates sit above the template_declaration wrapper
    docstring = _extract_doxygen_comment(template_parent or node, source)
    signature = _extract_function_signature(node, source)

    # Determine if this is a method (inside class body or qualified with class scope)
    is_method = bool(class_stack) or (is_cpp and bool(scope_parts))

    parent_type_name: str | None = None
    if is_method:
        # Method inside class body or qualified (ClassName::method, or
        # Outer::Inner::method for nested-scope definitions)
        actual_class = class_stack[-1] if class_stack else (scope_parts[-1] if scope_parts else "")
        kind = _method_callable_kind(name, actual_class, tags)

        if not class_stack and scope_parts:
            # Out-of-line definition — the class usually lives in another file
            # (header/impl split), so emit its NAME for post-batch resolution
            # (GraphClient.resolve_member_defines).  Fallback parent is this
            # file's Module (namespaces have no nodes).  parent_type_name is
            # the bare innermost class name (last scope part) — the resolver
            # matches on TypeDef name, not a '::'-qualified chain.
            qn = _build_qn(module_qn, namespace_parts, scope_parts, name)
            parent_qn_str = f"{project_name}:{module_qn}"
            parent_type_name = scope_parts[-1]
        else:
            qn = _build_qn(module_qn, namespace_parts, class_stack, name)
            parent_qn_str = _parent_qn(project_name, module_qn, namespace_parts, class_stack)
        visibility = current_visibility
    else:
        kind = CallableKind.FUNCTION
        qn = _build_qn(module_qn, namespace_parts, class_stack, name)
        parent_qn_str = _parent_qn(project_name, module_qn, namespace_parts, class_stack)
        # File-scope static → PRIVATE (internal linkage)
        visibility = Visibility.PRIVATE if "static" in tags and not class_stack else current_visibility

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

    # DEFINES relationship
    relationships.append(
        ParsedRelationship(
            from_qualified_name=parent_qn_str,
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
            properties={"parent_type_name": parent_type_name} if parent_type_name else {},
        )
    )

    # Extract CALLS from function body
    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, f"{project_name}:{qn}", relationships)


def _process_declaration(
    node: Node,
    *,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    is_cpp: bool,
    namespace_parts: list[str],
    class_stack: list[str],
    current_visibility: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a declaration node (global variables, forward declarations, etc.).

    Also handles function declarations embedded inside declarations and
    struct/class/enum specifiers within declarations.
    """
    # Check if this declaration contains a type specifier with a body (inline struct/enum/union/class def)
    for child in node.children:
        if child.type in _TYPE_DEF_NODES and child.child_by_field_name("body") is not None:
            _process_type_def(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=current_visibility,
                entities=entities,
                relationships=relationships,
            )
            return

    # Check if this is a function declaration (has a function_declarator but no body on this node)
    # We only care about actual variable declarations here
    for child in node.children:
        if child.type == "function_declarator":
            # This is a function forward declaration — skip (we only track definitions)
            return

    # Extract declarator name for variable declarations
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return

    # Skip function declarations, including pointer/reference-returning prototypes
    # (`int* alloc(int);` wraps the function_declarator in a pointer_declarator)
    if declarator.type == "function_declarator" or _prototype_declarator(declarator) is not None:
        return

    name = _get_declarator_name(declarator)
    if not name:
        return

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    docstring = _extract_doxygen_comment(node, source)
    qn = _build_qn(module_qn, namespace_parts, class_stack, name)
    tags = _extract_tags(node)

    # File-scope static → PRIVATE
    visibility = Visibility.PRIVATE if "static" in tags and not class_stack else current_visibility

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.VALUE,
            kind=ValueKind.VARIABLE,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            source=node_text(node),
            visibility=visibility,
            tags=tags,
        )
    )

    parent_full_qn = _parent_qn(project_name, module_qn, namespace_parts, class_stack)
    relationships.append(
        ParsedRelationship(
            from_qualified_name=parent_full_qn,
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )


def _process_field_declaration(
    node: Node,
    *,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    namespace_parts: list[str],
    class_stack: list[str],
    current_visibility: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a field_declaration inside a struct/class body."""
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return
    name = _get_declarator_name(declarator)
    if not name:
        return

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    qn = _build_qn(module_qn, namespace_parts, class_stack, name)

    if _prototype_declarator(declarator) is not None:
        # Method declaration without a body (`void draw() const;`) — a Callable,
        # not a field.  Function-pointer members stay on the field path.
        tags = _extract_tags(node)
        kind = _method_callable_kind(name, class_stack[-1] if class_stack else "", tags)
        entities.append(
            ParsedEntity(
                name=name,
                qualified_name=f"{project_name}:{qn}",
                label=NodeLabel.CALLABLE,
                kind=kind,
                line_start=line_start,
                line_end=line_end,
                file_path=path,
                docstring=_extract_doxygen_comment(node, source),
                signature=_extract_function_signature(node, source),
                visibility=current_visibility,
                tags=tags,
            )
        )
    else:
        entities.append(
            ParsedEntity(
                name=name,
                qualified_name=f"{project_name}:{qn}",
                label=NodeLabel.VALUE,
                kind=ValueKind.FIELD,
                line_start=line_start,
                line_end=line_end,
                file_path=path,
                visibility=current_visibility,
            )
        )

    parent_full_qn = _parent_qn(project_name, module_qn, namespace_parts, class_stack)
    relationships.append(
        ParsedRelationship(
            from_qualified_name=parent_full_qn,
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )


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
                    if call_name:
                        relationships.append(
                            ParsedRelationship(
                                from_qualified_name=from_qn,
                                rel_type=RelType.CALLS,
                                to_name=call_name,
                            )
                        )
                elif func.type == "field_expression":
                    # obj.method() — extract method name
                    field = func.child_by_field_name("field")
                    if field is not None:
                        call_name = node_text(field)
                        if call_name:
                            relationships.append(
                                ParsedRelationship(
                                    from_qualified_name=from_qn,
                                    rel_type=RelType.CALLS,
                                    to_name=call_name,
                                )
                            )
                elif func.type == "qualified_identifier":
                    # ns::func() — extract full qualified name
                    call_name = node_text(func)
                    if call_name:
                        relationships.append(
                            ParsedRelationship(
                                from_qualified_name=from_qn,
                                rel_type=RelType.CALLS,
                                to_name=call_name,
                            )
                        )
        # Recurse but don't descend into nested function definitions
        if child.type not in ("function_definition", "class_specifier", "struct_specifier"):
            _extract_calls(child, from_qn, relationships)


# ---------------------------------------------------------------------------
# Language registration
# ---------------------------------------------------------------------------

if _C_AVAILABLE:
    register_language(
        LanguageConfig(
            name="c",
            extensions=_C_EXTENSIONS,
            language=_C_LANGUAGE,
            query=_C_QUERY,
            parse_func=_parse_cpp,
        )
    )

if _CPP_AVAILABLE:
    register_language(
        LanguageConfig(
            name="cpp",
            extensions=_CPP_EXTENSIONS,
            language=_CPP_LANGUAGE,
            query=_CPP_QUERY,
            parse_func=_parse_cpp,
        )
    )
