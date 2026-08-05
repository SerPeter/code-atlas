"""Java and C# language support — tree-sitter parsers for JVM-family languages."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from tree_sitter import Language, Query

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
    from collections.abc import Iterable, Iterator

    from tree_sitter import Node

# ---------------------------------------------------------------------------
# Tree-sitter queries (minimal — we rely on walk-based extraction)
# ---------------------------------------------------------------------------

_EMPTY_QUERY_SRC = "(program) @root"
_EMPTY_CS_QUERY_SRC = "(compilation_unit) @root"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_VISIBILITY_KEYWORDS: frozenset[str] = frozenset({"public", "private", "protected", "internal"})

_MODIFIER_TAGS: frozenset[str] = frozenset(
    {
        "abstract",
        "static",
        "final",
        "sealed",
        "partial",
        "async",
        "virtual",
        "override",
        "readonly",
        "volatile",
        "transient",
        "native",
        "synchronized",
        "default",
        "strictfp",
    }
)


def _module_qualified_name(file_path: str) -> str:
    """Convert file path to a module qualified name.

    ``src/com/example/MyClass.java`` -> ``src.com.example.MyClass``
    ``src/Models/User.cs`` -> ``src.Models.User``
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    parts = list(p.parts)
    if parts and parts[-1].endswith((".java", ".cs")):
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return ".".join(parts)


# Conventional Java source-root directory markers (Maven/Gradle 'src/main/java',
# 'src/test/java', or a bare 'src') that are not part of the package/import
# namespace. Checked longest-first so 'src/main/java/...' strips as one unit
# rather than partially matching the bare 'src' marker. Java-only: unlike Python
# and C#, Java's import path is package-based (not file-path-based), so leaving
# the source-root prefix in the stored qualified_name makes intra-project imports
# structurally unable to match (jvm.py finding: Java intra-project imports never
# resolve internally). Mirrors python.py's leading-component `_SOURCE_ROOTS` strip
# (S2) — same convention-based, parse-time-only approach.
_JAVA_SOURCE_ROOT_MARKERS: tuple[tuple[str, ...], ...] = (
    ("src", "main", "java"),
    ("src", "test", "java"),
    ("src",),
)


def _strip_java_source_root(parts: list[str]) -> list[str]:
    """Strip a leading conventional Java source-root marker from module-qn *parts*.

    Only strips when more components follow (a file literally named ``src.java``
    at the repo root keeps qn ``src``, never emptied).
    """
    for marker in _JAVA_SOURCE_ROOT_MARKERS:
        n = len(marker)
        if len(parts) > n and tuple(parts[:n]) == marker:
            return parts[n:]
    return parts


_VISIBILITY_MAP: dict[str, str] = {
    "public": Visibility.PUBLIC,
    "private": Visibility.PRIVATE,
    "protected": Visibility.PROTECTED,
    "internal": Visibility.INTERNAL,
}


def _extract_visibility(node: Node, *, default: str = Visibility.INTERNAL) -> str:
    """Extract visibility from modifier keywords on a declaration node.

    Scans the node's direct children for a ``modifiers`` / ``modifier`` node
    containing visibility keywords.  Returns the default if none found.
    """
    for child in node.children:
        if child.type == "modifiers":
            for mod_child in child.children:
                vis = _VISIBILITY_MAP.get(node_text(mod_child).strip())
                if vis is not None:
                    return vis
        # C# declarations can have individual modifier nodes as direct children
        if child.type == "modifier":
            vis = _VISIBILITY_MAP.get(node_text(child).strip())
            if vis is not None:
                return vis
    return default


def _extract_modifier_tags(node: Node) -> list[str]:
    """Extract modifier keywords (abstract, static, final, etc.) as tags."""
    tags: list[str] = []
    for child in node.children:
        if child.type in ("modifiers", "modifier"):
            for mod_child in child.children:
                text = node_text(mod_child).strip()
                if text in _MODIFIER_TAGS:
                    tags.append(text)
        if child.type == "modifier":
            text = node_text(child).strip()
            if text in _MODIFIER_TAGS:
                tags.append(text)
    return tags


def _extract_annotations_java(node: Node) -> list[str]:
    """Extract Java annotations as tags (e.g. ``annotation:Override``)."""
    tags: list[str] = []
    for child in node.children:
        if child.type == "modifiers":
            for mod_child in child.children:
                if mod_child.type in ("marker_annotation", "annotation"):
                    name_node = mod_child.child_by_field_name("name")
                    if name_node is not None:
                        tags.append(f"annotation:{node_text(name_node)}")
    return tags


def _extract_attributes_csharp(node: Node) -> list[str]:
    """Extract C# attributes as tags (e.g. ``attribute:Serializable``)."""
    tags: list[str] = []
    # Attributes can appear as preceding siblings or children
    prev = node.prev_named_sibling
    while prev is not None and prev.type == "attribute_list":
        for attr_child in prev.children:
            if attr_child.type == "attribute":
                name_node = attr_child.child_by_field_name("name")
                if name_node is not None:
                    tags.append(f"attribute:{node_text(name_node)}")
        prev = prev.prev_named_sibling
    # Also check direct children (some declarations embed attribute_list)
    for child in node.children:
        if child.type == "attribute_list":
            for attr_child in child.children:
                if attr_child.type == "attribute":
                    name_node = attr_child.child_by_field_name("name")
                    if name_node is not None:
                        tags.append(f"attribute:{node_text(name_node)}")
    return tags


def _get_preceding_doc_comment_java(node: Node, source: bytes) -> str | None:
    """Extract Javadoc block comment (``/** ... */``) preceding a declaration."""
    prev = node.prev_named_sibling
    # Also check unnamed siblings
    if prev is None:
        prev = node.prev_sibling
    while prev is not None:
        if prev.type in ("block_comment", "comment"):
            text = source[prev.start_byte : prev.end_byte].decode("utf-8", errors="replace")
            if text.startswith("/**"):
                # Strip /** and */ and leading * on each line
                text = text[3:]
                text = text.removesuffix("*/")
                lines = text.split("\n")
                cleaned = []
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("*"):
                        stripped = stripped[1:].strip()
                    cleaned.append(stripped)
                result = "\n".join(cleaned).strip()
                return result or None
            break
        if prev.type in ("modifiers", "marker_annotation", "annotation"):
            prev = prev.prev_named_sibling
            if prev is None:
                prev_unnamed = node.prev_sibling
                while prev_unnamed is not None and prev_unnamed.type not in ("block_comment", "comment"):
                    prev_unnamed = prev_unnamed.prev_sibling
                prev = prev_unnamed
            continue
        break
    return None


_XML_TAG_RE = re.compile(r"<[^>]+>")


def _get_preceding_doc_comment_csharp(node: Node, source: bytes) -> str | None:
    """Extract C# XML doc comments (``/// ...``) preceding a declaration.

    Strips XML tags to produce a clean docstring.
    """
    doc_lines: list[str] = []
    prev = node.prev_sibling
    # Collect consecutive comment nodes (may skip attribute_list siblings)
    while prev is not None:
        if prev.type == "comment":
            text = source[prev.start_byte : prev.end_byte].decode("utf-8", errors="replace").strip()
            if text.startswith("///"):
                doc_lines.append(text[3:].strip())
                prev = prev.prev_sibling
                continue
            break
        if prev.type == "attribute_list":
            prev = prev.prev_sibling
            continue
        break
    if not doc_lines:
        return None
    doc_lines.reverse()
    raw = " ".join(doc_lines)
    # Strip XML tags
    cleaned = _XML_TAG_RE.sub("", raw).strip()
    # Normalize whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned or None


def _extract_signature_from_node(node: Node, source: bytes) -> str | None:
    """Extract method/constructor signature (declaration line without body)."""
    body = node.child_by_field_name("body")
    if body is not None:
        sig_bytes = source[node.start_byte : body.start_byte].rstrip()
        sig = sig_bytes.decode("utf-8", errors="replace").strip()
        return sig or None
    # No body — use the full node text (abstract/interface methods)
    text = node_text(node).strip()
    # Remove trailing semicolon for cleaner signature
    text = text.removesuffix(";").strip()
    return text or None


# Node types whose same-named siblings form an overload set (Java and C# share these).
_OVERLOADABLE_MEMBER_TYPES: frozenset[str] = frozenset({"method_declaration", "constructor_declaration"})

_TYPE_QUALIFIER_RE = re.compile(r"[\w$]+(?:\.|::)")


def _normalize_type_text(text: str) -> str:
    """Normalize a parameter type for overload suffixes: strip whitespace and package/namespace qualifiers."""
    return _TYPE_QUALIFIER_RE.sub("", "".join(text.split()))


def _overloaded_callable_names(children: Iterable[Node]) -> frozenset[str]:
    """Names declared by 2+ method/constructor declarations among *children*.

    Takes the member sequence rather than the container so C# can pass the
    ``#if``-flattened view (see ``_cs_members``): an overload set split across a
    conditional-compilation boundary is still one overload set.
    """
    counts: Counter[str] = Counter()
    for child in children:
        if child.type in _OVERLOADABLE_MEMBER_TYPES:
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                counts[node_text(name_node)] += 1
    return frozenset(name for name, c in counts.items() if c > 1)


def _java_param_types(node: Node) -> str:
    """Comma-joined normalized parameter types for a Java method/constructor declaration."""
    params = node.child_by_field_name("parameters")
    if params is None:
        return ""
    types: list[str] = []
    for child in params.named_children:
        if child.type == "formal_parameter":
            t = child.child_by_field_name("type")
            if t is not None:
                types.append(_normalize_type_text(node_text(t)))
        elif child.type == "spread_parameter":
            # No fields in the grammar: type is the first named child that isn't modifiers/declarator.
            # Varargs render as 'T[]', not 'T...': the suffix must stay dot-free (dots in
            # qualified_name separate scope segments only), and Java forbids a same-erasure
            # T[] overload next to T..., so '[]' can never collide within an overload set.
            for sc in child.named_children:
                if sc.type not in ("modifiers", "variable_declarator"):
                    types.append(_normalize_type_text(node_text(sc)) + "[]")
                    break
    return ",".join(types)


def _csharp_param_types(node: Node) -> str:
    """Comma-joined normalized parameter types for a C# method/constructor declaration."""
    params = node.child_by_field_name("parameters")
    if params is None:
        return ""
    types: list[str] = []
    pending_params = False
    for i, child in enumerate(params.children):
        if child.type == "parameter":
            t = child.child_by_field_name("type")
            if t is None:
                continue
            mods = [node_text(m) for m in child.children if m.type == "modifier"]
            types.append(" ".join([*mods, _normalize_type_text(node_text(t))]))
        elif child.type == "params":
            pending_params = True
        elif params.field_name_for_child(i) == "type":
            # `params T[] name` is spliced directly into parameter_list (tree-sitter-c-sharp 0.23)
            prefix = "params " if pending_params else ""
            types.append(prefix + _normalize_type_text(node_text(child)))
            pending_params = False
    return ",".join(types)


def _overload_suffix(node: Node, param_types: str) -> str:
    """Disambiguating qn suffix for an overloaded callable: ``[<TypeParams>](<param types>)``."""
    tp = node.child_by_field_name("type_parameters")
    tp_text = "".join(node_text(tp).split()) if tp is not None else ""
    return f"{tp_text}({param_types})"


def _defines_source(parent_qn: str | None, module_qn: str) -> str:
    """The qn a DEFINES edge should originate from, given a naming scope.

    An anonymous inner class contributes a ``$Base`` segment to the qualified
    names of the members it declares, but has no node of its own (ADR-0031: no
    entity for a form with no name a developer could refer to). An edge from
    that segment would point at nothing, so it is stripped back to the nearest
    scope that was actually emitted — the enclosing method, field or type.
    """
    if parent_qn is None:
        return module_qn
    parts = parent_qn.split(".")
    while parts and parts[-1].startswith("$"):
        parts.pop()
    return ".".join(parts) or module_qn


# Nodes that own their members' calls themselves, so a containing body must not
# claim them: a nested declaration is walked separately and gets its own from_qn.
_CALL_SCOPE_BOUNDARIES: frozenset[str] = frozenset(
    {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
        "record_struct_declaration",
        "annotation_type_declaration",
        "struct_declaration",
        "method_declaration",
        "constructor_declaration",
        # Java: the body of `new Base() { ... }`. Its methods are named and become
        # entities of their own, so their calls are not the enclosing method's.
        "class_body",
        # C# local functions carry a name, so ADR-0031 makes each one an entity.
        "local_function_statement",
    }
)


def _emit_call(node: Node, from_qn: str, relationships: list[ParsedRelationship]) -> None:
    """Emit a CALLS edge if *node* is itself a call or an instantiation."""
    if node.type in ("method_invocation", "invocation_expression"):
        # Java: method_invocation has name field or object.method pattern
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=from_qn,
                    rel_type=RelType.CALLS,
                    to_name=node_text(name_node),
                    # Java puts the receiver on the invocation itself.
                    properties=call_receiver_props(node.child_by_field_name("object")),
                )
            )
            return
        # Handle C# invocation expressions via function field
        func = node.child_by_field_name("function")
        if func is None:
            return
        if func.type == "member_access_expression":
            name_part = func.child_by_field_name("name")
            if name_part is not None:
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=from_qn,
                        rel_type=RelType.CALLS,
                        to_name=node_text(name_part),
                        properties=call_receiver_props(func.child_by_field_name("expression")),
                    )
                )
        elif func.type == "identifier":
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=from_qn,
                    rel_type=RelType.CALLS,
                    to_name=node_text(func),
                )
            )
        return

    # Java's object creation: `new Foo()`
    if node.type == "object_creation_expression":
        type_node = node.child_by_field_name("type")
        if type_node is not None:
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=from_qn,
                    rel_type=RelType.CALLS,
                    to_name=node_text(type_node),
                )
            )


def _extract_calls(
    node: Node,
    source: bytes,
    from_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Recursively extract call expressions (method_invocation / invocation_expression).

    *node* itself is a candidate, not just its children: a field initializer is
    handed here as the expression it assigns, and ``Foo f = new Foo();`` passes
    the ``new`` in as the root. Checking children only dropped every such call.

    Lambdas and anonymous methods are walked *through*, not stopped at: their
    calls attribute to the nearest enclosing named scope (ADR-0031).
    """
    _emit_call(node, from_qn, relationships)
    for child in node.children:
        # Recurse but don't descend into nested declarations that own their own calls
        if child.type not in _CALL_SCOPE_BOUNDARIES:
            _extract_calls(child, source, from_qn, relationships)


def _extract_type_refs(
    node: Node,
    from_qn: str,
    rel_type: RelType,
    relationships: list[ParsedRelationship],
) -> None:
    """Recursively find type_identifier / scoped_type_identifier nodes and emit relationships.

    Handles nested structures like ``super_interfaces -> type_list -> type_identifier``
    and ``superclass -> type_identifier``.
    """
    for child in node.children:
        if child.type in ("type_identifier", "scoped_type_identifier"):
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=from_qn,
                    rel_type=rel_type,
                    to_name=node_text(child),
                )
            )
        elif child.type == "generic_type":
            # For generic types like List<String>, extract the base type name
            for gt_child in child.children:
                if gt_child.type in ("type_identifier", "scoped_type_identifier"):
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=from_qn,
                            rel_type=rel_type,
                            to_name=node_text(gt_child),
                        )
                    )
                    break
        elif child.type in ("type_list", "interface_type_list"):
            _extract_type_refs(child, from_qn, rel_type, relationships)


# ---------------------------------------------------------------------------
# Java parser
# ---------------------------------------------------------------------------

# Java type declaration node types -> TypeDefKind mapping
_JAVA_TYPE_NODES: dict[str, str] = {
    "class_declaration": TypeDefKind.CLASS,
    "interface_declaration": TypeDefKind.INTERFACE,
    "enum_declaration": TypeDefKind.ENUM,
    "annotation_type_declaration": TypeDefKind.ANNOTATION,
    "record_declaration": TypeDefKind.RECORD,
}


def _parse_java(
    path: str,
    source: bytes,
    root: Node,
    project_name: str,
) -> ParsedFile:
    """Extract entities and relationships from a Java parse tree."""
    module_qn = ".".join(_strip_java_source_root(_module_qualified_name(path).split(".")))

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

    _walk_java_node(root, path, source, project_name, module_qn, entities, relationships, parent_qn=None)

    # Process imports at top level
    _extract_java_imports(root, project_name, module_qn, relationships)

    return ParsedFile(
        file_path=path,
        language="java",
        entities=entities,
        relationships=relationships,
    )


def _extract_java_imports(
    root: Node,
    project_name: str,
    module_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Extract import declarations from the Java root node."""
    for child in root.children:
        if child.type == "import_declaration":
            # The import path is in a scoped_identifier or identifier child
            for imp_child in child.children:
                if imp_child.type in ("scoped_identifier", "identifier"):
                    import_name = node_text(imp_child)
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=f"{project_name}:{module_qn}",
                            rel_type=RelType.IMPORTS,
                            to_name=import_name,
                        )
                    )
                    break


def _walk_java_node(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    parent_qn: str | None,
) -> None:
    """Recursively walk Java AST nodes to extract entities."""
    overloaded = _overloaded_callable_names(node.children)
    for child in node.children:
        # Type declarations (class, interface, enum, annotation, record)
        if child.type in _JAVA_TYPE_NODES:
            _process_java_type(child, path, source, project_name, module_qn, entities, relationships, parent_qn)
            continue

        # Methods
        if child.type == "method_declaration":
            _process_java_method(
                child, path, source, project_name, module_qn, entities, relationships, parent_qn, overloaded
            )
            continue

        # Constructors
        if child.type == "constructor_declaration":
            _process_java_constructor(
                child, path, source, project_name, module_qn, entities, relationships, parent_qn, overloaded
            )
            continue

        # Fields. An interface's fields are `constant_declaration`, not
        # `field_declaration` — same shape, different node type.
        if child.type in ("field_declaration", "constant_declaration"):
            _process_java_field(child, path, source, project_name, module_qn, entities, relationships, parent_qn)
            continue

        # Enum constants
        if child.type == "enum_constant":
            _process_java_enum_constant(
                child, path, source, project_name, module_qn, entities, relationships, parent_qn
            )
            continue

        # Static and instance initializer blocks run as part of constructing the
        # type, so what they call is the type's, not nobody's.
        if child.type in ("static_initializer", "block"):
            owner = _defines_source(parent_qn, module_qn)
            _extract_calls(child, source, f"{project_name}:{owner}", relationships)
            _scan_java_nested(child, path, source, project_name, module_qn, entities, relationships, owner)
            continue

        # Recurse into class_body, enum_body, interface_body, annotation_type_body.
        # `enum_body_declarations` holds everything after an enum's constants — its
        # methods, constructors and nested types were invisible without it.
        if child.type in (
            "class_body",
            "enum_body",
            "enum_body_declarations",
            "interface_body",
            "annotation_type_body",
        ):
            _walk_java_node(child, path, source, project_name, module_qn, entities, relationships, parent_qn)


def _java_anon_scope(node: Node, seen: Counter[str]) -> str:
    """Naming segment for the anonymous class in ``new Base(...) { ... }``.

    The base type is the only stable handle the source offers — Java's own
    ``Outer$1`` numbering and any line-derived name churn the uid on every edit
    above the expression, which a uid may not do. Generics and package
    qualifiers are dropped so the segment stays a single dot-free token.

    One owner can instantiate the same base twice (``UnsafeAllocator.create``
    tries four ``new UnsafeAllocator()`` bodies in sequence), so repeats within
    that owner take a ``$2``, ``$3`` suffix. That counter is source order inside
    one scope, not a line number: editing anything outside the owner leaves it
    alone, which is the property ADR-0031 requires of a uid.
    """
    type_node = node.child_by_field_name("type")
    name = "anon"
    if type_node is not None:
        text = node_text(type_node).split("<", 1)[0].strip().rsplit(".", 1)[-1].strip()
        name = text or "anon"
    seen[name] += 1
    return f"${name}" if seen[name] == 1 else f"${name}${seen[name]}"


def _scan_java_nested(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    owner_qn: str,
    seen: Counter[str] | None = None,
) -> None:
    """Find named declarations buried inside a Java statement or expression body.

    ``_walk_java_node`` only reaches members of a type body, which leaves three
    real shapes uncaptured: a local class declared inside a method, a record or
    class declared inside a lambda, and — the largest by an order of magnitude —
    the methods of an anonymous inner class. Those methods carry names, so
    ADR-0031 makes them entities; the class holding them does not, so it
    contributes a ``$Base`` naming segment and no node.

    *seen* counts anonymous bases already named under this owner and is created
    per owner, so the disambiguating suffix cannot leak between methods.
    """
    if seen is None:
        seen = Counter()

    if node.type in _JAVA_TYPE_NODES:
        _process_java_type(node, path, source, project_name, module_qn, entities, relationships, owner_qn)
        return

    if node.type == "object_creation_expression":
        body = next((c for c in node.children if c.type == "class_body"), None)
        if body is not None:
            _walk_java_node(
                body,
                path,
                source,
                project_name,
                module_qn,
                entities,
                relationships,
                parent_qn=f"{owner_qn}.{_java_anon_scope(node, seen)}",
            )
        # A second anonymous class can sit in the constructor arguments of the first.
        args = node.child_by_field_name("arguments")
        if args is not None:
            _scan_java_nested(args, path, source, project_name, module_qn, entities, relationships, owner_qn, seen)
        return

    for child in node.children:
        _scan_java_nested(child, path, source, project_name, module_qn, entities, relationships, owner_qn, seen)


def _process_java_type(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    parent_qn: str | None,
) -> None:
    """Process a Java type declaration (class, interface, enum, annotation, record)."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    kind = _JAVA_TYPE_NODES[node.type]
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    qn = f"{module_qn}.{name}" if parent_qn is None else f"{parent_qn}.{name}"
    full_qn = f"{project_name}:{qn}"

    visibility = _extract_visibility(node, default=Visibility.INTERNAL)
    tags = _extract_modifier_tags(node) + _extract_annotations_java(node)
    docstring = _get_preceding_doc_comment_java(node, source)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=full_qn,
            label=NodeLabel.TYPE_DEF,
            kind=kind,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            visibility=visibility,
            tags=tags,
        )
    )

    # DEFINES relationship from parent -> this type
    parent_full_qn = f"{project_name}:{_defines_source(parent_qn, module_qn)}"
    relationships.append(
        ParsedRelationship(
            from_qualified_name=parent_full_qn,
            rel_type=RelType.DEFINES,
            to_name=full_qn,
        )
    )

    # Superclass -> INHERITS (Java: superclass field)
    superclass = node.child_by_field_name("superclass")
    if superclass is not None:
        _extract_type_refs(superclass, full_qn, RelType.INHERITS, relationships)

    # Interfaces -> IMPLEMENTS
    interfaces = node.child_by_field_name("interfaces")
    if interfaces is not None:
        _extract_type_refs(interfaces, full_qn, RelType.IMPLEMENTS, relationships)

    # For interfaces: "extends" means INHERITS for the extending interface
    if node.type == "interface_declaration":
        extends_node = node.child_by_field_name("extends")
        if extends_node is not None:
            _extract_type_refs(extends_node, full_qn, RelType.INHERITS, relationships)

    # Recurse into body
    body = node.child_by_field_name("body")
    if body is not None:
        _walk_java_node(body, path, source, project_name, module_qn, entities, relationships, parent_qn=qn)


def _process_java_method(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    parent_qn: str | None,
    overloaded: frozenset[str],
) -> None:
    """Process a Java method_declaration."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    is_method = parent_qn is not None
    kind = CallableKind.METHOD if is_method else CallableKind.FUNCTION
    qn_name = f"{name}{_overload_suffix(node, _java_param_types(node))}" if name in overloaded else name
    qn = f"{parent_qn}.{qn_name}" if parent_qn else f"{module_qn}.{qn_name}"
    full_qn = f"{project_name}:{qn}"

    # Check for static -> STATIC_METHOD
    mod_tags = _extract_modifier_tags(node)
    if is_method and "static" in mod_tags:
        kind = CallableKind.STATIC_METHOD

    visibility = _extract_visibility(node, default=Visibility.INTERNAL)
    tags = mod_tags + _extract_annotations_java(node)
    docstring = _get_preceding_doc_comment_java(node, source)
    signature = _extract_signature_from_node(node, source)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=full_qn,
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
    define_from = f"{project_name}:{_defines_source(parent_qn, module_qn)}"
    relationships.append(
        ParsedRelationship(
            from_qualified_name=define_from,
            rel_type=RelType.DEFINES,
            to_name=full_qn,
        )
    )

    # Extract calls from body
    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, source, full_qn, relationships)
        _scan_java_nested(body, path, source, project_name, module_qn, entities, relationships, qn)


def _process_java_constructor(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    parent_qn: str | None,
    overloaded: frozenset[str],
) -> None:
    """Process a Java constructor_declaration."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    qn_name = f"{name}{_overload_suffix(node, _java_param_types(node))}" if name in overloaded else name
    qn = f"{parent_qn}.{qn_name}" if parent_qn else f"{module_qn}.{qn_name}"
    full_qn = f"{project_name}:{qn}"

    visibility = _extract_visibility(node, default=Visibility.INTERNAL)
    tags = _extract_modifier_tags(node) + _extract_annotations_java(node)
    docstring = _get_preceding_doc_comment_java(node, source)
    signature = _extract_signature_from_node(node, source)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=full_qn,
            label=NodeLabel.CALLABLE,
            kind=CallableKind.CONSTRUCTOR,
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

    define_from = f"{project_name}:{_defines_source(parent_qn, module_qn)}"
    relationships.append(
        ParsedRelationship(
            from_qualified_name=define_from,
            rel_type=RelType.DEFINES,
            to_name=full_qn,
        )
    )

    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, source, full_qn, relationships)
        _scan_java_nested(body, path, source, project_name, module_qn, entities, relationships, qn)


def _process_java_field(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    parent_qn: str | None,
) -> None:
    """Process a Java field_declaration or interface constant_declaration."""
    mod_tags = _extract_modifier_tags(node)
    # An interface's fields are implicitly public static final, and say so by
    # carrying no modifiers at all.
    in_interface = node.type == "constant_declaration"
    is_static = in_interface or "static" in mod_tags
    is_final = in_interface or "final" in mod_tags

    kind = ValueKind.CONSTANT if (is_static and is_final) else ValueKind.FIELD
    visibility = _extract_visibility(node, default=Visibility.PUBLIC if in_interface else Visibility.INTERNAL)
    tags = mod_tags + _extract_annotations_java(node)

    # field_declaration -> declarator: variable_declarator with name
    for child in node.children:
        if child.type == "variable_declarator":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            name = node_text(name_node)
            line_start = child.start_point[0] + 1
            line_end = child.end_point[0] + 1

            qn = f"{parent_qn}.{name}" if parent_qn else f"{module_qn}.{name}"
            full_qn = f"{project_name}:{qn}"

            entities.append(
                ParsedEntity(
                    name=name,
                    qualified_name=full_qn,
                    label=NodeLabel.VALUE,
                    kind=kind,
                    line_start=line_start,
                    line_end=line_end,
                    file_path=path,
                    source=node_text(node),
                    visibility=visibility,
                    tags=tags,
                )
            )

            define_from = f"{project_name}:{_defines_source(parent_qn, module_qn)}"
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=define_from,
                    rel_type=RelType.DEFINES,
                    to_name=full_qn,
                )
            )

            # An initializer runs when the type is constructed or loaded, and it is
            # where `new Base() { ... }` most often appears — the field is the
            # nearest named scope, so it owns both the calls and the anonymous
            # class's members.
            value = child.child_by_field_name("value")
            if value is not None:
                _extract_calls(value, source, full_qn, relationships)
                _scan_java_nested(value, path, source, project_name, module_qn, entities, relationships, qn)


def _process_java_enum_constant(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    parent_qn: str | None,
) -> None:
    """Process a Java enum_constant."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    qn = f"{parent_qn}.{name}" if parent_qn else f"{module_qn}.{name}"
    full_qn = f"{project_name}:{qn}"

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=full_qn,
            label=NodeLabel.VALUE,
            kind=ValueKind.ENUM_MEMBER,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            visibility=Visibility.PUBLIC,
        )
    )

    define_from = f"{project_name}:{_defines_source(parent_qn, module_qn)}"
    relationships.append(
        ParsedRelationship(
            from_qualified_name=define_from,
            rel_type=RelType.DEFINES,
            to_name=full_qn,
        )
    )

    # `CONSTANT { ... }` is a per-constant subclass. Its methods are named and the
    # constant is, so they nest under it rather than needing an anonymous segment.
    body = node.child_by_field_name("body")
    if body is not None:
        _walk_java_node(body, path, source, project_name, module_qn, entities, relationships, parent_qn=qn)
    args = node.child_by_field_name("arguments")
    if args is not None:
        _extract_calls(args, source, full_qn, relationships)
        _scan_java_nested(args, path, source, project_name, module_qn, entities, relationships, qn)


# ---------------------------------------------------------------------------
# C# parser
# ---------------------------------------------------------------------------

# C# type declaration node types -> TypeDefKind mapping
_CS_TYPE_NODES: dict[str, str] = {
    "class_declaration": TypeDefKind.CLASS,
    "struct_declaration": TypeDefKind.STRUCT,
    "interface_declaration": TypeDefKind.INTERFACE,
    "enum_declaration": TypeDefKind.ENUM,
    "record_declaration": TypeDefKind.RECORD,
    "record_struct_declaration": TypeDefKind.RECORD,
}


def _parse_csharp(
    path: str,
    source: bytes,
    root: Node,
    project_name: str,
) -> ParsedFile:
    """Extract entities and relationships from a C# parse tree."""
    module_qn = _module_qualified_name(path)

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

    _walk_csharp_node(root, path, source, project_name, module_qn, entities, relationships, parent_qn=None)

    # Process using directives at top level
    _extract_csharp_usings(root, project_name, module_qn, relationships)

    return ParsedFile(
        file_path=path,
        language="csharp",
        entities=entities,
        relationships=relationships,
    )


def _extract_csharp_usings(
    root: Node,
    project_name: str,
    module_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Extract using directives from C# compilation unit."""
    for child in root.children:
        if child.type == "using_directive":
            # The namespace/type name is typically in an identifier or qualified_name child
            for uc in child.children:
                if uc.type in ("qualified_name", "identifier"):
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=f"{project_name}:{module_qn}",
                            rel_type=RelType.IMPORTS,
                            to_name=node_text(uc),
                        )
                    )
                    break


# `#if` / `#else` / `#elif` are not statements in tree-sitter-c-sharp — everything a
# region guards is nested *inside* the preproc node. A walker that reads direct
# children therefore stops at the `#if` and never sees the namespace, type or member
# behind it, which on Newtonsoft.Json hid 2,777 of 7,339 named declarations.
_CS_PREPROC_BRANCHES: frozenset[str] = frozenset({"preproc_if", "preproc_else", "preproc_elif"})


def _cs_members(node: Node) -> Iterator[Node]:
    """*node*'s children with conditional-compilation regions flattened away.

    Every branch is walked, not just the first: which one compiles depends on a
    build configuration the indexer does not have, and a graph of one
    configuration is the wrong graph for whoever is reading the other.
    """
    for child in node.children:
        if child.type in _CS_PREPROC_BRANCHES:
            yield from _cs_members(child)
        else:
            yield child


def _walk_csharp_namespace(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    parent_qn: str | None,
    current_ns_qn: str,
) -> str:
    """Walk a namespace declaration; return the naming scope its siblings inherit.

    The namespace name folds into a naming prefix for any top-level (non-nested)
    type declared within. Namespaces have no graph node of their own (unlike
    Python packages), so DEFINES/parent_qn semantics are unaffected — the module
    (file) entity stays the structural parent; only the *naming* prefix changes
    (fixes same-named types in different namespaces colliding on uid).
    """
    name_node = node.child_by_field_name("name")
    ns_name = node_text(name_node) if name_node is not None else None
    child_ns_qn = f"{current_ns_qn}.{ns_name}" if current_ns_qn and ns_name else (ns_name or current_ns_qn)
    body = node.child_by_field_name("body")
    if body is None:
        # File-scoped namespace ('namespace Foo;', no braces): there is no body
        # node — the declarations it covers are the remaining siblings, so the
        # caller extends its scope instead of recursing.
        return child_ns_qn
    # Braced namespace: recurse into its own scope only.
    _walk_csharp_node(body, path, source, project_name, module_qn, entities, relationships, parent_qn, child_ns_qn)
    return current_ns_qn


def _walk_csharp_node(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    parent_qn: str | None,
    namespace_qn: str = "",
) -> None:
    """Recursively walk C# AST nodes to extract entities."""
    members = list(_cs_members(node))
    overloaded = _overloaded_callable_names(members)
    current_ns_qn = namespace_qn
    for child in members:
        if child.type in ("namespace_declaration", "file_scoped_namespace_declaration"):
            current_ns_qn = _walk_csharp_namespace(
                child, path, source, project_name, module_qn, entities, relationships, parent_qn, current_ns_qn
            )
            continue

        # Type declarations
        if child.type in _CS_TYPE_NODES:
            _process_csharp_type(
                child, path, source, project_name, module_qn, entities, relationships, parent_qn, current_ns_qn
            )
            continue

        # Methods
        if child.type == "method_declaration":
            _process_csharp_method(
                child, path, source, project_name, module_qn, entities, relationships, parent_qn, overloaded
            )
            continue

        # Constructors
        if child.type == "constructor_declaration":
            _process_csharp_constructor(
                child, path, source, project_name, module_qn, entities, relationships, parent_qn, overloaded
            )
            continue

        # Destructors
        if child.type == "destructor_declaration":
            _process_csharp_destructor(child, path, source, project_name, module_qn, entities, relationships, parent_qn)
            continue

        # Properties
        if child.type == "property_declaration":
            _process_csharp_property(child, path, source, project_name, module_qn, entities, relationships, parent_qn)
            continue

        # Fields
        if child.type == "field_declaration":
            _process_csharp_field(child, path, source, project_name, module_qn, entities, relationships, parent_qn)
            continue

        # Enum members
        if child.type == "enum_member_declaration":
            _process_csharp_enum_member(child, path, project_name, module_qn, entities, relationships, parent_qn)
            continue

        # Operators, conversions and indexers get no entity of their own, but their
        # bodies are full of calls. The nearest named scope enclosing them is the
        # type, so that is where those calls belong (ADR-0031).
        if child.type in (
            "operator_declaration",
            "conversion_operator_declaration",
            "indexer_declaration",
            "event_declaration",
        ):
            _extract_calls(child, source, f"{project_name}:{parent_qn or module_qn}", relationships)
            continue

        # Recurse into declaration_list and similar containers
        if child.type in ("declaration_list", "global_statement"):
            _walk_csharp_node(
                child, path, source, project_name, module_qn, entities, relationships, parent_qn, current_ns_qn
            )


# A C# declaration can never be named with a reserved word, so one that is did not
# come from a declaration. tree-sitter-c-sharp recovers `else if (x) { ... }` at the
# head of a `#if` branch as a local_function_statement returning `else` and named
# `if` — 15 times across Newtonsoft.Json (DictionaryWrapper.cs:131, JsonArrayContract
# .cs:189, IsoDateTimeConverter.cs:94, ...). Emitting those raises the measured
# capture rate while putting Callables named `if` in the graph, which is worse than
# the miss.
_CS_RESERVED_NAMES: frozenset[str] = frozenset(
    {
        "if",
        "else",
        "for",
        "foreach",
        "while",
        "do",
        "switch",
        "case",
        "try",
        "catch",
        "finally",
        "return",
        "lock",
        "using",
        "fixed",
        "checked",
        "unchecked",
    }
)


def _scan_csharp_local_functions(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    owner_qn: str,
) -> None:
    """Emit entities for the ``local_function_statement`` nodes inside a body.

    A C# local function has a name callers in that scope use, so ADR-0031 makes
    it an entity — the same treatment Python's nested ``def`` already gets. It
    nests under the callable that declares it, and its own body is scanned in
    turn, so ``Outer.Inner.Innermost`` falls out of the recursion.
    """
    for child in node.children:
        if child.type in _CS_TYPE_NODES or child.type == "method_declaration":
            continue
        if child.type == "local_function_statement":
            name_node = child.child_by_field_name("name")
            name = node_text(name_node) if name_node is not None else ""
            if not name or name in _CS_RESERVED_NAMES:
                # Not a declaration — a recovered statement. It still holds real
                # calls, and `_extract_calls` will not enter it on the owner's
                # behalf (local_function_statement is a call-scope boundary), so
                # attribute them here before moving on.
                _extract_calls(child, source, f"{project_name}:{owner_qn}", relationships)
                _scan_csharp_local_functions(
                    child, path, source, project_name, module_qn, entities, relationships, owner_qn
                )
                continue
            qn = f"{owner_qn}.{name}"
            full_qn = f"{project_name}:{qn}"
            entities.append(
                ParsedEntity(
                    name=name,
                    qualified_name=full_qn,
                    label=NodeLabel.CALLABLE,
                    kind=CallableKind.FUNCTION,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    file_path=path,
                    signature=_extract_signature_from_node(child, source),
                    source=node_text(child),
                    visibility=Visibility.PRIVATE,
                    tags=_extract_modifier_tags(child),
                )
            )
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=f"{project_name}:{owner_qn}",
                    rel_type=RelType.DEFINES,
                    to_name=full_qn,
                )
            )
            body = child.child_by_field_name("body")
            if body is not None:
                _extract_calls(body, source, full_qn, relationships)
                _scan_csharp_local_functions(body, path, source, project_name, module_qn, entities, relationships, qn)
            continue
        _scan_csharp_local_functions(child, path, source, project_name, module_qn, entities, relationships, owner_qn)


def _process_csharp_type(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    parent_qn: str | None,
    namespace_qn: str = "",
) -> None:
    """Process a C# type declaration."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    kind = _CS_TYPE_NODES[node.type]
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    ns_prefix = f"{namespace_qn}." if namespace_qn else ""
    qn = f"{parent_qn}.{name}" if parent_qn is not None else f"{module_qn}.{ns_prefix}{name}"
    full_qn = f"{project_name}:{qn}"

    visibility = _extract_visibility(node, default=Visibility.PRIVATE)
    mod_tags = _extract_modifier_tags(node)
    attr_tags = _extract_attributes_csharp(node)
    tags = mod_tags + attr_tags
    docstring = _get_preceding_doc_comment_csharp(node, source)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=full_qn,
            label=NodeLabel.TYPE_DEF,
            kind=kind,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            visibility=visibility,
            tags=tags,
        )
    )

    # DEFINES relationship
    parent_full_qn = f"{project_name}:{parent_qn}" if parent_qn else f"{project_name}:{module_qn}"
    relationships.append(
        ParsedRelationship(
            from_qualified_name=parent_full_qn,
            rel_type=RelType.DEFINES,
            to_name=full_qn,
        )
    )

    # Base list -> INHERITS / IMPLEMENTS
    # base_list is a child node (not a named field in tree-sitter-c-sharp)
    bases = None
    for child in node.children:
        if child.type == "base_list":
            bases = child
            break
    if bases is not None:
        for base_child in bases.children:
            if base_child.type in ("identifier", "qualified_name", "generic_name"):
                base_name = node_text(base_child)
                # For generic_name, extract just the identifier part
                if base_child.type == "generic_name":
                    id_node = base_child.child_by_field_name("name")
                    if id_node is not None:
                        base_name = node_text(id_node)
                # In C#, interfaces conventionally start with 'I' and are listed in bases
                # For interfaces extending other interfaces, use INHERITS
                if node.type == "interface_declaration":
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=full_qn,
                            rel_type=RelType.INHERITS,
                            to_name=base_name,
                        )
                    )
                elif base_name.startswith("I") and len(base_name) > 1 and base_name[1].isupper():
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=full_qn,
                            rel_type=RelType.IMPLEMENTS,
                            to_name=base_name,
                        )
                    )
                else:
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=full_qn,
                            rel_type=RelType.INHERITS,
                            to_name=base_name,
                        )
                    )

    # Recurse into body
    body = node.child_by_field_name("body")
    if body is not None:
        _walk_csharp_node(body, path, source, project_name, module_qn, entities, relationships, parent_qn=qn)

    # Note: for enums, body is enum_member_declaration_list, already handled above


def _process_csharp_method(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    parent_qn: str | None,
    overloaded: frozenset[str],
) -> None:
    """Process a C# method_declaration."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    is_method = parent_qn is not None
    mod_tags = _extract_modifier_tags(node)
    if is_method and "static" in mod_tags:
        kind = CallableKind.STATIC_METHOD
    else:
        kind = CallableKind.METHOD if is_method else CallableKind.FUNCTION

    qn_name = f"{name}{_overload_suffix(node, _csharp_param_types(node))}" if name in overloaded else name
    qn = f"{parent_qn}.{qn_name}" if parent_qn else f"{module_qn}.{qn_name}"
    full_qn = f"{project_name}:{qn}"

    visibility = _extract_visibility(node, default=Visibility.PRIVATE)
    attr_tags = _extract_attributes_csharp(node)
    tags = mod_tags + attr_tags
    docstring = _get_preceding_doc_comment_csharp(node, source)
    signature = _extract_signature_from_node(node, source)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=full_qn,
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

    define_from = f"{project_name}:{parent_qn}" if parent_qn else f"{project_name}:{module_qn}"
    relationships.append(
        ParsedRelationship(
            from_qualified_name=define_from,
            rel_type=RelType.DEFINES,
            to_name=full_qn,
        )
    )

    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, source, full_qn, relationships)
        _scan_csharp_local_functions(body, path, source, project_name, module_qn, entities, relationships, qn)


def _process_csharp_constructor(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    parent_qn: str | None,
    overloaded: frozenset[str],
) -> None:
    """Process a C# constructor_declaration."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    qn_name = f"{name}{_overload_suffix(node, _csharp_param_types(node))}" if name in overloaded else name
    qn = f"{parent_qn}.{qn_name}" if parent_qn else f"{module_qn}.{qn_name}"
    full_qn = f"{project_name}:{qn}"

    visibility = _extract_visibility(node, default=Visibility.PRIVATE)
    tags = _extract_modifier_tags(node) + _extract_attributes_csharp(node)
    docstring = _get_preceding_doc_comment_csharp(node, source)
    signature = _extract_signature_from_node(node, source)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=full_qn,
            label=NodeLabel.CALLABLE,
            kind=CallableKind.CONSTRUCTOR,
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

    define_from = f"{project_name}:{parent_qn}" if parent_qn else f"{project_name}:{module_qn}"
    relationships.append(
        ParsedRelationship(
            from_qualified_name=define_from,
            rel_type=RelType.DEFINES,
            to_name=full_qn,
        )
    )

    # A constructor's initializer (`: base(...)` / `: this(...)`) is a call, and it
    # sits outside the body — scanning the whole node covers both.
    _extract_calls(node, source, full_qn, relationships)
    body = node.child_by_field_name("body")
    if body is not None:
        _scan_csharp_local_functions(body, path, source, project_name, module_qn, entities, relationships, qn)


def _process_csharp_destructor(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    parent_qn: str | None,
) -> None:
    """Process a C# destructor_declaration (~ClassName())."""
    name_node = node.child_by_field_name("name")
    name = node_text(name_node) if name_node is not None else "~Finalize"
    # Prefix with ~ for clarity
    display_name = f"~{name}"
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    qn = f"{parent_qn}.{display_name}" if parent_qn else f"{module_qn}.{display_name}"
    full_qn = f"{project_name}:{qn}"

    signature = _extract_signature_from_node(node, source)

    entities.append(
        ParsedEntity(
            name=display_name,
            qualified_name=full_qn,
            label=NodeLabel.CALLABLE,
            kind=CallableKind.DESTRUCTOR,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            signature=signature,
            source=node_text(node),
            visibility=Visibility.PRIVATE,
        )
    )

    define_from = f"{project_name}:{parent_qn}" if parent_qn else f"{project_name}:{module_qn}"
    relationships.append(
        ParsedRelationship(
            from_qualified_name=define_from,
            rel_type=RelType.DEFINES,
            to_name=full_qn,
        )
    )

    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, source, full_qn, relationships)


def _process_csharp_property(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    parent_qn: str | None,
) -> None:
    """Process a C# property_declaration."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    qn = f"{parent_qn}.{name}" if parent_qn else f"{module_qn}.{name}"
    full_qn = f"{project_name}:{qn}"

    visibility = _extract_visibility(node, default=Visibility.PRIVATE)
    tags = _extract_modifier_tags(node) + _extract_attributes_csharp(node)
    docstring = _get_preceding_doc_comment_csharp(node, source)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=full_qn,
            label=NodeLabel.CALLABLE,
            kind=CallableKind.PROPERTY,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            source=node_text(node),
            visibility=visibility,
            tags=tags,
        )
    )

    define_from = f"{project_name}:{parent_qn}" if parent_qn else f"{project_name}:{module_qn}"
    relationships.append(
        ParsedRelationship(
            from_qualified_name=define_from,
            rel_type=RelType.DEFINES,
            to_name=full_qn,
        )
    )

    # A property is already a Callable here, so its accessors' calls have an owner
    # to attribute to. Scanning the whole node also picks up an expression body
    # (`=> Foo()`) and an initializer, both of which live outside `accessor_list`.
    _extract_calls(node, source, full_qn, relationships)


def _process_csharp_field(
    node: Node,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    parent_qn: str | None,
) -> None:
    """Process a C# field_declaration."""
    mod_tags = _extract_modifier_tags(node)
    is_static = "static" in mod_tags
    is_readonly = "readonly" in mod_tags

    kind = ValueKind.CONSTANT if (is_static and is_readonly) else ValueKind.FIELD
    visibility = _extract_visibility(node, default=Visibility.PRIVATE)
    tags = mod_tags + _extract_attributes_csharp(node)

    # field_declaration -> declaration (variable_declaration) -> variable_declarator(s)
    for child in node.children:
        if child.type == "variable_declaration":
            for var_decl in child.children:
                if var_decl.type == "variable_declarator":
                    name_node = var_decl.child_by_field_name("name")
                    # tree-sitter-c-sharp: variable_declarator has identifier as first child
                    if name_node is None:
                        for vc in var_decl.children:
                            if vc.type == "identifier":
                                name_node = vc
                                break
                    if name_node is None:
                        continue
                    name = node_text(name_node)
                    line_start = var_decl.start_point[0] + 1
                    line_end = var_decl.end_point[0] + 1

                    qn = f"{parent_qn}.{name}" if parent_qn else f"{module_qn}.{name}"
                    full_qn = f"{project_name}:{qn}"

                    entities.append(
                        ParsedEntity(
                            name=name,
                            qualified_name=full_qn,
                            label=NodeLabel.VALUE,
                            kind=kind,
                            line_start=line_start,
                            line_end=line_end,
                            file_path=path,
                            source=node_text(node),
                            visibility=visibility,
                            tags=tags,
                        )
                    )

                    define_from = f"{project_name}:{parent_qn}" if parent_qn else f"{project_name}:{module_qn}"
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=define_from,
                            rel_type=RelType.DEFINES,
                            to_name=full_qn,
                        )
                    )

                    # An initializer runs at construction or type-load time. The field
                    # is the nearest named scope, so it owns what the initializer calls.
                    # The value is spliced straight after the `=` token — there is no
                    # wrapper node to hand to `_extract_calls`.
                    after_eq = False
                    for vc in var_decl.children:
                        if after_eq:
                            _extract_calls(vc, source, full_qn, relationships)
                        elif vc.type == "=":
                            after_eq = True


def _process_csharp_enum_member(
    node: Node,
    path: str,
    project_name: str,
    module_qn: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    parent_qn: str | None,
) -> None:
    """Process a C# enum_member_declaration."""
    name_node = node.child_by_field_name("name")
    # tree-sitter-c-sharp: enum_member_declaration has identifier as first named child
    if name_node is None:
        for child in node.children:
            if child.type == "identifier":
                name_node = child
                break
    if name_node is None:
        return
    name = node_text(name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1

    qn = f"{parent_qn}.{name}" if parent_qn else f"{module_qn}.{name}"
    full_qn = f"{project_name}:{qn}"

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=full_qn,
            label=NodeLabel.VALUE,
            kind=ValueKind.ENUM_MEMBER,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            visibility=Visibility.PUBLIC,
        )
    )

    define_from = f"{project_name}:{parent_qn}" if parent_qn else f"{project_name}:{module_qn}"
    relationships.append(
        ParsedRelationship(
            from_qualified_name=define_from,
            rel_type=RelType.DEFINES,
            to_name=full_qn,
        )
    )


# ---------------------------------------------------------------------------
# Language registration — wrapped in try/except for optional grammars
# ---------------------------------------------------------------------------

try:
    import tree_sitter_java as ts_java

    _JAVA_LANGUAGE = Language(ts_java.language())
    _JAVA_QUERY = Query(_JAVA_LANGUAGE, _EMPTY_QUERY_SRC)

    register_language(
        LanguageConfig(
            name="java",
            extensions=frozenset({".java"}),
            language=_JAVA_LANGUAGE,
            query=_JAVA_QUERY,
            parse_func=_parse_java,
        )
    )
except ImportError:
    pass

try:
    import tree_sitter_c_sharp as ts_cs

    _CS_LANGUAGE = Language(ts_cs.language())
    _CS_QUERY = Query(_CS_LANGUAGE, _EMPTY_CS_QUERY_SRC)

    register_language(
        LanguageConfig(
            name="csharp",
            extensions=frozenset({".cs"}),
            language=_CS_LANGUAGE,
            query=_CS_QUERY,
            parse_func=_parse_csharp,
        )
    )
except ImportError:
    pass
