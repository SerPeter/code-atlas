"""HCL / Terraform support — tree-sitter parser for HCL configuration files.

Extracts the Terraform module namespace from ``.tf`` files (resources, data
sources, variables, outputs, locals, module calls, providers) plus the
statically-resolvable references between them, and variable assignments from
``.tfvars``.

Grammar notes (measured, tree-sitter-hcl ABI 14):
  - This is *base HCL*, not a Terraform dialect. The tree is generic:
    ``config_file > body > block(identifier, string_lit, ..., body)``.
  - There are no ``resource``/``module``/``variable`` node types. Those words
    appear only as a block's first ``identifier`` child, so Terraform semantics
    must be recovered from that identifier plus the block's string labels.
  - A reference is a ``variable_expr`` head followed by ``get_attr`` *siblings*
    (``var`` + ``.region``). The siblings are **not** always wrapped in an
    ``expression`` node — inside ``binary_operation`` (``var.a + local.b``) the
    heads and their ``get_attr`` chains are direct children of the operation.
    That is why reference collection anchors on ``variable_expr`` and walks
    ``next_sibling``, rather than on ``expression``.
  - Interpolations nest an ``expression`` under
    ``template_expr > quoted_template > template_interpolation`` (and the same
    under ``heredoc_template``), so the same walk covers ``"${local.x}-logs"``
    and heredoc bodies with no special casing.

Modeling decisions:
  - **Every address-bearing declaration is a TypeDef**, discriminated by
    ``kind`` (``terraform_resource``, ``terraform_variable``, ...). Two reasons.
    (1) A Terraform block *is* a typed declaration — ``resource
    "aws_s3_bucket" "logs"`` declares an object of type ``aws_s3_bucket``, and a
    ``variable`` block declares a type constraint, not a value (the value
    arrives from ``.tfvars``/CLI at apply time; that assignment is the Value —
    see below). (2) Mechanically, ``USES_TYPE`` is the only existing
    relationship that resolves a *bare name* to a node post-batch, and its
    resolver only matches TypeDef targets — labeling ``var.region`` a Value
    would silently drop every ``var.region`` reference edge.
  - ``.tfvars`` assignments are Values (``terraform_var_value``): they are
    settings, not declarations, and nothing can reference them.
  - **Entity names are Terraform addresses** (``aws_s3_bucket.logs``,
    ``var.region``, ``data.aws_ami.ubuntu``) because the reference resolver
    matches on ``name`` — and because an address is what an engineer searches
    for.
  - **uids are namespaced by directory, not by file**: all ``.tf`` files in one
    directory share a single flat Terraform namespace (``main.tf`` /
    ``variables.tf`` is pure convention), so ``infra/main.tf`` and
    ``infra/variables.tf`` both contribute to ``{project}:infra.<address>``.
    Terraform itself forbids duplicate addresses within a directory, so this
    cannot collide, and moving a block between files in the same module keeps
    its uid.
  - Graph nodes are config *blocks*, not applied instances: one ``resource``
    block with ``for_each`` over 20 subnets stays exactly one node.

Deliberately not extracted (needs runtime state or cross-file resolution the
parser cannot do): ``count``/``each``/``self``/``path``/``terraform`` reference
heads, ``for_each`` expansion, provisioners, ``try()``/``lookup()``
indirection, and the ``module`` block's argument-to-callee-``variable``
data flow (that one needs a post-batch resolver keyed on the callee directory,
which lives in the graph client, not here).

``.hcl`` gets the file-level Module entity only. It is a family of unrelated
dialects (Nomad job specs, Packer templates, Terragrunt, Consul agent config)
where the same block words do not mean Terraform things, so emitting
``terraform_*`` kinds for them would be a lie.
"""

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
from code_atlas.schema import NodeLabel, RelType

if TYPE_CHECKING:
    from tree_sitter import Node

_EXTENSIONS = frozenset({".tf", ".tfvars", ".hcl"})

# Terraform files get a distinct kind from plain HCL (Consul/Nomad/Packer
# configs) so queries can separate them without re-deriving it from the path.
_KINDS: dict[str, str] = {
    ".tf": "terraform_file",
    ".tfvars": "terraform_vars",
    ".hcl": "hcl_file",
}

# Top-level block identifier -> (entity kind, label count, address prefix).
# ``locals`` and ``terraform`` are absent on purpose: neither declares an
# addressable object, so both are handled separately.
_BLOCK_SPECS: dict[str, tuple[str, int, str]] = {
    "resource": ("terraform_resource", 2, ""),
    "data": ("terraform_data", 2, "data"),
    "variable": ("terraform_variable", 1, "var"),
    "output": ("terraform_output", 1, "output"),
    "module": ("terraform_module_call", 1, "module"),
    "provider": ("terraform_provider", 1, "provider"),
}

# Reference heads naming something outside the module namespace: per-instance
# loop state (count/each/self) and interpreter metadata (path/terraform).
# There is no static node to point them at.
_RUNTIME_HEADS = frozenset({"count", "each", "self", "path", "terraform"})

# Reserved reference head -> how many path segments complete the address.
# Anything else at the head position is a managed-resource TYPE, whose address
# is ``TYPE.NAME`` (one segment).
_REF_HEAD_SEGMENTS: dict[str, int] = {"var": 1, "local": 1, "module": 1, "data": 2}


# ---------------------------------------------------------------------------
# Path / naming helpers
# ---------------------------------------------------------------------------


def _module_qualified_name(file_path: str) -> str:
    """Convert a file path to a dotted qualified name, extension folded in.

    ``infra/main.tf`` -> ``infra.main_tf``;  ``us.east/main.tf`` -> ``us_east.main_tf``

    Unlike the code-language modules, the extension is *preserved* (its dot
    replaced) rather than stripped. ``qualified_name`` IS the graph uid, and
    these formats routinely co-exist under one stem — ``main.tf`` beside
    ``main.tfvars``, ``build.sh`` beside ``build.py``. Stripping would make two
    different files claim the same uid, and the later upsert would silently
    overwrite the earlier one.

    Dots are folded in *every* segment, not just the basename, for that same
    reason: ``.`` is the separator being built here, so a directory named
    ``a.b`` would fake a nesting level and make ``a.b/X.tf`` and ``a/b/X.tf``
    claim one uid. This is also what ``_namespace_prefix`` already does, so a
    file's module uid and the namespace of the blocks inside it agree.
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    return ".".join(part.replace(".", "_") for part in p.parts)


def _directory_parts(file_path: str) -> list[str]:
    """Directory components of *file_path* — real path segments, dots intact."""
    parent = PurePosixPath(file_path).parent
    return [part for part in parent.parts if part not in {"", "."}]


def _namespace_prefix(file_path: str) -> str:
    """Dotted uid prefix for the Terraform module *directory* containing the file.

    ``infra/prod/main.tf`` -> ``infra.prod.``;  ``main.tf`` -> ``""``.

    Dots inside a directory name are folded to underscores: a directory named
    ``us.east`` would otherwise be indistinguishable from two nested ones once
    joined into a dotted namespace.
    """
    parts = [part.replace(".", "_") for part in _directory_parts(file_path)]
    return ".".join(parts) + "." if parts else ""


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------


def _top_level(root: Node, node_type: str) -> list[Node]:
    """Direct ``config_file > body`` children of *node_type* (``block``/``attribute``)."""
    body = next((child for child in root.children if child.type == "body"), None)
    if body is None:
        return []
    return [child for child in body.children if child.type == node_type]


def _child_of_type(node: Node | None, node_type: str) -> Node | None:
    if node is None:
        return None
    return next((child for child in node.children if child.type == node_type), None)


def _block_parts(block: Node) -> tuple[str, list[str], Node | None]:
    """Split a block into ``(block type, string labels, body)``.

    Only *direct* children are inspected, so a nested block's identifier and
    labels never leak into its parent's.
    """
    block_type = ""
    labels: list[str] = []
    body: Node | None = None
    for child in block.children:
        if child.type == "identifier" and not block_type:
            block_type = node_text(child)
        elif child.type == "string_lit":
            labels.append(_string_lit_text(child))
        elif child.type == "body":
            body = child
    return block_type, labels, body


def _string_lit_text(string_lit: Node) -> str:
    """Text inside a ``string_lit``, without the surrounding quotes."""
    return "".join(node_text(child) for child in string_lit.children if child.type == "template_literal")


def _string_value(expr: Node | None) -> str | None:
    """Literal string value of an expression, or ``None`` if it is not a plain literal.

    An interpolated string parses as ``template_expr``, not ``literal_value``,
    so it correctly yields ``None`` — its value is not knowable statically.
    """
    node = expr
    while node is not None and node.type in {"expression", "literal_value"}:
        node = next((child for child in node.children if child.type in {"literal_value", "string_lit"}), None)
    if node is None or node.type != "string_lit":
        return None
    return _string_lit_text(node)


def _attribute_name(attribute: Node) -> str | None:
    ident = _child_of_type(attribute, "identifier")
    return node_text(ident) if ident is not None else None


def _attributes(body: Node | None) -> dict[str, Node]:
    """Map direct attribute name -> expression node (last assignment wins)."""
    if body is None:
        return {}
    attrs: dict[str, Node] = {}
    for child in body.children:
        if child.type != "attribute":
            continue
        name = _attribute_name(child)
        expr = _child_of_type(child, "expression")
        if name is not None and expr is not None:
            attrs[name] = expr
    return attrs


def _object_string(expr: Node | None, key: str) -> str | None:
    """Value of *key* in an object expression, when both key and value are literal.

    Used for ``required_providers { aws = { source = "hashicorp/aws" } }``.
    """
    obj = _child_of_type(_child_of_type(expr, "collection_value"), "object")
    if obj is None:
        return None
    for elem in obj.children:
        if elem.type != "object_elem":
            continue
        exprs = [child for child in elem.children if child.type == "expression"]
        if len(exprs) != 2:
            continue
        # An unquoted object key parses as a bare variable_expr; a quoted one as
        # a string literal. Both are legal HCL for the same key.
        ident = _child_of_type(_child_of_type(exprs[0], "variable_expr"), "identifier")
        elem_key = node_text(ident) if ident is not None else _string_value(exprs[0])
        if elem_key == key:
            return _string_value(exprs[1])
    return None


def _node_source(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _block_header(block: Node, source: bytes) -> str | None:
    """``resource "aws_s3_bucket" "logs"`` — everything before the opening brace."""
    start = _child_of_type(block, "block_start")
    if start is None:
        return None
    return source[block.start_byte : start.start_byte].decode("utf-8", errors="replace").strip()


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------


def _chain(head_node: Node) -> tuple[str, list[str]]:
    """Read a reference chain starting at a ``variable_expr`` head.

    Returns ``("var", ["region"])`` for ``var.region``. The walk stops at the
    first non-``get_attr`` sibling, so ``aws_subnet.private[*].id`` yields
    ``("aws_subnet", ["private"])`` — enough to name the address, and the splat
    tail is deliberately dropped.
    """
    ident = _child_of_type(head_node, "identifier")
    head = node_text(ident) if ident is not None else ""
    path: list[str] = []
    sibling = head_node.next_sibling
    while sibling is not None and sibling.type == "get_attr":
        attr = _child_of_type(sibling, "identifier")
        if attr is None:
            break
        path.append(node_text(attr))
        sibling = sibling.next_sibling
    return head, path


def _address(head: str, path: list[str]) -> str | None:
    """Classify a reference chain into a Terraform address, or ``None`` to skip."""
    if not head or head in _RUNTIME_HEADS:
        return None
    segments = _REF_HEAD_SEGMENTS.get(head)
    if segments is None:
        # Managed resource: TYPE.NAME. A bare identifier with no path is a type
        # keyword (`type = string`) or a for-expression binding, not a reference.
        return f"{head}.{path[0]}" if path else None
    if len(path) < segments:
        return None
    return ".".join([head, *path[:segments]])


def _collect_references(node: Node, out: list[str]) -> None:
    """Depth-first collect every Terraform address referenced under *node*.

    Iterative rather than recursive: HCL expression depth is input-controlled
    (nested parentheses, objects, function calls, conditionals all nest one
    grammar node per level, and none of it shows up as indentation the
    framework's pre-parse block-depth guard can see). A recursive walk hits
    Python's frame limit somewhere around 1000 levels and raises RecursionError
    out of ``parse_file``, which drops the entire file rather than one
    expression. The explicit stack is bounded by heap, not by the C stack, so
    raising ``sys.setrecursionlimit`` is not an equivalent fix.

    ``reversed`` keeps children on the stack in document order, which the
    ``dict.fromkeys`` dedup in ``_reference_rels`` relies on.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "variable_expr":
            address = _address(*_chain(current))
            if address is not None:
                out.append(address)
        stack.extend(reversed(current.children))


def _provider_address(expr: Node) -> str | None:
    """``provider = aws.west`` -> ``provider.aws.west``; ``provider = aws`` -> ``provider.aws``.

    Handled separately from the generic walk because the right-hand side of a
    ``provider`` argument is a *provider* address whose head is a local provider
    name — the generic classifier would read ``aws.west`` as a managed resource.
    """
    var = _child_of_type(expr, "variable_expr")
    if var is None:
        return None
    head, path = _chain(var)
    if not head:
        return None
    return ".".join(["provider", head, *path[:1]])


def _body_references(body: Node | None) -> list[str]:
    """Addresses referenced anywhere in a block body, nested blocks included.

    Nested blocks (``lifecycle``, ``dynamic``, ``content``, ...) are walked
    rather than skipped: their references belong to the enclosing declaration,
    which is the node the graph has.
    """
    if body is None:
        return []
    addresses: list[str] = []
    for child in body.children:
        if child.type == "attribute" and _attribute_name(child) == "provider":
            expr = _child_of_type(child, "expression")
            address = _provider_address(expr) if expr is not None else None
            if address is not None:
                addresses.append(address)
            continue
        _collect_references(child, addresses)
    return addresses


def _reference_rels(from_uid: str, addresses: list[str], own_address: str) -> list[ParsedRelationship]:
    """USES_TYPE edges for *addresses*, deduped, minus self-references.

    ``to_name`` is a bare Terraform address, because USES_TYPE is resolved
    post-batch by matching it against TypeDef ``name``. That resolver accepts a
    same-file match or a project-wide *unique* one, so a reference from
    ``main.tf`` to a ``variable`` in ``variables.tf`` resolves as long as no
    second module directory declares the same address — several modules each
    declaring ``var.region`` make the cross-file case ambiguous and the resolver
    drops it. Fixing that needs a directory-scoped resolver in the graph client.

    Unresolvable candidates (a reference to a resource in another project, a
    false positive from an exotic expression) match nothing and create no edge,
    which is the second guard behind the reserved-head filter.
    """
    return [
        ParsedRelationship(from_qualified_name=from_uid, rel_type=RelType.USES_TYPE, to_name=address)
        for address in dict.fromkeys(addresses)
        if address != own_address
    ]


# ---------------------------------------------------------------------------
# Module sources
# ---------------------------------------------------------------------------


def _registry_address(source: str) -> str | None:
    """Registry address (``hashicorp/aws``, ``ns/name/provider``) or ``None``.

    Host-prefixed (``app.terraform.io/ns/name/aws``) and fetcher-style
    (``git::https://...``, ``github.com/org/repo``) sources are rejected: they
    are external too, but ``resolve_imports`` derives the ExternalPackage name
    from the first dot-separated segment, which would mint nodes named
    ``app`` / ``git::https://github``. Better no edge than a garbage one.
    """
    base = source.partition("//")[0]
    segments = base.split("/")
    if len(segments) not in {2, 3} or not all(segments) or ":" in base or "." in segments[0]:
        return None
    return base


def _source_type(source: str) -> str:
    if source.startswith(("./", "../")):
        return "local"
    if _registry_address(source) is not None:
        return "registry"
    return "remote"


def _local_source_path(file_path: str, source: str) -> str:
    """Normalize a local module source against the calling file's directory.

    ``infra/main.tf`` + ``../modules/vpc`` -> ``modules/vpc``.
    """
    parts = _directory_parts(file_path)
    for segment in source.split("/"):
        if segment in {"", "."}:
            continue
        if segment == "..":
            if parts and parts[-1] != "..":
                parts.pop()
            else:
                parts.append("..")
        else:
            parts.append(segment)
    return "/".join(parts)


def _module_source_properties(file_path: str, attrs: dict[str, Node]) -> dict[str, str]:
    """``module_source``/``module_source_type`` (+ ``module_source_path`` for local).

    The ``module_`` prefix is not decoration. ``extra_properties`` is applied to
    the node as ``SET n += e.extra_properties`` *after* every ParsedEntity field
    has been written (``GraphClient._batch_create_entities`` /
    ``_batch_update_entities``), so any key that collides with a ParsedEntity
    field name silently overwrites it. A bare ``source`` key would replace the
    node's ``source`` property — the block's own text, which feeds BM25 and
    vector search — with a Terraform module address, with no error anywhere.
    ``source_type``/``source_path`` do not collide today, but they read as
    qualifiers of that reserved ``source`` property rather than of the
    Terraform ``source`` argument, so they move with it.
    """
    source = _string_value(attrs.get("source"))
    if source is None:
        return {}
    source_type = _source_type(source)
    props = {"module_source": source, "module_source_type": source_type}
    if source_type == "local":
        # The resolved directory, recorded as a property rather than an edge:
        # no relationship type routes "file -> directory", and the child
        # module's own nodes live under that directory's namespace prefix.
        props["module_source_path"] = _local_source_path(file_path, source)
    return props


def _module_source_imports(entity: ParsedEntity) -> list[ParsedRelationship]:
    """IMPORTS edge from a module call to its registry source (an ExternalPackage)."""
    if entity.extra_properties.get("module_source_type") != "registry":
        return []
    address = _registry_address(entity.extra_properties["module_source"])
    if address is None:
        return []
    return [
        ParsedRelationship(
            from_qualified_name=entity.qualified_name,
            rel_type=RelType.IMPORTS,
            to_name=address,
        )
    ]


def _required_provider_imports(body: Node | None, from_uid: str) -> list[ParsedRelationship]:
    """IMPORTS edges for ``terraform { required_providers { aws = { source = ... } } }``."""
    if body is None:
        return []
    rels: list[ParsedRelationship] = []
    for child in body.children:
        if child.type != "block":
            continue
        nested_type, _labels, nested_body = _block_parts(child)
        if nested_type != "required_providers":
            continue
        for expr in _attributes(nested_body).values():
            source = _object_string(expr, "source")
            address = _registry_address(source) if source else None
            if address is None:
                continue
            rels.append(
                ParsedRelationship(
                    from_qualified_name=from_uid,
                    rel_type=RelType.IMPORTS,
                    to_name=address,
                )
            )
    return rels


# ---------------------------------------------------------------------------
# Entity construction
# ---------------------------------------------------------------------------


def _defines(module_uid: str, entity: ParsedEntity) -> ParsedRelationship:
    return ParsedRelationship(
        from_qualified_name=module_uid,
        rel_type=RelType.DEFINES,
        to_name=entity.qualified_name,
    )


def _declaration_entity(
    block: Node,
    block_type: str,
    labels: list[str],
    body: Node | None,
    source: bytes,
    path: str,
    project_name: str,
    prefix: str,
) -> ParsedEntity | None:
    """Build the TypeDef for one addressable top-level block."""
    kind, arity, address_prefix = _BLOCK_SPECS[block_type]
    if len(labels) < arity:
        # Half-typed or malformed block (`resource "aws_s3_bucket" {`) — it has
        # no address, so there is nothing to name a node after.
        return None
    address = ".".join([part for part in (address_prefix, *labels[:arity]) if part])
    attrs = _attributes(body)

    if block_type == "provider":
        alias = _string_value(attrs.get("alias"))
        if alias:
            address = f"{address}.{alias}"

    extra: dict[str, str] = {}
    if block_type in {"resource", "data"}:
        extra["resource_type"] = labels[0]
    elif block_type == "module":
        extra.update(_module_source_properties(path, attrs))

    return ParsedEntity(
        name=address,
        qualified_name=f"{project_name}:{prefix}{address}",
        label=NodeLabel.TYPE_DEF,
        kind=kind,
        line_start=block.start_point[0] + 1,
        line_end=block.end_point[0] + 1,
        file_path=path,
        docstring=_string_value(attrs.get("description")),
        signature=_block_header(block, source),
        source=_node_source(block, source),
        extra_properties=extra,
    )


def _emit_locals(
    body: Node | None,
    source: bytes,
    path: str,
    project_name: str,
    prefix: str,
    module_uid: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """One entity per attribute of a ``locals`` block — ``local.<name>`` each.

    A ``locals`` block is a container, not a declaration: the addressable thing
    is each attribute inside it, and a module may have several ``locals`` blocks
    contributing to one flat namespace.
    """
    if body is None:
        return
    for attribute in body.children:
        if attribute.type != "attribute":
            continue
        name = _attribute_name(attribute)
        if name is None:
            continue
        address = f"local.{name}"
        entity = ParsedEntity(
            name=address,
            qualified_name=f"{project_name}:{prefix}{address}",
            label=NodeLabel.TYPE_DEF,
            kind="terraform_local",
            line_start=attribute.start_point[0] + 1,
            line_end=attribute.end_point[0] + 1,
            file_path=path,
            source=_node_source(attribute, source),
        )
        entities.append(entity)
        relationships.append(_defines(module_uid, entity))
        expr = _child_of_type(attribute, "expression")
        if expr is not None:
            addresses: list[str] = []
            _collect_references(expr, addresses)
            relationships.extend(_reference_rels(entity.qualified_name, addresses, address))


def _extract_terraform(
    root: Node,
    source: bytes,
    path: str,
    project_name: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Extract the Terraform namespace of one ``.tf`` file."""
    prefix = _namespace_prefix(path)
    module_uid = f"{project_name}:{_module_qualified_name(path)}"

    for block in _top_level(root, "block"):
        block_type, labels, body = _block_parts(block)
        if block_type == "locals":
            _emit_locals(body, source, path, project_name, prefix, module_uid, entities, relationships)
        elif block_type == "terraform":
            # Nothing here is addressable; the payload is the provider
            # dependency list. `backend`/`required_version` are not modeled.
            relationships.extend(_required_provider_imports(body, module_uid))
        elif block_type in _BLOCK_SPECS:
            entity = _declaration_entity(block, block_type, labels, body, source, path, project_name, prefix)
            if entity is None:
                continue
            entities.append(entity)
            relationships.append(_defines(module_uid, entity))
            relationships.extend(_reference_rels(entity.qualified_name, _body_references(body), entity.name))
            relationships.extend(_module_source_imports(entity))


def _extract_tfvars(
    root: Node,
    path: str,
    module_uid: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Extract top-level assignments from a ``.tfvars`` file.

    ``.tfvars`` is attributes only — no blocks — and each assignment feeds the
    ``variable`` block of the same name, hence the USES_TYPE edge.

    Neither ``source`` nor ``signature`` is populated: ``.tfvars`` routinely
    holds live credentials, and this parser is not going to copy them into a
    BM25/vector-searchable node. The key, the file and the line are enough to
    answer "where is this variable set".
    """
    for attribute in _top_level(root, "attribute"):
        name = _attribute_name(attribute)
        if name is None:
            continue
        entity = ParsedEntity(
            # File-scoped uid, unlike .tf blocks: `dev.tfvars` and
            # `prod.tfvars` in one directory legitimately set the same key.
            name=name,
            qualified_name=f"{module_uid}.{name}",
            label=NodeLabel.VALUE,
            kind="terraform_var_value",
            line_start=attribute.start_point[0] + 1,
            line_end=attribute.end_point[0] + 1,
            file_path=path,
        )
        entities.append(entity)
        relationships.append(_defines(module_uid, entity))
        relationships.append(
            ParsedRelationship(
                from_qualified_name=entity.qualified_name,
                rel_type=RelType.USES_TYPE,
                to_name=f"var.{name}",
            )
        )


def _parse_hcl(path: str, source: bytes, root: Node, project_name: str) -> ParsedFile:
    """Extract entities from an HCL file."""
    norm_path = path.replace("\\", "/")
    suffix = PurePosixPath(norm_path).suffix.lower()
    language = "hcl"

    if not source.strip():
        # No Module node for an empty file — it would be an unsearchable stub
        # that still costs an embedding.
        return ParsedFile(file_path=norm_path, language=language, entities=[], relationships=[])

    module_uid = f"{project_name}:{_module_qualified_name(norm_path)}"
    entities = [
        ParsedEntity(
            name=PurePosixPath(norm_path).name,
            qualified_name=module_uid,
            label=NodeLabel.MODULE,
            kind=_KINDS.get(suffix, "hcl_file"),
            line_start=1,
            line_end=root.end_point[0] + 1,
            file_path=norm_path,
        )
    ]
    relationships: list[ParsedRelationship] = []

    if suffix == ".tf":
        _extract_terraform(root, source, norm_path, project_name, entities, relationships)
    elif suffix == ".tfvars":
        _extract_tfvars(root, norm_path, module_uid, entities, relationships)

    return ParsedFile(file_path=norm_path, language=language, entities=entities, relationships=relationships)


# ---------------------------------------------------------------------------
# Language registration
# ---------------------------------------------------------------------------

try:
    import tree_sitter_hcl as _ts_hcl
    from tree_sitter import Language, Query

    _HCL_LANGUAGE = Language(_ts_hcl.language())
    _HCL_QUERY = Query(_HCL_LANGUAGE, "(config_file) @root")

    register_language(
        LanguageConfig(
            name="hcl",
            extensions=_EXTENSIONS,
            language=_HCL_LANGUAGE,
            query=_HCL_QUERY,
            parse_func=_parse_hcl,
            comment_node_types=frozenset({"comment"}),
        )
    )
except ImportError:
    pass
