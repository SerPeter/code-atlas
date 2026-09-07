"""HTML-family markup — Lightning Web Component templates today, Aura and Visualforce next.

A Salesforce component's composition lives in its markup and nowhere else: the
``.js`` says what the component *does*, the ``.html`` says what it is *made of*.
Until this module existed ``.html`` reached no parser at all, so the component
tree — the thing "how is this app built" mostly means — was invisible.

Why a grammar rather than a regex
---------------------------------
An LWC template is HTML5, not XML, and the difference is not cosmetic. Measured
over 387 real templates from public repositories, the ``tree-sitter-xml`` grammar
this repo already depends on fails **364 of them** with 4,181 error nodes. Three
constructs do it, each legal HTML and each fatal to an XML parser: an unquoted
attribute value (``value={greeting}``), a valueless attribute (``lwc:else``), and
a void element (``<br>``) — the last of which swallows the rest of the document.
A regex was rejected for the same reason from the other direction: 1,977 real
start tags span more than one line and 7 custom-tag matches sit inside comments.

Why the tag -> module direction
-------------------------------
Salesforce publishes both halves of the name conversion and they are **not**
inverses. The documented one, ``generateCustomElementTagName`` (folder -> tag),
inserts a hyphen only between a lowercase letter and a capital, so
``rd2ChangeEntryItem`` renders ``rd2change-entry-item`` — while NPSP's own
template writes ``<c-rd2-change-entry-item>``. Measured against 476 real
references, folder -> tag resolves 462; the compiler's ``kebabcaseToCamelcase``
(tag -> module), reimplemented here, resolves **476 of 476**. So the direction
that decides module resolution in the compiler is the direction used here.

Scope guard
-----------
``.html`` is indexed everywhere, not only in Salesforce trees, so a bare
``<template>`` root is not enough to start minting ``lwc.*`` edges. A Lightning
Web Component *must* live in a directory named ``lwc``; that is a platform rule,
not a convention, which makes it a sound gate rather than a heuristic. A
``.html`` anywhere else gets its ``Module`` and nothing more.

The floor is one ``Module`` per file, always. Returning ``None`` would produce an
empty ``ParsedFile`` with no ``Module``, hence no ``file_hash`` — and only
``FILE_HASH_LABELS`` nodes carry one — so every ordinary web page in every
indexed repository would be re-read and re-parsed on every pass, forever, with no
error anywhere.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from tree_sitter import Language, Query

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    register_language,
)
from code_atlas.parsing.languages.apex import (
    APEX_NAMESPACE,
    COMPONENT_NAMESPACE,
    LABEL_NAMESPACE,
    PAGE_NAMESPACE,
    SOBJECT_NAMESPACE,
)
from code_atlas.parsing.languages.salesforce import AURA_NAMESPACE, LWC_NAMESPACE
from code_atlas.schema import NodeLabel, RelType

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tree_sitter import Node

_API_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_RAW_TAG_NAME = re.compile(r"</?\s*([A-Za-z_][\w:.\-]*)")

_LANGUAGE_NAME = "html"

_TEMPLATE_ROOT = "template"
_LWC_DIRECTORY = "lwc"
_DEFAULT_LWC_NAMESPACE = "c"

_TAG_PARENT_TYPES = frozenset({"start_tag", "self_closing_tag"})

_TEMPLATE_KIND = "lwc_template"
_DOCUMENT_KIND = "html_document"

_AURA_DIRECTORY = "aura"
_AURA_ROOTS = frozenset({"aura:component", "aura:application", "aura:event", "aura:interface"})
_AURA_SUFFIXES = frozenset({".cmp", ".app", ".evt", ".intf"})
_AURA_KIND = "aura_component"
_AURA_MODULE_KIND = "aura_markup"

_VISUALFORCE_KINDS: dict[str, tuple[str, str]] = {
    ".page": (PAGE_NAMESPACE, "vf_page"),
    ".component": (COMPONENT_NAMESPACE, "vf_component"),
}
_VISUALFORCE_MODULE_KIND = "visualforce_markup"

_CUSTOM_TAG_PREFIX = "c:"

# `{!$Label.c.commonAdminPermissionErrorTitle}` in Aura, `{!$Label.X}` in Visualforce.
# The namespace segment is optional and the label is always last.
_MARKUP_LABEL = re.compile(r"\$Label\.(?:([A-Za-z_]\w*)\.)?([A-Za-z_]\w*)")
# `{!$ObjectType.Account.Fields.Name}` -- the ONLY object-qualified field reference in
# the whole Salesforce markup surface, and therefore the only one that can form a valid
# `sobject.<Object>.<Field>` uid without guessing an owner.
_MARKUP_OBJECT_FIELD = re.compile(r"\$ObjectType\.([A-Za-z_]\w*)\.Fields\.([A-Za-z_]\w*)")


def _module_qualified_name(file_path: str) -> str:
    """Dotted qualified name with dots folded in every segment.

    The same rule ``config.py`` and ``salesforce.py`` use, and deliberately **not**
    ``typescript._module_qualified_name``, which strips the extension instead: that
    one gives ``lwc/x/x.html`` and ``lwc/x/x.js`` the *same* qualified name, and
    therefore the same uid. Two files sharing a uid means each one's re-parse
    silently deletes the other's edges (ADR-0032).
    """
    parts = PurePosixPath(file_path.replace("\\", "/")).parts
    return ".".join(part.replace(".", "_") for part in parts)


def kebab_to_module(tag: str) -> str:
    """``c-error-panel`` -> ``c/errorPanel``, the LWC template compiler's own rule.

    Reimplemented from ``@lwc/template-compiler``'s ``kebabcaseToCamelcase``: the
    first hyphen becomes the namespace separator, every later hyphen uppercases the
    character after it. This is the direction that resolves 476 of 476 real
    references; see the module docstring for the one that does not.
    """
    out: list[str] = []
    namespace_found = False
    upper_next = False
    for char in tag:
        if char == "-":
            if not namespace_found:
                namespace_found = True
                out.append("/")
            else:
                upper_next = True
            continue
        out.append(char.upper() if upper_next else char)
        upper_next = False
    return "".join(out)


def _import_target(tag: str) -> str | None:
    """The graph target for a custom element, or ``None`` if it is not one.

    A custom element must contain a hyphen and a built-in HTML element never does,
    which is the whole of the test. ``c`` is the default namespace and resolves to
    the node ``salesforce.py`` mints from the bundle's ``.js-meta.xml``; every other
    namespace keeps the module specifier (``lightning/recordForm``), which is the
    same string an LWC ``.js`` would import, so the two converge on one stub.
    """
    if "-" not in tag:
        return None
    module = kebab_to_module(tag)
    namespace, _, name = module.partition("/")
    if not name:
        return None
    if namespace == _DEFAULT_LWC_NAMESPACE:
        return f"{LWC_NAMESPACE}.{name}"
    return module


def _owning_bundle(file_path: str) -> str | None:
    """The bundle directory under ``lwc/``, or ``None`` if this is not an LWC tree.

    Handles a template in a subdirectory (``lwc/<b>/templates/empty.html``), which
    real bundles use when ``render()`` picks between several — so the bundle is the
    part *after* ``lwc``, never simply the parent directory.
    """
    parts = PurePosixPath(file_path.replace("\\", "/")).parts
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == _LWC_DIRECTORY:
            return parts[index + 1]
    return None


def _root_element(root: Node) -> Node | None:
    return next((child for child in root.children if child.type == "element"), None)


def _tag_name(element: Node) -> str | None:
    """The element's tag name, read from the raw tag text rather than the `tag_name` node.

    The HTML grammar truncates a tag name at an underscore — `<c:UTIL_Message />`
    parses as `tag_name 'c:UTIL'` plus an `attribute '_Message'`, and plain
    `<my_widget>` yields `my`. That is defensible for HTML, where an underscore is
    not valid in a custom-element name, and wrong for the two markup dialects that
    are not HTML: Aura and Visualforce component names are full of underscores
    (`c:STG_Container`, `c:UTIL_Message`), so trusting the node would silently drop
    most Salesforce component references.

    Reading the tag's own bytes costs one regex and gives the same answer for every
    well-formed name, hyphenated LWC tags included.
    """
    for child in element.children:
        if child.type not in _TAG_PARENT_TYPES:
            continue
        if child.text is None:
            return None
        match = _RAW_TAG_NAME.match(child.text.decode("utf-8", "replace"))
        return match.group(1) if match else None
    return None


def _iter_tag_names(node: Node) -> list[str]:
    """Every element's tag name in the document, depth-first."""
    found: list[str] = []
    stack = list(node.children)
    while stack:
        current = stack.pop()
        if current.type == "element":
            tag = _tag_name(current)
            if tag is not None:
                found.append(tag)
        stack.extend(current.children)
    return found


def _attributes(element: Node) -> dict[str, str]:
    """``{name: value}`` for one element's start tag, lower-cased keys.

    Markup attribute names are case-insensitive and real files mix
    ``standardController`` with ``standardcontroller``.
    """
    found: dict[str, str] = {}
    for child in element.children:
        if child.type not in _TAG_PARENT_TYPES:
            continue
        for attribute in child.children:
            if attribute.type != "attribute":
                continue
            name = value = None
            for part in attribute.children:
                if part.type == "attribute_name" and part.text is not None:
                    name = part.text.decode("utf-8", "replace").lower()
                elif part.type in {"attribute_value", "quoted_attribute_value"}:
                    raw = part.text.decode("utf-8", "replace") if part.text is not None else ""
                    value = raw.strip("\"'")
            if name is not None:
                found[name] = value or ""
        break
    return found


def _iter_elements(node: Node) -> Iterator[Node]:
    """Every element in the document, depth-first."""
    stack = list(node.children)
    while stack:
        current = stack.pop()
        if current.type == "element":
            yield current
        stack.extend(current.children)


def _api_names(value: str | None) -> list[str]:
    """Split a comma-separated attribute into well-formed API names.

    `extensions="A,B"` is the only attribute that takes a list, and real files put
    spaces after the commas.
    """
    if not value:
        return []
    return [part for raw in value.split(",") if (part := raw.strip()) and _API_NAME.match(part)]


def _markup_text_references(source: bytes) -> set[str]:
    """`$Label.X` and `$ObjectType.O.Fields.F` targets in a markup document.

    Both live inside `{!...}` expressions rather than in attributes or elements, so
    they are read from the raw text. Every other merge-field construct is skipped:
    `{!v.attr}`, `{!c.handler}` and `{!account.Name}` all name something local to
    the component or to a controller variable, and a wrong uid is worse than a
    missing entity (ADR-0032).
    """
    text = source.decode("utf-8", "replace")
    targets = {f"{LABEL_NAMESPACE}.{match.group(2)}" for match in _MARKUP_LABEL.finditer(text)}
    targets |= {
        f"{SOBJECT_NAMESPACE}.{match.group(1)}.{match.group(2)}" for match in _MARKUP_OBJECT_FIELD.finditer(text)
    }
    return targets


def _custom_component_targets(tag: str) -> list[str]:
    """`<c:foo>` -> the component it names, which the file does not say enough to identify.

    Salesforce shares one `c` namespace between Aura and LWC -- you cannot have an
    Aura and an LWC component of the same name -- so `<c:foo>` is unambiguous *in an
    org* and completely ambiguous *in a file*. A parser sees one file and has no way
    to know which kind `foo` is, and this codebase mints the two under different
    namespaces (`aura.foo`, `lwc.foo`).

    Both are emitted. `resolve_imports` matches the one that exists and mints an
    `ext/` stub for the other, so every real edge lands and the cost is a stub per
    reference. The alternative -- picking one -- loses a real edge every time it
    guesses wrong, and in one measured corpus 38 of 132 Aura `<c:X>` references
    point at LWC bundles, so neither choice is rare enough to ignore.

    The clean fix is a shared `cmp.<Name>` namespace minted by both handlers, which
    is what the platform itself has. That would change uids ATL-173 already shipped,
    so it is a follow-up rather than a detour.
    """
    name = tag.partition(":")[2]
    if not _API_NAME.match(name):
        return []
    return [f"{AURA_NAMESPACE}.{name}", f"{LWC_NAMESPACE}.{name}"]


def _aura_bundle(file_path: str) -> str | None:
    """The bundle directory under ``aura/``, or ``None`` outside an Aura tree."""
    parts = PurePosixPath(file_path.replace("\\", "/")).parts
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == _AURA_DIRECTORY:
            return parts[index + 1]
    return None


def _aura_references(root: Node, source: bytes) -> set[str]:
    """Everything an Aura bundle names: its controller, its children, its labels."""
    targets = _markup_text_references(source)
    for element in _iter_elements(root):
        tag = _tag_name(element)
        if tag is None:
            continue
        if tag.startswith(_CUSTOM_TAG_PREFIX):
            targets.update(_custom_component_targets(tag))
        elif tag in _AURA_ROOTS:
            attributes = _attributes(element)
            targets.update(f"{APEX_NAMESPACE}.{name}" for name in _api_names(attributes.get("controller")))
            # `extends="c:Base"` is IMPORTS, not INHERITS: `resolve_inherits` matches a
            # bare name rather than a qualified one and mints no stub, so a namespaced
            # INHERITS target never matches and a missing base vanishes silently.
            extends = attributes.get("extends", "")
            if extends.startswith(_CUSTOM_TAG_PREFIX):
                targets.update(_custom_component_targets(extends))
    return targets


def _visualforce_references(root: Node, source: bytes) -> set[str]:
    """Everything a Visualforce page names: controllers, components, labels, fields."""
    targets = _markup_text_references(source)
    for element in _iter_elements(root):
        tag = _tag_name(element)
        if tag is None:
            continue
        if tag.startswith(_CUSTOM_TAG_PREFIX):
            name = tag.partition(":")[2]
            if _API_NAME.match(name):
                targets.add(f"{COMPONENT_NAMESPACE}.{name}")
            continue
        attributes = _attributes(element)
        for attribute in ("controller", "extensions"):
            targets.update(f"{APEX_NAMESPACE}.{name}" for name in _api_names(attributes.get(attribute)))
        targets.update(f"{SOBJECT_NAMESPACE}.{name}" for name in _api_names(attributes.get("standardcontroller")))
    return targets


def _parse_markup(path: str, source: bytes, root: Node, project_name: str) -> ParsedFile | None:
    """One ``Module`` always, plus whatever the markup dialect this file is says.

    Four dialects share the grammar: an LWC template, an Aura bundle, a Visualforce
    page or component, and ordinary HTML — which is the floor and mints only the
    ``Module``.
    """
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in _VISUALFORCE_KINDS:
        return _parse_visualforce(path, source, root, project_name, suffix=suffix)
    if suffix in _AURA_SUFFIXES:
        return _parse_aura(path, source, root, project_name)
    return _parse_lwc_template(path, source, root, project_name)


def _module_entity(path: str, root: Node, project_name: str, kind: str) -> tuple[ParsedEntity, str]:
    """The file's ``Module`` and its uid — every branch needs one, always."""
    module_qn = f"{project_name}:{_module_qualified_name(path)}"
    return (
        ParsedEntity(
            name=PurePosixPath(path).name,
            qualified_name=module_qn,
            label=NodeLabel.MODULE,
            kind=kind,
            line_start=1,
            line_end=root.end_point[0] + 1,
            file_path=path,
        ),
        module_qn,
    )


def _component_file(
    path: str,
    root: Node,
    project_name: str,
    *,
    name: str | None,
    namespace: str,
    module_kind: str,
    kind: str,
    references: set[str],
) -> ParsedFile:
    """A markup file that declares one addressable component, plus its references."""
    module, module_qn = _module_entity(path, root, project_name, module_kind)
    entities = [module]
    source_uid = module_qn
    if name is not None:
        component_qn = f"{project_name}:{namespace}.{name}"
        entities.append(
            ParsedEntity(
                name=name,
                qualified_name=component_qn,
                label=NodeLabel.TYPE_DEF,
                kind=kind,
                line_start=1,
                line_end=root.end_point[0] + 1,
                file_path=path,
            )
        )
        source_uid = component_qn
    relationships = (
        [ParsedRelationship(from_qualified_name=module_qn, rel_type=RelType.DEFINES, to_name=source_uid)]
        if name is not None
        else []
    )
    relationships += [
        ParsedRelationship(from_qualified_name=source_uid, rel_type=RelType.IMPORTS, to_name=target)
        for target in sorted(references)
    ]
    return ParsedFile(file_path=path, language=_LANGUAGE_NAME, entities=entities, relationships=relationships)


def _parse_aura(path: str, source: bytes, root: Node, project_name: str) -> ParsedFile | None:
    """``aura/<B>/<B>.cmp|.app|.evt|.intf`` -> ``aura.<B>`` plus what it names.

    A bundle folder holds exactly one of the four primary suffixes, so whichever is
    present mints the component.

    Guarded on the ``aura/`` directory because ``.app`` is two different formats:
    ``applications/Foo.app`` is XML ``<CustomApplication>`` metadata, not markup.
    Such a file gets its ``Module`` and nothing else; a handler for it belongs here,
    since this module owns the suffix, and is ATL-180's to add.
    """
    bundle = _aura_bundle(path)
    if bundle is None or not _API_NAME.match(bundle):
        module, _ = _module_entity(path, root, project_name, _DOCUMENT_KIND)
        return ParsedFile(file_path=path, language=_LANGUAGE_NAME, entities=[module], relationships=[])
    return _component_file(
        path,
        root,
        project_name,
        name=bundle,
        namespace=AURA_NAMESPACE,
        module_kind=_AURA_MODULE_KIND,
        kind=_AURA_KIND,
        references=_aura_references(root, source),
    )


def _parse_visualforce(path: str, source: bytes, root: Node, project_name: str, *, suffix: str) -> ParsedFile | None:
    """``pages/<N>.page`` and ``components/<N>.component`` -> one addressable node.

    Named for the file, which is what Apex's ``Page.<N>`` and a WebLink's ``<page>``
    both write.
    """
    namespace, kind = _VISUALFORCE_KINDS[suffix]
    stem = PurePosixPath(path).stem
    return _component_file(
        path,
        root,
        project_name,
        name=stem if _API_NAME.match(stem) else None,
        namespace=namespace,
        module_kind=_VISUALFORCE_MODULE_KIND,
        kind=kind,
        references=_visualforce_references(root, source),
    )


def _parse_lwc_template(path: str, source: bytes, root: Node, project_name: str) -> ParsedFile | None:  # noqa: ARG001
    """One ``Module`` always; component-composition ``IMPORTS`` when it is an LWC template."""
    bundle = _owning_bundle(path)
    root_element = _root_element(root)
    is_template = bundle is not None and root_element is not None and _tag_name(root_element) == _TEMPLATE_ROOT

    module_qn = f"{project_name}:{_module_qualified_name(path)}"
    module = ParsedEntity(
        name=PurePosixPath(path).name,
        qualified_name=module_qn,
        label=NodeLabel.MODULE,
        kind=_TEMPLATE_KIND if is_template else _DOCUMENT_KIND,
        line_start=1,
        line_end=root.end_point[0] + 1,
        file_path=path,
    )
    if not is_template:
        return ParsedFile(file_path=path, language=_LANGUAGE_NAME, entities=[module], relationships=[])

    targets = {target for tag in _iter_tag_names(root) if (target := _import_target(tag)) is not None}
    # The template's route back to its own bundle, which matters most for a template
    # in a subdirectory: without it such a file has no edge to the component at all.
    targets.add(f"{LWC_NAMESPACE}.{bundle}")

    relationships = [
        ParsedRelationship(from_qualified_name=module_qn, rel_type=RelType.IMPORTS, to_name=target)
        for target in sorted(targets)
    ]
    return ParsedFile(
        file_path=path,
        language=_LANGUAGE_NAME,
        entities=[module],
        relationships=relationships,
    )


try:
    import tree_sitter_html as _ts_html

    _HTML_LANGUAGE = Language(_ts_html.language())
    register_language(
        LanguageConfig(
            name=_LANGUAGE_NAME,
            extensions=frozenset({".html", ".htm", ".cmp", ".app", ".evt", ".intf", ".page", ".component"}),
            language=_HTML_LANGUAGE,
            query=Query(_HTML_LANGUAGE, "(document) @root"),
            parse_func=_parse_markup,
            comment_node_types=frozenset({"comment"}),
        )
    )
except ImportError:
    pass
