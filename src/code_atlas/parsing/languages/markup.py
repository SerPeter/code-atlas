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
from code_atlas.parsing.languages.salesforce import LWC_NAMESPACE
from code_atlas.schema import NodeLabel, RelType

if TYPE_CHECKING:
    from tree_sitter import Node

_LANGUAGE_NAME = "html"

_TEMPLATE_ROOT = "template"
_LWC_DIRECTORY = "lwc"
_DEFAULT_LWC_NAMESPACE = "c"

_TAG_PARENT_TYPES = frozenset({"start_tag", "self_closing_tag"})

_TEMPLATE_KIND = "lwc_template"
_DOCUMENT_KIND = "html_document"


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
    for child in element.children:
        if child.type in _TAG_PARENT_TYPES:
            for part in child.children:
                if part.type == "tag_name":
                    return part.text.decode("utf-8", "replace") if part.text is not None else None
            return None
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


def _parse_markup(path: str, source: bytes, root: Node, project_name: str) -> ParsedFile | None:  # noqa: ARG001
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
            extensions=frozenset({".html", ".htm"}),
            language=_HTML_LANGUAGE,
            query=Query(_HTML_LANGUAGE, "(document) @root"),
            parse_func=_parse_markup,
            comment_node_types=frozenset({"comment"}),
        )
    )
except ImportError:
    pass
