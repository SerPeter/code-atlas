"""LWC template composition — the edges that make the component tree visible.

Every fixture here is a shape taken from a real public template and checked by parsing
that template, not invented to match the parser. The naming rule in particular is
verified against the one real case that discriminates between the two published
algorithms: NPSP writes `<c-rd2-change-entry-item>` for a folder named
`rd2ChangeEntryItem`, which the *documented* folder-to-tag rule cannot produce.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_html", reason="tree-sitter-html not installed")

from code_atlas.parsing.ast import ParsedFile, parse_file
from code_atlas.parsing.languages.markup import kebab_to_module
from code_atlas.parsing.languages.salesforce import LWC_NAMESPACE
from code_atlas.schema import NodeLabel, RelType

PROJECT = "test_project"
LWC = "force-app/main/default/lwc"


def _parse(source: str, path: str) -> ParsedFile:
    parsed = parse_file(path, source.encode(), PROJECT)
    assert parsed is not None
    return parsed


def _imports(parsed: ParsedFile) -> set[str]:
    return {r.to_name for r in parsed.relationships if r.rel_type is RelType.IMPORTS}


@pytest.mark.parametrize(
    ("tag", "module"),
    [
        # The discriminating case: a digit before a capital. The documented
        # `generateCustomElementTagName` splits only between [a-z] and [A-Z], so it
        # renders this folder as `rd2change-entry-item` -- which no real file writes.
        ("c-rd2-change-entry-item", "c/rd2ChangeEntryItem"),
        ("c-error-panel", "c/errorPanel"),
        ("c-contact-tile", "c/contactTile"),
        ("lightning-record-form", "lightning/recordForm"),
        ("lightning-card", "lightning/card"),
    ],
)
def test_a_tag_converts_to_its_module_specifier(tag: str, module: str):
    assert kebab_to_module(tag) == module


def test_a_template_imports_the_components_it_composes():
    """`compositionBasics` in trailheadapps/lwc-recipes, trimmed."""
    source = """\
<template>
    <lightning-card title="CompositionBasics">
        <c-contact-tile contact={contact}></c-contact-tile>
        <c-view-source source="lwc/compositionBasics"></c-view-source>
    </lightning-card>
</template>
"""
    parsed = _parse(source, f"{LWC}/compositionBasics/compositionBasics.html")
    assert _imports(parsed) == {
        f"{LWC_NAMESPACE}.contactTile",
        f"{LWC_NAMESPACE}.viewSource",
        # Its own bundle, so a template always has a route back to its component.
        f"{LWC_NAMESPACE}.compositionBasics",
        # A namespaced base component keeps the module specifier an LWC .js would
        # import, so markup and script converge on one stub.
        "lightning/card",
    }


def test_a_template_in_a_subdirectory_still_finds_its_bundle():
    """Real bundles keep alternate templates in `templates/` and pick one in render().

    The bundle is the directory *under* `lwc/`, never simply the parent.
    """
    parsed = _parse("<template><p>nothing</p></template>", f"{LWC}/errorPanel/templates/noDataIllustration.html")
    assert _imports(parsed) == {f"{LWC_NAMESPACE}.errorPanel"}


def test_a_plain_html_page_mints_exactly_one_module():
    """The floor, and it is load-bearing.

    `.html` is indexed in every repository now. A page that produced no `Module`
    would carry no `file_hash` -- only FILE_HASH_LABELS nodes do -- and would be
    re-read and re-parsed on every indexing pass, forever, with no error anywhere.
    """
    parsed = _parse("<html><body><my-widget></my-widget></body></html>", "src/web/index.html")
    assert len(parsed.entities) == 1
    assert parsed.entities[0].label is NodeLabel.MODULE
    assert parsed.entities[0].kind == "html_document"
    assert parsed.relationships == []


def test_a_template_outside_an_lwc_directory_mints_no_component_edges():
    """A `<template>` root is not proof; the `lwc/` directory is.

    An LWC bundle *must* live in a directory named `lwc` -- a platform rule, not a
    convention -- so requiring it is exact rather than heuristic. Without this guard
    any framework using a `<template>` root would mint `lwc.*` edges to nothing.
    """
    parsed = _parse("<template><my-widget></my-widget></template>", "src/components/Thing.html")
    assert parsed.relationships == []
    assert parsed.entities[0].kind == "html_document"


def test_the_template_and_its_script_do_not_collide_on_one_uid():
    """`typescript._module_qualified_name` strips the extension; this one folds it.

    Copying the TypeScript rule would give `x.html` and `x.js` the same qualified
    name, so two files would fight over one node and each re-parse would delete the
    other's edges (ADR-0032).
    """
    template = _parse("<template></template>", f"{LWC}/errorPanel/errorPanel.html")
    script = parse_file(
        f"{LWC}/errorPanel/errorPanel.js",
        b"import { LightningElement } from 'lwc';\nexport default class ErrorPanel extends LightningElement {}\n",
        PROJECT,
    )
    assert script is not None
    template_uids = {e.qualified_name for e in template.entities}
    script_uids = {e.qualified_name for e in script.entities}
    assert not (template_uids & script_uids)


def test_html_comments_do_not_contribute_components():
    """A regex over `<c-...>` matched 7 commented-out tags in a real corpus."""
    source = """\
<template>
    <!-- <c-legacy-tile></c-legacy-tile> -->
    <c-product-tile product={product}></c-product-tile>
</template>
"""
    parsed = _parse(source, f"{LWC}/productTileList/productTileList.html")
    assert f"{LWC_NAMESPACE}.legacyTile" not in _imports(parsed)
    assert f"{LWC_NAMESPACE}.productTile" in _imports(parsed)
