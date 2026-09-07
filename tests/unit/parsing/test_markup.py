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
from code_atlas.parsing.languages.apex import (
    APEX_NAMESPACE,
    COMPONENT_NAMESPACE,
    LABEL_NAMESPACE,
    PAGE_NAMESPACE,
    SOBJECT_NAMESPACE,
)
from code_atlas.parsing.languages.markup import kebab_to_module
from code_atlas.parsing.languages.salesforce import AURA_NAMESPACE, LWC_NAMESPACE
from code_atlas.schema import CUSTOM_COMPONENT_PREFIX, NodeLabel, RelType

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


# ---------------------------------------------------------------------------
# Aura bundles and Visualforce markup
# ---------------------------------------------------------------------------

AURA = "force-app/main/default/aura"
PAGES = "force-app/main/default/pages"


def test_an_aura_bundle_names_its_controller_children_and_labels():
    """SalesforceFoundation/NPSP `BDI_ManageAdvancedMapping.cmp`, trimmed."""
    source = """\
<aura:component implements="lightning:isUrlAddressable" controller="BDI_ManageAdvancedMappingCtrl">
    <aura:handler name="init" value="{!this}" action="{!c.doInit}" />
    <div class="slds-grid">
        <c:bdiObjectMappings ondeployment="{!c.handleDeploymentNotification}" />
        <c:utilIllustration title="{!$Label.c.commonAdminPermissionErrorTitle}" />
    </div>
</aura:component>
"""
    path = f"{AURA}/BDI_ManageAdvancedMapping/BDI_ManageAdvancedMapping.cmp"
    parsed = _parse(source, path)
    component = next(e for e in parsed.entities if e.kind == "aura_component")
    assert component.qualified_name == f"{PROJECT}:{AURA_NAMESPACE}.BDI_ManageAdvancedMapping"
    targets = _imports(parsed)
    assert f"{APEX_NAMESPACE}.BDI_ManageAdvancedMappingCtrl" in targets
    assert f"{LABEL_NAMESPACE}.commonAdminPermissionErrorTitle" in targets
    # `{!c.doInit}` names a controller action and `{!this}` a component reference;
    # neither is a graph target.
    assert not any(target.endswith((".doInit", ".this")) for target in targets)


def test_a_custom_tag_takes_one_kind_agnostic_target():
    """`<c:foo>` cannot say whether `foo` is Aura or LWC, so it names neither.

    Salesforce shares one `c` namespace between the two and forbids them holding the
    same name, so the reference is unambiguous in an org and ambiguous in a file.
    Emitting `aura.foo` AND `lwc.foo` -- which shipped first -- resolved the true
    edge and left an `ext/` stub asserting a component that was never referenced.
    `resolve_imports` widens the single `cmp.` target instead.
    """
    source = "<aura:component>\n    <c:bdiObjectMappings />\n</aura:component>\n"
    parsed = _parse(source, f"{AURA}/Wrapper/Wrapper.cmp")
    targets = _imports(parsed)
    assert f"{CUSTOM_COMPONENT_PREFIX}bdiObjectMappings" in targets
    assert f"{AURA_NAMESPACE}.bdiObjectMappings" not in targets
    assert f"{LWC_NAMESPACE}.bdiObjectMappings" not in targets


def test_extends_names_an_aura_component_directly():
    """`extends` is the one component reference a single file CAN resolve.

    Only an Aura component can be extended -- an LWC cannot be, and an Aura
    component cannot extend an LWC -- so this one does not need widening.
    """
    parsed = _parse(
        '<aura:component extends="c:BaseCmp"></aura:component>\n',
        f"{AURA}/Child/Child.cmp",
    )
    assert f"{AURA_NAMESPACE}.BaseCmp" in _imports(parsed)
    assert f"{CUSTOM_COMPONENT_PREFIX}BaseCmp" not in _imports(parsed)


def test_a_visualforce_page_names_its_controllers_and_object():
    """SalesforceFoundation/NPSP `ACCT_ViewOverride.page`, verbatim.

    `extensions` is the only attribute that takes a list, and real files put spaces
    after the commas.
    """
    source = """\
<apex:page standardController="Account" extensions="ACCT_ViewOverride_CTRL, ACCT_Second_CTRL">
    <apex:pageMessages />
    <c:UTIL_Message />
</apex:page>
"""
    parsed = _parse(source, f"{PAGES}/ACCT_ViewOverride.page")
    page = next(e for e in parsed.entities if e.kind == "vf_page")
    assert page.qualified_name == f"{PROJECT}:{PAGE_NAMESPACE}.ACCT_ViewOverride"
    assert _imports(parsed) == {
        f"{SOBJECT_NAMESPACE}.Account",
        f"{APEX_NAMESPACE}.ACCT_ViewOverride_CTRL",
        f"{APEX_NAMESPACE}.ACCT_Second_CTRL",
        f"{COMPONENT_NAMESPACE}.UTIL_Message",
    }


def test_object_type_is_the_only_field_reference_markup_can_resolve():
    """`$ObjectType.O.Fields.F` names both halves; `{!account.Name}` names neither.

    It is the only object-qualified field reference in the Salesforce markup
    surface, and therefore the only one that can form a valid uid without guessing
    an owner.
    """
    source = """\
<apex:page standardController="Account">
    <apex:outputText value="{!$ObjectType.Account.Fields.Rating.Label}" />
    <apex:outputField value="{!account.Industry}" />
</apex:page>
"""
    parsed = _parse(source, f"{PAGES}/Ratings.page")
    targets = _imports(parsed)
    assert f"{SOBJECT_NAMESPACE}.Account.Rating" in targets
    assert f"{SOBJECT_NAMESPACE}.Account.Industry" not in targets


def test_an_application_dot_app_is_not_an_aura_bundle():
    """`.app` is two formats and only the directory tells them apart.

    `applications/<N>.app` is XML `<CustomApplication>` metadata; treating it as
    markup would mint an `aura.<N>` node for something that is not a component.
    """
    source = """\
<?xml version="1.0" encoding="UTF-8"?>
<CustomApplication xmlns="http://soap.sforce.com/2006/04/metadata">
    <label>Education Data Architecture</label>
</CustomApplication>
"""
    parsed = _parse(source, "unpackaged/config/dev/applications/Education_Data_Architecture.app")
    assert [e.kind for e in parsed.entities] == ["html_document"]
    assert parsed.relationships == []


def test_an_aura_application_does_mint_its_bundle():
    """`aura/<B>/<B>.app` is the one bundle shape whose primary file is not `.cmp`."""
    source = '<aura:application extends="force:slds">\n    <c:STG_Container />\n</aura:application>\n'
    parsed = _parse(source, f"{AURA}/STG_App/STG_App.app")
    component = next(e for e in parsed.entities if e.kind == "aura_component")
    assert component.qualified_name == f"{PROJECT}:{AURA_NAMESPACE}.STG_App"


def test_an_underscore_in_a_tag_name_survives_the_grammar():
    """The HTML grammar truncates a tag name at an underscore, and Salesforce uses them.

    `<c:UTIL_Message />` parses as `tag_name 'c:UTIL'` plus an `attribute '_Message'`,
    and plain `<my_widget>` yields `my`. Defensible for HTML, where an underscore is
    invalid in a custom-element name -- and wrong for Aura and Visualforce, which are
    not HTML. Trusting the node would have dropped most Salesforce component
    references, since `UTIL_`/`STG_`/`ACCT_` prefixes are the house style.
    """
    parsed = _parse(
        "<apex:page>\n    <c:UTIL_Message />\n    <c:STG_Container></c:STG_Container>\n</apex:page>\n",
        f"{PAGES}/Underscores.page",
    )
    assert _imports(parsed) == {
        f"{COMPONENT_NAMESPACE}.UTIL_Message",
        f"{COMPONENT_NAMESPACE}.STG_Container",
    }
