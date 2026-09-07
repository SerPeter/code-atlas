"""Tests for SFDX metadata parsing — the XML half of Salesforce support.

Three concerns, in priority order:

1. **The join with ``apex.py``.** An SObject named one way by the metadata
   parser and another way by the Apex parser produces a graph where nothing
   connects, and *nothing* fails loudly when that happens. The identity tests
   below are the only signal there is.
2. Dispatch — which files this parser claims and, just as importantly, which it
   declines back to ``config.py``'s generic structural parse.
3. Robustness — a metadata directory holds tens of thousands of files, so a
   handler that raises on one unusual document poisons the whole batch.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_xml", reason="tree-sitter-xml not installed")

from code_atlas.parsing.ast import ParsedEntity, ParsedFile, parse_file
from code_atlas.parsing.languages.apex import APEX_NAMESPACE, SOBJECT_NAMESPACE
from code_atlas.parsing.languages.salesforce import (
    GLOBAL_VALUE_SET_NAMESPACE,
    LABEL_NAMESPACE,
    LWC_NAMESPACE,
    PAGE_NAMESPACE,
    PERMISSION_SET_NAMESPACE,
    TAB_NAMESPACE,
    looks_like_salesforce_metadata,
)
from code_atlas.schema import NodeLabel, RelType

PROJECT = "test_project"

OBJECTS = "force-app/main/default/objects"
FLOWS = "force-app/main/default/flows"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(source: str, path: str) -> ParsedFile:
    result = parse_file(path, source.encode("utf-8"), PROJECT)
    assert result is not None, f"{path} produced no ParsedFile"
    return result


def _by_kind(parsed: ParsedFile, kind: str) -> list[ParsedEntity]:
    return [entity for entity in parsed.entities if entity.kind == kind]


def _one(parsed: ParsedFile, kind: str) -> ParsedEntity:
    matches = _by_kind(parsed, kind)
    assert len(matches) == 1, f"expected 1 {kind!r}, got {[(e.kind, e.name) for e in parsed.entities]}"
    return matches[0]


def _qns(parsed: ParsedFile) -> set[str]:
    """Qualified names with the ``{project}:`` prefix stripped."""
    return {entity.qualified_name.split(":", 1)[1] for entity in parsed.entities}


def _rels(parsed: ParsedFile, rel_type: RelType) -> set[tuple[str, str]]:
    return {(r.from_qualified_name, r.to_name) for r in parsed.relationships if r.rel_type == rel_type}


def _targets(parsed: ParsedFile, from_uid: str, rel_type: RelType) -> set[str]:
    return {to for frm, to in _rels(parsed, rel_type) if frm == from_uid}


def _uid(qualified_name: str) -> str:
    return f"{PROJECT}:{qualified_name}"


# ---------------------------------------------------------------------------
# Fixtures — trimmed but structurally faithful to trailheadapps/dreamhouse-lwc
# ---------------------------------------------------------------------------

PROPERTY_OBJECT = """\
<?xml version="1.0" encoding="UTF-8"?>
<CustomObject xmlns="http://soap.sforce.com/2006/04/metadata">
    <label>Property</label>
    <pluralLabel>Properties</pluralLabel>
    <sharingModel>ReadWrite</sharingModel>
    <deploymentStatus>Deployed</deploymentStatus>
    <description>A house for sale</description>
    <enableReports>true</enableReports>
</CustomObject>
"""

BROKER_LOOKUP_FIELD = """\
<?xml version="1.0" encoding="UTF-8"?>
<CustomField xmlns="http://soap.sforce.com/2006/04/metadata">
    <fullName>Broker__c</fullName>
    <label>Broker</label>
    <type>Lookup</type>
    <referenceTo>Broker__c</referenceTo>
    <relationshipName>Properties</relationshipName>
    <deleteConstraint>SetNull</deleteConstraint>
    <required>false</required>
</CustomField>
"""

CREATE_PROPERTY_FLOW = """\
<?xml version="1.0" encoding="UTF-8"?>
<Flow xmlns="http://soap.sforce.com/2006/04/metadata">
    <label>Create Property</label>
    <processType>AutoLaunchedFlow</processType>
    <status>Active</status>
    <actionCalls>
        <name>Geocode</name>
        <actionName>GeocodingService</actionName>
        <actionType>apex</actionType>
    </actionCalls>
    <actionCalls>
        <name>Notify</name>
        <actionName>Property__c.NewListing</actionName>
        <actionType>emailAlert</actionType>
    </actionCalls>
    <apexPluginCalls>
        <name>Legacy</name>
        <apexClass>LegacyPlugin</apexClass>
    </apexPluginCalls>
    <subflows>
        <name>Sub</name>
        <flowName>Notify_Broker</flowName>
    </subflows>
    <recordCreates>
        <name>NewProperty</name>
        <object>Property__c</object>
    </recordCreates>
    <recordUpdates>
        <name>TouchBroker</name>
        <object>Broker__c</object>
    </recordUpdates>
    <recordLookups>
        <name>GetContact</name>
        <object>Contact</object>
    </recordLookups>
    <variables>
        <name>acct</name>
        <objectType>Account</objectType>
    </variables>
    <start>
        <object>Property__c</object>
        <triggerType>RecordAfterSave</triggerType>
        <recordTriggerType>Create</recordTriggerType>
    </start>
</Flow>
"""


# ---------------------------------------------------------------------------
# 1. Identity — the contract with apex.py and typescript.py
# ---------------------------------------------------------------------------


def test_sobject_qualified_name_is_the_apex_import_target():
    """The SObject's qualified name must be *exactly* what ``apex.py`` imports.

    ``GraphClient.resolve_imports`` matches an IMPORTS target against internal
    entities' ``qualified_name`` and only mints an ``ext/`` stub on a miss. So
    the string ``apex.py`` builds for ``[SELECT ... FROM Property__c]`` —
    ``f"{SOBJECT_NAMESPACE}.Property__c"`` — has to be the qualified name minted
    here, or the Apex tier and the metadata tier end up on two disconnected
    nodes with nothing anywhere reporting a problem.
    """
    parsed = _parse(PROPERTY_OBJECT, f"{OBJECTS}/Property__c/Property__c.object-meta.xml")

    sobject = _one(parsed, "sobject")
    assert sobject.qualified_name == _uid(f"{SOBJECT_NAMESPACE}.Property__c")
    assert sobject.label == NodeLabel.TYPE_DEF
    assert sobject.name == "Property__c"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("[SELECT Id FROM Property__c]", f"{SOBJECT_NAMESPACE}.Property__c"),
        ("insert new Property__c();", f"{SOBJECT_NAMESPACE}.Property__c"),
    ],
    ids=["soql", "dml"],
)
def test_apex_sobject_reference_resolves_to_the_metadata_node(source: str, expected: str):
    """End-to-end: the Apex parser's target string equals the metadata uid.

    Asserted against a real Apex parse rather than against the constant alone,
    because the join breaks just as thoroughly if ``apex.py`` changes how it
    builds the target as it does if this module changes how it builds the uid.
    """
    pytest.importorskip("tree_sitter_java", reason="tree-sitter-java not installed")

    apex = _parse(
        f"public class PropertyController {{\n    public void run() {{\n        {source}\n    }}\n}}\n",
        "force-app/main/default/classes/PropertyController.cls",
    )
    imports = {r.to_name for r in apex.relationships if r.rel_type == RelType.IMPORTS}
    assert expected in imports

    metadata = _parse(PROPERTY_OBJECT, f"{OBJECTS}/Property__c/Property__c.object-meta.xml")
    assert expected in _qns(metadata)


def test_flow_apex_reference_targets_the_apex_class_qualified_name():
    """``actionType=apex`` names a class, and ``apex.py`` stores it as ``apex.<Class>``."""
    parsed = _parse(CREATE_PROPERTY_FLOW, f"{FLOWS}/Create_Property.flow-meta.xml")
    flow_uid = _one(parsed, "flow").qualified_name

    assert f"{APEX_NAMESPACE}.GeocodingService" in _targets(parsed, flow_uid, RelType.IMPORTS)

    pytest.importorskip("tree_sitter_java", reason="tree-sitter-java not installed")
    apex = _parse(
        "public class GeocodingService {\n    @InvocableMethod\n    public static void geocode() {}\n}\n",
        "force-app/main/default/classes/GeocodingService.cls",
    )
    assert f"{APEX_NAMESPACE}.GeocodingService" in _qns(apex)


# ---------------------------------------------------------------------------
# 2. CustomObject
# ---------------------------------------------------------------------------


def test_object_file_mints_a_module_and_an_sobject():
    parsed = _parse(PROPERTY_OBJECT, f"{OBJECTS}/Property__c/Property__c.object-meta.xml")

    module = _one(parsed, "sf_object")
    assert module.label == NodeLabel.MODULE
    assert module.line_start == 1

    sobject = _one(parsed, "sobject")
    # Every human-readable string the component declares, not one of them. label and
    # pluralLabel are in extra_properties too, which BM25 sees but build_embed_text
    # does not -- so before this they were invisible to semantic search.
    assert sobject.docstring == "Property\nProperties\nA house for sale"
    assert sobject.extra_properties == {
        "sobject_type": "custom",
        "sobject_label": "Property",
        "plural_label": "Properties",
        "sharing_model": "ReadWrite",
        "deployment_status": "Deployed",
        "enable_reports": True,
    }
    assert _rels(parsed, RelType.DEFINES) == {(module.qualified_name, sobject.qualified_name)}


@pytest.mark.parametrize(
    ("api_name", "expected"),
    [
        ("Account", "standard"),
        ("Property__c", "custom"),
        ("Trigger_Config__mdt", "customMetadataType"),
        ("Order_Placed__e", "platformEvent"),
        ("Legacy__x", "externalObject"),
        ("Telemetry__b", "bigObject"),
        ("Property__Share", "system"),
    ],
)
def test_sobject_type_comes_from_the_api_name_suffix(api_name: str, expected: str):
    """The suffix is the platform's own type marker, and it is all there is."""
    source = '<?xml version="1.0"?>\n<CustomObject xmlns="http://soap.sforce.com/2006/04/metadata"/>\n'
    parsed = _parse(source, f"{OBJECTS}/{api_name}/{api_name}.object-meta.xml")
    assert _one(parsed, "sobject").extra_properties["sobject_type"] == expected


def test_custom_setting_is_distinguished_from_a_plain_custom_object():
    """A custom setting is a ``__c`` object; only ``<customSettingsType>`` tells them apart."""
    source = """\
<?xml version="1.0"?>
<CustomObject xmlns="http://soap.sforce.com/2006/04/metadata">
    <customSettingsType>Hierarchy</customSettingsType>
</CustomObject>
"""
    parsed = _parse(source, f"{OBJECTS}/App_Config__c/App_Config__c.object-meta.xml")
    assert _one(parsed, "sobject").extra_properties["sobject_type"] == "customSetting"


def test_inline_fields_are_contained_by_uid():
    """The non-decomposed layout keeps object and fields in one file.

    Both ends are then present in the same parse, so containment is a plain
    uid-routed DEFINES with no post-batch resolution involved.
    """
    source = """\
<?xml version="1.0"?>
<CustomObject xmlns="http://soap.sforce.com/2006/04/metadata">
    <label>Property</label>
    <fields>
        <fullName>Price__c</fullName>
        <type>Currency</type>
    </fields>
    <fields>
        <fullName>Broker__c</fullName>
        <type>MasterDetail</type>
        <referenceTo>Broker__c</referenceTo>
    </fields>
</CustomObject>
"""
    parsed = _parse(source, f"{OBJECTS}/Property__c/Property__c.object-meta.xml")
    object_uid = _uid(f"{SOBJECT_NAMESPACE}.Property__c")

    assert _qns(parsed) >= {
        f"{SOBJECT_NAMESPACE}.Property__c",
        f"{SOBJECT_NAMESPACE}.Property__c.Price__c",
        f"{SOBJECT_NAMESPACE}.Property__c.Broker__c",
    }
    assert _targets(parsed, object_uid, RelType.DEFINES) == {
        _uid(f"{SOBJECT_NAMESPACE}.Property__c.Price__c"),
        _uid(f"{SOBJECT_NAMESPACE}.Property__c.Broker__c"),
    }
    # No parent_type_name: the parent is right here, so nothing is deferred.
    assert all(rel.properties == {} for rel in parsed.relationships if rel.rel_type == RelType.DEFINES)
    # The master-detail target is still an object reference.
    assert _targets(parsed, _uid(f"{SOBJECT_NAMESPACE}.Property__c.Broker__c"), RelType.IMPORTS) == {
        f"{SOBJECT_NAMESPACE}.Broker__c"
    }


# ---------------------------------------------------------------------------
# 3. CustomField (decomposed)
# ---------------------------------------------------------------------------


def test_decomposed_field_is_scoped_to_its_owning_object():
    """Field API names repeat across objects, so the uid must carry the owner."""
    parsed = _parse(BROKER_LOOKUP_FIELD, f"{OBJECTS}/Property__c/fields/Broker__c.field-meta.xml")

    field = _one(parsed, "sobject_field")
    assert field.label == NodeLabel.VALUE
    assert field.qualified_name == _uid(f"{SOBJECT_NAMESPACE}.Property__c.Broker__c")
    assert field.name == "Broker__c"
    assert field.extra_properties == {
        "sobject": "Property__c",
        "field_type": "Lookup",
        "required": False,
        "reference_to": "Broker__c",
        "relationship_name": "Properties",
        "delete_constraint": "SetNull",
    }


def test_decomposed_field_defers_containment_to_the_post_batch_resolver():
    """A cross-file DEFINES must carry ``parent_type_name``.

    Anchoring it on the parent's uid instead would make the edge depend on the
    object file being upserted before the field file — true by luck of
    alphabetical ordering today, silently false the moment batching changes.
    ``parent_type_name`` routes it through ``resolve_member_defines``, which runs
    after every file in the batch is in the graph.
    """
    parsed = _parse(BROKER_LOOKUP_FIELD, f"{OBJECTS}/Property__c/fields/Broker__c.field-meta.xml")

    module = _one(parsed, "sf_field")
    field = _one(parsed, "sobject_field")
    defines = [rel for rel in parsed.relationships if rel.rel_type == RelType.DEFINES]

    assert len(defines) == 1
    assert defines[0].from_qualified_name == module.qualified_name
    assert defines[0].to_name == field.qualified_name
    assert defines[0].properties == {"parent_type_name": "Property__c"}


def test_field_file_module_imports_its_owning_object():
    """The owner link that survives a standard object having no source file.

    ``Account.object-meta.xml`` does not exist in most repos, so
    ``resolve_member_defines`` has no ``TypeDef`` to attach the field to and
    falls back to the field's own module. The module-level IMPORTS still reaches
    the shared ``sobject.Account`` identity — the very node ``apex.py``'s SOQL
    references land on.
    """
    parsed = _parse(BROKER_LOOKUP_FIELD, f"{OBJECTS}/Account/fields/Broker__c.field-meta.xml")
    module = _one(parsed, "sf_field")
    assert f"{SOBJECT_NAMESPACE}.Account" in _targets(parsed, module.qualified_name, RelType.IMPORTS)


def test_lookup_reference_is_anchored_on_the_field():
    """Object-to-object reachability is two hops, and deliberately so.

    ``_recreate_file_relationships`` deletes edges by their source node's
    ``file_path``, so an edge sourced at ``sobject.Property__c`` but contributed
    by ``fields/Broker__c.field-meta.xml`` would be wiped whenever the object
    file alone was re-parsed, with nothing to restore it. Anchoring the edge in
    the file that states the fact keeps its lifetime right.
    """
    parsed = _parse(BROKER_LOOKUP_FIELD, f"{OBJECTS}/Property__c/fields/Broker__c.field-meta.xml")
    field_uid = _one(parsed, "sobject_field").qualified_name

    assert _targets(parsed, field_uid, RelType.IMPORTS) == {f"{SOBJECT_NAMESPACE}.Broker__c"}
    assert not _targets(parsed, _uid(f"{SOBJECT_NAMESPACE}.Property__c"), RelType.IMPORTS)


def test_formula_body_is_searchable_source():
    source = """\
<?xml version="1.0"?>
<CustomField xmlns="http://soap.sforce.com/2006/04/metadata">
    <fullName>Days_On_Market__c</fullName>
    <type>Number</type>
    <formula>TODAY() - Date_Listed__c</formula>
</CustomField>
"""
    parsed = _parse(source, f"{OBJECTS}/Property__c/fields/Days_On_Market__c.field-meta.xml")
    assert _one(parsed, "sobject_field").source == "TODAY() - Date_Listed__c"


def test_rollup_summary_references_the_child_object_and_its_fields():
    """A roll-up names the child field it aggregates, not only the child object.

    Both halves are kept: the object is the only answer when the field is standard
    and has no file of its own.
    """
    source = """\
<?xml version="1.0"?>
<CustomField xmlns="http://soap.sforce.com/2006/04/metadata">
    <fullName>Total_Price__c</fullName>
    <type>Summary</type>
    <summarizedField>Property__c.Price__c</summarizedField>
    <summaryForeignKey>Property__c.Broker__c</summaryForeignKey>
</CustomField>
"""
    parsed = _parse(source, f"{OBJECTS}/Broker__c/fields/Total_Price__c.field-meta.xml")
    field_uid = _one(parsed, "sobject_field").qualified_name
    assert _targets(parsed, field_uid, RelType.IMPORTS) == {
        f"{SOBJECT_NAMESPACE}.Property__c",
        f"{SOBJECT_NAMESPACE}.Property__c.Price__c",
        f"{SOBJECT_NAMESPACE}.Property__c.Broker__c",
    }


def test_a_rollup_states_every_reference_it_carries():
    """SalesforceFoundation/NPSP `Opportunity/Next_Grant_Deadline_Due_Date__c`, verbatim.

    Four field-level references in one file. Before this story the graph held the
    object half of one of them.
    """
    source = """\
<?xml version="1.0" encoding="UTF-8"?>
<CustomField xmlns="http://soap.sforce.com/2006/04/metadata">
    <fullName>Next_Grant_Deadline_Due_Date__c</fullName>
    <label>Next Deliverable Date</label>
    <summarizedField>Grant_Deadline__c.Grant_Deadline_Due_Date__c</summarizedField>
    <summaryFilterItems>
        <field>Grant_Deadline__c.Grant_Deadline_Due_Date__c</field>
        <operation>notEqual</operation>
    </summaryFilterItems>
    <summaryFilterItems>
        <field>Grant_Deadline__c.Grant_Deliverable_Close_Date__c</field>
        <operation>equals</operation>
    </summaryFilterItems>
    <summaryForeignKey>Grant_Deadline__c.Opportunity__c</summaryForeignKey>
    <summaryOperation>min</summaryOperation>
    <type>Summary</type>
</CustomField>
"""
    path = f"{OBJECTS}/Opportunity/fields/Next_Grant_Deadline_Due_Date__c.field-meta.xml"
    parsed = _parse(source, path)
    assert _targets(parsed, _one(parsed, "sobject_field").qualified_name, RelType.IMPORTS) == {
        f"{SOBJECT_NAMESPACE}.Grant_Deadline__c",
        f"{SOBJECT_NAMESPACE}.Grant_Deadline__c.Grant_Deadline_Due_Date__c",
        f"{SOBJECT_NAMESPACE}.Grant_Deadline__c.Grant_Deliverable_Close_Date__c",
        f"{SOBJECT_NAMESPACE}.Grant_Deadline__c.Opportunity__c",
    }


def test_a_lookup_filter_names_the_fields_it_filters_on():
    """SalesforceFoundation/NPSP `Allocation__c/General_Accounting_Unit__c`, trimmed.

    `lookupFilter` wraps its own `filterItems`, one level deeper than
    `summaryFilterItems`, which holds `field` directly.
    """
    source = """\
<?xml version="1.0"?>
<CustomField xmlns="http://soap.sforce.com/2006/04/metadata">
    <fullName>General_Accounting_Unit__c</fullName>
    <type>Lookup</type>
    <referenceTo>General_Accounting_Unit__c</referenceTo>
    <lookupFilter>
        <active>true</active>
        <filterItems>
            <field>General_Accounting_Unit__c.Active__c</field>
            <operation>equals</operation>
        </filterItems>
    </lookupFilter>
</CustomField>
"""
    path = f"{OBJECTS}/Allocation__c/fields/General_Accounting_Unit__c.field-meta.xml"
    parsed = _parse(source, path)
    targets = _targets(parsed, _one(parsed, "sobject_field").qualified_name, RelType.IMPORTS)
    assert f"{SOBJECT_NAMESPACE}.General_Accounting_Unit__c.Active__c" in targets


def test_a_bare_field_reference_is_relative_to_the_owning_object():
    """`controllingField` carries a bare name where the roll-up elements carry a dotted one.

    Reading the dot rather than the element name means one rule covers both shapes,
    which matters because three of the eight elements handled occur in none of the
    400 real field files sampled and their form could not be confirmed.
    """
    source = """\
<?xml version="1.0"?>
<CustomField xmlns="http://soap.sforce.com/2006/04/metadata">
    <fullName>Sub_Type__c</fullName>
    <type>Picklist</type>
    <valueSet>
        <controllingField>Type__c</controllingField>
        <restricted>true</restricted>
    </valueSet>
</CustomField>
"""
    parsed = _parse(source, f"{OBJECTS}/Account/fields/Sub_Type__c.field-meta.xml")
    targets = _targets(parsed, _one(parsed, "sobject_field").qualified_name, RelType.IMPORTS)
    assert f"{SOBJECT_NAMESPACE}.Account.Type__c" in targets


def test_a_global_value_set_reference_is_read_from_inside_value_set():
    """SalesforceFoundation/NPSP `DataImport__c/Payment_ACH_Code__c`, verbatim.

    `valueSetName` is a child of `<valueSet>`, and `_text_of` reads direct children
    only -- so reading it off the `CustomField` root produced nothing at all, and
    `extra_properties["value_set_name"]` could never be populated.
    """
    source = """\
<?xml version="1.0" encoding="UTF-8"?>
<CustomField xmlns="http://soap.sforce.com/2006/04/metadata">
    <fullName>Payment_ACH_Code__c</fullName>
    <label>Payment ACH Code</label>
    <type>Picklist</type>
    <valueSet>
        <restricted>true</restricted>
        <valueSetName>Payment_ACH_Code</valueSetName>
    </valueSet>
</CustomField>
"""
    path = f"{OBJECTS}/DataImport__c/fields/Payment_ACH_Code__c.field-meta.xml"
    parsed = _parse(source, path)
    field = _one(parsed, "sobject_field")
    assert field.extra_properties["value_set_name"] == "Payment_ACH_Code"
    assert f"{GLOBAL_VALUE_SET_NAMESPACE}.Payment_ACH_Code" in _targets(parsed, field.qualified_name, RelType.IMPORTS)


def test_a_global_value_set_is_named_for_its_file_not_its_label():
    """The documentation says `masterLabel` and the documentation is wrong.

    NPSP's file declares `<masterLabel>Payment ACH Code</masterLabel>` while every
    field referencing it writes `Payment_ACH_Code` -- the filename. Minting from the
    label would join nothing.
    """
    source = """\
<?xml version="1.0" encoding="UTF-8"?>
<GlobalValueSet xmlns="http://soap.sforce.com/2006/04/metadata">
    <customValue><fullName>PPD</fullName><label>Prearranged</label></customValue>
    <customValue><fullName>CCD</fullName><label>Cash Concentration</label></customValue>
    <masterLabel>Payment ACH Code</masterLabel>
    <sorted>false</sorted>
</GlobalValueSet>
"""
    path = "force-app/main/default/globalValueSets/Payment_ACH_Code.globalValueSet-meta.xml"
    parsed = _parse(source, path)
    value_set = _one(parsed, "global_value_set")
    assert value_set.qualified_name == _uid(f"{GLOBAL_VALUE_SET_NAMESPACE}.Payment_ACH_Code")
    assert value_set.extra_properties["master_label"] == "Payment ACH Code"
    # The individual picklist entries are not nodes, for the same reason record
    # types' picklistValues are not.
    assert len(parsed.entities) == 2


def test_the_name_field_becomes_a_real_field_node():
    """`sobject.<Obj>.Name` is the target of every `SELECT Name` and had no definition."""
    source = """\
<?xml version="1.0"?>
<CustomObject xmlns="http://soap.sforce.com/2006/04/metadata">
    <label>Broker</label>
    <nameField>
        <label>Broker Name</label>
        <type>Text</type>
    </nameField>
    <pluralLabel>Brokers</pluralLabel>
</CustomObject>
"""
    parsed = _parse(source, f"{OBJECTS}/Broker__c/Broker__c.object-meta.xml")
    names = {entity.qualified_name for entity in parsed.entities}
    assert _uid(f"{SOBJECT_NAMESPACE}.Broker__c.Name") in names


def test_field_outside_the_decomposed_layout_falls_back_to_the_generic_parse():
    """Without ``objects/<Object>/fields/`` there is no owner, and no unique name."""
    parsed = _parse(BROKER_LOOKUP_FIELD, "retrieved/Broker__c.field-meta.xml")
    assert _by_kind(parsed, "sobject_field") == []
    assert _by_kind(parsed, "xml_element")


# ---------------------------------------------------------------------------
# 4. Flow
# ---------------------------------------------------------------------------


def test_flow_is_one_callable_carrying_its_references():
    """One node per flow, not one per flow element — the references are the value."""
    parsed = _parse(CREATE_PROPERTY_FLOW, f"{FLOWS}/Create_Property.flow-meta.xml")

    flow = _one(parsed, "flow")
    assert flow.label == NodeLabel.CALLABLE
    assert flow.qualified_name == _uid("flow.Create_Property")
    assert flow.extra_properties == {
        "flow_label": "Create Property",
        "process_type": "AutoLaunchedFlow",
        "status": "Active",
        "trigger_object": "Property__c",
        "trigger_type": "RecordAfterSave",
        "record_trigger_type": "Create",
        # Read/write direction is kept here because IMPORTS edge properties do
        # not survive resolve_imports.
        "sobjects_read": ["Account", "Contact", "Property__c"],
        "sobjects_written": ["Broker__c", "Property__c"],
    }
    # Two entities only: the file and the flow. No per-element explosion.
    assert len(parsed.entities) == 2


def test_flow_imports_every_sobject_it_touches():
    parsed = _parse(CREATE_PROPERTY_FLOW, f"{FLOWS}/Create_Property.flow-meta.xml")
    flow_uid = _one(parsed, "flow").qualified_name

    assert _targets(parsed, flow_uid, RelType.IMPORTS) == {
        f"{SOBJECT_NAMESPACE}.Account",
        f"{SOBJECT_NAMESPACE}.Broker__c",
        f"{SOBJECT_NAMESPACE}.Contact",
        f"{SOBJECT_NAMESPACE}.Property__c",
        f"{APEX_NAMESPACE}.GeocodingService",
        f"{APEX_NAMESPACE}.LegacyPlugin",
    }


def test_flow_calls_its_subflows():
    parsed = _parse(CREATE_PROPERTY_FLOW, f"{FLOWS}/Create_Property.flow-meta.xml")
    flow_uid = _one(parsed, "flow").qualified_name
    assert _targets(parsed, flow_uid, RelType.CALLS) == {"Notify_Broker"}


def test_action_call_of_an_unmodelled_type_produces_nothing():
    """``emailAlert``'s ``Object.AlertName`` is not an Apex class or a flow.

    ``InvocableActionType`` has ~100 members; treating an unrecognised one as an
    Apex class would mint a junk ``ext/apex.Property__c.NewListing`` stub.
    """
    parsed = _parse(CREATE_PROPERTY_FLOW, f"{FLOWS}/Create_Property.flow-meta.xml")
    all_targets = {to for _, to in _rels(parsed, RelType.IMPORTS) | _rels(parsed, RelType.CALLS)}
    assert not any("NewListing" in target for target in all_targets)


def test_action_type_matching_is_case_insensitive():
    source = """\
<?xml version="1.0"?>
<Flow xmlns="http://soap.sforce.com/2006/04/metadata">
    <actionCalls>
        <actionName>GeocodingService</actionName>
        <actionType>Apex</actionType>
    </actionCalls>
    <actionCalls>
        <actionName>Notify_Broker</actionName>
        <actionType>FLOW</actionType>
    </actionCalls>
</Flow>
"""
    parsed = _parse(source, f"{FLOWS}/Odd_Case.flow-meta.xml")
    flow_uid = _one(parsed, "flow").qualified_name
    assert _targets(parsed, flow_uid, RelType.IMPORTS) == {f"{APEX_NAMESPACE}.GeocodingService"}
    assert _targets(parsed, flow_uid, RelType.CALLS) == {"Notify_Broker"}


def test_screen_flow_with_no_data_access_still_parses():
    source = """\
<?xml version="1.0"?>
<Flow xmlns="http://soap.sforce.com/2006/04/metadata">
    <label>Just Screens</label>
    <processType>Flow</processType>
    <screens>
        <name>Welcome</name>
        <fields>
            <name>Hello</name>
            <extensionName>c:navigateToRecord</extensionName>
        </fields>
    </screens>
</Flow>
"""
    parsed = _parse(source, f"{FLOWS}/Just_Screens.flow-meta.xml")
    flow = _one(parsed, "flow")
    assert "sobjects_read" not in flow.extra_properties
    assert "sobjects_written" not in flow.extra_properties
    # A screen flow touches no data and still states one thing: which component it
    # puts on the screen. The `c:` prefix is stripped so the target meets the bundle
    # wherever it is minted -- `_api_name` rejects the colon outright.
    assert _rels(parsed, RelType.IMPORTS) == {(flow.qualified_name, f"{LWC_NAMESPACE}.navigateToRecord")}


def test_dml_object_resolves_through_a_flow_local_variable():
    """The shape that made 11.5% of real flows report a write as a read.

    ``recordUpdates`` with no ``<object>`` — 43% of them in a 419-flow corpus — names
    a flow variable instead. Shape from navikt/crm-hot-tolk
    ``HOT_AssignInterestedResource``, which writes ServiceAppointment and, before
    this, said so nowhere.
    """
    source = """\
<?xml version="1.0"?>
<Flow xmlns="http://soap.sforce.com/2006/04/metadata">
    <variables>
        <name>appointmentVar</name>
        <objectType>ServiceAppointment</objectType>
    </variables>
    <recordUpdates>
        <name>Update_Appointment</name>
        <inputReference>appointmentVar</inputReference>
        <inputAssignments>
            <field>Status</field>
            <value><stringValue>Assigned</stringValue></value>
        </inputAssignments>
    </recordUpdates>
</Flow>
"""
    parsed = _parse(source, f"{FLOWS}/Assign.flow-meta.xml")
    flow = _one(parsed, "flow")
    assert flow.extra_properties["sobjects_written"] == ["ServiceAppointment"]
    assert flow.extra_properties["fields_written"] == ["ServiceAppointment.Status"]
    assert f"{SOBJECT_NAMESPACE}.ServiceAppointment.Status" in _targets(parsed, flow.qualified_name, RelType.IMPORTS)


def test_dollar_record_resolves_to_the_triggering_object():
    """``$Record`` in a record-triggered flow is the object named by ``start``."""
    source = """\
<?xml version="1.0"?>
<Flow xmlns="http://soap.sforce.com/2006/04/metadata">
    <start>
        <object>Case</object>
        <recordTriggerType>CreateAndUpdate</recordTriggerType>
        <triggerType>RecordAfterSave</triggerType>
    </start>
    <recordUpdates>
        <name>Close_It</name>
        <inputReference>$Record</inputReference>
        <inputAssignments>
            <field>Status__c</field>
            <value><stringValue>Closed</stringValue></value>
        </inputAssignments>
    </recordUpdates>
</Flow>
"""
    parsed = _parse(source, f"{FLOWS}/Close_Case.flow-meta.xml")
    flow = _one(parsed, "flow")
    assert flow.extra_properties["fields_written"] == ["Case.Status__c"]


def test_field_writes_and_reads_are_separated():
    """``inputAssignments`` is a write; ``filters`` on the same element is a read.

    Shape from trailheadapps/dreamhouse-lwc ``Create_property``, whose ``<object>``
    is a *later* sibling than the assignments that depend on it.
    """
    source = """\
<?xml version="1.0"?>
<Flow xmlns="http://soap.sforce.com/2006/04/metadata">
    <recordCreates>
        <name>create_property</name>
        <inputAssignments>
            <field>Address__c</field>
            <value><elementReference>property_address.street</elementReference></value>
        </inputAssignments>
        <object>Property__c</object>
    </recordCreates>
    <recordLookups>
        <name>find_broker</name>
        <object>Broker__c</object>
        <filters>
            <field>Email__c</field>
            <operator>EqualTo</operator>
        </filters>
        <sortField>Name</sortField>
    </recordLookups>
</Flow>
"""
    parsed = _parse(source, f"{FLOWS}/Create_property.flow-meta.xml")
    flow = _one(parsed, "flow")
    assert flow.extra_properties["fields_written"] == ["Property__c.Address__c"]
    assert flow.extra_properties["fields_read"] == ["Broker__c.Email__c", "Broker__c.Name"]
    # `elementReference` names a flow variable, not a field, and is never mined:
    # 37-38% of them are unresolvable and a wrong uid beats no uid to nothing.
    assert not any("street" in target for target in _targets(parsed, flow.qualified_name, RelType.IMPORTS))


def test_object_field_reference_resolves_through_the_symbol_table():
    """``objectFieldReference`` is ``<recordVariable>.<Field>``, never ``<Object>.<Field>``."""
    source = """\
<?xml version="1.0"?>
<Flow xmlns="http://soap.sforce.com/2006/04/metadata">
    <variables>
        <name>curKnowledge</name>
        <objectType>Knowledge__kav</objectType>
    </variables>
    <screens>
        <name>Review</name>
        <fields>
            <name>Outer</name>
            <fields>
                <name>Inner</name>
                <objectFieldReference>curKnowledge.Title</objectFieldReference>
            </fields>
        </fields>
    </screens>
</Flow>
"""
    parsed = _parse(source, f"{FLOWS}/Review.flow-meta.xml")
    flow = _one(parsed, "flow")
    assert flow.extra_properties["fields_read"] == ["Knowledge__kav.Title"]


def test_orchestration_steps_and_overrides_are_calls():
    """Three flow-to-flow spellings a direct-children sweep of ``actionCalls`` misses.

    ``steppedStages`` appears in no published WSDL and was found only by reading real
    files (UnofficialSF ``Automation_Orchestration``); ``overriddenFlow`` is in the
    WSDL but not in the HTML field table (trailheadapps/coral-cloud
    ``Send_Verification_Code``).
    """
    source = """\
<?xml version="1.0"?>
<Flow xmlns="http://soap.sforce.com/2006/04/metadata">
    <overriddenFlow>SvcCopilotTmpl__SendVerificationCode</overriddenFlow>
    <sourceTemplate>runtime_revenue_arcflows__Auto_Renew</sourceTemplate>
    <steppedStages>
        <name>Submit_Content</name>
        <steps>
            <name>Submit_Content_for_Approval</name>
            <actionName>testflow</actionName>
            <actionType>createWorkItem</actionType>
        </steps>
    </steppedStages>
    <orchestratedStages>
        <name>Stage_One</name>
        <stageSteps>
            <name>Step_A</name>
            <actionName>Nested_Screen_Flow</actionName>
            <actionType>stepInteractive</actionType>
        </stageSteps>
    </orchestratedStages>
</Flow>
"""
    parsed = _parse(source, f"{FLOWS}/Orchestration.flow-meta.xml")
    flow_uid = _one(parsed, "flow").qualified_name
    assert _targets(parsed, flow_uid, RelType.CALLS) == {
        "SvcCopilotTmpl__SendVerificationCode",
        "runtime_revenue_arcflows__Auto_Renew",
        "testflow",
        "Nested_Screen_Flow",
    }


def test_screen_component_is_found_three_levels_deep():
    """70 of 381 real ``extensionName`` sites sit three ``fields`` levels down."""
    source = """\
<?xml version="1.0"?>
<Flow xmlns="http://soap.sforce.com/2006/04/metadata">
    <screens>
        <name>Wizard</name>
        <fields>
            <name>Section</name>
            <fields>
                <name>Column</name>
                <fields>
                    <name>Widget</name>
                    <extensionName>c:hot_flowFooterButtons</extensionName>
                </fields>
            </fields>
        </fields>
    </screens>
</Flow>
"""
    parsed = _parse(source, f"{FLOWS}/Wizard.flow-meta.xml")
    flow_uid = _one(parsed, "flow").qualified_name
    assert _targets(parsed, flow_uid, RelType.IMPORTS) == {f"{LWC_NAMESPACE}.hot_flowFooterButtons"}


def test_apex_class_on_a_variable_is_an_import():
    """``variables/apexClass`` — 46 real occurrences, none reachable via actionCalls."""
    source = """\
<?xml version="1.0"?>
<Flow xmlns="http://soap.sforce.com/2006/04/metadata">
    <variables>
        <name>request</name>
        <apexClass>QuoteRequest</apexClass>
    </variables>
</Flow>
"""
    parsed = _parse(source, f"{FLOWS}/Apex_Var.flow-meta.xml")
    flow_uid = _one(parsed, "flow").qualified_name
    assert _targets(parsed, flow_uid, RelType.IMPORTS) == {f"{APEX_NAMESPACE}.QuoteRequest"}


# ---------------------------------------------------------------------------
# 4b. LightningComponentBundle
# ---------------------------------------------------------------------------

LWC = "force-app/main/default/lwc"


def test_a_bundle_with_no_template_still_gets_a_node():
    """`trailheadapps/lwc-recipes` `errorPanel` has no `errorPanel.html` at all.

    It picks a template in `render()`. So the template cannot be what mints the
    component, and a real bundle's meta file can be this bare.
    """
    source = """\
<?xml version="1.0" encoding="UTF-8" ?>
<LightningComponentBundle xmlns="http://soap.sforce.com/2006/04/metadata">
    <apiVersion>66.0</apiVersion>
    <isExposed>false</isExposed>
</LightningComponentBundle>
"""
    parsed = _parse(source, f"{LWC}/errorPanel/errorPanel.js-meta.xml")
    component = _one(parsed, "lwc_component")
    assert component.qualified_name == f"{PROJECT}:{LWC_NAMESPACE}.errorPanel"
    assert component.name == "errorPanel"
    assert component.extra_properties["is_exposed"] is False


def test_the_bundle_is_named_for_its_directory_not_its_master_label():
    """Every referencing surface spells the folder name, and `masterLabel` is prose."""
    source = """\
<?xml version="1.0" encoding="UTF-8" ?>
<LightningComponentBundle xmlns="http://soap.sforce.com/2006/04/metadata">
    <apiVersion>67.0</apiVersion>
    <isExposed>true</isExposed>
    <masterLabel>Product Tile List</masterLabel>
    <targets>
        <target>lightning__AppPage</target>
        <target>lightningCommunity__Default</target>
    </targets>
</LightningComponentBundle>
"""
    parsed = _parse(source, f"{LWC}/productTileList/productTileList.js-meta.xml")
    component = _one(parsed, "lwc_component")
    assert component.qualified_name == f"{PROJECT}:{LWC_NAMESPACE}.productTileList"
    assert component.extra_properties["master_label"] == "Product Tile List"
    # Opaque strings: `lightningCommunity__Default` and `lightning_VoiceExtension`
    # coexist in real files, so a validated enum would be wrong within a release.
    assert component.extra_properties["targets"] == ["lightning__AppPage", "lightningCommunity__Default"]


def test_target_config_objects_and_apex_datasource_become_imports():
    """`<objects>` nests two levels in; `datasource` is an attribute, not text.

    Shapes from `trailheadapps/ebikes-lwc` `productTileList` and `hero`.
    """
    source = """\
<?xml version="1.0" encoding="UTF-8" ?>
<LightningComponentBundle xmlns="http://soap.sforce.com/2006/04/metadata">
    <isExposed>true</isExposed>
    <targets>
        <target>lightning__RecordPage</target>
    </targets>
    <targetConfigs>
        <targetConfig targets="lightning__RecordPage">
            <property name="heroDetailsPosition" type="String"
                      datasource="apex://HeroDetailsPositionCustomPicklist" />
            <objects>
                <object>Order__c</object>
            </objects>
        </targetConfig>
    </targetConfigs>
</LightningComponentBundle>
"""
    parsed = _parse(source, f"{LWC}/hero/hero.js-meta.xml")
    component = _one(parsed, "lwc_component")
    assert _targets(parsed, component.qualified_name, RelType.IMPORTS) == {
        f"{SOBJECT_NAMESPACE}.Order__c",
        f"{APEX_NAMESPACE}.HeroDetailsPositionCustomPicklist",
    }


def test_the_bundle_node_and_its_module_are_two_nodes_from_one_file():
    """The meta file mints both, and nothing else may mint either.

    `uid = project:qualified_name` and `_recreate_file_relationships` deletes edges
    by their source node's `file_path`. Two files minting one uid means each
    re-parse silently deletes the other's edges -- the ADR-0032 failure mode.
    """
    source = """\
<?xml version="1.0" encoding="UTF-8" ?>
<LightningComponentBundle xmlns="http://soap.sforce.com/2006/04/metadata">
    <isExposed>true</isExposed>
</LightningComponentBundle>
"""
    path = f"{LWC}/eDRD_lwc_RelatedPhysicians/eDRD_lwc_RelatedPhysicians.js-meta.xml"
    parsed = _parse(source, path)
    assert {entity.file_path for entity in parsed.entities} == {path}
    # Underscored folder names are real -- 13 of them in one public repo.
    assert _one(parsed, "lwc_component").qualified_name.endswith("lwc.eDRD_lwc_RelatedPhysicians")
    modules = [entity for entity in parsed.entities if entity.label is NodeLabel.MODULE]
    assert len(modules) == 1


# ---------------------------------------------------------------------------
# 4c. Decomposed children of CustomObject
# ---------------------------------------------------------------------------

OBJECTS = "force-app/main/default/objects"


def test_a_validation_rule_carries_the_admins_error_message():
    """SalesforceFoundation/EDA `Relationship__c/Related_Contact_Do_Not_Change`, verbatim.

    `errorMessage` is why this type earns a node: it is the admin's own explanation
    of a business rule, and it had nowhere to live.
    """
    source = """\
<?xml version="1.0" encoding="UTF-8"?>
<ValidationRule xmlns="http://soap.sforce.com/2006/04/metadata">
    <fullName>Related_Contact_Do_Not_Change</fullName>
    <active>true</active>
    <description>Do not allow user to change Related Contact value</description>
    <errorConditionFormula>and(not( ISNEW()), ISCHANGED( RelatedContact__c ))</errorConditionFormula>
    <errorDisplayField>RelatedContact__c</errorDisplayField>
    <errorMessage>Instead of changing the Contacts, delete this record.</errorMessage>
</ValidationRule>
"""
    path = f"{OBJECTS}/Relationship__c/validationRules/Related_Contact_Do_Not_Change.validationRule-meta.xml"
    parsed = _parse(source, path)
    rule = _one(parsed, "validation_rule")
    assert rule.qualified_name == _uid("validationrule.Relationship__c.Related_Contact_Do_Not_Change")
    assert "Instead of changing the Contacts" in (rule.docstring or "")
    assert rule.source is not None
    assert "ISCHANGED" in rule.source
    assert _targets(parsed, rule.qualified_name, RelType.IMPORTS) == {
        f"{SOBJECT_NAMESPACE}.Relationship__c",
        f"{SOBJECT_NAMESPACE}.Relationship__c.RelatedContact__c",
    }


def test_only_dollar_label_is_mined_out_of_a_formula():
    """A formula's bare identifiers are not edges; `$Label.X` is unambiguous.

    `ISCHANGED`, `Contact__c` and `Account.Owner.Name` all look alike to a regex,
    and inventing a target is worse than missing one (ADR-0032).
    """
    source = """\
<?xml version="1.0"?>
<ValidationRule xmlns="http://soap.sforce.com/2006/04/metadata">
    <fullName>Needs_Reason</fullName>
    <errorConditionFormula>AND(ISBLANK(Reason__c), $Label.Reason_Required,
        ISPICKVAL(Stage__c, "X"))</errorConditionFormula>
    <errorMessage>A reason is required.</errorMessage>
</ValidationRule>
"""
    path = f"{OBJECTS}/Case/validationRules/Needs_Reason.validationRule-meta.xml"
    parsed = _parse(source, path)
    targets = _targets(parsed, _one(parsed, "validation_rule").qualified_name, RelType.IMPORTS)
    assert f"{LABEL_NAMESPACE}.Reason_Required" in targets
    # Neither the function names nor the bare field references become edges.
    assert not any(t.endswith(("Reason__c", "Stage__c", "ISBLANK", "ISPICKVAL")) for t in targets)


def test_a_record_type_does_not_mint_its_picklist_values():
    """266 picklist values sit inside 8 real record-type files -- a 33x multiplier."""
    source = """\
<?xml version="1.0" encoding="UTF-8"?>
<RecordType xmlns="http://soap.sforce.com/2006/04/metadata">
    <fullName>Certificate</fullName>
    <active>true</active>
    <description>A certification related to the completion of specific courses.</description>
    <label>Certificate</label>
    <picklistValues>
        <picklist>Academic_Level__c</picklist>
        <values><fullName>Adult Education</fullName><default>false</default></values>
        <values><fullName>Doctoral</fullName><default>false</default></values>
    </picklistValues>
</RecordType>
"""
    path = f"{OBJECTS}/Academic_Certification__c/recordTypes/Certificate.recordType-meta.xml"
    parsed = _parse(source, path)
    # The Module and the record type, and nothing per picklist value.
    assert len(parsed.entities) == 2
    assert _one(parsed, "record_type").qualified_name == _uid("recordtype.Academic_Certification__c.Certificate")


def test_a_field_set_uses_displayed_fields_not_available_ones():
    """`availableFields` outnumbers `displayedFields` 53 to 7 in a real file.

    It means "an admin could add this", not "this is shown", so edges for it would
    drown the ones that answer "which UI reads this field".
    """
    source = """\
<?xml version="1.0" encoding="UTF-8"?>
<FieldSet xmlns="http://soap.sforce.com/2006/04/metadata">
    <fullName>BDE_Entry_FS</fullName>
    <availableFields>
        <field>AccountNumber</field>
        <isFieldManaged>false</isFieldManaged>
    </availableFields>
    <displayedFields>
        <field>Name</field>
        <isFieldManaged>false</isFieldManaged>
    </displayedFields>
    <label>BDE Entry</label>
</FieldSet>
"""
    path = f"{OBJECTS}/Account/fieldSets/BDE_Entry_FS.fieldSet-meta.xml"
    parsed = _parse(source, path)
    targets = _targets(parsed, _one(parsed, "field_set").qualified_name, RelType.IMPORTS)
    assert f"{SOBJECT_NAMESPACE}.Account.Name" in targets
    assert f"{SOBJECT_NAMESPACE}.Account.AccountNumber" not in targets


def test_a_web_link_opens_a_visualforce_page():
    """SalesforceFoundation/NPSP `Account/Manage_Household`, verbatim."""
    source = """\
<?xml version="1.0" encoding="UTF-8"?>
<WebLink xmlns="http://soap.sforce.com/2006/04/metadata">
    <fullName>Manage_Household</fullName>
    <displayType>button</displayType>
    <linkType>page</linkType>
    <masterLabel>Manage Household</masterLabel>
    <page>HH_ManageHHAccount</page>
</WebLink>
"""
    path = f"{OBJECTS}/Account/webLinks/Manage_Household.webLink-meta.xml"
    parsed = _parse(source, path)
    assert f"{PAGE_NAMESPACE}.HH_ManageHHAccount" in _targets(
        parsed, _one(parsed, "web_link").qualified_name, RelType.IMPORTS
    )


def test_a_web_link_that_is_not_a_page_names_nothing():
    """`linkType` url/javascript names something this module does not model."""
    source = """\
<?xml version="1.0"?>
<WebLink xmlns="http://soap.sforce.com/2006/04/metadata">
    <fullName>Open_Docs</fullName>
    <linkType>url</linkType>
    <url>https://example.invalid/docs</url>
</WebLink>
"""
    path = f"{OBJECTS}/Account/webLinks/Open_Docs.webLink-meta.xml"
    parsed = _parse(source, path)
    targets = _targets(parsed, _one(parsed, "web_link").qualified_name, RelType.IMPORTS)
    assert targets == {f"{SOBJECT_NAMESPACE}.Account"}


def test_a_list_view_skips_legacy_report_tokens():
    """`columns` mixes field API names with `NAME`, `RECORDTYPE`, `CORE.USERS.ALIAS`.

    Those name no field; each would mint an `ext/` stub for something that does not
    exist. Standard fields are lost with them, which is the price of not inventing
    82 targets out of 225.
    """
    source = """\
<?xml version="1.0" encoding="UTF-8"?>
<ListView xmlns="http://soap.sforce.com/2006/04/metadata">
    <fullName>All</fullName>
    <columns>NAME</columns>
    <columns>Issuer__c</columns>
    <columns>RECORDTYPE</columns>
    <columns>CORE.USERS.ALIAS</columns>
    <filterScope>Everything</filterScope>
    <label>All</label>
</ListView>
"""
    path = f"{OBJECTS}/Academic_Certification__c/listViews/All.listView-meta.xml"
    parsed = _parse(source, path)
    targets = _targets(parsed, _one(parsed, "list_view").qualified_name, RelType.IMPORTS)
    assert targets == {
        f"{SOBJECT_NAMESPACE}.Academic_Certification__c",
        f"{SOBJECT_NAMESPACE}.Academic_Certification__c.Issuer__c",
    }


@pytest.mark.parametrize("kind", ["validation_rule", "record_type", "field_set", "web_link", "list_view"])
def test_a_decomposed_child_is_never_a_callable(kind: str):
    """None of these may be a `Callable`, and the reason is not stylistic.

    `resolve_calls` builds `name_to_callables` from every Callable in the project
    and matches on **bare name**, project-wide. A validation rule named `validate`
    would be a candidate for an Apex `validate()` call, and if it were the only
    one it would resolve at confidence `resolved` -- a confidently wrong edge,
    which is exactly what ADR-0032 exists to prevent.

    A `Flow` is a Callable because `subflows` genuinely invokes one by name.
    Nothing invokes a validation rule or a list view by name, so `Callable` buys
    nothing here and costs that. Every IMPORTS edge is unaffected by the label.
    """
    sources = {
        "validation_rule": (
            "validationRules",
            "V",
            "<ValidationRule xmlns='{ns}'><fullName>V</fullName></ValidationRule>",
        ),
        "record_type": ("recordTypes", "R", "<RecordType xmlns='{ns}'><fullName>R</fullName></RecordType>"),
        "field_set": ("fieldSets", "F", "<FieldSet xmlns='{ns}'><fullName>F</fullName></FieldSet>"),
        "web_link": ("webLinks", "W", "<WebLink xmlns='{ns}'><fullName>W</fullName></WebLink>"),
        "list_view": ("listViews", "L", "<ListView xmlns='{ns}'><fullName>L</fullName></ListView>"),
    }
    child_dir, name, template = sources[kind]
    suffix = {
        "validation_rule": "validationRule",
        "record_type": "recordType",
        "field_set": "fieldSet",
        "web_link": "webLink",
        "list_view": "listView",
    }[kind]
    body = template.format(ns="http://soap.sforce.com/2006/04/metadata")
    parsed = _parse(
        f'<?xml version="1.0"?>\n{body}\n',
        f"{OBJECTS}/Account/{child_dir}/{name}.{suffix}-meta.xml",
    )
    assert _one(parsed, kind).label is NodeLabel.VALUE


# ---------------------------------------------------------------------------
# 4d. PermissionSet and Profile
# ---------------------------------------------------------------------------

PERMSETS = "force-app/main/default/permissionsets"
PROFILES = "force-app/main/default/profiles"


def test_a_permission_set_is_one_node_and_its_grants_are_edges():
    """The shape from bcgov/MoH-SAT `Salesforce_Backup_Administrator`, trimmed.

    That real 427 KB file became 1,947 nodes under the generic parse -- 1,944 of
    them all named `fieldPermissions` with empty source. Every fact is in the edges.
    """
    source = """\
<?xml version="1.0" encoding="UTF-8"?>
<PermissionSet xmlns="http://soap.sforce.com/2006/04/metadata">
    <label>Backup Administrator</label>
    <hasActivationRequired>false</hasActivationRequired>
    <fieldPermissions>
        <editable>true</editable>
        <field>Employee.AlternateEmail</field>
        <readable>true</readable>
    </fieldPermissions>
    <objectPermissions>
        <allowRead>true</allowRead>
        <object>AIInsightReason</object>
    </objectPermissions>
    <classAccesses>
        <apexClass>BackupController</apexClass>
        <enabled>true</enabled>
    </classAccesses>
    <userPermissions>
        <enabled>true</enabled>
        <name>AssignTopics</name>
    </userPermissions>
</PermissionSet>
"""
    parsed = _parse(source, f"{PERMSETS}/Salesforce_Backup_Administrator.permissionset-meta.xml")
    node = _one(parsed, "permission_set")
    assert node.qualified_name == _uid(f"{PERMISSION_SET_NAMESPACE}.Salesforce_Backup_Administrator")
    # The Module and the permission set, and nothing per row.
    assert len(parsed.entities) == 2
    assert _targets(parsed, node.qualified_name, RelType.IMPORTS) == {
        f"{SOBJECT_NAMESPACE}.Employee.AlternateEmail",
        f"{SOBJECT_NAMESPACE}.AIInsightReason",
        f"{APEX_NAMESPACE}.BackupController",
    }
    # A non-referential grant is rolled up, not dropped.
    assert node.extra_properties["user_permissions"] == ["AssignTopics"]
    assert node.extra_properties["fields_editable"] == ["Employee.AlternateEmail"]


def test_an_all_false_permission_row_grants_nothing():
    """A permission set can only add, so an all-false row is noise by construction.

    One real 6,441-line permission set carries 1,276 such rows and 8 real facts.
    """
    source = """\
<?xml version="1.0"?>
<PermissionSet xmlns="http://soap.sforce.com/2006/04/metadata">
    <label>External Committee User</label>
    <fieldPermissions>
        <editable>false</editable>
        <field>Account.Rating</field>
        <readable>false</readable>
    </fieldPermissions>
    <fieldPermissions>
        <editable>false</editable>
        <field>Account.Industry</field>
        <readable>true</readable>
    </fieldPermissions>
</PermissionSet>
"""
    parsed = _parse(source, f"{PERMSETS}/External_Committee_User.permissionset-meta.xml")
    node = _one(parsed, "permission_set")
    targets = _targets(parsed, node.qualified_name, RelType.IMPORTS)
    assert targets == {f"{SOBJECT_NAMESPACE}.Account.Industry"}
    assert node.extra_properties["fields_readable"] == ["Account.Industry"]


def test_a_cumulusci_namespace_token_is_folded_away():
    """CumulusCI templates a package namespace into the metadata it ships.

    An unmanaged build substitutes the empty string, and that is what the field
    file in the same repo is named -- so folding the token is what makes the target
    meet a real node instead of `ext/%%%NAMESPACE%%%Course_Enrollment__c`.
    """
    source = """\
<?xml version="1.0"?>
<PermissionSet xmlns="http://soap.sforce.com/2006/04/metadata">
    <label>EDA</label>
    <fieldPermissions>
        <editable>true</editable>
        <field>%%%NAMESPACE%%%Course_Enrollment__c.%%%NAMESPACE%%%Grade__c</field>
        <readable>true</readable>
    </fieldPermissions>
</PermissionSet>
"""
    parsed = _parse(source, f"{PERMSETS}/EDA.permissionset-meta.xml")
    assert _targets(parsed, _one(parsed, "permission_set").qualified_name, RelType.IMPORTS) == {
        f"{SOBJECT_NAMESPACE}.Course_Enrollment__c.Grade__c"
    }


def test_a_profile_reads_tab_visibilities_where_a_permission_set_reads_tab_settings():
    """Two element names for one concept, and neither doc mentions the other.

    A shared handler reading one name returns nothing at all for the other type,
    and the file still parses fine -- so the failure is silent.
    """
    permset = """\
<?xml version="1.0"?>
<PermissionSet xmlns="http://soap.sforce.com/2006/04/metadata">
    <label>Ops</label>
    <tabSettings><tab>Broker__c</tab><visibility>Visible</visibility></tabSettings>
</PermissionSet>
"""
    profile = """\
<?xml version="1.0"?>
<Profile xmlns="http://soap.sforce.com/2006/04/metadata">
    <custom>true</custom>
    <tabVisibilities><tab>Broker__c</tab><visibility>DefaultOn</visibility></tabVisibilities>
</Profile>
"""
    parsed_set = _parse(permset, f"{PERMSETS}/Ops.permissionset-meta.xml")
    parsed_profile = _parse(profile, f"{PROFILES}/Admin.profile-meta.xml")
    assert f"{TAB_NAMESPACE}.Broker__c" in _targets(
        parsed_set, _one(parsed_set, "permission_set").qualified_name, RelType.IMPORTS
    )
    assert f"{TAB_NAMESPACE}.Broker__c" in _targets(
        parsed_profile, _one(parsed_profile, "profile").qualified_name, RelType.IMPORTS
    )


def test_a_hidden_tab_is_not_a_grant():
    source = """\
<?xml version="1.0"?>
<PermissionSet xmlns="http://soap.sforce.com/2006/04/metadata">
    <label>Ops</label>
    <tabSettings><tab>Broker__c</tab><visibility>Hidden</visibility></tabSettings>
</PermissionSet>
"""
    parsed = _parse(source, f"{PERMSETS}/Ops.permissionset-meta.xml")
    assert _targets(parsed, _one(parsed, "permission_set").qualified_name, RelType.IMPORTS) == set()


@pytest.mark.parametrize("kind", ["permission_set", "profile"])
def test_permission_documents_keep_their_vector(kind: str):
    """Excluding these was considered and rejected.

    The volume argument is already answered -- one node per file instead of 1,947 --
    and what remains carries the admin's own `label` and `description` as its
    docstring. The grants are properties and edges, not text, so there is nothing
    table-shaped to exclude.
    """
    from code_atlas.search.embeddings import DEFAULT_EXCLUDE_KINDS

    assert kind not in DEFAULT_EXCLUDE_KINDS


# ---------------------------------------------------------------------------
# 5. CustomLabels and CustomMetadata
# ---------------------------------------------------------------------------

CUSTOM_LABELS = """\
<?xml version="1.0"?>
<CustomLabels xmlns="http://soap.sforce.com/2006/04/metadata">
    <labels>
        <fullName>Greeting</fullName>
        <language>en_US</language>
        <protected>false</protected>
        <shortDescription>Home page greeting</shortDescription>
        <value>Welcome!</value>
    </labels>
    <labels>
        <fullName>Farewell</fullName>
        <value>Bye</value>
    </labels>
</CustomLabels>
"""


def test_custom_labels_produce_one_value_each():
    parsed = _parse(CUSTOM_LABELS, "force-app/main/default/labels/CustomLabels.labels-meta.xml")

    module = _one(parsed, "sf_labels")
    labels = _by_kind(parsed, "custom_label")
    assert [label.name for label in labels] == ["Greeting", "Farewell"]
    assert labels[0].label == NodeLabel.VALUE
    assert labels[0].qualified_name == _uid("label.Greeting")
    assert labels[0].docstring == "Home page greeting"
    assert labels[0].source == "Welcome!"
    assert _targets(parsed, module.qualified_name, RelType.DEFINES) == {
        _uid("label.Greeting"),
        _uid("label.Farewell"),
    }


def test_one_file_many_labels_is_capped():
    """The one Tier-1 type where a single file can hold thousands of components."""

    entries = "".join(f"    <labels><fullName>L{i}</fullName><value>v</value></labels>\n" for i in range(1500))
    source = f'<?xml version="1.0"?>\n<CustomLabels xmlns="http://soap.sforce.com/2006/04/metadata">\n{entries}</CustomLabels>\n'

    parsed = _parse(source, "force-app/main/default/labels/CustomLabels.labels-meta.xml")
    # CustomLabels has its own budget: one file holds every label in the org, so its
    # component count is a property of the org, not of the document. NPSP's declares
    # 2,046 and used to lose 1,047 of them in source order.
    assert len(parsed.entities) == 1501
    # Every surviving label still got its containment edge — no dangling refs to
    # entities the budget cut.
    minted = {entity.qualified_name for entity in parsed.entities}
    assert all(to in minted for _, to in _rels(parsed, RelType.DEFINES))


CUSTOM_METADATA_RECORD = """\
<?xml version="1.0"?>
<CustomMetadata xmlns="http://soap.sforce.com/2006/04/metadata" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <label>Account trigger handler</label>
    <protected>false</protected>
    <values>
        <field>Class__c</field>
        <value xsi:type="xsd:string">MDTAccountTriggerHandler</value>
    </values>
    <values>
        <field>Object__c</field>
        <value xsi:type="xsd:string">Account</value>
    </values>
</CustomMetadata>
"""


def test_custom_metadata_record_infers_its_type_and_links_to_it():
    """The filename omits ``__mdt``; the type is still the SObject the record instantiates."""
    parsed = _parse(
        CUSTOM_METADATA_RECORD,
        "force-app/main/default/customMetadata/Metadata_Driven_Trigger.MDTAccountTriggerHandler.md-meta.xml",
    )

    record = _one(parsed, "custom_metadata_record")
    assert record.label == NodeLabel.VALUE
    assert record.qualified_name == _uid("cmdt.Metadata_Driven_Trigger__mdt.MDTAccountTriggerHandler")
    assert record.name == "MDTAccountTriggerHandler"
    assert record.docstring == "Account trigger handler"
    assert record.extra_properties["metadata_type"] == "Metadata_Driven_Trigger__mdt"
    # Field values go into `source`, not properties: the field names differ per
    # type and would pollute the node schema.
    assert record.source == "Class__c=MDTAccountTriggerHandler\nObject__c=Account"
    assert _targets(parsed, record.qualified_name, RelType.IMPORTS) == {
        f"{SOBJECT_NAMESPACE}.Metadata_Driven_Trigger__mdt"
    }


def test_custom_metadata_filename_that_already_carries_mdt_is_not_doubled():
    parsed = _parse(
        CUSTOM_METADATA_RECORD,
        "force-app/main/default/customMetadata/Trigger_Config__mdt.Handler.md-meta.xml",
    )
    assert _one(parsed, "custom_metadata_record").qualified_name == _uid("cmdt.Trigger_Config__mdt.Handler")


# ---------------------------------------------------------------------------
# 6. Dispatch — what this parser must NOT claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (
            "force-app/main/default/workflows/Account.workflow-meta.xml",
            (
                '<?xml version="1.0"?>\n<Workflow xmlns="http://soap.sforce.com/2006/04/metadata">\n'
                "  <fullName>Account</fullName>\n</Workflow>\n"
            ),
        ),
        (
            "force-app/main/default/flexipages/Property_Record_Page.flexipage-meta.xml",
            (
                '<?xml version="1.0"?>\n<FlexiPage xmlns="http://soap.sforce.com/2006/04/metadata">\n'
                "  <masterLabel>Property Record Page</masterLabel>\n</FlexiPage>\n"
            ),
        ),
        ("pom.xml", "<project>\n  <artifactId>acme</artifactId>\n</project>\n"),
    ],
    ids=["workflow", "flexipage", "maven"],
)
def test_unmodelled_root_elements_fall_through_to_the_generic_parse(path: str, source: str):
    parsed = _parse(source, path)
    assert _by_kind(parsed, "xml_document")
    assert _by_kind(parsed, "xml_element")


def test_a_bundle_whose_folder_is_not_an_api_name_falls_through():
    """The component is named for its directory, so an unusable directory declines.

    Declining hands the file to the generic structural parse, which is what it got
    before this handler existed -- never an empty ParsedFile.
    """
    source = (
        '<?xml version="1.0"?>\n<LightningComponentBundle xmlns="http://soap.sforce.com/2006/04/metadata">\n'
        "  <isExposed>true</isExposed>\n</LightningComponentBundle>\n"
    )
    parsed = _parse(source, "force-app/main/default/lwc/9lives/9lives.js-meta.xml")
    assert _by_kind(parsed, "lwc_component") == []
    assert _by_kind(parsed, "xml_document")


# ---------------------------------------------------------------------------
# 6b. The dialect route (ADR-0048)
# ---------------------------------------------------------------------------

_SFDX_WORKFLOW = (
    '<?xml version="1.0"?>\n<Workflow xmlns="http://soap.sforce.com/2006/04/metadata">\n'
    "  <fullName>Account</fullName>\n</Workflow>\n"
)

_SFDX_FLOW = (
    '<?xml version="1.0"?>\n<Flow xmlns="http://soap.sforce.com/2006/04/metadata">\n  <label>Loose</label>\n</Flow>\n'
)


@pytest.mark.parametrize(
    ("head", "claimed", "why"),
    [
        (_SFDX_FLOW, True, "a modelled root element plus the metadata namespace"),
        (
            '<?xml version="1.0"?>\n<Flow xmlns="urn:metadata.tooling.soap.sforce.com">\n</Flow>\n',
            True,
            (
                "the tooling namespace is a substring match, not exact equality -- "
                "a real .cls-meta.xml sidecar declares this one"
            ),
        ),
        (
            '<?xml version="1.0"?>\n<Flow>\n  <label>An orchestration</label>\n</Flow>\n',
            False,
            "a bare <Flow> root is generic enough for BPMN exports and workflow engines",
        ),
        (
            '<?xml version="1.0"?>\n<project><artifactId>acme</artifactId></project>\n',
            False,
            "an ordinary XML document",
        ),
        (
            _SFDX_WORKFLOW,
            False,
            (
                "SFDX, but a root element this module models no handler for -- claiming "
                "it would buy nothing and the generic parse already handles it"
            ),
        ),
    ],
    ids=["modelled", "tooling-namespace", "no-namespace", "not-salesforce", "unmodelled-type"],
)
def test_the_sniff_claims_only_what_it_models(head: str, claimed: bool, why: str):
    assert looks_like_salesforce_metadata(head.encode()) is claimed, why


def test_a_claimed_file_the_handler_declines_still_gets_its_entities():
    """The load-bearing property of this route.

    ADR-0048: a dialect that claims a file and then declines gets an **empty**
    `ParsedFile`, not a fallback -- which would delete the file's entities from the
    graph. The sniff sees bytes only, so it cannot anticipate the handler's
    path-shaped declines: this document has a modelled root element and the right
    namespace, and is declined solely because its name is not `*-meta.xml`.
    """
    parsed = _parse(_SFDX_FLOW, "exports/Loose.xml")
    assert _by_kind(parsed, "flow") == []
    # Exactly what it produced before the dialect route existed.
    assert _by_kind(parsed, "xml_document")
    assert _by_kind(parsed, "xml_element")


def test_a_non_salesforce_flow_document_is_not_claimed():
    """``<Flow>`` is a generic enough root that neither the tag nor the name alone is proof."""
    source = "<Flow>\n  <label>An orchestration</label>\n</Flow>\n"
    parsed = _parse(source, "workflows/pipeline.xml")
    assert _by_kind(parsed, "flow") == []
    assert _by_kind(parsed, "xml_document")


def test_the_metadata_namespace_alone_is_not_enough_without_an_sfdx_filename():
    """The API name comes from the filename; with no ``*-meta.xml`` there is none."""
    parsed = _parse(CREATE_PROPERTY_FLOW, "retrieved/Create_Property.xml")
    assert _by_kind(parsed, "flow") == []
    assert _by_kind(parsed, "xml_document")


# ---------------------------------------------------------------------------
# 7. Robustness — a metadata tree holds tens of thousands of files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        '<?xml version="1.0"?>\n<CustomObject xmlns="http://soap.sforce.com/2006/04/metadata">\n',
        '<?xml version="1.0"?>\n<CustomObject><label>Unclosed\n</CustomObject>\n',
        '<?xml version="1.0"?>\n<CustomObject><<>></CustomObject>\n',
        "<CustomObject/>",
        "not xml at all",
        '<?xml version="1.0"?>\n<CustomObject>\xef\xbf\xbd</CustomObject>\n',
    ],
    ids=["unterminated", "unclosed-child", "garbage-markup", "empty-element", "not-xml", "replacement-char"],
)
def test_malformed_documents_never_raise(source: str):
    """Tree-sitter recovers from anything; the handler must not care that it did."""
    result = parse_file(
        f"{OBJECTS}/Property__c/Property__c.object-meta.xml",
        source.encode("utf-8"),
        PROJECT,
    )
    assert result is not None


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (
            f"{OBJECTS}/Property__c/fields/X.field-meta.xml",
            (
                '<?xml version="1.0"?>\n<CustomField xmlns="http://soap.sforce.com/2006/04/metadata">\n'
                "  <fullName></fullName>\n  <type></type>\n</CustomField>\n"
            ),
        ),
        (
            f"{FLOWS}/Empty.flow-meta.xml",
            (
                '<?xml version="1.0"?>\n<Flow xmlns="http://soap.sforce.com/2006/04/metadata">\n'
                "  <recordCreates><object/></recordCreates>\n"
                "  <subflows><flowName/></subflows>\n"
                "  <actionCalls><actionType>apex</actionType></actionCalls>\n</Flow>\n"
            ),
        ),
        (
            "force-app/main/default/labels/CustomLabels.labels-meta.xml",
            (
                '<?xml version="1.0"?>\n<CustomLabels xmlns="http://soap.sforce.com/2006/04/metadata">\n'
                "  <labels><value>orphan</value></labels>\n</CustomLabels>\n"
            ),
        ),
    ],
    ids=["nameless-field", "empty-flow-refs", "nameless-label"],
)
def test_empty_and_missing_elements_produce_no_junk_edges(path: str, source: str):
    """An empty ``<object/>`` names nothing, and must not become a node or an edge."""
    parsed = _parse(source, path)
    for rel in parsed.relationships:
        assert rel.to_name, f"empty edge target from {rel.from_qualified_name}"
        assert not rel.to_name.endswith("."), rel.to_name


def test_unknown_child_elements_are_ignored_not_fatal():
    """Salesforce adds metadata fields every release; unknown ones cost nothing."""
    source = """\
<?xml version="1.0"?>
<CustomObject xmlns="http://soap.sforce.com/2006/04/metadata">
    <label>Property</label>
    <someFutureSetting>
        <nested>value</nested>
    </someFutureSetting>
    <compactLayouts><fullName>Compact</fullName></compactLayouts>
</CustomObject>
"""
    parsed = _parse(source, f"{OBJECTS}/Property__c/Property__c.object-meta.xml")
    assert len(parsed.entities) == 2
    assert _one(parsed, "sobject").extra_properties["sobject_label"] == "Property"


def test_merge_field_syntax_is_not_mistaken_for_an_api_name():
    """A record-triggered flow's ``<object>`` can hold a template in hand-edited XML."""
    source = """\
<?xml version="1.0"?>
<Flow xmlns="http://soap.sforce.com/2006/04/metadata">
    <recordLookups><object>{!$Record.Type}</object></recordLookups>
    <recordCreates><object>Has Spaces</object></recordCreates>
    <recordUpdates><object>Valid__c</object></recordUpdates>
</Flow>
"""
    parsed = _parse(source, f"{FLOWS}/Templated.flow-meta.xml")
    flow_uid = _one(parsed, "flow").qualified_name
    assert _targets(parsed, flow_uid, RelType.IMPORTS) == {f"{SOBJECT_NAMESPACE}.Valid__c"}


def test_every_entity_is_hashed_and_positioned():
    """The framework contract: a content hash and a sane line span on every node."""
    for path, source in (
        (f"{OBJECTS}/Property__c/Property__c.object-meta.xml", PROPERTY_OBJECT),
        (f"{OBJECTS}/Property__c/fields/Broker__c.field-meta.xml", BROKER_LOOKUP_FIELD),
        (f"{FLOWS}/Create_Property.flow-meta.xml", CREATE_PROPERTY_FLOW),
        ("force-app/main/default/labels/CustomLabels.labels-meta.xml", CUSTOM_LABELS),
    ):
        parsed = _parse(source, path)
        assert parsed.entities
        for entity in parsed.entities:
            assert entity.content_hash, entity.qualified_name
            assert entity.file_path == path
            assert entity.qualified_name.startswith(f"{PROJECT}:")
            assert 1 <= entity.line_start <= entity.line_end
        qualified_names = [entity.qualified_name for entity in parsed.entities]
        assert len(set(qualified_names)) == len(qualified_names), qualified_names


def test_a_field_indexes_every_string_a_person_wrote():
    """docstring was `description or label` -- either/or -- and inlineHelpText was read
    nowhere. That last one is written for the end user who does not understand the
    field, which makes it the most searchable string in a metadata tree."""
    parsed = _parse(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<CustomField xmlns="http://soap.sforce.com/2006/04/metadata">\n'
        "    <fullName>Commission_Rate__c</fullName>\n"
        "    <label>Commission Rate</label>\n"
        "    <description>Percentage of sale price paid on completion.</description>\n"
        "    <inlineHelpText>Enter a whole number. 3 means 3%.</inlineHelpText>\n"
        "    <type>Percent</type>\n"
        "</CustomField>\n",
        f"{OBJECTS}/Broker__c/fields/Commission_Rate__c.field-meta.xml",
    )
    field = _one(parsed, "sobject_field")
    doc = field.docstring or ""
    assert "Commission Rate" in doc
    assert "Percentage of sale price paid on completion." in doc
    assert "Enter a whole number. 3 means 3%." in doc


def test_repeated_prose_is_not_indexed_twice():
    """label and masterLabel routinely repeat each other, and a docstring saying the
    same phrase twice wastes the embedding it costs."""
    parsed = _parse(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<CustomField xmlns="http://soap.sforce.com/2006/04/metadata">\n'
        "    <fullName>Stage__c</fullName>\n"
        "    <label>Stage</label>\n"
        "    <description>Stage</description>\n"
        "    <type>Picklist</type>\n"
        "</CustomField>\n",
        f"{OBJECTS}/Broker__c/fields/Stage__c.field-meta.xml",
    )
    assert (_one(parsed, "sobject_field").docstring or "") == "Stage"
