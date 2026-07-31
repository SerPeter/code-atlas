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
    assert sobject.docstring == "A house for sale"
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


def test_rollup_summary_references_the_child_object():
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
    assert _targets(parsed, field_uid, RelType.IMPORTS) == {f"{SOBJECT_NAMESPACE}.Property__c"}


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
    assert _rels(parsed, RelType.IMPORTS) == set()


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
    from code_atlas.parsing.languages.salesforce import _MAX_ENTITIES_PER_FILE

    entries = "".join(f"    <labels><fullName>L{i}</fullName><value>v</value></labels>\n" for i in range(1500))
    source = f'<?xml version="1.0"?>\n<CustomLabels xmlns="http://soap.sforce.com/2006/04/metadata">\n{entries}</CustomLabels>\n'

    parsed = _parse(source, "force-app/main/default/labels/CustomLabels.labels-meta.xml")
    assert len(parsed.entities) == _MAX_ENTITIES_PER_FILE
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
            "web/lwc/propertyTile/propertyTile.js-meta.xml",
            '<?xml version="1.0"?>\n<LightningComponentBundle xmlns="http://soap.sforce.com/2006/04/metadata">\n'
            "  <isExposed>true</isExposed>\n</LightningComponentBundle>\n",
        ),
        (
            "force-app/main/default/permissionsets/Admin.permissionset-meta.xml",
            '<?xml version="1.0"?>\n<PermissionSet xmlns="http://soap.sforce.com/2006/04/metadata">\n'
            "  <label>Admin</label>\n</PermissionSet>\n",
        ),
        ("pom.xml", "<project>\n  <artifactId>acme</artifactId>\n</project>\n"),
    ],
    ids=["lwc-bundle", "permission-set", "maven"],
)
def test_unmodelled_root_elements_fall_through_to_the_generic_parse(path: str, source: str):
    parsed = _parse(source, path)
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
            '<?xml version="1.0"?>\n<CustomField xmlns="http://soap.sforce.com/2006/04/metadata">\n'
            "  <fullName></fullName>\n  <type></type>\n</CustomField>\n",
        ),
        (
            f"{FLOWS}/Empty.flow-meta.xml",
            '<?xml version="1.0"?>\n<Flow xmlns="http://soap.sforce.com/2006/04/metadata">\n'
            "  <recordCreates><object/></recordCreates>\n"
            "  <subflows><flowName/></subflows>\n"
            "  <actionCalls><actionType>apex</actionType></actionCalls>\n</Flow>\n",
        ),
        (
            "force-app/main/default/labels/CustomLabels.labels-meta.xml",
            '<?xml version="1.0"?>\n<CustomLabels xmlns="http://soap.sforce.com/2006/04/metadata">\n'
            "  <labels><value>orphan</value></labels>\n</CustomLabels>\n",
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
