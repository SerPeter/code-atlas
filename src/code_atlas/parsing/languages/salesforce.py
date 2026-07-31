"""Salesforce SFDX metadata support — the declarative half of a Salesforce org.

``parsing/languages/apex.py`` covers the *code* half (``.cls`` / ``.trigger``);
this module covers the *metadata* half, which in SFDX source format is XML.
Every Salesforce metadata file — ``Account.object-meta.xml``,
``Broker__c.field-meta.xml``, ``Create_Property.flow-meta.xml`` — has the plain
``.xml`` suffix, so they all arrive at ``config.py``'s XML handler.
:func:`parse_salesforce_metadata` is the Salesforce-aware branch that handler
tries first; declining (returning ``None``) falls back to the generic
one-level-deep structural parse that was there before.

Dispatch is content-first, convention-second — the same shape ``config.py``
already uses to tell a Kubernetes manifest from a compose file:

* the **root element name** picks the handler (``CustomObject``, ``CustomField``,
  ``Flow``, ``CustomLabels``, ``CustomMetadata``), and
* the file must additionally *look* like SFDX metadata — either the
  ``*-meta.xml`` filename convention or the ``soap.sforce.com`` metadata
  namespace on the root element. Without that guard a ``<Flow>`` document in a
  BPMN or workflow-engine repo would be read as a Salesforce flow.

Naming — why it matters more here than anywhere else
----------------------------------------------------
``apex.py`` already coined two namespaces, and they are *contracts*, not local
conventions::

    apex.<ClassName>[.<member>]   Apex classes and their members  (real entities)
    sobject.<ObjectApiName>       every SObject reference Apex/LWC makes

``apex.py`` emits ``IMPORTS -> sobject.Account`` for SOQL and DML;
``typescript.py`` emits the same target for ``@salesforce/schema/Account.Name``.
``GraphClient.resolve_imports`` matches an import target against the
*qualified_name* of real entities first and only mints an
``ExternalPackage``/``ExternalSymbol`` stub when nothing matches.  So by giving
the SObject minted from ``Account.object-meta.xml`` the qualified name
``sobject.Account``, every Apex SOQL query, every LWC schema import and the
object's own definition converge on **one** node — and in a repo with no
``Account.object-meta.xml`` (standard objects have no source file) they converge
on the ``ext/sobject.Account`` stub instead.  Either way, one node.

Two namespaces are added here, following the same rule:

    flow.<FlowApiName>            Flows
    cmdt.<Type__mdt>.<Record>     CustomMetadata records

Schema mapping (no new NodeLabels, no new RelTypes — both have import-time
validators that RuntimeError):

===========================  ==================  ============================
metadata                     label               kind
===========================  ==================  ============================
the file                     ``Module``          ``sf_object`` / ``sf_field`` /
                                                 ``sf_flow`` / ``sf_labels`` /
                                                 ``sf_custom_metadata``
CustomObject                 ``TypeDef``         ``sobject``
CustomField                  ``Value``           ``sobject_field``
Flow                         ``Callable``        ``flow``
CustomLabel                  ``Value``           ``custom_label``
CustomMetadata record        ``Value``           ``custom_metadata_record``
===========================  ==================  ============================

An SObject is a ``TypeDef`` because that is what ``apex.py`` makes an Apex class
and what ``resolve_type_refs``/``resolve_member_defines`` resolve against; a
field is a ``Value`` because that is what ``jvm.py`` (and therefore ``apex.py``)
makes a class field, and a Salesforce field is likewise a named data slot rather
than an invokable.  A Flow is a ``Callable`` for the same reason ``config.py``
makes a CI job and an Ansible task one: it is an invokable unit, and being a
``Callable`` is what lets ``resolve_calls`` wire a ``subflows`` reference to it.
The variety of SObject (custom / platform event / custom metadata type / …)
lives in ``extra_properties["sobject_type"]`` rather than in ``kind``, so that
``kind = 'sobject'`` stays one stable predicate for "this is a Salesforce
object".

Edge mapping, constrained by ``GraphClient``'s routing registries:

* ``DEFINES`` (uid-routed) for file -> component, and for a field declared
  *inline* in its object file.
* ``DEFINES`` carrying ``parent_type_name`` (post-batch,
  ``resolve_member_defines``) for a *decomposed* field —
  ``objects/X/fields/Y.field-meta.xml`` is a different file from
  ``objects/X/X.object-meta.xml``, and a uid-routed edge would silently drop
  whenever the field file happens to be upserted first.
* ``IMPORTS`` (post-batch, ``resolve_imports``) for every cross-component
  reference: field -> ``sobject.<referenceTo>``, flow -> ``sobject.<object>``,
  flow -> ``apex.<Class>``, CMDT record -> ``sobject.<Type__mdt>``, and the
  field *file*'s module -> ``sobject.<owner>``.  IMPORTS is chosen over
  ``USES_TYPE`` throughout because it matches on the full namespaced qualified
  name (so ``sobject.Account`` cannot collide with an Apex class called
  ``Account``) and because it mints a stub for a target with no source file —
  which is the normal case for standard objects and managed-package classes.
  ``USES_TYPE`` would silently drop both.
* ``CALLS`` (post-batch, ``resolve_calls``) for flow -> subflow.

Known limitations, all deliberate:

* **Lookup/master-detail edges hang off the field, not the object.** The
  object-to-object reachability is therefore two hops,
  ``(SObject)-[:DEFINES]->(Value{sobject_field})-[:IMPORTS]->(SObject)``.
  Emitting the edge straight from the owning ``SObject`` was rejected because
  ``_recreate_file_relationships`` deletes edges by their *source node's*
  ``file_path``: an edge sourced at ``sobject.Account`` but contributed by
  ``fields/Broker__c.field-meta.xml`` would be wiped the next time
  ``Account.object-meta.xml`` alone was re-parsed, and nothing would ever
  restore it.  Anchoring each edge in the file that states the fact keeps the
  edge's lifetime equal to the fact's.
* **Field API names are only unique within their object**, so a field's
  qualified name is ``sobject.<Object>.<Field>``.  A field on a standard object
  with no ``.object-meta.xml`` in the repo has no ``TypeDef`` to attach to, so
  ``resolve_member_defines`` falls back to a ``DEFINES`` from the field's own
  module; ``extra_properties["sobject"]`` still records the owner, and the
  module's ``IMPORTS -> sobject.<owner>`` still reaches the shared stub.
* **API names are matched verbatim, not case-folded.**  Salesforce API names are
  case-insensitive, so ``FROM ACCOUNT`` in Apex will not meet ``Account`` here.
  Normalising would be worse, not better: it would break the join with
  ``apex.py``/``typescript.py``, which do not normalise either.  If this is ever
  fixed it has to be fixed in all three at once.
* Permission sets, profiles, layouts, flexipages, record types, validation
  rules, formula/validation-rule field references, and the "Apex class name
  living in a CustomMetadata string value" heuristic are all out of scope.

Robustness: real orgs have tens of thousands of metadata files and some are
enormous.  Every handler is written to tolerate missing, empty, repeated and
unknown elements without raising, and declines (``None``) rather than guessing
when it cannot identify the component.  The size and nesting-depth guards in
``parsing/ast.py`` already run before any of this, and are not duplicated here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from loguru import logger

from code_atlas.parsing.ast import ParsedEntity, ParsedFile, ParsedRelationship
from code_atlas.parsing.languages.apex import APEX_NAMESPACE, SOBJECT_NAMESPACE
from code_atlas.schema import NodeLabel, RelType

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from tree_sitter import Node

# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

FLOW_NAMESPACE = "flow"
"""Root qualified-name segment for Flows — ``flow.Create_Property``.

Flows, like Apex classes, are addressed org-wide by bare API name (a
``subflows`` element names ``Create_Property``, not a path), so a path-derived
qualified name would make those references structurally unmatchable.  Same
reasoning as ``apex.APEX_NAMESPACE``.
"""

CMDT_NAMESPACE = "cmdt"
"""Root qualified-name segment for CustomMetadata *records* — ``cmdt.Type__mdt.Record``.

The record's *type* is an ordinary SObject and lives under ``sobject.``; only
the record instances live here.
"""

LABEL_NAMESPACE = "label"
"""Root qualified-name segment for CustomLabels — ``label.Greeting``.

Nothing points at these yet: ``typescript.py`` leaves ``@salesforce/label/c.X``
as an ordinary external import and ``apex.py`` does not extract
``System.Label.X``.  The namespace is chosen so that whichever side is taught to
emit it first meets the definitions here.
"""

_METADATA_NS = "http://soap.sforce.com/2006/04/metadata"
_META_XML_SUFFIX = "-meta.xml"

# Salesforce API names are ``[A-Za-z][A-Za-z0-9_]*`` (``__c``/``__r`` suffixes and
# ``ns__`` managed-package prefixes are just underscores). Anything else in a
# slot that should hold one — an empty element, a merge field, a formula
# fragment, `{!$Record.Id}` — is not a component name and must not become a node
# or an edge target.
_API_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

_QN_UNSAFE_RE = re.compile(r"[^0-9A-Za-z_-]+")

_LANGUAGE = "xml"
"""``ParsedFile.language`` for everything emitted here.

``"xml"``, not ``"salesforce"``: the value has to be a *registered* language name
(``test_languages_init`` asserts it), and these files reach the parser through
``config.py``'s ``.xml`` registration.  This module registers no language of its
own — there is nothing for it to register, since SFDX gives every metadata file
the same plain ``.xml`` suffix.
"""

_MAX_ENTITIES_PER_FILE = 1000
"""Cap on components minted from one file.

Only ``CustomLabels`` can realistically reach it — one
``CustomLabels.labels-meta.xml`` holds every label in the org, and large orgs
run to thousands.  Every other Tier-1 type is one component per file.
"""


def _module_qualified_name(file_path: str) -> str:
    """Convert a file path to a dotted qualified name, dots folded in every segment.

    Byte-for-byte the same rule as ``config._module_qualified_name`` — the two
    must agree, because a Salesforce metadata file reaches this module through
    ``config._parse_xml`` and must claim the same Module uid whichever branch
    handles it.  It is duplicated rather than imported to keep the dependency
    between the two modules one-way (``config`` -> ``salesforce``); an import
    the other way would be a cycle.
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    return ".".join(part.replace(".", "_") for part in p.parts)


def _qn_segment(text: str) -> str:
    """Fold arbitrary text into one safe dotted-qualified-name segment."""
    cleaned = _QN_UNSAFE_RE.sub("_", text.strip()).strip("_")
    return cleaned or "unnamed"


def _api_name(value: str | None) -> str | None:
    """*value* if it is a well-formed Salesforce API name, else ``None``.

    Phrased as a narrowing filter rather than a predicate so callers can feed a
    possibly-absent element straight through without a second ``is not None``.
    """
    return value if value is not None and _API_NAME_RE.match(value) is not None else None


# ---------------------------------------------------------------------------
# XML element-tree primitives
#
# Shared with ``config._parse_xml``'s generic structural parse, which imports
# them from here.  They live in this module rather than in ``config`` so that
# the import between the two runs in exactly one direction.
# ---------------------------------------------------------------------------

_XML_TAG_NODES = frozenset({"STag", "EmptyElemTag"})


def _node_str(node: Node) -> str:
    text = node.text
    return text.decode("utf-8", errors="replace") if text is not None else ""


def xml_tag(element: Node) -> str | None:
    """The element's tag name, read from its ``STag``/``EmptyElemTag`` ``Name`` child."""
    for child in element.children:
        if child.type in _XML_TAG_NODES:
            for part in child.children:
                if part.type == "Name":
                    return _node_str(part)
            return None
    return None


def _xml_content(element: Node) -> Node | None:
    return next((child for child in element.children if child.type == "content"), None)


def xml_child_elements(element: Node) -> list[Node]:
    content = _xml_content(element)
    if content is None:
        return []
    return [child for child in content.children if child.type == "element"]


def xml_text(element: Node) -> str | None:
    """Concatenated character data directly inside *element*, or ``None`` if blank."""
    content = _xml_content(element)
    if content is None:
        return None
    parts: list[str] = []
    for child in content.children:
        if child.type == "CharData":
            parts.append(_node_str(child))
        elif child.type == "CDSect":
            parts.extend(_node_str(piece) for piece in child.children if piece.type == "CData")
    joined = "".join(parts).strip()
    return joined or None


def _xml_attributes(element: Node) -> dict[str, str]:
    """Attributes on the element's start tag, quotes stripped."""
    attributes: dict[str, str] = {}
    for child in element.children:
        if child.type not in _XML_TAG_NODES:
            continue
        for part in child.children:
            if part.type != "Attribute":
                continue
            name: str | None = None
            value = ""
            for piece in part.children:
                if piece.type == "Name" and name is None:
                    name = _node_str(piece)
                elif piece.type == "AttValue":
                    value = _node_str(piece).strip("\"'")
            if name is not None:
                attributes[name] = value
        break
    return attributes


# ---------------------------------------------------------------------------
# Typed child accessors
#
# Salesforce metadata is a flat-ish tree of repeated single-purpose elements, so
# everything below is phrased as "the children named X" rather than as a walk.
# ---------------------------------------------------------------------------


def _children(element: Node, tag: str) -> list[Node]:
    return [child for child in xml_child_elements(element) if xml_tag(child) == tag]


def _child(element: Node, tag: str) -> Node | None:
    return next((child for child in xml_child_elements(element) if xml_tag(child) == tag), None)


def _text_of(element: Node, tag: str) -> str | None:
    child = _child(element, tag)
    return xml_text(child) if child is not None else None


def _texts_of(element: Node, tag: str) -> list[str]:
    return [text for text in (xml_text(child) for child in _children(element, tag)) if text]


def _bool_of(element: Node, tag: str) -> bool | None:
    text = _text_of(element, tag)
    if text is None:
        return None
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return None


def _lines(element: Node) -> tuple[int, int]:
    return element.start_point[0] + 1, element.end_point[0] + 1


def _compact(properties: dict[str, Any]) -> dict[str, Any]:
    """Drop ``None`` values so absent metadata does not become a null property."""
    return {key: value for key, value in properties.items() if value is not None}


# ---------------------------------------------------------------------------
# Emission accumulator
# ---------------------------------------------------------------------------


@dataclass
class _Emit:
    """Entity/relationship accumulator with per-file deduping and a node budget."""

    file_path: str
    project_name: str
    entities: list[ParsedEntity] = field(default_factory=list)
    relationships: list[ParsedRelationship] = field(default_factory=list)
    seen_qns: set[str] = field(default_factory=set)
    seen_rels: set[tuple[str, str, str]] = field(default_factory=set)
    truncated: bool = False

    @property
    def full(self) -> bool:
        return len(self.entities) >= _MAX_ENTITIES_PER_FILE

    def add(
        self,
        *,
        name: str,
        qn_suffix: str,
        label: NodeLabel,
        kind: str,
        line_start: int,
        line_end: int,
        docstring: str | None = None,
        source: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str | None:
        """Append an entity; returns its uid, or ``None`` if the budget is spent."""
        if self.full:
            self.truncated = True
            return None
        qn = f"{self.project_name}:{qn_suffix}"
        if qn in self.seen_qns:
            counter = 2
            while f"{qn}#{counter}" in self.seen_qns:
                counter += 1
            qn = f"{qn}#{counter}"
        self.seen_qns.add(qn)
        self.entities.append(
            ParsedEntity(
                name=name,
                qualified_name=qn,
                label=label,
                kind=kind,
                line_start=line_start,
                line_end=max(line_start, line_end),
                file_path=self.file_path,
                docstring=docstring,
                source=source,
                extra_properties=extra or {},
            )
        )
        return qn

    def rel(
        self,
        from_uid: str | None,
        rel_type: RelType,
        to_name: str | None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Append a relationship, ignoring exact duplicates and unanchored edges.

        Both endpoints are nullable because ``add`` returns ``None`` once the
        per-file node budget is spent; an edge to a node that was never minted is
        dropped here rather than guarded at every call site.
        """
        if from_uid is None or not to_name:
            return
        key = (from_uid, rel_type.value, to_name)
        if key in self.seen_rels:
            return
        self.seen_rels.add(key)
        self.relationships.append(
            ParsedRelationship(
                from_qualified_name=from_uid,
                rel_type=rel_type,
                to_name=to_name,
                properties=properties or {},
            )
        )

    def imports_sobject(self, from_uid: str | None, api_name: str | None) -> None:
        """``IMPORTS -> sobject.<Name>`` — the shared identity ``apex.py`` also targets."""
        name = _api_name(api_name)
        if name is not None:
            self.rel(from_uid, RelType.IMPORTS, f"{SOBJECT_NAMESPACE}.{name}")

    def result(self) -> ParsedFile:
        if self.truncated:
            logger.warning(
                "salesforce: {} declares more than {} components — the rest were skipped",
                self.file_path,
                _MAX_ENTITIES_PER_FILE,
            )
        return ParsedFile(
            file_path=self.file_path,
            language=_LANGUAGE,
            entities=self.entities,
            relationships=self.relationships,
        )


# ---------------------------------------------------------------------------
# SFDX file-name conventions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MetaFile:
    """A ``<base>.<suffix>-meta.xml`` filename, split into its two halves."""

    base: str
    """Everything before the type suffix — ``Account``, ``Broker__c``,
    ``Metadata_Driven_Trigger.MDTAccountTriggerHandler``."""
    suffix: str
    """The metadata type suffix — ``object``, ``field``, ``flow``, ``labels``, ``md``."""


def _meta_file(path: str) -> _MetaFile | None:
    """Split an SFDX ``*-meta.xml`` basename, or ``None`` if it is not one."""
    name = PurePosixPath(path).name
    if not name.endswith(_META_XML_SUFFIX):
        return None
    stem = name[: -len(_META_XML_SUFFIX)]
    base, _, suffix = stem.rpartition(".")
    return _MetaFile(base=base, suffix=suffix) if base and suffix else None


def _decomposed_owner(path: str, child_dir: str) -> str | None:
    """Owner object for ``objects/<Object>/<child_dir>/<Name>.<x>-meta.xml``.

    SFDX decomposes ``CustomObject`` into a directory tree, so a field file's
    owner is not in the file at all — it is the grandparent directory.  Returns
    ``None`` for any other shape, which makes the caller decline the file rather
    than invent an owner.
    """
    parts = PurePosixPath(path).parts
    if len(parts) < 4 or parts[-2] != child_dir or parts[-4] != "objects":
        return None
    return _api_name(parts[-3])


# Checked in order, and the order is load-bearing only for `__c`: it must come
# last, because `Foo__mdt`/`Foo__e`/`Foo__b` do not end in `__c` but a naive
# "contains a double underscore" rule would claim them.
_SOBJECT_TYPE_BY_SUFFIX: tuple[tuple[str, str], ...] = (
    ("__mdt", "customMetadataType"),
    ("__e", "platformEvent"),
    ("__x", "externalObject"),
    ("__b", "bigObject"),
    ("__Share", "system"),
    ("__History", "system"),
    ("__Feed", "system"),
    ("__ChangeEvent", "system"),
    ("__c", "custom"),
)


def _sobject_type(api_name: str, element: Node) -> str:
    """Classify an object from its API-name suffix — the platform's own convention."""
    for suffix, kind in _SOBJECT_TYPE_BY_SUFFIX:
        if api_name.endswith(suffix):
            # A custom setting is a `__c` object distinguished only by the
            # presence of `<customSettingsType>`; there is no name suffix for it.
            if kind == "custom" and _text_of(element, "customSettingsType"):
                return "customSetting"
            return kind
    return "standard"


# ---------------------------------------------------------------------------
# CustomObject
# ---------------------------------------------------------------------------

_OBJECT_MODULE_KIND = "sf_object"
_SOBJECT_KIND = "sobject"


def _parse_object(emit: _Emit, element: Node, path: str, meta: _MetaFile) -> ParsedFile | None:
    """``objects/<Object>/<Object>.object-meta.xml`` -> one ``TypeDef``.

    Fields declared *inline* (the non-decomposed Metadata-API layout, still legal
    and still common in retrieved packages) are emitted here too, with a
    uid-routed ``DEFINES`` because both ends are in this file.
    """
    api_name = _api_name(meta.base)
    if api_name is None:
        return None

    module_uid = _module(emit, path, _OBJECT_MODULE_KIND, element)
    line_start, line_end = _lines(element)
    object_uid = emit.add(
        name=api_name,
        qn_suffix=f"{SOBJECT_NAMESPACE}.{api_name}",
        label=NodeLabel.TYPE_DEF,
        kind=_SOBJECT_KIND,
        line_start=line_start,
        line_end=line_end,
        docstring=_text_of(element, "description"),
        extra=_compact(
            {
                "sobject_type": _sobject_type(api_name, element),
                "sobject_label": _text_of(element, "label"),
                "plural_label": _text_of(element, "pluralLabel"),
                "sharing_model": _text_of(element, "sharingModel"),
                "deployment_status": _text_of(element, "deploymentStatus"),
                "enable_reports": _bool_of(element, "enableReports"),
            }
        ),
    )
    emit.rel(module_uid, RelType.DEFINES, object_uid)

    for field_element in _children(element, "fields"):
        _emit_field(emit, field_element, owner=api_name, module_uid=module_uid, parent_uid=object_uid)

    return emit.result()


# ---------------------------------------------------------------------------
# CustomField
# ---------------------------------------------------------------------------

_FIELD_MODULE_KIND = "sf_field"
_FIELD_KIND = "sobject_field"


def _parse_field(emit: _Emit, element: Node, path: str, meta: _MetaFile) -> ParsedFile | None:
    """``objects/<Object>/fields/<Field>.field-meta.xml`` -> one ``Value``.

    Declines when the path is not the decomposed-object shape: the owning object
    is *only* recoverable from the directory, and a field with no owner has no
    unique name (``Name``, ``Status__c`` and ``Picture__c`` recur across dozens
    of objects in any real org).
    """
    owner = _decomposed_owner(path, "fields")
    if owner is None:
        return None

    module_uid = _module(emit, path, _FIELD_MODULE_KIND, element)
    # The owner link that always resolves. `resolve_member_defines` below can
    # only attach the field to a real `TypeDef`, which does not exist for a
    # custom field on a standard object; this reaches the shared
    # `sobject.<Owner>` identity — internal node or `ext/` stub — regardless.
    emit.imports_sobject(module_uid, owner)
    _emit_field(emit, element, owner=owner, module_uid=module_uid, parent_uid=None, fallback_name=meta.base)
    return emit.result()


def _emit_field(
    emit: _Emit,
    element: Node,
    *,
    owner: str,
    module_uid: str | None,
    parent_uid: str | None,
    fallback_name: str | None = None,
) -> None:
    """Emit one field ``Value`` plus its containment and reference edges.

    *parent_uid* is set only when the object is declared in this same file, in
    which case containment is a plain uid-routed ``DEFINES``.  Otherwise the
    ``DEFINES`` carries ``parent_type_name`` and is resolved post-batch by
    ``GraphClient.resolve_member_defines``, which is ordering-independent — a
    uid-routed edge would be dropped whenever the field file happened to be
    upserted before its object file.
    """
    api_name = _api_name(_text_of(element, "fullName") or fallback_name)
    if api_name is None:
        return

    line_start, line_end = _lines(element)
    field_type = _text_of(element, "type")
    formula = _text_of(element, "formula")
    reference_to = _texts_of(element, "referenceTo")
    field_uid = emit.add(
        name=api_name,
        qn_suffix=f"{SOBJECT_NAMESPACE}.{owner}.{_qn_segment(api_name)}",
        label=NodeLabel.VALUE,
        kind=_FIELD_KIND,
        line_start=line_start,
        line_end=line_end,
        docstring=_text_of(element, "description") or _text_of(element, "label"),
        # The formula is the only part of a field that is *code*; putting it in
        # `source` is what makes it reachable from BM25 and vector search.
        source=formula,
        extra=_compact(
            {
                "sobject": owner,
                "field_type": field_type,
                "required": _bool_of(element, "required"),
                "unique": _bool_of(element, "unique"),
                "external_id": _bool_of(element, "externalId"),
                "reference_to": reference_to[0] if reference_to else None,
                "relationship_name": _text_of(element, "relationshipName"),
                "delete_constraint": _text_of(element, "deleteConstraint"),
                "value_set_name": _text_of(element, "valueSetName"),
            }
        ),
    )
    if field_uid is None:
        return

    if parent_uid is not None:
        emit.rel(parent_uid, RelType.DEFINES, field_uid)
    else:
        emit.rel(
            module_uid,
            RelType.DEFINES,
            field_uid,
            {"parent_type_name": owner},
        )

    # Lookup / master-detail / hierarchy targets — the declarative data model's
    # backbone. Anchored on the field rather than on the owning object; see the
    # module docstring for why.
    for target in reference_to:
        emit.imports_sobject(field_uid, target)
    # A roll-up summary names its child object as `Object.Field`.
    summary_key = _text_of(element, "summaryForeignKey")
    if summary_key and "." in summary_key:
        emit.imports_sobject(field_uid, summary_key.split(".", 1)[0])


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

_FLOW_MODULE_KIND = "sf_flow"
_FLOW_KIND = "flow"

_FLOW_READ_ELEMENTS: tuple[tuple[str, str], ...] = (
    ("recordLookups", "object"),
    ("dynamicChoiceSets", "object"),
    ("dynamicChoiceSets", "picklistObject"),
    ("variables", "objectType"),
    ("start", "object"),
)
"""``(flow element name, child element holding an SObject API name)`` — read side."""

_FLOW_WRITE_ELEMENTS: tuple[tuple[str, str], ...] = (
    ("recordCreates", "object"),
    ("recordUpdates", "object"),
    ("recordDeletes", "object"),
)
"""Same, for elements that perform DML."""


def _parse_flow(emit: _Emit, element: Node, path: str, meta: _MetaFile) -> ParsedFile | None:
    """``flows/<Flow>.flow-meta.xml`` -> one ``Callable`` plus its references.

    Deliberately *not* one node per flow element.  A large orchestration flow has
    hundreds of them and modelling each would multiply the graph's node count for
    questions nobody asks; what is worth extracting is which SObjects, Apex
    classes and subflows the flow reaches, rolled up to the flow itself.  The
    read/write split that would otherwise be lost (``IMPORTS`` edges carry no
    properties through ``resolve_imports``) is preserved on the flow node as
    ``sobjects_read`` / ``sobjects_written``.
    """
    api_name = _api_name(meta.base)
    if api_name is None:
        return None

    module_uid = _module(emit, path, _FLOW_MODULE_KIND, element)
    start = _child(element, "start")

    reads = _flow_sobjects(element, _FLOW_READ_ELEMENTS)
    writes = _flow_sobjects(element, _FLOW_WRITE_ELEMENTS)

    line_start, line_end = _lines(element)
    flow_uid = emit.add(
        name=api_name,
        qn_suffix=f"{FLOW_NAMESPACE}.{api_name}",
        label=NodeLabel.CALLABLE,
        kind=_FLOW_KIND,
        line_start=line_start,
        line_end=line_end,
        docstring=_text_of(element, "description"),
        extra=_compact(
            {
                "flow_label": _text_of(element, "label"),
                "process_type": _text_of(element, "processType"),
                "status": _text_of(element, "status"),
                "run_in_mode": _text_of(element, "runInMode"),
                "trigger_object": _text_of(start, "object") if start is not None else None,
                "trigger_type": _text_of(start, "triggerType") if start is not None else None,
                "record_trigger_type": _text_of(start, "recordTriggerType") if start is not None else None,
                "sobjects_read": sorted(reads) or None,
                "sobjects_written": sorted(writes) or None,
            }
        ),
    )
    emit.rel(module_uid, RelType.DEFINES, flow_uid)

    for api in sorted(reads | writes):
        emit.imports_sobject(flow_uid, api)
    for apex_class in sorted(_flow_apex_classes(element)):
        emit.rel(flow_uid, RelType.IMPORTS, f"{APEX_NAMESPACE}.{apex_class}")
    for subflow in sorted(_flow_subflows(element)):
        emit.rel(flow_uid, RelType.CALLS, subflow)

    return emit.result()


def _flow_sobjects(element: Node, sources: Iterable[tuple[str, str]]) -> set[str]:
    """SObject API names named by the given ``(element, child)`` pairs."""
    found: set[str] = set()
    for parent_tag, child_tag in sources:
        for parent in _children(element, parent_tag):
            for value in _texts_of(parent, child_tag):
                name = _api_name(value)
                if name is not None:
                    found.add(name)
    return found


def _flow_apex_classes(element: Node) -> set[str]:
    """Apex classes the flow invokes.

    Two spellings: the modern ``actionCalls`` with ``actionType=apex`` (whose
    ``actionName`` is the class holding the ``@InvocableMethod``) and the legacy
    ``apexPluginCalls`` with an explicit ``apexClass``.  Every other
    ``actionType`` — ``emailAlert``, ``quickAction``, ``lwcComponent``,
    ``externalService`` and ~100 more — names something this module does not
    model, and is skipped rather than guessed at.
    """
    classes = {_api_name(_text_of(call, "actionName")) for call in _action_calls(element, "apex")}
    classes |= {_api_name(_text_of(call, "apexClass")) for call in _children(element, "apexPluginCalls")}
    return {name for name in classes if name is not None}


def _flow_subflows(element: Node) -> set[str]:
    """Flows this flow invokes — ``subflows.flowName`` and ``actionType=flow``."""
    names = {_api_name(_text_of(subflow, "flowName")) for subflow in _children(element, "subflows")}
    names |= {_api_name(_text_of(call, "actionName")) for call in _action_calls(element, "flow")}
    return {name for name in names if name is not None}


def _action_calls(element: Node, action_type: str) -> list[Node]:
    """``actionCalls`` children of the given ``actionType``, matched case-insensitively."""
    return [
        call
        for call in _children(element, "actionCalls")
        if (_text_of(call, "actionType") or "").strip().lower() == action_type
    ]


# ---------------------------------------------------------------------------
# CustomLabels
# ---------------------------------------------------------------------------

_LABELS_MODULE_KIND = "sf_labels"
_LABEL_KIND = "custom_label"


def _parse_labels(emit: _Emit, element: Node, path: str, meta: _MetaFile) -> ParsedFile | None:  # noqa: ARG001
    """``labels/CustomLabels.labels-meta.xml`` -> one ``Value`` per ``<labels>``.

    The one Tier-1 type where a single file holds many components, so editing one
    label re-diffs all of them.  Correctness is unaffected (the per-entity
    ``content_hash`` gate still skips the unchanged ones); only churn is higher.
    """
    module_uid = _module(emit, path, _LABELS_MODULE_KIND, element)
    for label_element in _children(element, "labels"):
        api_name = _api_name(_text_of(label_element, "fullName"))
        if api_name is None:
            continue
        line_start, line_end = _lines(label_element)
        label_uid = emit.add(
            name=api_name,
            qn_suffix=f"{LABEL_NAMESPACE}.{api_name}",
            label=NodeLabel.VALUE,
            kind=_LABEL_KIND,
            line_start=line_start,
            line_end=line_end,
            docstring=_text_of(label_element, "shortDescription"),
            source=_text_of(label_element, "value"),
            extra=_compact(
                {
                    "language": _text_of(label_element, "language"),
                    "protected": _bool_of(label_element, "protected"),
                    "categories": _text_of(label_element, "categories"),
                }
            ),
        )
        if label_uid is None:
            break
        emit.rel(module_uid, RelType.DEFINES, label_uid)
    return emit.result()


# ---------------------------------------------------------------------------
# CustomMetadata records
# ---------------------------------------------------------------------------

_CMDT_MODULE_KIND = "sf_custom_metadata"
_CMDT_KIND = "custom_metadata_record"
_MDT_SUFFIX = "__mdt"


def _parse_custom_metadata(emit: _Emit, element: Node, path: str, meta: _MetaFile) -> ParsedFile | None:
    """``customMetadata/<Type>.<Record>.md-meta.xml`` -> one ``Value``.

    The type is the part before the first dot, with ``__mdt`` implied when the
    filename omits it (which it usually does).  The record's ``<values>`` are
    flattened into ``source`` rather than into properties: the field names differ
    per type, so they would pollute the node schema, and full-text search over
    ``field=value`` is what makes the config content findable at all.
    """
    full_name = _text_of(element, "fullName") or meta.base
    raw_type, _, raw_record = full_name.partition(".")
    if not raw_record:
        return None
    type_name = _api_name(raw_type if raw_type.endswith(_MDT_SUFFIX) else raw_type + _MDT_SUFFIX)
    record_name = _api_name(raw_record)
    if type_name is None or record_name is None:
        return None

    module_uid = _module(emit, path, _CMDT_MODULE_KIND, element)
    line_start, line_end = _lines(element)
    record_uid = emit.add(
        name=record_name,
        qn_suffix=f"{CMDT_NAMESPACE}.{type_name}.{_qn_segment(record_name)}",
        label=NodeLabel.VALUE,
        kind=_CMDT_KIND,
        line_start=line_start,
        line_end=line_end,
        docstring=_text_of(element, "label"),
        source="\n".join(_cmdt_values(element)) or None,
        extra=_compact(
            {
                "metadata_type": type_name,
                "protected": _bool_of(element, "protected"),
            }
        ),
    )
    emit.rel(module_uid, RelType.DEFINES, record_uid)
    emit.imports_sobject(record_uid, type_name)
    return emit.result()


def _cmdt_values(element: Node) -> list[str]:
    """``<values><field>X</field><value>Y</value></values>`` -> ``["X=Y", ...]``."""
    rendered: list[str] = []
    for entry in _children(element, "values"):
        name = _text_of(entry, "field")
        if not name:
            continue
        rendered.append(f"{name}={_text_of(entry, 'value') or ''}")
    return rendered


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, Callable[[_Emit, Node, str, _MetaFile], ParsedFile | None]] = {
    "CustomObject": _parse_object,
    "CustomField": _parse_field,
    "Flow": _parse_flow,
    "CustomLabels": _parse_labels,
    "CustomMetadata": _parse_custom_metadata,
}
"""Root element name -> handler.  This *is* the supported-type list.

Every other Salesforce root element — ``LightningComponentBundle`` on an LWC's
``.js-meta.xml``, ``PermissionSet``, ``Layout``, ``FlexiPage``, ``ApexClass`` on
a ``.cls-meta.xml`` sidecar — falls through to ``config.py``'s generic
structural parse, which is the pre-existing behaviour for all of them.
"""


def _module(emit: _Emit, path: str, kind: str, element: Node) -> str | None:
    """Mint the file's ``Module`` node.

    Added before any component so that it, and not a component, owns the
    undeduplicated file-level qualified name — the same rule ``config.py``
    follows.
    """
    return emit.add(
        name=PurePosixPath(path).name,
        qn_suffix=_module_qualified_name(path),
        label=NodeLabel.MODULE,
        kind=kind,
        line_start=1,
        line_end=element.end_point[0] + 1,
    )


def _looks_like_sfdx(path: str, element: Node) -> bool:
    """Corroborate the root element name with a Salesforce-specific signal.

    A bare ``<Flow>`` or ``<CustomObject>`` root is not proof: those names are
    generic enough to appear in BPMN exports, ORM mapping files and hand-rolled
    schema definitions.  Either the SFDX filename convention or the Metadata API
    namespace makes it Salesforce; neither makes it something else's.
    """
    if path.endswith(_META_XML_SUFFIX):
        return True
    return any(value == _METADATA_NS for key, value in _xml_attributes(element).items() if key.startswith("xmlns"))


def parse_salesforce_metadata(path: str, root: Node, project_name: str) -> ParsedFile | None:
    """Parse an SFDX metadata document, or return ``None`` to decline it.

    ``None`` means "not Salesforce metadata I model" and hands the file back to
    ``config._parse_xml``'s generic structural parse — it is *not* a rejection of
    the file.  Declining happens for an unknown root element, a non-SFDX
    filename with no metadata namespace, and any recognised type whose component
    identity cannot be established from the path (an orphaned field file, a
    CustomMetadata file with no record name).
    """
    element = next((child for child in root.children if child.type == "element"), None)
    if element is None:
        return None
    tag = xml_tag(element)
    if tag not in _HANDLERS or not _looks_like_sfdx(path, element):
        return None

    meta = _meta_file(path)
    if meta is None:
        # Namespace-identified but not SFDX-named: there is no filename to take
        # the component's API name from, and for CustomField/CustomMetadata no
        # directory either. The generic parse still indexes the content.
        return None

    emit = _Emit(file_path=path, project_name=project_name)
    return _HANDLERS[tag](emit, element, path, meta)
