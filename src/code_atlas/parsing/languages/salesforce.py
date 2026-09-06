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
    lwc.<BundleFolderName>        Lightning Web Component bundles

A namespace constant lives in the module that *mints* the node it addresses, and
every other module imports it — ``apex.py`` owns ``APEX_NAMESPACE`` and
``SOBJECT_NAMESPACE`` and this module imports both, while ``typescript.py``
imports ``LWC_NAMESPACE`` from here to rewrite a ``c/<name>`` sibling import.
The rule exists because an unreconciled namespace is silent: every edge lands on
a different ``ext/`` stub and nothing ever converges.

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
    from collections.abc import Callable, Iterable, Iterator

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

LWC_NAMESPACE = "lwc"
"""Root qualified-name segment for Lightning Web Component bundles — ``lwc.errorPanel``.

The bundle folder name, verbatim. A flow screen names a component as
``c:navigateToRecord`` and an Aura component as ``<c:navigateToRecord>``; both
strip to the same folder name, so both meet the bundle wherever it is minted —
or, until it is, the same ``ext/lwc.navigateToRecord`` stub. That convergence is
the whole reason to spell it out here rather than emit the raw ``c:`` string.
"""

VALIDATION_RULE_NAMESPACE = "validationrule"
RECORD_TYPE_NAMESPACE = "recordtype"
FIELD_SET_NAMESPACE = "fieldset"
WEB_LINK_NAMESPACE = "weblink"
LIST_VIEW_NAMESPACE = "listview"
"""Namespaces for the decomposed children of a ``CustomObject``.

Each is ``<kind>.<Object>.<Name>`` because none of these names is unique across
objects — every second object has a ``ListView`` called ``All`` — and the owner is
taken from the ``objects/<Object>/<child>/`` path rather than from the document,
which never states it.
"""

PAGE_NAMESPACE = "page"
"""Root qualified-name segment for Visualforce pages — ``page.HH_ManageHHAccount``.

Declared here and minted by nothing yet: a ``WebLink`` names the page it opens
before any parser reads ``pages/*.page`` (ATL-182), so those edges rest on
``ext/page.<Name>`` stubs and join the real node the moment one exists.  It lives
in this module rather than in ``markup.py`` because ``markup`` already imports
from here, and the reverse would be a cycle.

``page.<Name>`` matches the syntax Apex uses to reference one (``Page.<Name>``),
so the Apex side can join here unchanged if it is ever taught to emit it.
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
"""Default cap on components minted from one file.

Every modelled type except ``CustomLabels`` is one component per file, or a
handful, so this is unreachable for them and exists only as a backstop against a
pathological document.
"""

_LABELS_MAX_ENTITIES = 20_000
"""``CustomLabels``' own cap, and the reason :class:`_Emit` takes a budget at all.

One ``CustomLabels.labels-meta.xml`` holds *every* label in the org, so unlike
every other type its component count is a property of the org rather than of the
document — the default cap read "large orgs run to thousands" and then stopped at
1,000.  NPSP's declares 2,046 in 777 KB and lost 1,047 of them, silently as far as
the graph was concerned and in source order, so *which* labels survived depended
on file layout.

Splitting the file was rejected: a label's uid is ``label.<Name>`` precisely so
that whichever module learns to emit ``System.Label.X`` or
``@salesforce/label/c.X`` meets the definition here, and a per-chunk uid would
break that.  This is one file per project, and a label is a short string, so the
node cost is bounded and small.
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
    budget: int = _MAX_ENTITIES_PER_FILE
    entities: list[ParsedEntity] = field(default_factory=list)
    relationships: list[ParsedRelationship] = field(default_factory=list)
    seen_qns: set[str] = field(default_factory=set)
    seen_rels: set[tuple[str, str, str]] = field(default_factory=set)
    truncated: bool = False

    @property
    def full(self) -> bool:
        return len(self.entities) >= self.budget

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
                self.budget,
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
        docstring=_prose(element),
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


_PROSE_TAGS: tuple[str, ...] = (
    "label",
    "pluralLabel",
    "description",
    "inlineHelpText",
    "masterLabel",
    "errorMessage",
)
"""Elements holding prose a person wrote for another person.

Salesforce spreads a component's documentation across several of these and a component
typically fills more than one: ``label`` is the UI name, ``description`` is for the
admin, ``inlineHelpText`` is the hover text written for the end user who does not
understand the field. That last one is the most searchable string in a metadata tree and
was read nowhere.
"""


def _prose(element: Node) -> str | None:
    """Every human-readable string this component declares, in one docstring.

    Was ``description or label`` -- either/or, so a field with both indexed only the
    first and a field with neither but an inlineHelpText indexed nothing. Measured, a
    representative field reached the index at 0.13 of its own bytes.

    Deduplicated because ``label`` and ``masterLabel`` frequently repeat each other, and
    a docstring that says the same phrase twice wastes the embedding it costs.
    """
    seen: list[str] = []
    for tag in _PROSE_TAGS:
        text = _text_of(element, tag)
        if text and text.strip() and text.strip() not in seen:
            seen.append(text.strip())
    return "\n".join(seen) or None


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
        docstring=_prose(element),
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

_FLOW_DML_ELEMENTS: tuple[str, ...] = ("recordCreates", "recordUpdates", "recordDeletes")
"""Elements that perform DML. Their object is resolved, not read — see :func:`_flow_element_object`."""

_FLOW_RECORD_ELEMENTS: tuple[str, ...] = (*_FLOW_DML_ELEMENTS, "recordLookups")
"""Every element that both names an SObject and doubles as a record variable of that type."""

_OBJECT_BEARING_TAGS: tuple[str, ...] = ("object", "picklistObject")
_REFERENCE_BEARING_TAGS: tuple[str, ...] = ("inputReference", "outputReference", "collectionReference")

_FIELD_CONTAINER_PATHS: tuple[tuple[str, str, bool], ...] = (
    ("recordCreates", "inputAssignments", True),
    ("recordUpdates", "inputAssignments", True),
    ("recordUpdates", "filters", False),
    ("recordLookups", "filters", False),
    ("recordDeletes", "filters", False),
    ("recordLookups", "outputAssignments", False),
    ("dynamicChoiceSets", "filters", False),
    ("dynamicChoiceSets", "outputAssignments", False),
)
"""``(element, repeated container holding <field>, is_write)`` — the field sites with a wrapper."""

_FIELD_DIRECT_PATHS: tuple[tuple[str, str], ...] = (
    ("recordLookups", "sortField"),
    ("dynamicChoiceSets", "picklistField"),
    ("dynamicChoiceSets", "displayField"),
    ("dynamicChoiceSets", "valueField"),
    ("dynamicChoiceSets", "sortField"),
)
"""``(element, child naming a field directly)`` — all reads."""

_ACTION_SITE_TAGS: tuple[tuple[str, str], ...] = (
    ("actionName", "actionType"),
    ("exitActionName", "exitActionType"),
    ("entryActionName", "entryActionType"),
)
"""``(name tag, type tag)`` pairs that together name an invocable, wherever they appear."""

_FLOW_ACTION_TYPES: frozenset[str] = frozenset(
    {"flow", "createWorkItem", "stepBackground", "stepInteractive", "stepApproval"}
)
"""``actionType`` values whose ``actionName`` is a Flow API name.

``flow`` is the ordinary subflow-as-action spelling; the other four are
Orchestrator steps, each of which runs a screen or autolaunched flow. Every other
value — ~310 of them — names something this module does not model and is skipped
rather than guessed at.
"""
_FLOW_ACTION_TYPES_FOLDED: frozenset[str] = frozenset(value.lower() for value in _FLOW_ACTION_TYPES)


def _iter_elements(element: Node) -> Iterator[Node]:
    """Every descendant element, depth-first, *excluding* ``element`` itself.

    Flow nests references arbitrarily deep — 70 of 381 ``extensionName`` sites sit
    three ``fields`` levels down, and orchestration hides ``actionName`` two levels
    below the root — so the reference sweeps descend rather than enumerate paths.
    Depth is bounded by the document, and ``parse_file`` already catches
    ``RecursionError``.
    """
    for child in xml_child_elements(element):
        yield child
        yield from _iter_elements(child)


def _flow_symbol_table(element: Node) -> dict[str, str]:
    """``flow-local name -> SObject API name``, for the whole document.

    A ``<field>`` names a field on its element's sibling ``<object>`` — except when
    the element carries ``inputReference`` / ``outputReference`` instead, which names
    a flow variable. 43% of ``recordUpdates`` elements in a 419-flow corpus have no
    ``<object>`` at all, so without this table 27% of the field references they state
    are unresolvable and, worse, the object they *write* is recorded only as a read.
    """
    table: dict[str, str] = {}
    for variable in _children(element, "variables"):
        name = _text_of(variable, "name")
        api = _api_name(_text_of(variable, "objectType"))
        if name and api:
            table[name.strip()] = api
    # A record element's own <name> is the implicit record variable holding its result
    # when storeOutputAutomatically is set, so it belongs in the table too.
    for tag in _FLOW_RECORD_ELEMENTS:
        for record_element in _children(element, tag):
            name = _text_of(record_element, "name")
            api = _api_name(_text_of(record_element, "object"))
            if name and api:
                table[name.strip()] = api
    start = _child(element, "start")
    triggering = _api_name(_text_of(start, "object")) if start is not None else None
    if triggering is not None:
        table["$Record"] = triggering
        table["$Record__Prior"] = triggering
    return table


def _flow_resolve_reference(reference: str | None, table: dict[str, str]) -> str | None:
    """The SObject a ``<recordVariable>[.<anything>]`` reference names, or ``None``."""
    if reference is None:
        return None
    return table.get(reference.strip().partition(".")[0])


def _flow_element_object(element: Node, table: dict[str, str]) -> str | None:
    """The SObject an element operates on — its own tag first, then a flow-local reference."""
    for tag in _OBJECT_BEARING_TAGS:
        direct = _api_name(_text_of(element, tag))
        if direct is not None:
            return direct
    for tag in _REFERENCE_BEARING_TAGS:
        resolved = _flow_resolve_reference(_text_of(element, tag), table)
        if resolved is not None:
            return resolved
    return None


def _parse_flow(emit: _Emit, element: Node, path: str, meta: _MetaFile) -> ParsedFile | None:
    """``flows/<Flow>.flow-meta.xml`` -> one ``Callable`` plus its references.

    Deliberately *not* one node per flow element.  A large orchestration flow has
    hundreds of them, and every reference they state resolves to an existing
    namespace — ``sobject.X``, ``sobject.X.Y``, ``apex.X``, ``flow.X``, ``lwc.X`` —
    from inside this one file.  The element is *where* a reference lives, not the
    thing anyone searches for: nobody asks for ``Decision_3``, they ask which flow
    writes ``Case.Status``, and that is an edge.  Measured on a 419-flow corpus, a
    node per element is 18x the flow count and answers nothing the edges do not.

    The read/write split is preserved on the node as ``sobjects_read`` /
    ``sobjects_written`` / ``fields_read`` / ``fields_written`` because
    ``resolve_imports`` builds its edge from ``{from_uid, to_uid}`` alone and drops
    every property — so the split cannot live on the edge.
    """
    api_name = _api_name(meta.base)
    if api_name is None:
        return None

    module_uid = _module(emit, path, _FLOW_MODULE_KIND, element)
    start = _child(element, "start")
    table = _flow_symbol_table(element)

    reads = _flow_sobjects(element, _FLOW_READ_ELEMENTS)
    writes = {
        owner
        for tag in _FLOW_DML_ELEMENTS
        for dml in _children(element, tag)
        if (owner := _flow_element_object(dml, table)) is not None
    }
    fields_read, fields_written = _flow_field_references(element, table)

    line_start, line_end = _lines(element)
    flow_uid = emit.add(
        name=api_name,
        qn_suffix=f"{FLOW_NAMESPACE}.{api_name}",
        label=NodeLabel.CALLABLE,
        kind=_FLOW_KIND,
        line_start=line_start,
        line_end=line_end,
        docstring=_prose(element),
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
                "fields_read": sorted(fields_read) or None,
                "fields_written": sorted(fields_written) or None,
            }
        ),
    )
    emit.rel(module_uid, RelType.DEFINES, flow_uid)

    for api in sorted(reads | writes):
        emit.imports_sobject(flow_uid, api)
    for qualified in sorted(fields_read | fields_written):
        emit.rel(flow_uid, RelType.IMPORTS, f"{SOBJECT_NAMESPACE}.{qualified}")
    for apex_class in sorted(_flow_apex_classes(element)):
        emit.rel(flow_uid, RelType.IMPORTS, f"{APEX_NAMESPACE}.{apex_class}")
    for bundle in sorted(_flow_lwc_bundles(element)):
        emit.rel(flow_uid, RelType.IMPORTS, f"{LWC_NAMESPACE}.{bundle}")
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


def _flow_field_references(element: Node, table: dict[str, str]) -> tuple[set[str], set[str]]:
    """``(read, written)`` sets of ``<Object>.<Field>``, from structured paths only.

    ``elementReference`` values and ``{!...}`` merge fields inside formulas and text
    templates are *not* mined: measured on a 419-flow corpus, 37-38% of them name a
    flow variable, a global or a screen component rather than a field, and a wrong
    uid is worse than a missing entity (ADR-0032).  Every path swept here is a slot
    whose contract is "an SObject field API name", and all of them resolve.
    """
    read: set[str] = set()
    written: set[str] = set()
    for owner, field_name, is_write in _iter_field_sites(element, table):
        api_owner = _api_name(owner)
        api_field = _api_name(field_name)
        if api_owner is not None and api_field is not None:
            (written if is_write else read).add(f"{api_owner}.{api_field}")
    return read, written


def _iter_field_sites(element: Node, table: dict[str, str]) -> Iterator[tuple[str | None, str, bool]]:
    """``(owning SObject, field API name, is_write)`` for every structured field slot."""
    yield from _wrapped_field_sites(element, table)
    yield from _direct_field_sites(element, table)
    yield from _nested_field_sites(element, table)
    yield from _object_field_reference_sites(element, table)


def _wrapped_field_sites(element: Node, table: dict[str, str]) -> Iterator[tuple[str | None, str, bool]]:
    """``<element>/<container>/<field>`` — assignments and filters."""
    for parent_tag, container_tag, is_write in _FIELD_CONTAINER_PATHS:
        for parent in _children(element, parent_tag):
            owner = _flow_element_object(parent, table)
            for container in _children(parent, container_tag):
                for value in _texts_of(container, "field"):
                    yield owner, value, is_write


def _direct_field_sites(element: Node, table: dict[str, str]) -> Iterator[tuple[str | None, str, bool]]:
    """``<element>/<sortField|picklistField|displayField|valueField>`` — all reads."""
    for parent_tag, field_tag in _FIELD_DIRECT_PATHS:
        for parent in _children(element, parent_tag):
            owner = _flow_element_object(parent, table)
            for value in _texts_of(parent, field_tag):
                yield owner, value, False


def _nested_field_sites(element: Node, table: dict[str, str]) -> Iterator[tuple[str | None, str, bool]]:
    """The two families whose container is not a direct child of the element."""
    start = _child(element, "start")
    if start is not None:
        owner = _flow_element_object(start, table)
        for criterion in _children(start, "filters"):
            for value in _texts_of(criterion, "field"):
                yield owner, value, False
    for processor in _children(element, "collectionProcessors"):
        owner = _flow_element_object(processor, table)
        for option in _children(processor, "sortOptions"):
            for value in _texts_of(option, "sortField"):
                yield owner, value, False


def _object_field_reference_sites(element: Node, table: dict[str, str]) -> Iterator[tuple[str | None, str, bool]]:
    """``objectFieldReference``, always ``<recordVariableName>.<FieldApiName>``.

    Never ``<Object>.<Field>``, so the head goes through the symbol table like any
    other reference.  Swept over the whole document rather than under ``screens``
    alone: the value's shape is unambiguous and a screen field can nest three deep.
    """
    for descendant in _iter_elements(element):
        for value in _texts_of(descendant, "objectFieldReference"):
            head, _, field_name = value.strip().partition(".")
            if field_name:
                yield table.get(head), field_name, False


def _flow_action_sites(element: Node) -> list[tuple[str, str]]:
    """Every ``(name, type)`` invocable reference in the document, at any depth.

    Four spellings carry one, and only the first is a direct child of ``<Flow>``:
    ``actionCalls``, ``orchestratedStages/stageSteps``, ``steppedStages/steps`` and
    ``screens/actions``.  ``steppedStages`` is not in the published WSDL at all and
    was found only by reading real files.  A descent costs nothing and also catches
    ``FlowStart.fanOutAction``, which is documented but absent from every public flow
    sampled, so its shape could not be confirmed.
    """
    sites: list[tuple[str, str]] = []
    for descendant in _iter_elements(element):
        for name_tag, type_tag in _ACTION_SITE_TAGS:
            raw = _text_of(descendant, name_tag)
            if raw is None:
                continue
            sites.append((raw.strip(), (_text_of(descendant, type_tag) or "").strip().lower()))
    return sites


def _flow_apex_classes(element: Node) -> set[str]:
    """Apex classes the flow invokes.

    Three spellings: an action site whose type is ``apex`` (whose name is the class
    holding the ``@InvocableMethod``), the legacy ``apexPluginCalls`` with an explicit
    ``apexClass``, and an ``apexClass`` on a ``variables`` or ``transforms`` element.
    """
    classes = {name for name, action_type in _flow_action_sites(element) if action_type == "apex"}
    for tag in ("apexPluginCalls", "variables", "transforms"):
        classes |= set(_texts_of_descendants(element, tag, "apexClass"))
    return {name for name in (_api_name(candidate) for candidate in classes) if name is not None}


def _texts_of_descendants(element: Node, parent_tag: str, child_tag: str) -> list[str]:
    """``child_tag`` texts under every direct ``parent_tag`` child."""
    return [text for parent in _children(element, parent_tag) for text in _texts_of(parent, child_tag)]


def _flow_subflows(element: Node) -> set[str]:
    """Flows this flow invokes, overrides or was templated from.

    ``subflows.flowName`` is the ordinary call.  An action site typed ``flow`` is the
    same thing spelled as an action, and the four Orchestrator step types each run a
    flow.  ``overriddenFlow`` and ``sourceTemplate`` are flow-to-flow references that
    the HTML documentation omits and the Metadata API WSDL declares — the HTML field
    table is an incomplete rendering of the WSDL, which is the real spec.

    Returns **bare API names, not** ``flow.<Name>``.  ``CALLS`` is resolved by
    ``resolve_calls``, which matches a Callable's *name*; a namespaced target matches
    nothing and every subflow edge silently disappears.  This is the opposite
    convention to the ``IMPORTS`` targets above, which ``resolve_imports`` matches on
    the full qualified name — the two resolvers do not agree and the difference is
    load-bearing.
    """
    names = set(_texts_of_descendants(element, "subflows", "flowName"))
    names |= {name for name, action_type in _flow_action_sites(element) if action_type in _FLOW_ACTION_TYPES_FOLDED}
    names |= {text for tag in ("overriddenFlow", "sourceTemplate") for text in _texts_of(element, tag)}
    return {name for name in (_api_name(candidate) for candidate in names) if name is not None}


def _flow_lwc_bundles(element: Node) -> set[str]:
    """LWC bundles placed on a screen, from ``screens//fields/extensionName``.

    The value is namespaced — ``c:navigateToRecord`` — and the prefix is stripped
    before validation, because ``_api_name`` rejects the colon and would drop every
    one of them.  Walked as a descent: 70 of 381 real occurrences sit three ``fields``
    levels deep, where a direct-children sweep sees nothing.
    """
    bundles: set[str] = set()
    for descendant in _iter_elements(element):
        for value in _texts_of(descendant, "extensionName"):
            name = _api_name(value.strip().rpartition(":")[2])
            if name is not None:
                bundles.add(name)
    return bundles


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

_VALIDATION_RULE_MODULE_KIND = "sf_validation_rule"
_VALIDATION_RULE_KIND = "validation_rule"
_RECORD_TYPE_MODULE_KIND = "sf_record_type"
_RECORD_TYPE_KIND = "record_type"
_FIELD_SET_MODULE_KIND = "sf_field_set"
_FIELD_SET_KIND = "field_set"
_WEB_LINK_MODULE_KIND = "sf_web_link"
_WEB_LINK_KIND = "web_link"
_LIST_VIEW_MODULE_KIND = "sf_list_view"
_LIST_VIEW_KIND = "list_view"

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
        docstring=_prose(element),
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
# LightningComponentBundle
# ---------------------------------------------------------------------------

_LWC_MODULE_KIND = "sf_lwc_bundle"
_LWC_KIND = "lwc_component"

_APEX_DATASOURCE_PREFIX = "apex://"
_SCHEMA_TYPE_PREFIX = "@salesforce/schema/"


def _parse_lwc_bundle(emit: _Emit, element: Node, path: str, meta: _MetaFile) -> ParsedFile | None:  # noqa: ARG001
    """``lwc/<b>/<b>.js-meta.xml`` -> one ``TypeDef`` named for the **bundle folder**.

    The bundle is three files and this is the one that mints the component, because
    it is the only one that is 1:1 with a bundle and it is the file that declares
    the component exists.  The ``.js`` cannot: 26% of real bundles do not export
    ``PascalCase(folder)`` and three export an anonymous class, so its TypeDef's
    name is unpredictable.  The ``.html`` cannot either — real bundles exist with
    no ``<name>.html`` at all, picking a template in ``render()`` instead.

    All three files keep minting their own ``Module``.  Only ``FILE_HASH_LABELS``
    nodes carry ``file_hash``, so a file that mints none is re-parsed on every pass
    forever (ADR-0049 §3).

    The name comes from the **directory**, not from ``masterLabel`` (absent on real
    bundles) and not from the filename: the directory is what every referencing
    surface spells.  A template writes ``<c-error-panel>``, a ``.js`` writes
    ``c/errorPanel``, an Aura file ``<c:errorPanel>``, a flow ``c:errorPanel`` and a
    FlexiPage ``errorPanel``; all five must land here.
    """
    bundle = _api_name(PurePosixPath(path).parent.name)
    if bundle is None:
        return None

    module_uid = _module(emit, path, _LWC_MODULE_KIND, element)
    targets_element = _child(element, "targets")
    targets = _texts_of(targets_element, "target") if targets_element is not None else []

    line_start, line_end = _lines(element)
    bundle_uid = emit.add(
        name=bundle,
        qn_suffix=f"{LWC_NAMESPACE}.{bundle}",
        label=NodeLabel.TYPE_DEF,
        kind=_LWC_KIND,
        line_start=line_start,
        line_end=line_end,
        docstring=_prose(element),
        extra=_compact(
            {
                "master_label": _text_of(element, "masterLabel"),
                "is_exposed": _bool_of(element, "isExposed"),
                "api_version": _text_of(element, "apiVersion"),
                # Opaque strings, deliberately: 21 distinct values across 406 real
                # files, including `lightning_VoiceExtension` with a single
                # underscore where every sibling has two. An enum would be wrong
                # within a release.
                "targets": targets or None,
            }
        ),
    )
    emit.rel(module_uid, RelType.DEFINES, bundle_uid)

    for api in sorted(_lwc_sobjects(element)):
        emit.imports_sobject(bundle_uid, api)
    for apex_class in sorted(_lwc_apex_classes(element)):
        emit.rel(bundle_uid, RelType.IMPORTS, f"{APEX_NAMESPACE}.{apex_class}")

    return emit.result()


def _lwc_sobjects(element: Node) -> set[str]:
    """SObjects the bundle is scoped to, from ``objects`` and schema-typed properties.

    ``<objects><object>`` sits two levels inside ``targetConfigs/targetConfig``, so
    this descends rather than reading direct children.
    """
    found: set[str] = set()
    for descendant in _iter_elements(element):
        if xml_tag(descendant) == "objects":
            for value in _texts_of(descendant, "object"):
                name = _api_name(value)
                if name is not None:
                    found.add(name)
        declared = _xml_attributes(descendant).get("type", "")
        if declared.startswith(_SCHEMA_TYPE_PREFIX):
            reference = declared.removeprefix(_SCHEMA_TYPE_PREFIX).strip("/")
            name = _api_name(reference.split(".")[0])
            if name is not None:
                found.add(name)
    return found


def _lwc_apex_classes(element: Node) -> set[str]:
    """Apex classes backing a design-time picklist — ``datasource="apex://<Class>"``.

    An **attribute**, not element text, which is why this cannot go through the
    ``_texts_of`` helpers the rest of the module uses.
    """
    found: set[str] = set()
    for descendant in _iter_elements(element):
        datasource = _xml_attributes(descendant).get("datasource", "")
        if datasource.startswith(_APEX_DATASOURCE_PREFIX):
            name = _api_name(datasource.removeprefix(_APEX_DATASOURCE_PREFIX).strip())
            if name is not None:
                found.add(name)
    return found


# ---------------------------------------------------------------------------
# Decomposed children of CustomObject
#
# SFDX splits an object into a directory tree, so each of these files states one
# component of an object named nowhere in the document — only in the path. They
# had no handler and fell to the generic structural parse, which walks the whole
# body: measured across seven public repos, 1,271 such files minted 20,819 nodes
# against 10,165 SFDX files minting 20,330. 11% of the files, 51% of the nodes,
# and every one contentless (`picklistValues#17`, no name, no source, no
# docstring). Handling them is node-count *negative*.
# ---------------------------------------------------------------------------

_LABEL_REFERENCE_RE = re.compile(r"\$Label\.([A-Za-z0-9_]+)")


def _formula_labels(formula: str | None) -> set[str]:
    """Custom labels a formula references as ``$Label.X``.

    The only thing mined out of formula text.  Bare identifiers in a formula are
    *not*: a field, a function name, a global and a cross-object path all look
    alike, and a wrong uid is worse than a missing entity (ADR-0032).
    ``$Label.`` is unambiguous — nothing else is spelled that way.
    """
    if not formula:
        return set()
    return {match.group(1) for match in _LABEL_REFERENCE_RE.finditer(formula)}


def _decomposed_component(
    emit: _Emit,
    element: Node,
    path: str,
    meta: _MetaFile,
    *,
    child_dir: str,
    namespace: str,
    module_kind: str,
    kind: str,
    source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[str | None, str] | None:
    """Mint the file's Module and its one component, or ``None`` to decline.

    The owner comes from the path — ``objects/<Object>/<child_dir>/<Name>...`` —
    because the document never names it.  The ``IMPORTS -> sobject.<Owner>`` is
    emitted from the component rather than from the object, so the edge lives in
    the file that states it: ``_recreate_file_relationships`` deletes edges by
    their *source* node's ``file_path``, and an edge sourced at the object would
    be wiped whenever the object file alone was re-parsed.
    """
    owner = _decomposed_owner(path, child_dir)
    name = _api_name(meta.base)
    if owner is None or name is None:
        return None
    module_uid = _module(emit, path, module_kind, element)
    line_start, line_end = _lines(element)
    uid = emit.add(
        name=name,
        qn_suffix=f"{namespace}.{owner}.{name}",
        label=NodeLabel.VALUE,
        kind=kind,
        line_start=line_start,
        line_end=line_end,
        docstring=_prose(element),
        source=source,
        extra=_compact({"sobject": owner, "active": _bool_of(element, "active"), **(extra or {})}),
    )
    emit.rel(module_uid, RelType.DEFINES, uid)
    emit.imports_sobject(uid, owner)
    return uid, owner


def _owned_field(emit: _Emit, uid: str | None, owner: str, value: str | None) -> None:
    """``IMPORTS -> sobject.<Owner>.<Field>`` for a field named on the owning object."""
    field = _api_name(value)
    if field is not None:
        emit.rel(uid, RelType.IMPORTS, f"{SOBJECT_NAMESPACE}.{owner}.{field}")


def _parse_validation_rule(emit: _Emit, element: Node, path: str, meta: _MetaFile) -> ParsedFile | None:
    """``objects/<Obj>/validationRules/<Name>.validationRule-meta.xml``.

    ``errorMessage`` is the reason this type is worth indexing at all: it is the
    admin's own explanation of a business rule, written for the user who tripped
    it, and it was the most searchable prose in the object tree with no node to
    hang on.  It reaches the docstring through ``_PROSE_TAGS``.

    Only ``errorDisplayField`` becomes a field edge.  It is a plain API name, so
    it costs no parsing and cannot be wrong; the formula beside it is mined for
    ``$Label.X`` and nothing else.
    """
    formula = _text_of(element, "errorConditionFormula")
    minted = _decomposed_component(
        emit,
        element,
        path,
        meta,
        child_dir="validationRules",
        namespace=VALIDATION_RULE_NAMESPACE,
        module_kind=_VALIDATION_RULE_MODULE_KIND,
        kind=_VALIDATION_RULE_KIND,
        source=formula,
        extra={"error_display_field": _text_of(element, "errorDisplayField")},
    )
    if minted is None:
        return None
    uid, owner = minted
    _owned_field(emit, uid, owner, _text_of(element, "errorDisplayField"))
    for label in sorted(_formula_labels(formula)):
        emit.rel(uid, RelType.IMPORTS, f"{LABEL_NAMESPACE}.{label}")
    return emit.result()


def _parse_record_type(emit: _Emit, element: Node, path: str, meta: _MetaFile) -> ParsedFile | None:
    """``objects/<Obj>/recordTypes/<Name>.recordType-meta.xml``.

    ``picklistValues`` is deliberately not walked.  A single real record type file
    carries 266 of them across eight files — a 33x node multiplier on a type that
    looks harmless — and a picklist value answers no question anybody asks of a
    code graph.
    """
    minted = _decomposed_component(
        emit,
        element,
        path,
        meta,
        child_dir="recordTypes",
        namespace=RECORD_TYPE_NAMESPACE,
        module_kind=_RECORD_TYPE_MODULE_KIND,
        kind=_RECORD_TYPE_KIND,
        extra={"compact_layout": _text_of(element, "compactLayoutAssignment")},
    )
    return emit.result() if minted is not None else None


def _parse_field_set(emit: _Emit, element: Node, path: str, meta: _MetaFile) -> ParsedFile | None:
    """``objects/<Obj>/fieldSets/<Name>.fieldSet-meta.xml``.

    A field set is what an LWC or Aura component iterates over, so this is the
    answer to "which UI reads this field".

    ``displayedFields`` only.  Its sibling ``availableFields`` is far more numerous
    — 53 against 7 in a real file — but means "an admin could add this", not "this
    is shown", and edges for what *might* be read would drown the ones for what is.
    """
    minted = _decomposed_component(
        emit,
        element,
        path,
        meta,
        child_dir="fieldSets",
        namespace=FIELD_SET_NAMESPACE,
        module_kind=_FIELD_SET_MODULE_KIND,
        kind=_FIELD_SET_KIND,
    )
    if minted is None:
        return None
    uid, owner = minted
    for displayed in _children(element, "displayedFields"):
        for value in _texts_of(displayed, "field"):
            _owned_field(emit, uid, owner, value)
    return emit.result()


def _parse_web_link(emit: _Emit, element: Node, path: str, meta: _MetaFile) -> ParsedFile | None:
    """``objects/<Obj>/webLinks/<Name>.webLink-meta.xml``.

    A button on a record page.  When its ``linkType`` is ``page`` the ``<page>``
    element names a Visualforce page by API name, which is a real cross-component
    edge; every other ``linkType`` (``url``, ``javascript``) names something this
    module does not model and is skipped rather than guessed at.
    """
    minted = _decomposed_component(
        emit,
        element,
        path,
        meta,
        child_dir="webLinks",
        namespace=WEB_LINK_NAMESPACE,
        module_kind=_WEB_LINK_MODULE_KIND,
        kind=_WEB_LINK_KIND,
        extra={
            "link_type": _text_of(element, "linkType"),
            "display_type": _text_of(element, "displayType"),
        },
    )
    if minted is None:
        return None
    uid, _owner = minted
    if (_text_of(element, "linkType") or "").strip().lower() == "page":
        page = _api_name(_text_of(element, "page"))
        if page is not None:
            emit.rel(uid, RelType.IMPORTS, f"{PAGE_NAMESPACE}.{page}")
    return emit.result()


def _parse_list_view(emit: _Emit, element: Node, path: str, meta: _MetaFile) -> ParsedFile | None:
    """``objects/<Obj>/listViews/<Name>.listView-meta.xml``.

    ``columns`` mixes real field API names with legacy report tokens —
    ``NAME``, ``RECORDTYPE``, ``CORE.USERS.ALIAS`` — which name no field and would
    each mint an ``ext/`` stub for a thing that does not exist.  Only ``__c``
    columns become edges: 143 of 225 in the sampled corpus, and the 82 skipped are
    exactly those tokens.  Standard fields are lost with them, which is the price
    of not inventing 82 targets.
    """
    minted = _decomposed_component(
        emit,
        element,
        path,
        meta,
        child_dir="listViews",
        namespace=LIST_VIEW_NAMESPACE,
        module_kind=_LIST_VIEW_MODULE_KIND,
        kind=_LIST_VIEW_KIND,
        extra={"filter_scope": _text_of(element, "filterScope")},
    )
    if minted is None:
        return None
    uid, owner = minted
    for column in _texts_of(element, "columns"):
        if column.endswith("__c"):
            _owned_field(emit, uid, owner, column)
    return emit.result()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, Callable[[_Emit, Node, str, _MetaFile], ParsedFile | None]] = {
    "CustomObject": _parse_object,
    "CustomField": _parse_field,
    "Flow": _parse_flow,
    "CustomLabels": _parse_labels,
    "CustomMetadata": _parse_custom_metadata,
    "LightningComponentBundle": _parse_lwc_bundle,
    "ValidationRule": _parse_validation_rule,
    "RecordType": _parse_record_type,
    "FieldSet": _parse_field_set,
    "WebLink": _parse_web_link,
    "ListView": _parse_list_view,
}
"""Root element name -> handler.  This *is* the supported-type list.

Every other Salesforce root element — ``PermissionSet``, ``Layout``,
``FlexiPage``, ``ApexClass`` on a ``.cls-meta.xml`` sidecar — falls through to
``config.py``'s generic structural parse, which is the pre-existing behaviour
for all of them.
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


_SNIFF_NAMESPACE_MARKER = b"soap.sforce.com"
_SNIFF_ROOT_RE = re.compile(rb"<\s*([A-Za-z_][\w.-]*)")


def looks_like_salesforce_metadata(head: bytes) -> bool:
    """Does this document declare itself as SFDX metadata this module models?

    The dialect sniff (ADR-0048). It sees the first 4 KiB and **never the path** —
    the signature is ``Callable[[bytes], bool]`` — so it can use only what the
    document says about itself: a root element in :data:`_HANDLERS`, corroborated
    by the Metadata API namespace.

    ``soap.sforce.com`` as a **substring**, not the exact ``_METADATA_NS``. A real
    ``.cls-meta.xml`` sidecar declares ``urn:metadata.tooling.soap.sforce.com``,
    which exact equality misses; today such a file is rescued by
    :func:`_looks_like_sfdx`'s filename branch, which a sniff cannot reach.

    Claiming is a commitment: a claimed file that the handler declines gets an
    **empty** ``ParsedFile``, which would delete its entities from the graph. That
    is why the caller pairs this with the generic structural parse rather than
    letting a decline stand — see ``config._parse_salesforce_xml``.

    4 KiB is ample: every SFDX file puts its XML declaration on line 1 and the root
    element on line 2, at byte 41 in every sample measured.
    """
    if _SNIFF_NAMESPACE_MARKER not in head:
        return False
    for match in _SNIFF_ROOT_RE.finditer(head):
        name = match.group(1).decode("ascii", "replace")
        if name.startswith(("?", "!")):
            continue
        return name in _HANDLERS
    return False


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


_BUDGETS: dict[str, int] = {"CustomLabels": _LABELS_MAX_ENTITIES}
"""Per-root-element node budget, for the one type whose default is wrong.

Keyed by root element rather than raised inside the handler, because ``_Emit`` is
built before dispatch and a handler that raised its own budget mid-parse would
have to re-admit entities it had already dropped.
"""


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

    emit = _Emit(file_path=path, project_name=project_name, budget=_BUDGETS.get(tag, _MAX_ENTITIES_PER_FILE))
    return _HANDLERS[tag](emit, element, path, meta)
