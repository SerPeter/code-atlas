"""Salesforce Apex language support — the Java grammar behind a length-preserving shim.

There is no tree-sitter Apex grammar on PyPI (upstream ``aheber/tree-sitter-sfapex``
ships node/rust/web bindings only), so rather than fall back to line-based regex
like other tools do, this module reuses the ``tree-sitter-java`` grammar this
project already depends on.

Apex is close enough to Java that a small set of *length-preserving* source
rewrites brings a real ``.cls`` file to a clean Java parse.  Every rewrite
overwrites bytes in place with an equal-length replacement (padding with spaces,
never touching newlines), so tree-sitter's byte offsets and line numbers on the
shimmed text are identical to the original file.  That is what lets the entity
walk run on the shimmed tree while docstrings, signatures and source text are
sliced out of the *original* bytes.

The rewrites, in application order (see :func:`_shim`):

1. ``=>`` (Apex map literals) -> ``,``.
2. ``Trigger.new`` -> ``TriggerNew`` — ``new`` is a reserved word in Java.
3. A trigger header + its opening brace -> a synthetic ``class N { void N__body ( ) {``
   wrapper.  A trigger body is a bare statement list, and Java rejects statements
   directly in a class body, so the header alone is not enough; the wrapper costs
   one extra closing brace, appended after the file's last ``}``.
4. Apex-only modifiers (``global``, ``webservice``, ``testmethod``, ``virtual``,
   ``override``, ``(with|without|inherited) sharing``) -> blanked, recaptured as tags.
5. Collection literals ``new Map<Id, Account>{...}`` -> ``new Map<Id, Account>(...)``.
6. Inline SOQL/SOSL ``[SELECT ...]`` / ``[FIND ...]`` -> ``null`` plus padding.
   Pure whitespace is wrong here: it leaves ``x = ;``, which does not parse.
7. DML statements (``insert acc;``) -> blanked, recaptured as SObject references.
8. Apex properties ``String Name { get; set; }`` -> ``String Name;``, relabelled
   back to a ``Callable`` of kind ``property`` afterwards.

Entities: classes/interfaces/enums -> ``TypeDef``; methods/constructors ->
``Callable``; properties -> ``Callable`` (kind ``property``); fields ->
``Value``; triggers -> ``Callable`` (kind ``trigger``).

SObjects (``Account``, ``Custom_Object__c``) are referenced, never defined, by
the code being indexed, so they are modelled the way every other referenced-but-
absent symbol is: as ``IMPORTS`` relationships to ``ext/sobject.<Name>``, which
``GraphClient.resolve_imports`` materialises as ExternalPackage/ExternalSymbol
stubs.  The LWC side (``typescript.py``) emits the *same* ``sobject.<Name>``
target, so an Apex method and a Lightning component that touch the same object
meet on one shared node.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from loguru import logger
from tree_sitter import Language, Parser, Query

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    node_text,
    register_language,
)
from code_atlas.parsing.languages.jvm import (
    _extract_calls,
    _walk_java_node,
)
from code_atlas.schema import CallableKind, NodeLabel, RelType, Visibility

if TYPE_CHECKING:
    from tree_sitter import Node

# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

APEX_NAMESPACE = "apex"
"""Root qualified-name segment for Apex *classes* and their members.

Apex has no packages and no import statements: every class in an org is
addressable by bare name, and Salesforce itself addresses class members as
``@salesforce/apex/ClassName.methodName``.  Deriving qualified names from the
SFDX directory layout instead (``force-app.main.default.classes.Foo``) would make
those LWC import specifiers structurally unable to match anything in the graph —
the same reasoning behind ``jvm.py``'s ``_strip_java_source_root``.  Storing
``apex.ClassName.methodName`` lets ``GraphClient.resolve_imports`` classify an
``@salesforce/apex/...`` import as *internal* and wire the LWC module straight to
the real Apex ``Callable``.

Triggers deliberately keep a path-derived qualified name: they are not
addressable from anywhere (no Apex, LWC or REST caller can name one), and a
trigger may legally share its name with a class in the same org, which would
collide here.
"""

SOBJECT_NAMESPACE = "sobject"
"""Import-target prefix for SObject references, shared with ``typescript.py``.

Produces ``ext/sobject.Account`` — one ExternalPackage (``ext/sobject``) holding
one ExternalSymbol per referenced object, named after the object.
"""

_TRIGGER_BODY_SUFFIX = "__body"
# Fixed-length synthetic wrapper method name — see _TriggerFacts.body_name for why
# this must NOT embed the trigger's name.
_TRIGGER_BODY_NAME = "__b"

# ---------------------------------------------------------------------------
# Byte-level regexes.  Bytes, not str: the shim must preserve *byte* offsets, and
# blanking a multi-byte character with a single space would shift every offset
# after it.  Byte patterns also keep ``\w``/``\b`` ASCII-only, which is what Apex
# identifiers are.
# ---------------------------------------------------------------------------

_MAP_ARROW = re.compile(rb"=>")
_TRIGGER_NEW = re.compile(rb"\bTrigger\.new\b", re.IGNORECASE)
_TRIGGER_HEADER = re.compile(
    rb"\btrigger[ \t\r\n]+(\w+)[ \t\r\n]+on[ \t\r\n]+(\w+)[ \t\r\n]*\(([^)]*)\)[ \t\r\n]*\{",
    re.IGNORECASE,
)
_APEX_MODIFIER = re.compile(
    rb"\b(?:global|webservice|testmethod|virtual|override|(?:with|without|inherited)[ \t\r\n]+sharing)\b",
    re.IGNORECASE,
)
_COLLECTION_LITERAL = re.compile(rb"\bnew[ \t\r\n]+[\w.]+[ \t\r\n]*<[^;{}]*>[ \t\r\n]*\{")
_SOQL = re.compile(rb"\[[ \t\r\n]*(?:SELECT|FIND)\b.*?\]", re.IGNORECASE | re.DOTALL)
# Every FROM in the literal, including a subquery's: `(SELECT Id FROM Contacts)`
# names a child *relationship*, not an object, so it yields `sobject.Contacts`
# rather than `sobject.Contact`. Left as-is deliberately — mapping relationship
# names back to objects needs org metadata this parser does not have, and the
# relationship name is still the truest thing the source says.
_SOQL_FROM = re.compile(rb"\bFROM[ \t\r\n]+([A-Za-z_]\w*)", re.IGNORECASE)
_SOSL_RETURNING = re.compile(rb"\bRETURNING\b(.*)", re.IGNORECASE | re.DOTALL)
_SOSL_OBJECT = re.compile(rb"([A-Za-z_]\w*)[ \t\r\n]*\(")
# Statement-anchored: a DML keyword only counts at the start of a line or right
# after a `;`/`{`/`}`.  Without the anchor the pattern eats prose in comments
# ("// insert the record;") and the inside of string literals, which breaks the
# parse it is meant to protect.  `Database.insert(x)` is excluded twice over — by
# the anchor and by the required whitespace after the keyword.
_DML = re.compile(
    rb"(?:(?<=[;{}])|(?m:^))[ \t]*(insert|update|upsert|delete|undelete|merge)\b[ \t]+([^;{}]*);",
    re.IGNORECASE,
)
_DML_NEW = re.compile(rb"\bnew[ \t\r\n]+([A-Za-z_]\w*)")
_DML_IDENT = re.compile(rb"([A-Za-z_]\w*)")
# `<type tokens> <name> {` — two-plus whitespace-separated tokens before a brace.
# Matches plenty of non-properties (`public class Foo {`); the accessor gate in
# _rewrite_properties is the real discriminator.
_PROPERTY_CANDIDATE = re.compile(rb"(?m)^[ \t]*(?:[\w<>,\[\]\.]+[ \t]+)+(\w+)[ \t]*\{")
_ACCESSOR_HEAD = re.compile(
    # A real accessor is `get;` / `set;` / `get {` / `set {`. The terminator is
    # load-bearing, not decoration: `\b` alone matches the `Set` in `Set<Id> ids`
    # (IGNORECASE makes `Set` match `set`, and `<` is a word boundary), so
    # `_PROPERTY_CANDIDATE` firing on `public class P {` — which it does by
    # design, the comment above says the gate is the discriminator — then had
    # the whole class body rewritten as an accessor block. Measured: a single
    # `Set<Id>` field erased every entity in the file.
    rb"^[ \t\r\n]*(?:(?:global|public|private|protected)[ \t\r\n]+)?(?:get|set)[ \t\r\n]*[;{]",
    re.IGNORECASE,
)
_COMMENT = re.compile(rb"//[^\n]*|/\*.*?\*/", re.DOTALL)

_NEWLINES = (0x0A, 0x0D)
_SPACE = 0x20
_QUOTE = 0x27
_SLASH = 0x2F
_STAR = 0x2A
_BACKSLASH = 0x5C
_OPEN_BRACE = 0x7B
_CLOSE_BRACE = 0x7D
_OPEN_BRACKET = 0x5B
_CLOSE_BRACKET = 0x5D

# Apex/Java built-ins and collection types that are never SObjects.
_NON_SOBJECT_TYPES: frozenset[str] = frozenset(
    {
        "blob",
        "boolean",
        "date",
        "datetime",
        "decimal",
        "double",
        "id",
        "integer",
        "list",
        "long",
        "map",
        "object",
        "set",
        "sobject",
        "string",
        "time",
        "void",
    }
)

# Apex annotations whose spelling varies by author (`@isTest` / `@IsTest`).
# Canonicalised so downstream tag matching is stable.
_ANNOTATION_CANONICAL: dict[str, str] = {
    "auraenabled": "AuraEnabled",
    "future": "future",
    "invocablemethod": "InvocableMethod",
    "invocablevariable": "InvocableVariable",
    "istest": "isTest",
    "remoteaction": "RemoteAction",
    "testsetup": "TestSetup",
    "testvisible": "TestVisible",
}

# ---------------------------------------------------------------------------
# Shim primitives
# ---------------------------------------------------------------------------


def _blank(buf: bytearray, start: int, end: int) -> None:
    """Overwrite ``buf[start:end]`` with spaces, leaving line breaks intact."""
    for i in range(start, end):
        if buf[i] not in _NEWLINES:
            buf[i] = _SPACE


def _overwrite(buf: bytearray, start: int, end: int, repl: bytes) -> bool:
    """Blank ``buf[start:end]`` and drop *repl* into a line-break-free slot inside it.

    Returns ``False`` (span merely blanked) when *repl* does not fit, or when
    every candidate slot straddles a line break — writing over a newline would
    shift every subsequent line number, which is exactly what this shim exists
    to avoid.
    """
    n = len(repl)
    if n > end - start:
        _blank(buf, start, end)
        return False
    slot = -1
    for i in range(start, end - n + 1):
        if not any(b in _NEWLINES for b in buf[i : i + n]):
            slot = i
            break
    _blank(buf, start, end)
    if slot < 0:
        return False
    buf[slot : slot + n] = repl
    return True


def _match_brace(buf: bytes, open_idx: int) -> int:
    """Index of the ``}`` closing the ``{`` at *open_idx*, or -1.

    Skips Apex string literals and comments so a brace inside either does not
    unbalance the count.  Returns -1 unless *open_idx* really is a ``{``: callers
    scan match positions captured before earlier rewrites ran, and matching from
    a byte an earlier rewrite already blanked would silently latch onto some
    unrelated block further down the file.
    """
    if open_idx >= len(buf) or buf[open_idx] != _OPEN_BRACE:
        return -1
    depth = 0
    i = open_idx
    n = len(buf)
    while i < n:
        char = buf[i]
        if char == _QUOTE:
            i += 1
            while i < n and buf[i] != _QUOTE:
                i += 2 if buf[i] == _BACKSLASH else 1
        elif char == _SLASH and i + 1 < n and buf[i + 1] == _SLASH:
            while i < n and buf[i] != 0x0A:
                i += 1
        elif char == _SLASH and i + 1 < n and buf[i + 1] == _STAR:
            close = buf.find(b"*/", i + 2)
            i = n if close < 0 else close + 1
        elif char == _OPEN_BRACE:
            depth += 1
        elif char == _CLOSE_BRACE:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _line_of(source: bytes, offset: int) -> int:
    """1-based line number of *offset* in *source*."""
    return source.count(b"\n", 0, offset) + 1


def _decode(chunk: bytes) -> str:
    return chunk.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Shim facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TriggerFact:
    """The header of an Apex trigger, recovered before the header is rewritten."""

    name: str
    sobject: str
    events: list[str]
    header: str
    line: int

    @property
    def body_name(self) -> str:
        """Name of the synthetic wrapper method the trigger body is parsed inside.

        A FIXED name, deliberately. It used to be ``f"{name}__body"``, which put the
        trigger's name in the wrapper *twice* while the real header contains it once —
        so the wrapper grew at 2x the header's rate and overflowed the span it has to
        fit inside. It failed on the canonical Salesforce shape: a long ``<Object>Trigger``
        name with a short SObject and a short event list. Measured, header vs wrapper:
        ``AccountTrigger`` 51 vs 54 (did NOT fit), so every in-body CALLS edge was lost.
        With a fixed name the wrapper grows at 1x and the budget is always positive —
        the shortest legal trigger header still leaves room.

        There is no collision risk: a trigger has no methods of its own, so this is the
        only method in the synthetic class.
        """
        return _TRIGGER_BODY_NAME


@dataclass
class _ShimFacts:
    """Everything the regex pass recovered before handing the residue to Java."""

    trigger: _TriggerFact | None = None
    modifiers: list[tuple[int, str]] = field(default_factory=list)
    """(line, Apex-only modifier keyword) — blanked out, re-attached as tags."""
    properties: list[tuple[int, str]] = field(default_factory=list)
    """(line, property name) — rewritten to fields, relabelled back to Callables."""
    sobjects: list[tuple[int, str]] = field(default_factory=list)
    """(line, SObject API name) — resolved directly from SOQL/SOSL or ``new X(...)`` DML."""
    dml: list[tuple[int, str]] = field(default_factory=list)
    """(line, variable name) — DML on a variable, resolved later via declared types."""


# ---------------------------------------------------------------------------
# Individual rewrites
# ---------------------------------------------------------------------------


def _rewrite_trigger(buf: bytearray, source: bytes, facts: _ShimFacts) -> bool:
    """Rewrite a trigger header into a synthetic class + method wrapper.

    Returns ``True`` when the caller must append a balancing ``}``.
    """
    match = _TRIGGER_HEADER.search(bytes(buf))
    if match is None:
        return False
    name = _decode(match.group(1))
    events = [" ".join(part.split()) for part in _decode(match.group(3)).split(",")]
    facts.trigger = _TriggerFact(
        name=name,
        sobject=_decode(match.group(2)),
        events=[event for event in events if event],
        header=_decode(source[match.start() : match.end() - 1]).strip(),
        line=_line_of(source, match.start()),
    )
    wrapper = f"class {name} {{ void {_TRIGGER_BODY_NAME} ( ) {{".encode()
    if not _overwrite(buf, match.start(), match.end(), wrapper):
        logger.warning("apex: trigger header for {} does not fit a single line; body will not parse", name)
        return False
    return True


def _rewrite_modifiers(buf: bytearray, source: bytes, facts: _ShimFacts) -> None:
    for match in list(_APEX_MODIFIER.finditer(bytes(buf))):
        keyword = " ".join(_decode(match.group(0)).lower().split())
        facts.modifiers.append((_line_of(source, match.start()), keyword))
        _blank(buf, match.start(), match.end())


def _rewrite_collection_literals(buf: bytearray) -> None:
    """``new Map<Id, Account>{...}`` -> ``new Map<Id, Account>(...)``.

    Apex collection literals are a brace-delimited argument list; Java only
    accepts parentheses there.  Swapping the two delimiters in place is
    length-preserving, and the ``=>`` inside map literals has already been
    turned into ``,`` by the time this runs.
    """
    for match in list(_COLLECTION_LITERAL.finditer(bytes(buf))):
        open_idx = match.end() - 1
        close_idx = _match_brace(bytes(buf), open_idx)
        if close_idx < 0:
            continue
        buf[open_idx] = 0x28  # (
        buf[close_idx] = 0x29  # )


def _soql_spans(buf: bytes) -> list[tuple[int, int]]:
    """``[SELECT ...]`` / ``[FIND ...]`` spans, skipping string literals and comments.

    A regex cannot do this. ``_SOQL`` was ``\\[...(?:SELECT|FIND)\\b.*?\\]`` under
    DOTALL, so a string containing ``'[SELECT oops'`` with no ``]`` of its own
    matched onward to the next ``]`` ANYWHERE in the file — measured erasing a
    whole method whose body happened to contain ``v[0]``. Scanning with the same
    quote/comment skipping ``_match_brace`` uses is the only correct fix.
    """
    spans: list[tuple[int, int]] = []
    i, n = 0, len(buf)
    while i < n:
        char = buf[i]
        if char == _QUOTE:
            i += 1
            while i < n and buf[i] != _QUOTE:
                i += 2 if buf[i] == _BACKSLASH else 1
        elif char == _SLASH and i + 1 < n and buf[i + 1] == _SLASH:
            while i < n and buf[i] != 0x0A:
                i += 1
        elif char == _SLASH and i + 1 < n and buf[i + 1] == _STAR:
            close = buf.find(b"*/", i + 2)
            i = n if close < 0 else close + 1
        elif char == _OPEN_BRACKET:
            head = i + 1
            while head < n and buf[head] in (_SPACE, 0x09, 0x0A, 0x0D):
                head += 1
            if buf[head : head + 6].upper() == b"SELECT" or buf[head : head + 4].upper() == b"FIND":
                end = _match_bracket(buf, i)
                if end > 0:
                    spans.append((i, end + 1))
                    i = end
        i += 1
    return spans


def _match_bracket(buf: bytes, open_idx: int) -> int:
    """Index of the ``]`` closing the ``[`` at *open_idx*, or -1. Quote/comment aware."""
    depth = 0
    i, n = open_idx, len(buf)
    while i < n:
        char = buf[i]
        if char == _QUOTE:
            i += 1
            while i < n and buf[i] != _QUOTE:
                i += 2 if buf[i] == _BACKSLASH else 1
        elif char == _OPEN_BRACKET:
            depth += 1
        elif char == _CLOSE_BRACKET:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _rewrite_soql(buf: bytearray, source: bytes, facts: _ShimFacts) -> None:
    for start, end in _soql_spans(bytes(buf)):
        literal = bytes(buf)[start:end]
        line = _line_of(source, start)
        for from_match in _SOQL_FROM.finditer(literal):
            facts.sobjects.append((line, _decode(from_match.group(1))))
        returning = _SOSL_RETURNING.search(literal)
        if returning is not None:
            for obj in _SOSL_OBJECT.finditer(returning.group(1)):
                facts.sobjects.append((line, _decode(obj.group(1))))
        _overwrite(buf, start, end, b"null")


def _rewrite_dml(buf: bytearray, source: bytes, facts: _ShimFacts) -> None:
    for match in list(_DML.finditer(bytes(buf))):
        expression = match.group(2)
        line = _line_of(source, match.start())
        constructed = _DML_NEW.search(expression)
        if constructed is not None:
            facts.sobjects.append((line, _decode(constructed.group(1))))
        else:
            identifier = _DML_IDENT.search(expression)
            if identifier is not None:
                facts.dml.append((line, _decode(identifier.group(1))))
        _blank(buf, match.start(), match.end())


def _rewrite_properties(buf: bytearray, source: bytes, facts: _ShimFacts) -> None:
    """``String Name { get; set; }`` -> ``String Name;`` (a Java field declaration).

    The resulting ``Value`` entity is relabelled to a ``Callable`` of kind
    ``property`` in :func:`_apply_property_kind`; going through a field keeps the
    type/visibility/annotation extraction that ``jvm.py`` already does.
    """
    for match in _PROPERTY_CANDIDATE.finditer(bytes(buf)):
        open_idx = match.end() - 1
        close_idx = _match_brace(bytes(buf), open_idx)
        if close_idx < 0:
            continue
        if _ACCESSOR_HEAD.match(bytes(buf[open_idx + 1 : close_idx])) is None:
            continue
        facts.properties.append((_line_of(source, match.start()), _decode(match.group(1))))
        _overwrite(buf, open_idx, close_idx + 1, b";")


def _shim(source: bytes, *, allow_trigger: bool) -> tuple[bytes, _ShimFacts]:
    """Rewrite Apex-only syntax out of *source*, returning parseable bytes plus facts.

    Byte offsets and line numbers are identical to *source*, with one exception:
    a trigger costs one appended ``}`` after the file's final brace, which shifts
    nothing before it.

    *allow_trigger* is driven by the file extension rather than by the header
    regex alone — only ``.trigger`` files may hold a trigger, and a commented-out
    header inside a ``.cls`` must not send the whole file down the trigger path.
    """
    facts = _ShimFacts()
    buf = bytearray(_MAP_ARROW.sub(b", ", source))
    for match in list(_TRIGGER_NEW.finditer(bytes(buf))):
        buf[match.start() : match.end()] = b"TriggerNew "

    needs_brace = _rewrite_trigger(buf, source, facts) if allow_trigger else False
    _rewrite_modifiers(buf, source, facts)
    _rewrite_collection_literals(buf)
    _rewrite_soql(buf, source, facts)
    _rewrite_dml(buf, source, facts)
    _rewrite_properties(buf, source, facts)

    if needs_brace:
        last = buf.rfind(b"}")
        if last >= 0:
            buf[last + 1 : last + 1] = b"}"
    return bytes(buf), facts


# ---------------------------------------------------------------------------
# Post-parse helpers
# ---------------------------------------------------------------------------


def _module_qualified_name(file_path: str) -> str:
    """``force-app/main/default/classes/Foo.cls`` -> ``force-app.main.default.classes.Foo``."""
    parts = list(PurePosixPath(file_path.replace("\\", "/")).parts)
    if parts and "." in parts[-1]:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return ".".join(parts)


def _innermost(entities: list[ParsedEntity], line: int, *, skip: int = 0) -> int | None:
    """Index of the smallest-span entity whose line range contains *line*.

    Mirrors ``ast.extract_rationale``'s attribution rule: a fact inside a method
    belongs to the method, not to the class that encloses it.  *skip* excludes
    leading entities (the module, which spans the whole file).
    """
    best: int | None = None
    for i in range(skip, len(entities)):
        entity = entities[i]
        if entity.line_start <= line <= entity.line_end and (
            best is None
            or (entity.line_end - entity.line_start) < (entities[best].line_end - entities[best].line_start)
        ):
            best = i
    return best


def _collect_declared_types(node: Node, declarations: list[tuple[int, str, str]]) -> None:
    """Collect ``(line, variable name, type name)`` for locals and parameters.

    An Apex DML statement usually names a *variable* (``insert acc;``), so the
    SObject it touches is only recoverable from the variable's declared type.
    """
    if node.type in ("local_variable_declaration", "formal_parameter"):
        type_node = node.child_by_field_name("type")
        if type_node is not None:
            type_name = _last_type_identifier(type_node)
            if type_name is not None:
                declarations.extend(
                    (name_node.start_point[0] + 1, node_text(name_node), type_name)
                    for name_node in _declarator_names(node)
                )
    for child in node.children:
        _collect_declared_types(child, declarations)


def _declarator_names(node: Node) -> list[Node]:
    """Name nodes declared by a local-variable declaration or formal parameter."""
    if node.type == "formal_parameter":
        name_node = node.child_by_field_name("name")
        return [name_node] if name_node is not None else []
    names: list[Node] = []
    for child in node.children:
        if child.type == "variable_declarator":
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                names.append(name_node)
    return names


def _last_type_identifier(node: Node) -> str | None:
    """Innermost element type of a (possibly generic) type node.

    ``Account`` -> ``Account``; ``List<Account>`` -> ``Account``;
    ``Map<Id, Account>`` -> ``Account``.  The last identifier wins because Apex
    collections put the payload type last.
    """
    if node.type == "type_identifier":
        return node_text(node)
    found: str | None = None
    for child in node.children:
        nested = _last_type_identifier(child)
        if nested is not None:
            found = nested
    return found


def _is_sobject_name(name: str) -> bool:
    return bool(name) and name.lower() not in _NON_SOBJECT_TYPES


def _apply_property_kind(entities: list[ParsedEntity], facts: _ShimFacts) -> None:
    """Relabel the ``Value`` produced by a shimmed property back to a ``Callable``.

    The uid is unchanged, so the ``DEFINES`` relationship emitted by ``jvm.py``
    still points at the right node.
    """
    for line, name in facts.properties:
        for i, entity in enumerate(entities):
            if entity.label == NodeLabel.VALUE and entity.name == name and entity.line_start <= line <= entity.line_end:
                entities[i] = replace(entity, label=NodeLabel.CALLABLE, kind=CallableKind.PROPERTY)
                break


def _apply_tags(entities: list[ParsedEntity], facts: _ShimFacts, source_lines: list[str]) -> None:
    """Re-attach blanked Apex modifiers, canonicalise annotations, restore source."""
    extra: dict[int, list[str]] = {}
    for line, keyword in facts.modifiers:
        owner = _innermost(entities, line, skip=1)
        if owner is not None:
            extra.setdefault(owner, []).append(keyword.replace(" ", "_"))

    for i, entity in enumerate(entities):
        tags = [_canonical_tag(tag) for tag in entity.tags]
        tags += [keyword for keyword in extra.get(i, []) if keyword not in tags]
        source = entity.source
        if source is not None:
            source = "\n".join(source_lines[entity.line_start - 1 : entity.line_end])
        if tags != entity.tags or source != entity.source:
            entities[i] = replace(entity, tags=tags, source=source)


def _canonical_tag(tag: str) -> str:
    """Normalise ``annotation:IsTest`` / ``annotation:istest`` to ``annotation:isTest``."""
    if not tag.startswith("annotation:"):
        return tag
    name = tag.removeprefix("annotation:")
    return f"annotation:{_ANNOTATION_CANONICAL.get(name.lower(), name)}"


def _sobject_relationships(
    entities: list[ParsedEntity],
    facts: _ShimFacts,
    declarations: list[tuple[int, str, str]],
    module_uid: str,
) -> list[ParsedRelationship]:
    """IMPORTS edges from the entity owning each SOQL/DML site to ``ext/sobject.<Name>``."""
    references: list[tuple[int, str]] = [(line, name) for line, name in facts.sobjects if _is_sobject_name(name)]
    for line, variable in facts.dml:
        declared = [decl for decl in declarations if decl[1] == variable and decl[0] <= line]
        if declared and _is_sobject_name(declared[-1][2]):
            references.append((line, declared[-1][2]))

    seen: set[tuple[str, str]] = set()
    relationships: list[ParsedRelationship] = []
    for line, name in references:
        owner = _innermost(entities, line, skip=1)
        from_uid = entities[owner].qualified_name if owner is not None else module_uid
        key = (from_uid, name)
        if key in seen:
            continue
        seen.add(key)
        relationships.append(
            ParsedRelationship(
                from_qualified_name=from_uid,
                rel_type=RelType.IMPORTS,
                to_name=f"{SOBJECT_NAMESPACE}.{name}",
            )
        )
    return relationships


def _warn_if_degenerate(path: str, source: bytes, entity_count: int) -> None:
    """Warn when a non-empty Apex file yielded no entities.

    Apex is case-insensitive but Java's grammar is not, so ``PUBLIC CLASS Foo``
    parses to nothing at all.  Failing loudly beats indexing an empty file.
    """
    if entity_count > 0:
        return
    stripped = _COMMENT.sub(b"", source).strip()
    if stripped:
        logger.warning(
            "apex: {} produced no entities from {} bytes of source — "
            "Apex keywords are case-insensitive but the Java grammar behind the shim is not",
            path,
            len(stripped),
        )


# ---------------------------------------------------------------------------
# Trigger extraction
# ---------------------------------------------------------------------------


def _find_trigger_body(root: Node, body_name: str) -> Node | None:
    """Locate the synthetic wrapper method's body in the shimmed tree."""
    for class_node in root.children:
        if class_node.type != "class_declaration":
            continue
        body = class_node.child_by_field_name("body")
        if body is None:
            continue
        for member in body.children:
            if member.type != "method_declaration":
                continue
            name_node = member.child_by_field_name("name")
            if name_node is not None and node_text(name_node) == body_name:
                return member.child_by_field_name("body")
    return None


def _build_trigger_entity(
    trigger: _TriggerFact,
    path: str,
    module_qn: str,
    project_name: str,
    source_lines: list[str],
) -> ParsedEntity:
    tags = ["apex:trigger"] + [f"trigger:{event.replace(' ', '_')}" for event in trigger.events]
    line_end = len(source_lines)
    return ParsedEntity(
        name=trigger.name,
        qualified_name=f"{project_name}:{module_qn}.{trigger.name}",
        label=NodeLabel.CALLABLE,
        kind="trigger",
        line_start=trigger.line,
        line_end=line_end,
        file_path=path,
        signature=trigger.header,
        source="\n".join(source_lines[trigger.line - 1 : line_end]),
        visibility=Visibility.PUBLIC,
        tags=tags,
    )


def _parse_trigger_file(
    path: str,
    source: bytes,
    shimmed_root: Node,
    project_name: str,
    facts: _ShimFacts,
    declarations: list[tuple[int, str, str]],
) -> ParsedFile:
    """Build the entity set for a ``.trigger`` file.

    The synthetic ``class N { void N__body() { ... } }`` wrapper exists only so
    the body parses; neither the class nor the wrapper method becomes an entity.
    """
    trigger = facts.trigger
    if trigger is None:  # pragma: no cover — the caller only routes here when set
        return _parse_class_file(path, source, shimmed_root, project_name, facts, declarations)
    module_qn = _module_qualified_name(path)
    module_uid = f"{project_name}:{module_qn}"
    source_lines = source.decode("utf-8", errors="replace").splitlines()

    entities = [
        ParsedEntity(
            name=module_qn.rsplit(".", 1)[-1],
            qualified_name=module_uid,
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=max(len(source_lines), 1),
            file_path=path,
        ),
        _build_trigger_entity(trigger, path, module_qn, project_name, source_lines),
    ]
    trigger_uid = entities[1].qualified_name

    relationships = [
        ParsedRelationship(from_qualified_name=module_uid, rel_type=RelType.DEFINES, to_name=trigger_uid),
        ParsedRelationship(
            from_qualified_name=trigger_uid,
            rel_type=RelType.IMPORTS,
            to_name=f"{SOBJECT_NAMESPACE}.{trigger.sobject}",
        ),
    ]
    body = _find_trigger_body(shimmed_root, trigger.body_name)
    if body is not None:
        _extract_calls(body, source, trigger_uid, relationships)
    relationships += _sobject_relationships(entities, facts, declarations, module_uid)

    return ParsedFile(file_path=path, language="apex", entities=entities, relationships=relationships)


# ---------------------------------------------------------------------------
# Class extraction
# ---------------------------------------------------------------------------


def _parse_class_file(
    path: str,
    source: bytes,
    shimmed_root: Node,
    project_name: str,
    facts: _ShimFacts,
    declarations: list[tuple[int, str, str]],
) -> ParsedFile:
    """Build the entity set for a ``.cls`` file by delegating to ``jvm.py``'s walk."""
    module_qn = _module_qualified_name(path)
    module_uid = f"{project_name}:{module_qn}"
    source_lines = source.decode("utf-8", errors="replace").splitlines()

    entities: list[ParsedEntity] = [
        ParsedEntity(
            name=module_qn.rsplit(".", 1)[-1],
            qualified_name=module_uid,
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=max(len(source_lines), 1),
            file_path=path,
        )
    ]
    relationships: list[ParsedRelationship] = []

    # `source` is the ORIGINAL bytes, not the shimmed ones: jvm.py slices it by
    # the shimmed tree's byte offsets for docstrings and signatures, and the shim
    # guarantees those offsets still address the same characters.
    _walk_java_node(
        shimmed_root,
        path,
        source,
        project_name,
        APEX_NAMESPACE,
        entities,
        relationships,
        parent_qn=None,
    )

    # Top-level types are DEFINED by the file, not by the synthetic `apex` root.
    namespace_uid = f"{project_name}:{APEX_NAMESPACE}"
    relationships = [
        ParsedRelationship(module_uid, rel.rel_type, rel.to_name, rel.properties)
        if rel.from_qualified_name == namespace_uid
        else rel
        for rel in relationships
    ]

    _apply_property_kind(entities, facts)
    _apply_tags(entities, facts, source_lines)
    relationships += _sobject_relationships(entities, facts, declarations, module_uid)

    _warn_if_degenerate(path, source, len(entities) - 1)
    return ParsedFile(file_path=path, language="apex", entities=entities, relationships=relationships)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_apex(
    path: str,
    source: bytes,
    root: Node,  # noqa: ARG001 — the raw-Apex tree is unusable; we reparse the shim
    project_name: str,
) -> ParsedFile:
    """Shim *source* into parseable Java, reparse it, and extract Apex entities."""
    is_trigger_file = PurePosixPath(path.replace("\\", "/")).suffix.lower() == ".trigger"
    shimmed, facts = _shim(source, allow_trigger=is_trigger_file)
    shimmed_root = Parser(_APEX_LANGUAGE).parse(shimmed).root_node

    declarations: list[tuple[int, str, str]] = []
    _collect_declared_types(shimmed_root, declarations)

    if facts.trigger is not None:
        return _parse_trigger_file(path, source, shimmed_root, project_name, facts, declarations)
    return _parse_class_file(path, source, shimmed_root, project_name, facts, declarations)


# ---------------------------------------------------------------------------
# Language registration — wrapped in try/except for optional grammars
# ---------------------------------------------------------------------------

try:
    import tree_sitter_java as _ts_java

    _APEX_LANGUAGE = Language(_ts_java.language())
    _APEX_QUERY = Query(_APEX_LANGUAGE, "(program) @root")

    register_language(
        LanguageConfig(
            name="apex",
            extensions=frozenset({".cls", ".trigger"}),
            language=_APEX_LANGUAGE,
            query=_APEX_QUERY,
            parse_func=_parse_apex,
        )
    )
except ImportError:
    pass
