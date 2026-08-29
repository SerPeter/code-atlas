"""Tests for the Apex parser (tree-sitter-java behind a length-preserving shim)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

pytest.importorskip("tree_sitter_java", reason="tree-sitter-java not installed")

from loguru import logger
from tree_sitter import Parser

from code_atlas.parsing.ast import ParsedFile, get_language_for_file, parse_file
from code_atlas.parsing.languages.apex import _APEX_LANGUAGE, _shim
from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind, ValueKind

PROJECT = "test_project"

CLASS_PATH = "force-app/main/default/classes/AccountService.cls"
TRIGGER_PATH = "force-app/main/default/triggers/AccountTrigger.trigger"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(source: str, path: str = CLASS_PATH) -> ParsedFile:
    result = parse_file(path, source.encode("utf-8"), PROJECT)
    assert result is not None
    return result


def _entity(parsed: ParsedFile, name: str):
    matches = [e for e in parsed.entities if e.name == name]
    assert len(matches) == 1, f"Expected 1 entity named {name!r}, got {[e.name for e in parsed.entities]}"
    return matches[0]


def _labelled(parsed: ParsedFile, label: NodeLabel):
    return [e for e in parsed.entities if e.label == label]


@contextmanager
def _captured_warnings() -> Generator[list[str]]:
    """Collect loguru WARNING output — the project logs through loguru, not stdlib logging."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _rel_targets(parsed: ParsedFile, rel_type: RelType, from_suffix: str = "") -> set[str]:
    return {
        r.to_name
        for r in parsed.relationships
        if r.rel_type == rel_type and r.from_qualified_name.endswith(from_suffix)
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_cls_routes_to_apex(self):
        config = get_language_for_file(CLASS_PATH)
        assert config is not None
        assert config.name == "apex"

    def test_trigger_routes_to_apex(self):
        config = get_language_for_file(TRIGGER_PATH)
        assert config is not None
        assert config.name == "apex"

    def test_java_still_routes_to_java(self):
        config = get_language_for_file("src/Main.java")
        assert config is not None
        assert config.name == "java"

    def test_language_name_is_apex(self):
        assert _parse("public class Foo {}\n").language == "apex"


# ---------------------------------------------------------------------------
# The shim itself
# ---------------------------------------------------------------------------


class TestShim:
    @pytest.mark.parametrize(
        "source",
        [
            "public with sharing class A {\n  global static void f() {\n    insert new Account();\n  }\n}\n",
            "trigger T on Account (before insert) {\n  update Trigger.new;\n}\n",
            "public class A {\n  public String P { get; private set; }\n}\n",
            "public class A {\n  void f() {\n    Map<Id, String> m = new Map<Id, String>{ 'a' => 'b' };\n  }\n}\n",
        ],
    )
    def test_line_count_is_preserved(self, source: str):
        """Line numbers must survive the shim, or every entity's position is wrong."""
        allow_trigger = source.startswith("trigger")
        shimmed, _ = _shim(source.encode("utf-8"), allow_trigger=allow_trigger)
        assert shimmed.count(b"\n") == source.encode("utf-8").count(b"\n")

    def test_offsets_are_preserved_for_non_ascii_source(self):
        """Blanking must be byte-wise: a multi-byte char blanked to one space shifts everything after it."""
        source = "public class A {\n  // café — note\n  void f() { insert new Account(); }\n}\n".encode()
        shimmed, _ = _shim(source, allow_trigger=False)
        assert len(shimmed) == len(source)

    def test_soql_becomes_null_not_whitespace(self):
        """`x = ;` does not parse — the SOQL literal must leave an expression behind."""
        source = b"public class A {\n  void f() { Account a = [SELECT Id FROM Account]; }\n}\n"
        shimmed, _ = _shim(source, allow_trigger=False)
        assert b"null" in shimmed

    def test_trigger_wrapper_adds_one_closing_brace(self):
        source = b"trigger T on Account (before insert) {\n  Integer i = 1;\n}\n"
        shimmed, _ = _shim(source, allow_trigger=True)
        assert len(shimmed) == len(source) + 1
        assert shimmed.count(b"{") == shimmed.count(b"}")

    def test_trigger_header_ignored_in_cls_files(self):
        """A commented-out trigger header inside a .cls must not hijack the file."""
        source = b"// trigger T on Account (before insert) {\npublic class A {}\n"
        _, facts = _shim(source, allow_trigger=False)
        assert facts.trigger is None


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


class TestApexClass:
    SOURCE = """\
/**
 * Reads accounts.
 */
public with sharing class AccountService implements Queueable {
    public static final Integer LIMIT_SIZE = 10;

    public String Region { get; private set; }

    public AccountService() { }

    @AuraEnabled(cacheable=true)
    global static List<Account> getAccounts(String prefix) {
        return [SELECT Id, Name FROM Account WHERE Name LIKE :prefix LIMIT 10];
    }

    public virtual override void execute(QueueableContext ctx) {
        System.debug('hi');
    }

    webservice static void ping() { }
}
"""

    @pytest.fixture
    def parsed(self) -> ParsedFile:
        return _parse(self.SOURCE)

    def test_module_entity_is_path_derived(self, parsed: ParsedFile):
        modules = _labelled(parsed, NodeLabel.MODULE)
        assert len(modules) == 1
        assert modules[0].qualified_name == f"{PROJECT}:force-app.main.default.classes.AccountService"

    def test_class_uses_the_apex_namespace_not_the_file_path(self, parsed: ParsedFile):
        klass = _labelled(parsed, NodeLabel.TYPE_DEF)[0]
        assert klass.kind == TypeDefKind.CLASS
        assert klass.qualified_name == f"{PROJECT}:apex.AccountService"

    def test_class_docstring_comes_from_original_bytes(self, parsed: ParsedFile):
        klass = _labelled(parsed, NodeLabel.TYPE_DEF)[0]
        assert klass.docstring == "Reads accounts."

    def test_defines_edge_points_at_the_module_not_the_namespace(self, parsed: ParsedFile):
        module_uid = f"{PROJECT}:force-app.main.default.classes.AccountService"
        defines = [r for r in parsed.relationships if r.rel_type == RelType.DEFINES]
        assert f"{PROJECT}:apex.AccountService" in {r.to_name for r in defines}
        assert all(r.from_qualified_name != f"{PROJECT}:apex" for r in defines)
        assert module_uid in {r.from_qualified_name for r in defines}

    def test_implements_is_extracted(self, parsed: ParsedFile):
        assert "Queueable" in _rel_targets(parsed, RelType.IMPLEMENTS)

    def test_sharing_modifier_becomes_a_tag(self, parsed: ParsedFile):
        klass = _labelled(parsed, NodeLabel.TYPE_DEF)[0]
        assert "with_sharing" in klass.tags

    def test_methods_are_callables(self, parsed: ParsedFile):
        method = _entity(parsed, "getAccounts")
        assert method.label == NodeLabel.CALLABLE
        assert method.kind == CallableKind.STATIC_METHOD
        assert method.qualified_name == f"{PROJECT}:apex.AccountService.getAccounts"

    def test_constructor_is_a_callable(self, parsed: ParsedFile):
        ctor = [e for e in parsed.entities if e.kind == CallableKind.CONSTRUCTOR]
        assert [e.name for e in ctor] == ["AccountService"]

    def test_field_is_a_value(self, parsed: ParsedFile):
        field = _entity(parsed, "LIMIT_SIZE")
        assert field.label == NodeLabel.VALUE
        assert field.kind == ValueKind.CONSTANT

    def test_property_is_a_callable(self, parsed: ParsedFile):
        prop = _entity(parsed, "Region")
        assert prop.label == NodeLabel.CALLABLE
        assert prop.kind == CallableKind.PROPERTY

    def test_apex_only_modifiers_attach_to_the_right_entity(self, parsed: ParsedFile):
        assert "global" in _entity(parsed, "getAccounts").tags
        assert "webservice" in _entity(parsed, "ping").tags
        execute = _entity(parsed, "execute")
        assert "virtual" in execute.tags
        assert "override" in execute.tags
        # ...and not onto the class that encloses them
        klass = _labelled(parsed, NodeLabel.TYPE_DEF)[0]
        assert "global" not in klass.tags

    def test_signature_and_source_come_from_the_original_text(self, parsed: ParsedFile):
        method = _entity(parsed, "getAccounts")
        assert method.signature is not None
        assert "@AuraEnabled(cacheable=true)" in method.signature
        assert method.source is not None
        assert "[SELECT Id, Name FROM Account" in method.source

    def test_calls_are_extracted(self, parsed: ParsedFile):
        assert "debug" in _rel_targets(parsed, RelType.CALLS, from_suffix="apex.AccountService.execute")


class TestAnnotations:
    @pytest.mark.parametrize(
        ("written", "canonical"),
        [
            ("@isTest", "annotation:isTest"),
            ("@IsTest", "annotation:isTest"),
            ("@ISTEST", "annotation:isTest"),
            ("@AuraEnabled", "annotation:AuraEnabled"),
            ("@auraenabled", "annotation:AuraEnabled"),
            ("@InvocableMethod", "annotation:InvocableMethod"),
        ],
    )
    def test_annotation_case_is_canonicalised(self, written: str, canonical: str):
        parsed = _parse(f"public class A {{\n    {written}\n    static void f() {{ }}\n}}\n")
        assert canonical in _entity(parsed, "f").tags

    def test_unknown_annotations_keep_their_spelling(self):
        parsed = _parse("public class A {\n    @MyCustomThing\n    static void f() { }\n}\n")
        assert "annotation:MyCustomThing" in _entity(parsed, "f").tags


class TestInterfacesAndEnums:
    def test_interface(self):
        parsed = _parse("public interface Payable {\n    void pay();\n}\n")
        iface = _entity(parsed, "Payable")
        assert iface.label == NodeLabel.TYPE_DEF
        assert iface.kind == TypeDefKind.INTERFACE

    def test_enum(self):
        parsed = _parse("public enum Season { WINTER, SPRING }\n")
        enum = _entity(parsed, "Season")
        assert enum.kind == TypeDefKind.ENUM
        assert _entity(parsed, "WINTER").kind == ValueKind.ENUM_MEMBER

    def test_inner_class_nests_under_the_outer_class(self):
        parsed = _parse("public class Outer {\n    public class Inner {\n    }\n}\n")
        assert _entity(parsed, "Inner").qualified_name == f"{PROJECT}:apex.Outer.Inner"


# ---------------------------------------------------------------------------
# SObject references
# ---------------------------------------------------------------------------


class TestSObjectReferences:
    def test_soql_from_target(self):
        parsed = _parse(
            "public class A {\n    void f() {\n        List<Account> rows = [SELECT Id FROM Account];\n    }\n}\n"
        )
        assert "sobject.Account" in _rel_targets(parsed, RelType.IMPORTS, from_suffix="apex.A.f")

    def test_sosl_returning_targets(self):
        parsed = _parse(
            "public class A {\n"
            "    void f() {\n"
            "        List<List<SObject>> r = [FIND 'x' IN ALL FIELDS RETURNING Account(Id), Contact(Id)];\n"
            "    }\n"
            "}\n"
        )
        targets = _rel_targets(parsed, RelType.IMPORTS, from_suffix="apex.A.f")
        assert {"sobject.Account", "sobject.Contact"} <= targets

    def test_dml_on_a_constructed_sobject(self):
        parsed = _parse("public class A {\n    void f() {\n        insert new Contact();\n    }\n}\n")
        assert "sobject.Contact" in _rel_targets(parsed, RelType.IMPORTS, from_suffix="apex.A.f")

    def test_dml_on_a_variable_resolves_through_its_declared_type(self):
        parsed = _parse(
            "public class A {\n"
            "    void f() {\n"
            "        List<Custom_Object__c> rows = new List<Custom_Object__c>();\n"
            "        update rows;\n"
            "    }\n"
            "}\n"
        )
        assert "sobject.Custom_Object__c" in _rel_targets(parsed, RelType.IMPORTS, from_suffix="apex.A.f")

    def test_dml_on_a_parameter_resolves_through_its_declared_type(self):
        parsed = _parse("public class A {\n    void f(Account a) {\n        insert a;\n    }\n}\n")
        assert "sobject.Account" in _rel_targets(parsed, RelType.IMPORTS, from_suffix="apex.A.f")

    def test_unresolvable_dml_target_emits_nothing(self):
        """An undeclared variable name must not become a fake SObject node."""
        parsed = _parse("public class A {\n    void f() {\n        delete mystery;\n    }\n}\n")
        assert _rel_targets(parsed, RelType.IMPORTS) == set()

    def test_primitive_types_are_never_sobjects(self):
        parsed = _parse("public class A {\n    void f() {\n        String s = 'x';\n        delete s;\n    }\n}\n")
        assert _rel_targets(parsed, RelType.IMPORTS) == set()

    def test_reference_is_attributed_to_the_innermost_method(self):
        parsed = _parse(
            "public class A {\n"
            "    void first() {\n"
            "        List<Account> a = [SELECT Id FROM Account];\n"
            "    }\n"
            "    void second() {\n"
            "        List<Contact> c = [SELECT Id FROM Contact];\n"
            "    }\n"
            "}\n"
        )
        assert _rel_targets(parsed, RelType.IMPORTS, from_suffix="apex.A.first") == {"sobject.Account"}
        assert _rel_targets(parsed, RelType.IMPORTS, from_suffix="apex.A.second") == {"sobject.Contact"}


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------


class TestTrigger:
    SOURCE = """\
trigger AccountTrigger on Account (before insert, after update) {
    for (Account a : Trigger.new) {
        a.Name = a.Name.toUpperCase();
    }
    List<Contact> related = [SELECT Id FROM Contact];
    delete related;
}
"""

    @pytest.fixture
    def parsed(self) -> ParsedFile:
        return _parse(self.SOURCE, path=TRIGGER_PATH)

    @staticmethod
    def _trigger(parsed: ParsedFile):
        callables = _labelled(parsed, NodeLabel.CALLABLE)
        assert len(callables) == 1
        return callables[0]

    def test_trigger_is_a_callable_with_a_trigger_kind(self, parsed: ParsedFile):
        trigger = self._trigger(parsed)
        assert trigger.name == "AccountTrigger"
        assert trigger.kind == "trigger"

    def test_no_synthetic_wrapper_entities_leak(self, parsed: ParsedFile):
        names = [e.name for e in parsed.entities]
        assert not any(name.endswith("__body") for name in names)
        assert not any(e.label == NodeLabel.TYPE_DEF for e in parsed.entities)

    def test_trigger_qualified_name_is_path_derived(self, parsed: ParsedFile):
        trigger = self._trigger(parsed)
        expected = f"{PROJECT}:force-app.main.default.triggers.AccountTrigger.AccountTrigger"
        assert trigger.qualified_name == expected

    def test_events_and_sobject_are_recovered_from_the_header(self, parsed: ParsedFile):
        trigger = self._trigger(parsed)
        assert "apex:trigger" in trigger.tags
        assert "trigger:before_insert" in trigger.tags
        assert "trigger:after_update" in trigger.tags
        assert trigger.signature == "trigger AccountTrigger on Account (before insert, after update)"

    def test_sobject_import_from_the_header(self, parsed: ParsedFile):
        assert "sobject.Account" in _rel_targets(parsed, RelType.IMPORTS)

    def test_body_calls_are_extracted(self, parsed: ParsedFile):
        """The synthetic method wrapper exists so the bare statement list parses at all."""
        assert "toUpperCase" in _rel_targets(parsed, RelType.CALLS)

    def test_body_soql_and_dml_are_extracted(self, parsed: ParsedFile):
        assert "sobject.Contact" in _rel_targets(parsed, RelType.IMPORTS)

    def test_module_defines_the_trigger(self, parsed: ParsedFile):
        module_uid = f"{PROJECT}:force-app.main.default.triggers.AccountTrigger"
        defines = [r for r in parsed.relationships if r.rel_type == RelType.DEFINES]
        assert defines[0].from_qualified_name == module_uid


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------


class TestDegenerateInput:
    def test_uppercase_keywords_warn_instead_of_silently_yielding_nothing(self):
        """Apex is case-insensitive; the Java grammar behind the shim is not."""
        with _captured_warnings() as warnings:
            parsed = _parse("PUBLIC CLASS Foo {\n    PUBLIC STATIC VOID bar() { }\n}\n")
        assert [e.label for e in parsed.entities] == [NodeLabel.MODULE]
        assert any("produced no entities" in message for message in warnings)

    def test_comment_only_file_does_not_warn(self):
        with _captured_warnings() as warnings:
            _parse("// nothing here\n/* not here either */\n")
        assert not any("produced no entities" in message for message in warnings)

    def test_empty_file_is_handled(self):
        parsed = _parse("")
        assert [e.label for e in parsed.entities] == [NodeLabel.MODULE]


# ---------------------------------------------------------------------------
# Apex-only syntax that would otherwise ERROR out of the Java grammar
# ---------------------------------------------------------------------------


class TestApexOnlySyntax:
    @staticmethod
    def _has_error(source: str) -> bool:
        """True when the shimmed source still contains a tree-sitter ERROR node."""
        shimmed, _ = _shim(source.encode("utf-8"), allow_trigger=source.lstrip().startswith("trigger"))
        root = Parser(_APEX_LANGUAGE).parse(shimmed).root_node
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type == "ERROR" or node.is_missing:
                return True
            stack.extend(node.children)
        return False

    @pytest.mark.parametrize(
        ("label", "source"),
        [
            (
                "sharing + global + soql + dml",
                "public with sharing class A {\n"
                "    @AuraEnabled\n"
                "    global static List<Account> f() {\n"
                "        List<Account> rows = [SELECT Id, Name FROM Account WHERE Name LIKE :x];\n"
                "        insert rows;\n"
                "        return rows;\n"
                "    }\n"
                "}\n",
            ),
            (
                "list literal",
                "public class A {\n    void f() {\n        List<String> xs = new List<String>{ 'a', 'b' };\n    }\n}\n",
            ),
            (
                "map literal with fat arrows",
                "public class A {\n"
                "    void f() {\n"
                "        Map<Id, String> m = new Map<Id, String>{ '1' => 'a', '2' => 'b' };\n"
                "    }\n"
                "}\n",
            ),
            (
                "auto-implemented property",
                "public class A {\n    public String Region { get; private set; }\n}\n",
            ),
            (
                "property with accessor bodies",
                "public class A {\n"
                "    public Integer Count {\n"
                "        get { return 1; }\n"
                "        set { this.x = value; }\n"
                "    }\n"
                "}\n",
            ),
            (
                "virtual/override/webservice/testmethod",
                "global virtual class Base {\n"
                "    public virtual override void run() { }\n"
                "    webservice static void ping() { }\n"
                "    static testmethod void check() { }\n"
                "}\n",
            ),
            (
                "trigger context variables",
                "trigger T on Account (before insert, before update) {\n"
                "    for (Account a : Trigger.new) {\n"
                "        a.Name = Trigger.oldMap.get(a.Id).Name;\n"
                "    }\n"
                "}\n",
            ),
            (
                "sosl",
                "public class A {\n"
                "    void f() {\n"
                "        List<List<SObject>> r = [FIND 'x*' IN NAME FIELDS RETURNING Account(Id, Name)];\n"
                "    }\n"
                "}\n",
            ),
        ],
    )
    def test_shimmed_source_parses_without_errors(self, label: str, source: str):
        assert not self._has_error(source), f"{label}: shimmed source still has ERROR nodes"

    def test_map_literal_does_not_break_the_enclosing_method(self):
        parsed = _parse(
            "public class A {\n"
            "    void f() {\n"
            "        Map<Id, String> m = new Map<Id, String>{ '1' => 'a' };\n"
            "        helper();\n"
            "    }\n"
            "}\n"
        )
        assert "helper" in _rel_targets(parsed, RelType.CALLS, from_suffix="apex.A.f")

    def test_property_with_accessor_bodies_is_still_a_callable(self):
        parsed = _parse(
            "public class A {\n"
            "    public Integer Count {\n"
            "        get { return 1; }\n"
            "        set { this.x = value; }\n"
            "    }\n"
            "}\n"
        )
        assert _entity(parsed, "Count").kind == CallableKind.PROPERTY


# ---------------------------------------------------------------------------
# The Apex <-> LWC contract
# ---------------------------------------------------------------------------


class TestLwcJoin:
    """The Apex qualified name IS the LWC import specifier, byte for byte.

    This is the whole reason Apex members are stored under ``apex.`` instead of a
    path-derived prefix: ``GraphClient.resolve_imports`` matches import targets
    against stored ``qualified_name`` exactly, so any divergence here silently
    downgrades every LWC -> Apex edge to an unjoinable ``ext/`` stub.
    """

    APEX = (
        "public with sharing class AccountService {\n"
        "    @AuraEnabled(cacheable=true)\n"
        "    public static List<Account> getAccounts() {\n"
        "        return [SELECT Id FROM Account];\n"
        "    }\n"
        "}\n"
    )
    LWC = (
        "import getAccounts from '@salesforce/apex/AccountService.getAccounts';\n"
        "import NAME from '@salesforce/schema/Account.Name';\n"
        "export default class AccountList extends LightningElement {}\n"
    )
    LWC_PATH = "force-app/main/default/lwc/accountList/accountList.js"

    @pytest.fixture
    def apex(self) -> ParsedFile:
        return _parse(self.APEX)

    @pytest.fixture
    def lwc(self) -> ParsedFile:
        pytest.importorskip("tree_sitter_javascript", reason="tree-sitter-javascript not installed")
        parsed = parse_file(self.LWC_PATH, self.LWC.encode("utf-8"), PROJECT)
        assert parsed is not None
        return parsed

    def test_lwc_apex_import_matches_an_apex_qualified_name(self, apex: ParsedFile, lwc: ParsedFile):
        apex_qns = {e.qualified_name for e in apex.entities}
        targets = {r.to_name for r in lwc.relationships if r.rel_type == RelType.IMPORTS}
        assert f"{PROJECT}:apex.AccountService.getAccounts" in apex_qns
        assert "apex.AccountService.getAccounts" in targets

    def test_both_tiers_reference_the_same_sobject_target(self, apex: ParsedFile, lwc: ParsedFile):
        apex_targets = {r.to_name for r in apex.relationships if r.rel_type == RelType.IMPORTS}
        lwc_targets = {r.to_name for r in lwc.relationships if r.rel_type == RelType.IMPORTS}
        assert "sobject.Account" in apex_targets & lwc_targets

    def test_dml_is_recognised_after_a_brace_on_the_same_line(self):
        parsed = _parse("public class A {\n    void f(Account a) {\n        if (a != null) { insert a; }\n    }\n}\n")
        assert "sobject.Account" in _rel_targets(parsed, RelType.IMPORTS, from_suffix="apex.A.f")

    def test_dml_word_in_a_comment_is_not_a_reference(self):
        parsed = _parse("public class A {\n    // insert the Account record;\n    void f() { }\n}\n")
        assert _rel_targets(parsed, RelType.IMPORTS) == set()


# ---------------------------------------------------------------------------
# Shim regressions found by adversarial review — both erased real code silently.
# ---------------------------------------------------------------------------


def test_set_field_does_not_erase_the_enclosing_class():
    """`Set<Id>` is ubiquitous in Apex and used to destroy the whole file.

    `_PROPERTY_CANDIDATE` matches `public class P {` by design (the accessor gate
    is meant to be the discriminator), and the gate's `(?:get|set)\b` matched the
    `Set` in `private Set<Id> ids` — IGNORECASE makes `Set` match `set`, and `<`
    is a word boundary. The class body was then rewritten as an accessor block.
    Measured before the fix: every entity in the file vanished.
    """
    parsed = _parse(
        "public class P {\n    private Set<Id> ids = new Set<Id>();\n    public void go() { System.debug(1); }\n}\n"
    )
    names = {e.name for e in parsed.entities}
    assert {"P", "ids", "go"} <= names, f"Set<Id> field erased entities: {names}"


def test_real_properties_still_rewrite():
    """The accessor-gate fix must not break the thing the gate exists for."""
    parsed = _parse(
        "public class S {\n"
        "    private Set<Id> ids;\n"
        "    public String Name { get; set; }\n"
        "    public Integer Cnt { get { return 1; } set; }\n"
        "}\n"
    )
    names = {e.name for e in parsed.entities}
    assert {"Name", "Cnt"} <= names, f"properties lost: {names}"


def test_soql_matcher_ignores_brackets_inside_string_literals():
    """A string containing `[SELECT` used to swallow code up to the next `]`.

    `_SOQL` ran under re.DOTALL with no string awareness, so `'[SELECT oops'`
    (no closing bracket of its own) matched onward to the `]` in a LATER
    method's `v[0]`, blanking everything between. Measured: method `b` gone.
    """
    parsed = _parse(
        "public class Q {\n"
        "    public void a() { String s = '[SELECT oops'; }\n"
        "    public void b() { List<Integer> v = new List<Integer>(); Integer z = v[0]; }\n"
        "    public void c() { Integer y = 2; }\n"
        "}\n"
    )
    names = {e.name for e in parsed.entities}
    assert {"a", "b", "c"} <= names, f"string literal swallowed code: {names}"
    sobjects = [r.to_name for r in parsed.relationships if r.to_name.startswith("sobject.")]
    assert sobjects == [], f"a string literal must not yield an SObject edge: {sobjects}"


def test_nested_soql_subquery_resolves_both_objects():
    """Bracket matching must nest — a subquery's `]` is not the outer query's."""
    parsed = _parse(
        "public class B {\n"
        "    public void go() { List<Contact> c = [SELECT Id, (SELECT Id FROM Cases) FROM Contact]; }\n"
        "}\n"
    )
    sobjects = {r.to_name for r in parsed.relationships if r.to_name.startswith("sobject.")}
    assert sobjects == {"sobject.Cases", "sobject.Contact"}, sobjects


def test_canonical_trigger_keeps_its_body_calls():
    """The synthetic wrapper must fit inside the header span it replaces.

    It used to be `class {name} { void {name}__body ( ) {` — the trigger name TWICE,
    while the real header `trigger {name} on {obj} (...)` contains it once. The wrapper
    therefore grew at 2x the header's rate and overflowed on the most ordinary
    Salesforce shape there is: a long `<Object>Trigger` name with a short SObject and a
    short event list. Measured for `AccountTrigger`: header 51 bytes, wrapper 54, no fit,
    body skipped, every in-body CALLS edge lost. Warned in logs but silent in the graph.
    """
    parsed = _parse(
        "trigger AccountTrigger on Account (before insert) {\n    AccountService.handle(Trigger.new);\n}\n",
        TRIGGER_PATH,
    )
    calls = [r.to_name for r in parsed.relationships if r.rel_type is RelType.CALLS]
    assert "handle" in calls, f"trigger body did not parse — wrapper overflowed the header span: {calls}"


@pytest.mark.parametrize(
    "header",
    [
        "trigger T on A (before insert)",  # shortest legal shape
        "trigger AccountTrigger on Account (before insert)",  # the canonical failure
        "trigger OpportunityLineItemTrigger on OpportunityLineItem (before insert, after update)",
    ],
)
def test_trigger_wrapper_always_fits_its_header(header: str):
    """Pin the length RELATIONSHIP, not one example.

    The wrapper must never be longer than the header it overwrites, for any trigger
    name. Asserting this directly is what stops a future edit from reintroducing a
    name-dependent term and only failing on names nobody happened to test.
    """
    from code_atlas.parsing.languages.apex import _TRIGGER_BODY_NAME

    name = header.split()[1]
    wrapper = f"class {name} {{ void {_TRIGGER_BODY_NAME} ( ) {{"
    assert len(wrapper) <= len(header) + 2, (  # +2 for the " {" the header's brace supplies
        f"wrapper ({len(wrapper)}) exceeds header ({len(header) + 2}) for {name!r}"
    )
