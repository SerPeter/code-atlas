"""Tests for Java and C# parsers."""

from __future__ import annotations

import pytest

from code_atlas.parsing.ast import ParsedFile, get_language_for_file, parse_file
from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind, ValueKind, Visibility

PROJECT = "test_project"


def _parse(source: str, path: str = "src/Example.java") -> ParsedFile:
    result = parse_file(path, source.encode("utf-8"), PROJECT)
    assert result is not None
    return result


def _entity_by_name(parsed: ParsedFile, name: str):
    matches = [e for e in parsed.entities if e.name == name]
    assert len(matches) == 1, (
        f"Expected 1 entity named {name!r}, got {len(matches)}: {[e.name for e in parsed.entities]}"
    )
    return matches[0]


def _rels_from(parsed: ParsedFile, from_qn_suffix: str, rel_type: RelType):
    return [
        r for r in parsed.relationships if r.from_qualified_name.endswith(from_qn_suffix) and r.rel_type == rel_type
    ]


# ===========================================================================
# Java tests
# ===========================================================================


class TestJavaLanguageDetection:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_java_detected(self):
        assert get_language_for_file("src/Main.java") is not None

    def test_java_extension_only(self):
        """Non-.java files are not matched to Java."""
        cfg = get_language_for_file("src/Main.kt")
        assert cfg is None or cfg.name != "java"


class TestJavaModule:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_module_entity(self):
        parsed = _parse("class Foo {}\n", path="src/com/example/MyClass.java")
        module = _entity_by_name(parsed, "MyClass")
        assert module.label == NodeLabel.MODULE
        assert module.kind == "module"
        assert module.qualified_name == f"{PROJECT}:src.com.example.MyClass"


class TestJavaClass:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_class_basic(self):
        parsed = _parse("public class MyClass {}\n")
        cls = _entity_by_name(parsed, "MyClass")
        assert cls.label == NodeLabel.TYPE_DEF
        assert cls.kind == TypeDefKind.CLASS
        assert cls.visibility == Visibility.PUBLIC

    def test_class_defines(self):
        parsed = _parse("public class MyClass {}\n")
        defines = _rels_from(parsed, "src.Example", RelType.DEFINES)
        targets = {r.to_name for r in defines}
        assert f"{PROJECT}:src.Example.MyClass" in targets


class TestJavaInterface:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_interface(self):
        parsed = _parse("public interface Runnable { void run(); }\n")
        iface = _entity_by_name(parsed, "Runnable")
        assert iface.label == NodeLabel.TYPE_DEF
        assert iface.kind == TypeDefKind.INTERFACE


class TestJavaEnum:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_enum(self):
        parsed = _parse("public enum Color { RED, GREEN, BLUE }\n")
        enum = _entity_by_name(parsed, "Color")
        assert enum.label == NodeLabel.TYPE_DEF
        assert enum.kind == TypeDefKind.ENUM

    def test_enum_members(self):
        parsed = _parse("public enum Color { RED, GREEN, BLUE }\n")
        red = _entity_by_name(parsed, "RED")
        assert red.label == NodeLabel.VALUE
        assert red.kind == ValueKind.ENUM_MEMBER

        green = _entity_by_name(parsed, "GREEN")
        assert green.kind == ValueKind.ENUM_MEMBER


class TestJavaMethod:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_method(self):
        source = """\
public class Foo {
    public void doWork(int x) {
        System.out.println(x);
    }
}
"""
        parsed = _parse(source)
        method = _entity_by_name(parsed, "doWork")
        assert method.label == NodeLabel.CALLABLE
        assert method.kind == CallableKind.METHOD
        assert method.visibility == Visibility.PUBLIC

    def test_static_method(self):
        source = """\
public class Foo {
    public static void main(String[] args) {}
}
"""
        parsed = _parse(source)
        method = _entity_by_name(parsed, "main")
        assert method.kind == CallableKind.STATIC_METHOD


class TestJavaConstructor:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_constructor(self):
        source = """\
public class Foo {
    public Foo(int x) {}
}
"""
        parsed = _parse(source)
        ctors = [e for e in parsed.entities if e.name == "Foo" and e.label == NodeLabel.CALLABLE]
        assert len(ctors) == 1
        assert ctors[0].kind == CallableKind.CONSTRUCTOR


class TestJavaVisibility:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_public(self):
        parsed = _parse("public class Foo { public void bar() {} }\n")
        bar = _entity_by_name(parsed, "bar")
        assert bar.visibility == Visibility.PUBLIC

    def test_private(self):
        parsed = _parse("public class Foo { private void bar() {} }\n")
        bar = _entity_by_name(parsed, "bar")
        assert bar.visibility == Visibility.PRIVATE

    def test_protected(self):
        parsed = _parse("public class Foo { protected void bar() {} }\n")
        bar = _entity_by_name(parsed, "bar")
        assert bar.visibility == Visibility.PROTECTED

    def test_package_private_default(self):
        """No modifier in Java -> INTERNAL (package-private)."""
        parsed = _parse("public class Foo { void bar() {} }\n")
        bar = _entity_by_name(parsed, "bar")
        assert bar.visibility == Visibility.INTERNAL


class TestJavaImports:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_import_statements(self):
        source = """\
import java.util.List;
import java.io.File;

public class Foo {}
"""
        parsed = _parse(source)
        imports = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
        imported = {r.to_name for r in imports}
        assert "java.util.List" in imported
        assert "java.io.File" in imported


class TestJavaInheritance:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_extends(self):
        parsed = _parse("public class Child extends Parent {}\n")
        inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
        assert any(r.to_name == "Parent" for r in inherits)

    def test_implements(self):
        parsed = _parse("public class Foo implements Runnable, Serializable {}\n")
        implements = [r for r in parsed.relationships if r.rel_type == RelType.IMPLEMENTS]
        impl_names = {r.to_name for r in implements}
        assert "Runnable" in impl_names
        assert "Serializable" in impl_names


class TestJavaDocstring:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_javadoc(self):
        source = """\
/**
 * A useful class.
 */
public class Documented {}
"""
        parsed = _parse(source)
        cls = _entity_by_name(parsed, "Documented")
        assert cls.docstring is not None
        assert "useful class" in cls.docstring


class TestJavaSignature:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_method_signature(self):
        source = """\
public class Foo {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        parsed = _parse(source)
        method = _entity_by_name(parsed, "add")
        assert method.signature is not None
        assert "add" in method.signature
        assert "int a" in method.signature
        # Body should not be in the signature
        assert "return" not in method.signature


class TestJavaFields:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_field(self):
        source = """\
public class Foo {
    private int count;
}
"""
        parsed = _parse(source)
        field = _entity_by_name(parsed, "count")
        assert field.label == NodeLabel.VALUE
        assert field.kind == ValueKind.FIELD
        assert field.visibility == Visibility.PRIVATE

    def test_constant(self):
        source = """\
public class Foo {
    public static final int MAX = 100;
}
"""
        parsed = _parse(source)
        const = _entity_by_name(parsed, "MAX")
        assert const.label == NodeLabel.VALUE
        assert const.kind == ValueKind.CONSTANT


class TestJavaAnnotationTags:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_annotation_tag(self):
        source = """\
public class Foo {
    @Override
    public String toString() {
        return "Foo";
    }
}
"""
        parsed = _parse(source)
        method = _entity_by_name(parsed, "toString")
        assert any("annotation:Override" in t for t in method.tags)

    def test_test_annotation(self):
        source = """\
public class FooTest {
    @Test
    public void testSomething() {}
}
"""
        parsed = _parse(source)
        method = _entity_by_name(parsed, "testSomething")
        assert any("annotation:Test" in t for t in method.tags)


class TestJavaDefines:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_defines_method(self):
        source = """\
public class Foo {
    public void bar() {}
}
"""
        parsed = _parse(source)
        defines = _rels_from(parsed, "src.Example.Foo", RelType.DEFINES)
        targets = {r.to_name for r in defines}
        assert f"{PROJECT}:src.Example.Foo.bar" in targets


class TestJavaCalls:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_method_calls(self):
        source = """\
public class Foo {
    public void doWork() {
        helper();
        System.out.println("hello");
    }
}
"""
        parsed = _parse(source)
        calls = _rels_from(parsed, "src.Example.Foo.doWork", RelType.CALLS)
        called = {r.to_name for r in calls}
        assert "helper" in called or "println" in called


class TestJavaContentHash:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_deterministic(self):
        source = "public class Foo { public void bar() {} }\n"
        p1 = _parse(source)
        p2 = _parse(source)
        for e1, e2 in zip(p1.entities, p2.entities, strict=True):
            assert e1.content_hash == e2.content_hash


class TestJavaEdgeCases:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_empty_file(self):
        parsed = _parse("")
        assert parsed is not None
        assert parsed.language == "java"
        assert len(parsed.entities) >= 1

    def test_syntax_error_tolerant(self):
        parsed = _parse("public class { broken }\n")
        assert parsed is not None


class TestJavaAnnotationType:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_annotation_type(self):
        source = "public @interface MyAnnotation {}\n"
        parsed = _parse(source)
        ann = _entity_by_name(parsed, "MyAnnotation")
        assert ann.label == NodeLabel.TYPE_DEF
        assert ann.kind == TypeDefKind.ANNOTATION


class TestJavaRecord:
    @pytest.fixture(autouse=True)
    def _require_java(self):
        pytest.importorskip("tree_sitter_java")

    def test_record(self):
        source = "public record Point(int x, int y) {}\n"
        parsed = _parse(source)
        rec = _entity_by_name(parsed, "Point")
        assert rec.label == NodeLabel.TYPE_DEF
        assert rec.kind == TypeDefKind.RECORD


# ===========================================================================
# C# tests
# ===========================================================================


class TestCSharpLanguageDetection:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_csharp_detected(self):
        assert get_language_for_file("src/User.cs") is not None

    def test_csharp_extension_only(self):
        cfg = get_language_for_file("src/User.vb")
        assert cfg is None or cfg.name != "csharp"


class TestCSharpModule:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_module_entity(self):
        parsed = _parse("class Foo {}\n", path="src/Models/User.cs")
        module = _entity_by_name(parsed, "User")
        assert module.label == NodeLabel.MODULE
        assert module.kind == "module"
        assert module.qualified_name == f"{PROJECT}:src.Models.User"


class TestCSharpClass:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_class_basic(self):
        parsed = _parse("public class MyClass {}\n", path="src/Example.cs")
        cls = _entity_by_name(parsed, "MyClass")
        assert cls.label == NodeLabel.TYPE_DEF
        assert cls.kind == TypeDefKind.CLASS
        assert cls.visibility == Visibility.PUBLIC

    def test_class_defines(self):
        parsed = _parse("public class MyClass {}\n", path="src/Example.cs")
        defines = _rels_from(parsed, "src.Example", RelType.DEFINES)
        targets = {r.to_name for r in defines}
        assert f"{PROJECT}:src.Example.MyClass" in targets


class TestCSharpInterface:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_interface(self):
        parsed = _parse("public interface IRunnable { void Run(); }\n", path="src/Example.cs")
        iface = _entity_by_name(parsed, "IRunnable")
        assert iface.label == NodeLabel.TYPE_DEF
        assert iface.kind == TypeDefKind.INTERFACE


class TestCSharpEnum:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_enum(self):
        parsed = _parse("public enum Color { Red, Green, Blue }\n", path="src/Example.cs")
        enum = _entity_by_name(parsed, "Color")
        assert enum.label == NodeLabel.TYPE_DEF
        assert enum.kind == TypeDefKind.ENUM

    def test_enum_members(self):
        parsed = _parse("public enum Color { Red, Green, Blue }\n", path="src/Example.cs")
        red = _entity_by_name(parsed, "Red")
        assert red.label == NodeLabel.VALUE
        assert red.kind == ValueKind.ENUM_MEMBER


class TestCSharpMethod:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_method(self):
        source = """\
public class Foo {
    public void DoWork(int x) {
        Console.WriteLine(x);
    }
}
"""
        parsed = _parse(source, path="src/Example.cs")
        method = _entity_by_name(parsed, "DoWork")
        assert method.label == NodeLabel.CALLABLE
        assert method.kind == CallableKind.METHOD
        assert method.visibility == Visibility.PUBLIC

    def test_static_method(self):
        source = """\
public class Foo {
    public static void Main(string[] args) {}
}
"""
        parsed = _parse(source, path="src/Example.cs")
        method = _entity_by_name(parsed, "Main")
        assert method.kind == CallableKind.STATIC_METHOD


class TestCSharpConstructor:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_constructor(self):
        source = """\
public class Foo {
    public Foo(int x) {}
}
"""
        parsed = _parse(source, path="src/Example.cs")
        ctors = [e for e in parsed.entities if e.label == NodeLabel.CALLABLE and e.kind == CallableKind.CONSTRUCTOR]
        assert len(ctors) == 1
        assert ctors[0].name == "Foo"


class TestCSharpVisibility:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_public(self):
        parsed = _parse("public class Foo { public void Bar() {} }\n", path="src/Example.cs")
        bar = _entity_by_name(parsed, "Bar")
        assert bar.visibility == Visibility.PUBLIC

    def test_private(self):
        parsed = _parse("public class Foo { private void Bar() {} }\n", path="src/Example.cs")
        bar = _entity_by_name(parsed, "Bar")
        assert bar.visibility == Visibility.PRIVATE

    def test_protected(self):
        parsed = _parse("public class Foo { protected void Bar() {} }\n", path="src/Example.cs")
        bar = _entity_by_name(parsed, "Bar")
        assert bar.visibility == Visibility.PROTECTED

    def test_internal(self):
        parsed = _parse("public class Foo { internal void Bar() {} }\n", path="src/Example.cs")
        bar = _entity_by_name(parsed, "Bar")
        assert bar.visibility == Visibility.INTERNAL

    def test_default_private(self):
        """No modifier in C# class -> PRIVATE."""
        parsed = _parse("public class Foo { void Bar() {} }\n", path="src/Example.cs")
        bar = _entity_by_name(parsed, "Bar")
        assert bar.visibility == Visibility.PRIVATE


class TestCSharpUsings:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_using_directives(self):
        source = """\
using System;
using System.Collections.Generic;

public class Foo {}
"""
        parsed = _parse(source, path="src/Example.cs")
        imports = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
        imported = {r.to_name for r in imports}
        assert "System" in imported
        assert "System.Collections.Generic" in imported


class TestCSharpInheritance:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_inherits(self):
        parsed = _parse("public class Child : Parent {}\n", path="src/Example.cs")
        inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
        assert any(r.to_name == "Parent" for r in inherits)

    def test_implements_interface(self):
        """C# interface naming convention: starts with I followed by uppercase."""
        parsed = _parse("public class Foo : IDisposable {}\n", path="src/Example.cs")
        implements = [r for r in parsed.relationships if r.rel_type == RelType.IMPLEMENTS]
        assert any(r.to_name == "IDisposable" for r in implements)


class TestCSharpDocstring:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_xml_doc_comment(self):
        source = """\
/// <summary>
/// A useful class.
/// </summary>
public class Documented {}
"""
        parsed = _parse(source, path="src/Example.cs")
        cls = _entity_by_name(parsed, "Documented")
        assert cls.docstring is not None
        assert "useful class" in cls.docstring


class TestCSharpSignature:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_method_signature(self):
        source = """\
public class Foo {
    public int Add(int a, int b) {
        return a + b;
    }
}
"""
        parsed = _parse(source, path="src/Example.cs")
        method = _entity_by_name(parsed, "Add")
        assert method.signature is not None
        assert "Add" in method.signature
        assert "return" not in method.signature


class TestCSharpFields:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_field(self):
        source = """\
public class Foo {
    private int _count;
}
"""
        parsed = _parse(source, path="src/Example.cs")
        field = _entity_by_name(parsed, "_count")
        assert field.label == NodeLabel.VALUE
        assert field.kind == ValueKind.FIELD
        assert field.visibility == Visibility.PRIVATE

    def test_constant(self):
        source = """\
public class Foo {
    public static readonly int Max = 100;
}
"""
        parsed = _parse(source, path="src/Example.cs")
        const = _entity_by_name(parsed, "Max")
        assert const.label == NodeLabel.VALUE
        assert const.kind == ValueKind.CONSTANT


class TestCSharpAttributeTags:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_attribute_tag(self):
        source = """\
[Serializable]
public class Foo {}
"""
        parsed = _parse(source, path="src/Example.cs")
        cls = _entity_by_name(parsed, "Foo")
        assert any("attribute:Serializable" in t for t in cls.tags)

    def test_test_attribute(self):
        source = """\
public class FooTest {
    [Test]
    public void TestSomething() {}
}
"""
        parsed = _parse(source, path="src/Example.cs")
        method = _entity_by_name(parsed, "TestSomething")
        assert any("attribute:Test" in t for t in method.tags)


class TestCSharpDefines:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_defines_method(self):
        source = """\
public class Foo {
    public void Bar() {}
}
"""
        parsed = _parse(source, path="src/Example.cs")
        defines = _rels_from(parsed, "src.Example.Foo", RelType.DEFINES)
        targets = {r.to_name for r in defines}
        assert f"{PROJECT}:src.Example.Foo.Bar" in targets


class TestCSharpCalls:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_method_calls(self):
        source = """\
public class Foo {
    public void DoWork() {
        Helper();
        Console.WriteLine("hello");
    }
}
"""
        parsed = _parse(source, path="src/Example.cs")
        calls = _rels_from(parsed, "src.Example.Foo.DoWork", RelType.CALLS)
        called = {r.to_name for r in calls}
        assert "Helper" in called or "WriteLine" in called


class TestCSharpContentHash:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_deterministic(self):
        source = "public class Foo { public void Bar() {} }\n"
        p1 = _parse(source, path="src/Example.cs")
        p2 = _parse(source, path="src/Example.cs")
        for e1, e2 in zip(p1.entities, p2.entities, strict=True):
            assert e1.content_hash == e2.content_hash


class TestCSharpEdgeCases:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_empty_file(self):
        parsed = _parse("", path="src/Example.cs")
        assert parsed is not None
        assert parsed.language == "csharp"
        assert len(parsed.entities) >= 1

    def test_syntax_error_tolerant(self):
        parsed = _parse("public class { broken }\n", path="src/Example.cs")
        assert parsed is not None


class TestCSharpStruct:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_struct(self):
        parsed = _parse("public struct Point { public int X; public int Y; }\n", path="src/Example.cs")
        st = _entity_by_name(parsed, "Point")
        assert st.label == NodeLabel.TYPE_DEF
        assert st.kind == TypeDefKind.STRUCT


class TestCSharpDestructor:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_destructor(self):
        source = """\
public class Foo {
    ~Foo() {}
}
"""
        parsed = _parse(source, path="src/Example.cs")
        destructors = [e for e in parsed.entities if e.kind == CallableKind.DESTRUCTOR]
        assert len(destructors) == 1
        assert destructors[0].label == NodeLabel.CALLABLE


class TestCSharpProperty:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_property(self):
        source = """\
public class Foo {
    public int Count { get; set; }
}
"""
        parsed = _parse(source, path="src/Example.cs")
        prop = _entity_by_name(parsed, "Count")
        assert prop.label == NodeLabel.CALLABLE
        assert prop.kind == CallableKind.PROPERTY
        assert prop.visibility == Visibility.PUBLIC


class TestCSharpPartialTag:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_partial_class(self):
        source = "public partial class Foo {}\n"
        parsed = _parse(source, path="src/Example.cs")
        cls = _entity_by_name(parsed, "Foo")
        assert "partial" in cls.tags


class TestCSharpNamespace:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_namespace_class(self):
        source = """\
namespace MyApp.Models {
    public class User {}
}
"""
        parsed = _parse(source, path="src/Example.cs")
        cls = _entity_by_name(parsed, "User")
        assert cls.label == NodeLabel.TYPE_DEF
        assert cls.kind == TypeDefKind.CLASS


class TestCSharpRecord:
    @pytest.fixture(autouse=True)
    def _require_csharp(self):
        pytest.importorskip("tree_sitter_c_sharp")

    def test_record(self):
        source = "public record Point(int X, int Y);\n"
        parsed = _parse(source, path="src/Example.cs")
        rec = _entity_by_name(parsed, "Point")
        assert rec.label == NodeLabel.TYPE_DEF
        assert rec.kind == TypeDefKind.RECORD
