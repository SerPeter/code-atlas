"""Tests for C and C++ parsers."""

from __future__ import annotations

import pytest

from code_atlas.parsing.ast import ParsedFile, get_language_for_file, parse_file
from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind, ValueKind, Visibility

PROJECT = "test_project"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(source: str, path: str = "src/example.c") -> ParsedFile:
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
# C TESTS
# ===========================================================================

ts_c = pytest.importorskip("tree_sitter_c")


# ---------------------------------------------------------------------------
# 1. Language detection
# ---------------------------------------------------------------------------


class TestCLanguageDetection:
    def test_c_extension(self):
        cfg = get_language_for_file("src/main.c")
        assert cfg is not None
        assert cfg.name == "c"

    def test_h_extension(self):
        cfg = get_language_for_file("include/utils.h")
        assert cfg is not None
        assert cfg.name == "c"


# ---------------------------------------------------------------------------
# 2. Module entity
# ---------------------------------------------------------------------------


class TestCModule:
    def test_module_entity(self):
        parsed = _parse("int x = 1;\n", path="src/server.c")
        module = _entity_by_name(parsed, "server")
        assert module.label == NodeLabel.MODULE
        assert module.kind == "module"
        assert module.qualified_name == f"{PROJECT}:src.server"

    def test_module_from_header(self):
        parsed = _parse("", path="include/utils.h")
        module = _entity_by_name(parsed, "utils")
        assert module.label == NodeLabel.MODULE
        assert module.qualified_name == f"{PROJECT}:include.utils"


# ---------------------------------------------------------------------------
# 3. Struct extraction
# ---------------------------------------------------------------------------


class TestCStruct:
    def test_struct_basic(self):
        parsed = _parse(
            """\
struct Point {
    int x;
    int y;
};
"""
        )
        s = _entity_by_name(parsed, "Point")
        assert s.label == NodeLabel.TYPE_DEF
        assert s.kind == TypeDefKind.STRUCT
        assert s.visibility == Visibility.PUBLIC

    def test_struct_defines_relationship(self):
        parsed = _parse("struct Foo { int a; };\n")
        defines = _rels_from(parsed, "src.example", RelType.DEFINES)
        target_names = {r.to_name for r in defines}
        assert f"{PROJECT}:src.example.Foo" in target_names


# ---------------------------------------------------------------------------
# 4. Enum extraction
# ---------------------------------------------------------------------------


class TestCEnum:
    def test_enum_basic(self):
        parsed = _parse(
            """\
enum Color {
    RED,
    GREEN,
    BLUE
};
"""
        )
        e = _entity_by_name(parsed, "Color")
        assert e.label == NodeLabel.TYPE_DEF
        assert e.kind == TypeDefKind.ENUM


# ---------------------------------------------------------------------------
# 5. Union extraction
# ---------------------------------------------------------------------------


class TestCUnion:
    def test_union_basic(self):
        parsed = _parse(
            """\
union Data {
    int i;
    float f;
};
"""
        )
        u = _entity_by_name(parsed, "Data")
        assert u.label == NodeLabel.TYPE_DEF
        assert u.kind == TypeDefKind.UNION


# ---------------------------------------------------------------------------
# 6. Typedef extraction
# ---------------------------------------------------------------------------


class TestCTypedef:
    def test_typedef_basic(self):
        parsed = _parse("typedef int MyInt;\n")
        td = _entity_by_name(parsed, "MyInt")
        assert td.label == NodeLabel.TYPE_DEF
        assert td.kind == TypeDefKind.TYPE_ALIAS

    def test_typedef_primitive_alias(self):
        """tree-sitter-c treats some names like size_t as primitive_type."""
        parsed = _parse("typedef unsigned long size_t;\n")
        td = _entity_by_name(parsed, "size_t")
        assert td.label == NodeLabel.TYPE_DEF
        assert td.kind == TypeDefKind.TYPE_ALIAS


# ---------------------------------------------------------------------------
# 7. Function extraction
# ---------------------------------------------------------------------------


class TestCFunction:
    def test_function_basic(self):
        parsed = _parse(
            """\
int add(int a, int b) {
    return a + b;
}
"""
        )
        func = _entity_by_name(parsed, "add")
        assert func.label == NodeLabel.CALLABLE
        assert func.kind == CallableKind.FUNCTION
        assert func.visibility == Visibility.PUBLIC

    def test_function_source(self):
        parsed = _parse(
            """\
int add(int a, int b) {
    return a + b;
}
"""
        )
        func = _entity_by_name(parsed, "add")
        assert func.source is not None
        assert "return a + b" in func.source


# ---------------------------------------------------------------------------
# 8. #include -> IMPORTS
# ---------------------------------------------------------------------------


class TestCIncludes:
    def test_system_include(self):
        parsed = _parse("#include <stdio.h>\n")
        imports = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
        imported = {r.to_name for r in imports}
        assert "stdio.h" in imported
        # No angle-bracket delimiters should leak into the import name
        assert "<stdio.h>" not in imported

    def test_local_include(self):
        parsed = _parse('#include "local.h"\n')
        imports = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
        imported = {r.to_name for r in imports}
        assert "local.h" in imported
        # No quote delimiters should leak into the import name
        assert '"local.h"' not in imported


# ---------------------------------------------------------------------------
# 9. Global variables
# ---------------------------------------------------------------------------


class TestCGlobalVariables:
    def test_global_variable(self):
        parsed = _parse("int global_count = 0;\n")
        var = _entity_by_name(parsed, "global_count")
        assert var.label == NodeLabel.VALUE
        assert var.kind == ValueKind.VARIABLE
        assert var.visibility == Visibility.PUBLIC

    def test_static_global_private(self):
        parsed = _parse("static int internal_count = 0;\n")
        var = _entity_by_name(parsed, "internal_count")
        assert var.label == NodeLabel.VALUE
        assert var.visibility == Visibility.PRIVATE
        assert "static" in var.tags


# ---------------------------------------------------------------------------
# 10. Struct fields
# ---------------------------------------------------------------------------


class TestCStructFields:
    def test_struct_fields(self):
        parsed = _parse(
            """\
struct Rect {
    int width;
    int height;
};
"""
        )
        w = _entity_by_name(parsed, "width")
        assert w.label == NodeLabel.VALUE
        assert w.kind == ValueKind.FIELD

        h = _entity_by_name(parsed, "height")
        assert h.kind == ValueKind.FIELD


# ---------------------------------------------------------------------------
# 11. Enum values
# ---------------------------------------------------------------------------


class TestCEnumValues:
    def test_enum_members(self):
        parsed = _parse(
            """\
enum Status {
    OK,
    ERROR
};
"""
        )
        ok = _entity_by_name(parsed, "OK")
        assert ok.label == NodeLabel.VALUE
        assert ok.kind == ValueKind.ENUM_MEMBER

        err = _entity_by_name(parsed, "ERROR")
        assert err.kind == ValueKind.ENUM_MEMBER


# ---------------------------------------------------------------------------
# 12. DEFINES relationships
# ---------------------------------------------------------------------------


class TestCDefines:
    def test_module_defines_function(self):
        parsed = _parse("int foo(void) { return 0; }\n")
        defines = _rels_from(parsed, "src.example", RelType.DEFINES)
        targets = {r.to_name for r in defines}
        assert f"{PROJECT}:src.example.foo" in targets

    def test_struct_defines_field(self):
        parsed = _parse("struct S { int x; };\n")
        defines = _rels_from(parsed, "src.example.S", RelType.DEFINES)
        targets = {r.to_name for r in defines}
        assert f"{PROJECT}:src.example.S.x" in targets

    def test_enum_defines_member(self):
        parsed = _parse("enum E { A, B };\n")
        defines = _rels_from(parsed, "src.example.E", RelType.DEFINES)
        targets = {r.to_name for r in defines}
        assert f"{PROJECT}:src.example.E.A" in targets
        assert f"{PROJECT}:src.example.E.B" in targets


# ---------------------------------------------------------------------------
# 13. CALLS extraction
# ---------------------------------------------------------------------------


class TestCCalls:
    def test_function_calls(self):
        parsed = _parse(
            """\
void foo(void) {
    printf("hello");
    bar();
}
"""
        )
        calls = _rels_from(parsed, "src.example.foo", RelType.CALLS)
        called = {r.to_name for r in calls}
        assert "printf" in called
        assert "bar" in called


# ---------------------------------------------------------------------------
# 14. Doxygen doc comments
# ---------------------------------------------------------------------------


class TestCDoxygen:
    def test_block_comment(self):
        parsed = _parse(
            """\
/** Adds two integers. */
int add(int a, int b) {
    return a + b;
}
"""
        )
        func = _entity_by_name(parsed, "add")
        assert func.docstring is not None
        assert "Adds two integers" in func.docstring

    def test_line_comment(self):
        parsed = _parse(
            """\
/// Computes the square.
int square(int x) {
    return x * x;
}
"""
        )
        func = _entity_by_name(parsed, "square")
        assert func.docstring is not None
        assert "square" in func.docstring.lower()


# ---------------------------------------------------------------------------
# 15. Signature extraction
# ---------------------------------------------------------------------------


class TestCSignature:
    def test_function_signature(self):
        parsed = _parse(
            """\
int add(int a, int b) {
    return a + b;
}
"""
        )
        func = _entity_by_name(parsed, "add")
        assert func.signature is not None
        assert "int add(int a, int b)" in func.signature
        # Signature should not contain the body
        assert "return" not in func.signature


# ---------------------------------------------------------------------------
# 16. Content hash determinism
# ---------------------------------------------------------------------------


class TestCContentHash:
    def test_hash_populated(self):
        parsed = _parse("int foo(void) { return 0; }\n")
        for entity in parsed.entities:
            assert entity.content_hash, f"Entity {entity.name!r} has empty content_hash"

    def test_hash_deterministic(self):
        source = "int foo(void) { return 0; }\n"
        parsed1 = _parse(source)
        parsed2 = _parse(source)
        for e1, e2 in zip(parsed1.entities, parsed2.entities, strict=True):
            assert e1.content_hash == e2.content_hash


# ---------------------------------------------------------------------------
# 17. Edge cases
# ---------------------------------------------------------------------------


class TestCEdgeCases:
    def test_empty_file(self):
        parsed = _parse("")
        assert parsed is not None
        assert parsed.language == "c"
        # Should have at least the module entity
        assert len(parsed.entities) >= 1

    def test_syntax_error_tolerant(self):
        """Tree-sitter is error-tolerant — malformed files don't crash."""
        parsed = _parse("int broken( { struct;\n")
        assert parsed is not None

    def test_anonymous_struct_skipped(self):
        """Anonymous structs should not create named entities."""
        parsed = _parse(
            """\
struct {
    int x;
} instance;
"""
        )
        # Should have module + the variable `instance`, but no named struct entity
        type_defs = [e for e in parsed.entities if e.label == NodeLabel.TYPE_DEF]
        assert len(type_defs) == 0


# ===========================================================================
# C++ TESTS
# ===========================================================================

ts_cpp = pytest.importorskip("tree_sitter_cpp")


# ---------------------------------------------------------------------------
# 18. Language detection (.cpp, .cc, .hpp)
# ---------------------------------------------------------------------------


class TestCppLanguageDetection:
    def test_cpp_extension(self):
        cfg = get_language_for_file("src/main.cpp")
        assert cfg is not None
        assert cfg.name == "cpp"

    def test_cc_extension(self):
        cfg = get_language_for_file("src/main.cc")
        assert cfg is not None
        assert cfg.name == "cpp"

    def test_hpp_extension(self):
        cfg = get_language_for_file("include/utils.hpp")
        assert cfg is not None
        assert cfg.name == "cpp"

    def test_cxx_extension(self):
        cfg = get_language_for_file("src/main.cxx")
        assert cfg is not None
        assert cfg.name == "cpp"

    def test_hxx_extension(self):
        cfg = get_language_for_file("include/utils.hxx")
        assert cfg is not None
        assert cfg.name == "cpp"

    def test_hh_extension(self):
        cfg = get_language_for_file("include/utils.hh")
        assert cfg is not None
        assert cfg.name == "cpp"


# ---------------------------------------------------------------------------
# 19. Class extraction
# ---------------------------------------------------------------------------


class TestCppClass:
    def test_class_basic(self):
        parsed = _parse(
            """\
class Animal {
public:
    int age;
};
""",
            path="src/animal.cpp",
        )
        cls = _entity_by_name(parsed, "Animal")
        assert cls.label == NodeLabel.TYPE_DEF
        assert cls.kind == TypeDefKind.CLASS
        assert cls.visibility == Visibility.PUBLIC

    def test_class_language(self):
        parsed = _parse("class Foo {};\n", path="src/foo.cpp")
        assert parsed.language == "cpp"


# ---------------------------------------------------------------------------
# 20. Namespace handling in qualified names
# ---------------------------------------------------------------------------


class TestCppNamespace:
    def test_namespace_function_qn(self):
        parsed = _parse(
            """\
namespace math {
    int add(int a, int b) {
        return a + b;
    }
}
""",
            path="src/math.cpp",
        )
        func = _entity_by_name(parsed, "add")
        assert func.qualified_name == f"{PROJECT}:src.math.math.add"

    def test_nested_namespace(self):
        parsed = _parse(
            """\
namespace outer {
    namespace inner {
        void work() {}
    }
}
""",
            path="src/ns.cpp",
        )
        func = _entity_by_name(parsed, "work")
        assert func.qualified_name == f"{PROJECT}:src.ns.outer.inner.work"

    def test_namespace_class_qn(self):
        parsed = _parse(
            """\
namespace net {
    class Server {
    public:
        void start() {}
    };
}
""",
            path="src/server.cpp",
        )
        cls = _entity_by_name(parsed, "Server")
        assert cls.qualified_name == f"{PROJECT}:src.server.net.Server"

        method = _entity_by_name(parsed, "start")
        assert method.qualified_name == f"{PROJECT}:src.server.net.Server.start"


# ---------------------------------------------------------------------------
# 21. Access specifier visibility
# ---------------------------------------------------------------------------


class TestCppAccessSpecifiers:
    def test_class_default_private(self):
        """Class members default to PRIVATE before any access specifier."""
        parsed = _parse(
            """\
class Foo {
    int secret;
public:
    int visible;
};
""",
            path="src/foo.cpp",
        )
        secret = _entity_by_name(parsed, "secret")
        assert secret.visibility == Visibility.PRIVATE

        visible = _entity_by_name(parsed, "visible")
        assert visible.visibility == Visibility.PUBLIC

    def test_struct_default_public(self):
        """Struct members default to PUBLIC."""
        parsed = _parse(
            """\
struct Bar {
    int field;
private:
    int hidden;
};
""",
            path="src/bar.cpp",
        )
        field = _entity_by_name(parsed, "field")
        assert field.visibility == Visibility.PUBLIC

        hidden = _entity_by_name(parsed, "hidden")
        assert hidden.visibility == Visibility.PRIVATE

    def test_protected(self):
        parsed = _parse(
            """\
class Base {
protected:
    int value;
};
""",
            path="src/base.cpp",
        )
        val = _entity_by_name(parsed, "value")
        assert val.visibility == Visibility.PROTECTED


# ---------------------------------------------------------------------------
# 22. Constructor / Destructor
# ---------------------------------------------------------------------------


class TestCppConstructorDestructor:
    def test_constructor(self):
        parsed = _parse(
            """\
class Widget {
public:
    Widget() {}
};
""",
            path="src/widget.cpp",
        )
        # There will be both the class and the constructor named "Widget"
        callables = [e for e in parsed.entities if e.label == NodeLabel.CALLABLE and e.name == "Widget"]
        assert len(callables) == 1
        assert callables[0].kind == CallableKind.CONSTRUCTOR

    def test_destructor(self):
        parsed = _parse(
            """\
class Widget {
public:
    ~Widget() {}
};
""",
            path="src/widget.cpp",
        )
        dtor = [e for e in parsed.entities if e.name == "~Widget"]
        assert len(dtor) == 1
        assert dtor[0].kind == CallableKind.DESTRUCTOR


# ---------------------------------------------------------------------------
# 23. Class inheritance -> INHERITS
# ---------------------------------------------------------------------------


class TestCppInheritance:
    def test_single_inheritance(self):
        parsed = _parse(
            """\
class Base {};
class Derived : public Base {};
""",
            path="src/inh.cpp",
        )
        inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
        assert len(inherits) >= 1
        assert any(r.to_name == "Base" for r in inherits)
        assert any(r.from_qualified_name.endswith("Derived") for r in inherits)

    def test_multiple_inheritance(self):
        parsed = _parse(
            """\
class A {};
class B {};
class C : public A, public B {};
""",
            path="src/multi.cpp",
        )
        inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
        c_inherits = [r for r in inherits if r.from_qualified_name.endswith("C")]
        base_names = {r.to_name for r in c_inherits}
        assert "A" in base_names
        assert "B" in base_names


# ---------------------------------------------------------------------------
# 24. Method vs function distinction
# ---------------------------------------------------------------------------


class TestCppMethodVsFunction:
    def test_method_inside_class(self):
        parsed = _parse(
            """\
class Foo {
public:
    void bar() {}
};

void baz() {}
""",
            path="src/mv.cpp",
        )
        bar = _entity_by_name(parsed, "bar")
        assert bar.kind == CallableKind.METHOD
        assert bar.qualified_name == f"{PROJECT}:src.mv.Foo.bar"

        baz = _entity_by_name(parsed, "baz")
        assert baz.kind == CallableKind.FUNCTION
        assert baz.qualified_name == f"{PROJECT}:src.mv.baz"


# ---------------------------------------------------------------------------
# 25. Virtual / override tags
# ---------------------------------------------------------------------------


class TestCppVirtualOverride:
    def test_virtual_tag(self):
        parsed = _parse(
            """\
class Base {
public:
    virtual void draw() {}
};
""",
            path="src/virt.cpp",
        )
        draw = _entity_by_name(parsed, "draw")
        assert "virtual" in draw.tags

    def test_override_tag(self):
        parsed = _parse(
            """\
class Derived : public Base {
public:
    void draw() override {}
};
""",
            path="src/ovr.cpp",
        )
        draw = _entity_by_name(parsed, "draw")
        assert "override" in draw.tags


# ---------------------------------------------------------------------------
# 26. Static file-scope -> PRIVATE visibility
# ---------------------------------------------------------------------------


class TestCppStaticFileScope:
    def test_static_function_private(self):
        parsed = _parse(
            """\
static void helper() {}
""",
            path="src/static_test.cpp",
        )
        func = _entity_by_name(parsed, "helper")
        assert func.visibility == Visibility.PRIVATE
        assert "static" in func.tags

    def test_static_variable_private(self):
        parsed = _parse(
            """\
static int counter = 0;
""",
            path="src/static_var.cpp",
        )
        func = _entity_by_name(parsed, "counter")
        assert func.visibility == Visibility.PRIVATE
        assert "static" in func.tags

    def test_non_static_function_public(self):
        parsed = _parse(
            """\
void public_func() {}
""",
            path="src/pub.cpp",
        )
        func = _entity_by_name(parsed, "public_func")
        assert func.visibility == Visibility.PUBLIC


# ---------------------------------------------------------------------------
# 27. Template declarations
# ---------------------------------------------------------------------------


class TestCppTemplates:
    def test_template_class_and_method(self):
        parsed = _parse(
            """\
template <typename T>
class Box {
public:
    T get() const { return value_; }
private:
    T value_;
};
""",
            path="include/box.hpp",
        )
        cls = _entity_by_name(parsed, "Box")
        assert cls.label == NodeLabel.TYPE_DEF
        assert cls.kind == TypeDefKind.CLASS
        assert cls.qualified_name == f"{PROJECT}:include.box.Box"
        assert "template" in cls.tags

        method = _entity_by_name(parsed, "get")
        assert method.label == NodeLabel.CALLABLE
        assert method.kind == CallableKind.METHOD
        assert method.qualified_name == f"{PROJECT}:include.box.Box.get"

        field = _entity_by_name(parsed, "value_")
        assert field.kind == ValueKind.FIELD

        defines = _rels_from(parsed, "include.box.Box", RelType.DEFINES)
        targets = {r.to_name for r in defines}
        assert f"{PROJECT}:include.box.Box.get" in targets
        assert f"{PROJECT}:include.box.Box.value_" in targets

    def test_template_function(self):
        parsed = _parse(
            """\
template <typename T>
T max_of(T a, T b) { return a > b ? a : b; }
""",
            path="src/algo.cpp",
        )
        func = _entity_by_name(parsed, "max_of")
        assert func.label == NodeLabel.CALLABLE
        assert func.kind == CallableKind.FUNCTION
        assert "template" in func.tags

    def test_out_of_line_template_method(self):
        parsed = _parse(
            """\
template <typename T>
T Box<T>::get() const { return value_; }
""",
            path="src/box.cpp",
        )
        method = _entity_by_name(parsed, "get")
        assert method.label == NodeLabel.CALLABLE
        assert method.kind == CallableKind.METHOD
        # Template arguments are stripped from the scope: Box<T> -> Box
        assert method.qualified_name == f"{PROJECT}:src.box.Box.get"

        rels = [r for r in parsed.relationships if r.rel_type == RelType.DEFINES and r.to_name == method.qualified_name]
        assert len(rels) == 1
        assert rels[0].from_qualified_name == f"{PROJECT}:src.box"
        assert rels[0].properties["parent_type_name"] == "Box"


# ---------------------------------------------------------------------------
# 28. In-class method prototypes
# ---------------------------------------------------------------------------


class TestCppMethodPrototypes:
    def test_prototype_is_callable(self):
        parsed = _parse(
            """\
class Widget {
public:
    void draw() const;
    int n_;
};
""",
            path="include/widget.hpp",
        )
        draw = _entity_by_name(parsed, "draw")
        assert draw.label == NodeLabel.CALLABLE
        assert draw.kind == CallableKind.METHOD
        assert draw.visibility == Visibility.PUBLIC

        n = _entity_by_name(parsed, "n_")
        assert n.label == NodeLabel.VALUE
        assert n.kind == ValueKind.FIELD

        defines = _rels_from(parsed, "include.widget.Widget", RelType.DEFINES)
        targets = {r.to_name for r in defines}
        assert f"{PROJECT}:include.widget.Widget.draw" in targets

    def test_pointer_and_reference_return_prototypes(self):
        parsed = _parse(
            """\
class Widget {
public:
    int* alloc_buffer(int n);
    int& ref_get();
};
""",
            path="include/widget.hpp",
        )
        alloc = _entity_by_name(parsed, "alloc_buffer")
        assert alloc.label == NodeLabel.CALLABLE
        assert alloc.kind == CallableKind.METHOD

        ref = _entity_by_name(parsed, "ref_get")
        assert ref.label == NodeLabel.CALLABLE
        assert ref.kind == CallableKind.METHOD

    def test_pure_virtual_and_static_prototypes(self):
        parsed = _parse(
            """\
class Shape {
public:
    virtual void render() = 0;
    static int count();
};
""",
            path="include/shape.hpp",
        )
        render = _entity_by_name(parsed, "render")
        assert render.label == NodeLabel.CALLABLE
        assert render.kind == CallableKind.METHOD
        assert "virtual" in render.tags

        count = _entity_by_name(parsed, "count")
        assert count.label == NodeLabel.CALLABLE
        assert count.kind == CallableKind.STATIC_METHOD

    def test_function_pointer_field_stays_field(self):
        parsed = _parse(
            """\
class Widget {
public:
    int (*cb)(int);
};
""",
            path="include/widget.hpp",
        )
        cb = _entity_by_name(parsed, "cb")
        assert cb.label == NodeLabel.VALUE
        assert cb.kind == ValueKind.FIELD

    def test_file_scope_pointer_prototype_skipped(self):
        """Pointer-returning free-function prototypes must not become Value variables."""
        parsed = _parse("int* alloc_buffer(int n);\n", path="src/proto.cpp")
        assert not [e for e in parsed.entities if e.name == "alloc_buffer"]


# ---------------------------------------------------------------------------
# 29. Out-of-line method definitions (S5 cross-file member contract)
# ---------------------------------------------------------------------------


class TestCppOutOfLineMethods:
    def test_out_of_line_method_emits_parent_type_name(self):
        parsed = _parse("void Widget::draw() { }\n", path="src/widget.cpp")
        draw = _entity_by_name(parsed, "draw")
        assert draw.label == NodeLabel.CALLABLE
        assert draw.kind == CallableKind.METHOD
        assert draw.qualified_name == f"{PROJECT}:src.widget.Widget.draw"

        rels = [r for r in parsed.relationships if r.rel_type == RelType.DEFINES and r.to_name == draw.qualified_name]
        assert len(rels) == 1
        assert rels[0].from_qualified_name == f"{PROJECT}:src.widget"
        assert rels[0].properties["parent_type_name"] == "Widget"
        # No rel may originate from the fabricated parent uid
        assert all(r.from_qualified_name != f"{PROJECT}:src.widget.Widget" for r in parsed.relationships)

        # Control: in-body methods keep plain uid-matched DEFINES with no parent_type_name
        header = _parse("class Widget {\npublic:\n    void resize() { }\n};\n", path="include/widget.hpp")
        resize = _entity_by_name(header, "resize")
        header_rels = [
            r for r in header.relationships if r.rel_type == RelType.DEFINES and r.to_name == resize.qualified_name
        ]
        assert len(header_rels) == 1
        assert header_rels[0].from_qualified_name == f"{PROJECT}:include.widget.Widget"
        assert "parent_type_name" not in header_rels[0].properties

    def test_out_of_line_constructor(self):
        parsed = _parse("Widget::Widget() { }\n", path="src/widget.cpp")
        ctor = [e for e in parsed.entities if e.label == NodeLabel.CALLABLE and e.name == "Widget"]
        assert len(ctor) == 1
        assert ctor[0].kind == CallableKind.CONSTRUCTOR

        rels = [
            r for r in parsed.relationships if r.rel_type == RelType.DEFINES and r.to_name == ctor[0].qualified_name
        ]
        assert len(rels) == 1
        assert rels[0].from_qualified_name == f"{PROJECT}:src.widget"
        assert rels[0].properties["parent_type_name"] == "Widget"

    def test_out_of_line_method_in_namespace_block(self):
        parsed = _parse(
            """\
namespace mylib {
void Widget::draw() { }
}
""",
            path="src/widget.cpp",
        )
        draw = _entity_by_name(parsed, "draw")
        assert draw.qualified_name == f"{PROJECT}:src.widget.mylib.Widget.draw"

        rels = [r for r in parsed.relationships if r.rel_type == RelType.DEFINES and r.to_name == draw.qualified_name]
        assert len(rels) == 1
        assert rels[0].from_qualified_name == f"{PROJECT}:src.widget"
        assert rels[0].properties["parent_type_name"] == "Widget"


# ---------------------------------------------------------------------------
# 30. Operator overloads
# ---------------------------------------------------------------------------


class TestCppOperatorOverloads:
    def test_in_class_operator_overload(self):
        """An in-class operator overload definition must produce a Callable entity."""
        parsed = _parse(
            """\
class Widget {
public:
    Widget operator+(const Widget& other) const { return *this; }
};
""",
            path="src/widget.cpp",
        )
        op = _entity_by_name(parsed, "operator+")
        assert op.label == NodeLabel.CALLABLE
        assert op.kind == CallableKind.METHOD
        assert op.qualified_name == f"{PROJECT}:src.widget.Widget.operator+"

    def test_out_of_line_operator_overload(self):
        """Control: out-of-line operator overloads already resolved via qualified_identifier."""
        parsed = _parse(
            "Widget Widget::operator+(const Widget& other) const { return *this; }\n",
            path="src/widget.cpp",
        )
        op = _entity_by_name(parsed, "operator+")
        assert op.label == NodeLabel.CALLABLE
        assert op.kind == CallableKind.METHOD
        assert op.qualified_name == f"{PROJECT}:src.widget.Widget.operator+"


# ---------------------------------------------------------------------------
# 31. Nested-scope out-of-line definitions (A::B::f)
# ---------------------------------------------------------------------------


class TestCppNestedScopeOutOfLine:
    def test_nested_scope_out_of_line_method(self):
        parsed = _parse("void Outer::Inner::method() {}\n", path="src/nested.cpp")
        method = _entity_by_name(parsed, "method")
        assert method.label == NodeLabel.CALLABLE
        # Qualified name must be dot-joined only — no '::' leaking from the scope chain
        assert method.qualified_name == f"{PROJECT}:src.nested.Outer.Inner.method"
        assert "::" not in method.qualified_name

        rels = [r for r in parsed.relationships if r.rel_type == RelType.DEFINES and r.to_name == method.qualified_name]
        assert len(rels) == 1
        assert rels[0].from_qualified_name == f"{PROJECT}:src.nested"
        # parent_type_name must be the bare innermost class name, not 'Inner::method'
        # or a '::'-qualified chain
        assert rels[0].properties["parent_type_name"] == "Inner"

    def test_nested_scope_out_of_line_constructor(self):
        """Constructor detection must use the bare innermost class name."""
        parsed = _parse("Outer::Inner::Inner() {}\n", path="src/nested.cpp")
        ctor = [e for e in parsed.entities if e.label == NodeLabel.CALLABLE and e.name == "Inner"]
        assert len(ctor) == 1
        assert ctor[0].kind == CallableKind.CONSTRUCTOR
        assert ctor[0].qualified_name == f"{PROJECT}:src.nested.Outer.Inner.Inner"


# ---------------------------------------------------------------------------
# 32. Preprocessor-conditional regions
# ---------------------------------------------------------------------------


class TestPreprocessorConditionals:
    """tree-sitter has no preprocessor, so ``#ifdef`` does not vanish — it nests.

    Everything between the directive and its ``#endif`` becomes a child of a
    ``preproc_ifdef``/``preproc_if`` node instead of sitting at file scope. A
    walker that only looks at its immediate children therefore misses the
    entire body of any guarded region, which in a header wrapped in an include
    guard is the whole file.
    """

    def test_ifdef_guarded_function_is_an_entity(self):
        parsed = _parse("#ifdef HAVE_POSIX\nvoid reap(void) { waitpid(); }\n#endif\n")
        reap = _entity_by_name(parsed, "reap")
        assert reap.label == NodeLabel.CALLABLE
        assert reap.qualified_name == f"{PROJECT}:src.example.reap"

    def test_include_guard_does_not_hide_the_whole_file(self):
        parsed = _parse(
            "#ifndef UTIL_H\n#define UTIL_H\nvoid helper(void) { inner(); }\n#endif\n",
            path="include/util.h",
        )
        helper = _entity_by_name(parsed, "helper")
        assert helper.label == NodeLabel.CALLABLE
        calls = _rels_from(parsed, "include.util.helper", RelType.CALLS)
        assert [r.to_name for r in calls] == ["inner"]

    def test_both_arms_of_an_if_else_are_indexed(self):
        """Without a preprocessor there is no way to know which arm the build picks."""
        parsed = _parse("#if FMT_USE_INT128\nvoid alpha(void) {}\n#else\nvoid beta(void) {}\n#endif\n")
        names = {e.name for e in parsed.entities if e.label == NodeLabel.CALLABLE}
        assert names == {"alpha", "beta"}

    def test_elif_arm_is_indexed(self):
        parsed = _parse("#if A\nvoid alpha(void) {}\n#elif B\nvoid gamma(void) {}\n#endif\n")
        assert _entity_by_name(parsed, "gamma").label == NodeLabel.CALLABLE

    def test_ifdef_inside_a_class_body_keeps_class_scope(self):
        parsed = _parse(
            "class Widget {\n public:\n#ifdef _WIN32\n  void win_only() { native(); }\n#endif\n};\n",
            path="src/widget.cpp",
        )
        method = _entity_by_name(parsed, "win_only")
        assert method.kind == CallableKind.METHOD
        assert method.qualified_name == f"{PROJECT}:src.widget.Widget.win_only"
        assert method.visibility == Visibility.PUBLIC


# ---------------------------------------------------------------------------
# 33. extern "C" linkage blocks
# ---------------------------------------------------------------------------


class TestLinkageSpecification:
    def test_extern_c_block_function_is_an_entity(self):
        parsed = _parse(
            'extern "C" {\nint c_api(int x) { return helper(x); }\n}\n',
            path="src/api.cpp",
        )
        fn = _entity_by_name(parsed, "c_api")
        assert fn.label == NodeLabel.CALLABLE
        assert fn.qualified_name == f"{PROJECT}:src.api.c_api"
        assert [r.to_name for r in _rels_from(parsed, "src.api.c_api", RelType.CALLS)] == ["helper"]

    def test_extern_c_single_declaration_still_works(self):
        parsed = _parse('extern "C" void bare(void) { go(); }\n', path="src/api.cpp")
        assert _entity_by_name(parsed, "bare").label == NodeLabel.CALLABLE


# ---------------------------------------------------------------------------
# 34. friend declarations
# ---------------------------------------------------------------------------


class TestFriendDeclaration:
    def test_friend_function_defined_in_class_belongs_to_the_enclosing_scope(self):
        """A friend is found by ADL — ``Point::distance`` does not name it."""
        parsed = _parse(
            "struct Point {\n  friend auto distance(Point p) -> double { return compute(p); }\n};\n",
            path="src/point.cpp",
        )
        fn = _entity_by_name(parsed, "distance")
        assert fn.label == NodeLabel.CALLABLE
        assert fn.qualified_name == f"{PROJECT}:src.point.distance"
        assert "Point" not in fn.qualified_name
        assert [r.to_name for r in _rels_from(parsed, "src.point.distance", RelType.CALLS)] == ["compute"]


# ---------------------------------------------------------------------------
# 35. Conversion operators
# ---------------------------------------------------------------------------


class TestConversionOperators:
    """``operator_cast`` names itself with a type, and its ``declarator`` field
    points at the (necessarily nameless) parameter list — so the generic
    declarator descent walks straight past the name and returns None."""

    def test_conversion_operator_is_a_named_method(self):
        parsed = _parse(
            "class Handle {\n public:\n  explicit operator bool() const { return valid(); }\n};\n",
            path="src/handle.cpp",
        )
        op = _entity_by_name(parsed, "operator bool")
        assert op.label == NodeLabel.CALLABLE
        assert op.kind == CallableKind.METHOD
        assert op.qualified_name == f"{PROJECT}:src.handle.Handle.operator bool"

    def test_conversion_operator_to_a_qualified_template_type(self):
        parsed = _parse(
            "class View {\n public:\n  operator std::basic_string_view<char>() const { return {}; }\n};\n",
            path="src/view.cpp",
        )
        assert _entity_by_name(parsed, "operator std::basic_string_view<char>").label == NodeLabel.CALLABLE

    def test_out_of_line_conversion_operator_name_excludes_the_parameter_list(self):
        parsed = _parse("Handle::operator bool() const { return true; }\n", path="src/handle.cpp")
        op = _entity_by_name(parsed, "operator bool")
        assert op.qualified_name == f"{PROJECT}:src.handle.Handle.operator bool"
        assert "(" not in op.name


# ---------------------------------------------------------------------------
# 36. Types nested inside a class body
# ---------------------------------------------------------------------------


class TestNestedTypeInClassBody:
    """At file scope a type definition is wrapped in a ``declaration``; inside a
    class body it is wrapped in a ``field_declaration`` instead, which needs the
    same unwrapping or the nested type and every method on it disappears."""

    def test_nested_struct_and_its_methods_are_captured(self):
        parsed = _parse(
            "class Outer {\n  struct Inner {\n    void go() { work(); }\n  };\n};\n",
            path="src/outer.cpp",
        )
        inner = _entity_by_name(parsed, "Inner")
        assert inner.label == NodeLabel.TYPE_DEF
        assert inner.kind == TypeDefKind.STRUCT
        assert inner.qualified_name == f"{PROJECT}:src.outer.Outer.Inner"

        go = _entity_by_name(parsed, "go")
        assert go.qualified_name == f"{PROJECT}:src.outer.Outer.Inner.go"
        assert [r.to_name for r in _rels_from(parsed, "src.outer.Outer.Inner.go", RelType.CALLS)] == ["work"]

    def test_nested_struct_is_not_also_emitted_as_a_field(self):
        parsed = _parse("class Outer {\n  struct Inner {\n    int x;\n  };\n};\n", path="src/outer.cpp")
        assert [e.label for e in parsed.entities if e.name == "Inner"] == [NodeLabel.TYPE_DEF]


# ---------------------------------------------------------------------------
# 37. Call attribution to the nearest enclosing named scope (ADR-0031)
# ---------------------------------------------------------------------------


class TestCallAttribution:
    def test_module_scope_call_attributes_to_the_module(self):
        parsed = _parse("int g = compute(1);\n")
        assert [r.to_name for r in _rels_from(parsed, "src.example", RelType.CALLS)] == ["compute"]

    def test_call_inside_a_lambda_attributes_to_the_enclosing_function(self):
        parsed = _parse("void run() {\n  auto f = [] { inner(); };\n}\n", path="src/run.cpp")
        assert [r.to_name for r in _rels_from(parsed, "src.run.run", RelType.CALLS)] == ["inner"]

    def test_call_inside_a_file_scope_lambda_attributes_to_the_module(self):
        parsed = _parse("auto handler = [] { setup(); };\n", path="src/run.cpp")
        assert [r.to_name for r in _rels_from(parsed, "src.run", RelType.CALLS)] == ["setup"]

    def test_call_inside_a_local_class_method_attributes_to_the_enclosing_function(self):
        """A local class gets no entity, so ADR-0031 sends its calls up to ``outer``."""
        parsed = _parse(
            "void outer() {\n  struct Local {\n    void go() { deep(); }\n  };\n}\n",
            path="src/local.cpp",
        )
        assert [r.to_name for r in _rels_from(parsed, "src.local.outer", RelType.CALLS)] == ["deep"]

    def test_field_initializer_call_reaches_the_graph(self):
        parsed = _parse("struct S {\n  int n = build();\n};\n", path="src/s.cpp")
        assert [r.to_name for r in _rels_from(parsed, "src.s", RelType.CALLS)] == ["build"]

    def test_a_macro_hidden_scope_is_recovered_and_owns_its_calls(self):
        """Superseded ATL-096's module-fallback expectation for this shape.

        ``FMT_BEGIN_NAMESPACE`` is not expandable, so tree-sitter used to mis-parse the
        specialisation that follows and leave its body as a bare block at file scope --
        and the honest answer then was to attribute the call to the module. ATL-143's
        shim recovers the scope, so the honest answer is now the method itself.
        """
        parsed = _parse(
            "FMT_BEGIN_NAMESPACE\n"
            "template <> struct formatter<my_type> {\n"
            "  auto format() -> int { return copy(); }\n"
            "};\n"
            "FMT_END_NAMESPACE\n",
            path="src/dbg.cpp",
        )
        assert [r.to_name for r in _rels_from(parsed, "src.dbg.formatter<my_type>.format", RelType.CALLS)] == ["copy"]
        assert _rels_from(parsed, "src.dbg", RelType.CALLS) == []

    def test_a_call_with_no_enclosing_callable_still_reaches_the_module(self):
        """The module fallback is not gone, only less often needed for macro debris.
        A file-scope initialiser has no enclosing callable at all, macros or not."""
        parsed = _parse("int x = compute();\n", path="src/low.cpp")
        assert [r.to_name for r in _rels_from(parsed, "src.low", RelType.CALLS)] == ["compute"]

    def test_a_call_is_emitted_exactly_once(self):
        """The structural walk and the call walk must not both claim a subtree."""
        parsed = _parse("void run() {\n  helper();\n}\n", path="src/run.cpp")
        assert [r.to_name for r in parsed.relationships if r.rel_type == RelType.CALLS] == ["helper"]

    def test_inline_struct_definition_calls_are_not_double_counted(self):
        parsed = _parse("struct S {\n  void m() { work(); }\n} instance;\n", path="src/s.cpp")
        calls = [r for r in parsed.relationships if r.rel_type == RelType.CALLS]
        assert [(r.from_qualified_name, r.to_name) for r in calls] == [(f"{PROJECT}:src.s.S.m", "work")]


# ---------------------------------------------------------------------------
# 38. Callee shapes
# ---------------------------------------------------------------------------


class TestCalleeShapes:
    def test_template_function_call_uses_the_template_name(self):
        parsed = _parse("void caller() {\n  auto v = max_value<int>();\n}\n", path="src/c.cpp")
        assert [r.to_name for r in _rels_from(parsed, "src.c.caller", RelType.CALLS)] == ["max_value"]

    def test_named_casts_are_not_calls(self):
        """``static_cast<T>(x)`` is shaped exactly like a call but is a keyword.

        Negative assertion on purpose: this suppresses output that the
        template_function handling above would otherwise produce, so a positive
        assertion elsewhere could never show the suppression had stopped working.
        """
        parsed = _parse(
            "void caller(void* p) {\n"
            "  auto a = static_cast<int>(1);\n"
            "  auto b = reinterpret_cast<char*>(p);\n"
            "  auto c = const_cast<int*>(p);\n"
            "  auto d = dynamic_cast<int*>(p);\n"
            "}\n",
            path="src/c.cpp",
        )
        assert [r.to_name for r in _rels_from(parsed, "src.c.caller", RelType.CALLS)] == []

    def test_parenthesized_callee_is_unwrapped(self):
        """``(T::min)()`` — the idiom for dodging the Windows min/max macros.

        Only the no-argument spelling reaches here: with arguments,
        ``(std::min)(a, b)`` is genuinely ambiguous with a C-style cast, and the
        grammar resolves it to ``cast_expression`` rather than to a call.
        """
        parsed = _parse("int caller() {\n  return (T::min)();\n}\n", path="src/c.cpp")
        assert [r.to_name for r in _rels_from(parsed, "src.c.caller", RelType.CALLS)] == ["T::min"]

    def test_callee_with_no_static_name_is_not_a_call(self):
        parsed = _parse("void caller() {\n  handlers[i]();\n}\n", path="src/c.cpp")
        assert [r.to_name for r in _rels_from(parsed, "src.c.caller", RelType.CALLS)] == []


# ---------------------------------------------------------------------------
# 39. Function-like macros that parse as function definitions
# ---------------------------------------------------------------------------


class TestGtestCaseNaming:
    """``TEST(Suite, Case) { ... }`` is a macro, but the grammar cannot tell it
    from a function definition whose name is ``TEST``.

    Left alone, every case in a file emits the same qualified name and upserts
    into one graph node carrying an arbitrary body and the union of every
    case's edges. fmt's base-test.cc alone had 47 of them. The macro arguments
    are the real name, and they are stable across edits in a way a line number
    would not be — ``Suite.Case`` is also exactly what gtest's own runner
    prints and what ``--gtest_filter`` accepts.
    """

    def test_gtest_case_is_named_from_its_macro_arguments(self):
        parsed = _parse("TEST(FormatTest, Escape) {\n  check();\n}\n", path="test/format-test.cc")
        case = _entity_by_name(parsed, "FormatTest.Escape")
        assert case.label == NodeLabel.CALLABLE
        assert case.qualified_name == f"{PROJECT}:test.format-test.FormatTest.Escape"
        assert "test" in case.tags
        assert [r.to_name for r in _rels_from(parsed, "FormatTest.Escape", RelType.CALLS)] == ["check"]

    def test_no_entity_is_named_after_the_macro(self):
        """Negative assertion: a positive one cannot catch a collision.

        Two cases in one file both being present says nothing — the failure
        being guarded is that they share a name, which only shows up as the
        *absence* of the macro-named node and the presence of two distinct ones.
        """
        parsed = _parse(
            "TEST(FormatTest, Escape) {\n  a();\n}\nTEST(FormatTest, Width) {\n  b();\n}\n",
            path="test/format-test.cc",
        )
        assert [e.name for e in parsed.entities if e.name == "TEST"] == []
        names = {e.name for e in parsed.entities if e.label == NodeLabel.CALLABLE}
        assert names == {"FormatTest.Escape", "FormatTest.Width"}

    def test_two_cases_in_one_file_get_distinct_qualified_names(self):
        parsed = _parse(
            "TEST(FormatTest, Escape) {\n  a();\n}\nTEST(FormatTest, Width) {\n  b();\n}\n",
            path="test/format-test.cc",
        )
        qns = [e.qualified_name for e in parsed.entities if e.label == NodeLabel.CALLABLE]
        assert len(qns) == len(set(qns)), f"colliding uids: {qns}"

    @pytest.mark.parametrize("macro", ["TEST", "TEST_F", "TEST_P", "TYPED_TEST", "TYPED_TEST_P"])
    def test_every_gtest_case_macro_is_named_from_its_arguments(self, macro):
        parsed = _parse(f"{macro}(SuiteName, CaseName) {{\n  go();\n}}\n", path="test/x-test.cc")
        assert _entity_by_name(parsed, "SuiteName.CaseName").label == NodeLabel.CALLABLE

    def test_a_stray_macro_before_the_case_does_not_hide_it(self):
        """fmt closes every file with ``FMT_END_NAMESPACE``, which the grammar
        absorbs as the following definition's *return type* — so the case no
        longer looks like a definition with no return type at all."""
        parsed = _parse(
            "FMT_END_NAMESPACE\nTEST(FormatTest, Escape) {\n  check();\n}\n",
            path="test/format-test.cc",
        )
        assert _entity_by_name(parsed, "FormatTest.Escape").label == NodeLabel.CALLABLE
        assert [e.name for e in parsed.entities if e.name == "TEST"] == []


class TestUnnameableMacroInvocations:
    """A function-like macro with no arguments worth reading gets no entity.

    ``FMT_CATCH(...) { ... }`` and gtest's ``GTEST_LOCK_EXCLUDED_(mu) { ... }``
    parse as definitions named after the macro. There is no sound name to give
    them, so they follow the same rule as a function stranded behind a
    mis-parse: emit nothing rather than something confidently wrong.
    """

    def test_macro_invocation_gets_no_entity(self):
        parsed = _parse("FMT_CATCH(...) {\n  report();\n}\n", path="src/fmt-c.cc")
        assert [e.name for e in parsed.entities if e.name == "FMT_CATCH"] == []
        assert [e.name for e in parsed.entities if e.label == NodeLabel.CALLABLE] == []

    def test_macro_invocation_body_still_reports_its_calls(self):
        """Suppressing the entity must not suppress the work it wraps."""
        parsed = _parse("FMT_CATCH(...) {\n  report();\n}\n", path="src/fmt-c.cc")
        assert [r.to_name for r in _rels_from(parsed, "src.fmt-c", RelType.CALLS)] == ["report"]

    def test_two_invocations_of_the_same_macro_do_not_collide(self):
        parsed = _parse(
            "FMT_CATCH(...) {\n  a();\n}\nFMT_CATCH(...) {\n  b();\n}\n",
            path="src/fmt-c.cc",
        )
        qns = [e.qualified_name for e in parsed.entities if e.label == NodeLabel.CALLABLE]
        assert qns == []
        assert {r.to_name for r in _rels_from(parsed, "src.fmt-c", RelType.CALLS)} == {"a", "b"}


class TestMacroDiscriminatorDoesNotEatRealCode:
    """The three C++ forms that legitimately have no return type must survive."""

    def test_in_class_constructor_survives(self):
        parsed = _parse("class Widget {\n public:\n  Widget() { init(); }\n};\n", path="src/w.cpp")
        # `Widget` names both the class TypeDef and its constructor.
        ctors = [e for e in parsed.entities if e.name == "Widget" and e.label == NodeLabel.CALLABLE]
        assert [e.kind for e in ctors] == [CallableKind.CONSTRUCTOR]
        assert [r.to_name for r in _rels_from(parsed, "src.w.Widget.Widget", RelType.CALLS)] == ["init"]

    def test_out_of_line_constructor_survives(self):
        parsed = _parse("Widget::Widget() { init(); }\n", path="src/w.cpp")
        assert _entity_by_name(parsed, "Widget").kind == CallableKind.CONSTRUCTOR

    def test_destructor_survives(self):
        parsed = _parse("class Widget {\n public:\n  ~Widget() { drop(); }\n};\n", path="src/w.cpp")
        assert _entity_by_name(parsed, "~Widget").kind == CallableKind.DESTRUCTOR

    def test_conversion_operator_survives(self):
        parsed = _parse("class H {\n public:\n  operator bool() { return ok(); }\n};\n", path="src/h.cpp")
        assert _entity_by_name(parsed, "operator bool").label == NodeLabel.CALLABLE

    def test_a_real_function_named_like_a_test_macro_is_untouched(self):
        """A return type means it is a function, whatever it is called."""
        parsed = _parse("int TEST(int a, int b) {\n  return add(a, b);\n}\n", path="src/t.cpp")
        assert _entity_by_name(parsed, "TEST").label == NodeLabel.CALLABLE


# ---------------------------------------------------------------------------
# 40. .h is shared between C and C++ — routed by content
# ---------------------------------------------------------------------------

_C_HEADER = b"""\
#ifndef UTIL_H
#define UTIL_H
#ifdef __cplusplus
extern "C" {
#endif
struct point { int x; int y; };
int add(int a, int b);
#ifdef __cplusplus
}
#endif
#endif
"""

_CPP_HEADER = b"""\
#ifndef WIDGET_H
#define WIDGET_H
namespace app {
class Widget {
 public:
  void draw() const { paint(); }
};
}  // namespace app
#endif
"""


def _dialect(path: str, source: bytes | None = None) -> str:
    """Which grammar does the router pick for this file?"""
    cfg = get_language_for_file(path, source) if source is not None else get_language_for_file(path)
    assert cfg is not None, f"no language registered for {path}"
    return cfg.name


class TestHeaderDialectRouting:
    """``.h`` is the standard C header extension *and* what most C++ projects
    call their headers. Routing it to C unconditionally left 23 of fmt's 25
    headers unparseable; routing it to C++ unconditionally would put the risk on
    C users, who are the status quo. So the content decides, and anything
    undecidable stays C.
    """

    def test_cpp_header_routes_to_the_cpp_grammar(self):
        assert _dialect("include/widget.h", _CPP_HEADER) == "cpp"

    def test_c_header_stays_on_the_c_grammar(self):
        """`extern "C"` is a C-header idiom, not a C++ marker — 216 of CPython's
        283 headers use it."""
        assert _dialect("include/util.h", _C_HEADER) == "c"

    def test_cpp_header_reports_itself_as_cpp(self):
        """The grammar and the recorded language have to agree, or the walker's
        C++ branches stay off for a file parsed as C++."""
        parsed = _parse(_CPP_HEADER.decode(), path="include/widget.h")
        assert parsed.language == "cpp"

    def test_c_header_still_reports_itself_as_c(self):
        parsed = _parse(_C_HEADER.decode(), path="include/util.h")
        assert parsed.language == "c"

    def test_cpp_header_yields_its_namespaced_members(self):
        """The point of the routing: under the C grammar none of this parses."""
        parsed = _parse(_CPP_HEADER.decode(), path="include/widget.h")
        draw = _entity_by_name(parsed, "draw")
        assert draw.qualified_name == f"{PROJECT}:include.widget.app.Widget.draw"
        assert [r.to_name for r in _rels_from(parsed, "app.Widget.draw", RelType.CALLS)] == ["paint"]

    def test_c_header_extraction_is_unchanged(self):
        parsed = _parse(_C_HEADER.decode(), path="include/util.h")
        assert _entity_by_name(parsed, "point").label == NodeLabel.TYPE_DEF

    # -- ambiguity resolves to C ------------------------------------------

    def test_a_comment_mentioning_class_does_not_flip_a_c_header(self):
        source = b"// Former class object interface\n/* a C++ class would go here */\nint f(void);\n"
        assert _dialect("include/note.h", source) == "c"

    def test_a_string_literal_mentioning_namespace_does_not_flip_a_c_header(self):
        source = b'static const char *msg = "namespace foo::bar";\nint f(void);\n'
        assert _dialect("include/msg.h", source) == "c"

    def test_cpp_words_used_as_c_identifiers_do_not_flip_a_header(self):
        """C reserves none of these, and CPython really does ship
        `namespaceSeparator` and `class_id` style fields."""
        source = b"struct s {\n  char *namespaceSeparator;\n  int class_id;\n  int template_count;\n};\n"
        assert _dialect("include/ident.h", source) == "c"

    def test_an_empty_header_stays_on_c(self):
        assert _dialect("include/empty.h", b"") == "c"

    # -- the mechanism ----------------------------------------------------

    def test_source_is_read_from_disk_when_the_caller_supplies_none(self, tmp_path):
        """The answer must not depend on who is asking — a caller that passes no
        source still has to agree with `parse_file`, or a measurement built on
        one and entities built on the other silently disagree."""
        header = tmp_path / "widget.h"
        header.write_bytes(_CPP_HEADER)
        assert _dialect(str(header)) == "cpp"

    def test_an_unreadable_path_falls_back_to_c(self, tmp_path):
        assert _dialect(str(tmp_path / "missing.h")) == "c"

    def test_unambiguous_extensions_are_not_sniffed(self):
        """`.hpp`/`.cpp`/`.c` say what they are; content must not override."""
        assert _dialect("include/w.hpp", _C_HEADER) == "cpp"
        assert _dialect("src/m.c", _CPP_HEADER) == "c"


# ---------------------------------------------------------------------------
# Overload uids (ADR-0032)
# ---------------------------------------------------------------------------

_OVERLOAD_SOURCE = """
#include <string>

namespace detail {
auto read(int v) -> int { return v; }
auto read(const std::string& s) -> int { return 0; }
template <typename T> auto read(T& v) -> int { return 0; }
auto only_once(int a) -> int { return a; }
}  // namespace detail

class Widget {
 public:
  Widget() {}
  Widget(int w) {}
  Widget(const Widget& other) {}
  Widget(Widget&& other) {}
  void draw() const {}
  auto value() & -> int { return 0; }
  auto value() const& -> int { return 0; }
};

#ifdef _WIN32
void probe(int a) {}
#else
void probe(double a) {}
#endif

template <typename... T>
void pack(T&... args) {}
void pack(int a) {}
"""


_OUT_OF_LINE_SOURCE = """
class Widget {
 public:
  void resize(int w, int h = 10);
  void resize(int w);
  void dup2(int fd);
  void dup2(int fd, std::error_code& ec);
};

void Widget::resize(int w, int h) {}
void Widget::dup2(int fd) {}
void Widget::dup2(int fd, std::error_code& ec) {}
"""


def _callable_uids(parsed: ParsedFile) -> list[str]:
    return [e.qualified_name for e in parsed.entities if e.label == NodeLabel.CALLABLE]


def _uid_ending(parsed: ParsedFile, suffix: str) -> str:
    matches = [u for u in _callable_uids(parsed) if u.endswith(suffix)]
    assert len(matches) == 1, f"expected exactly one uid ending {suffix!r}, got {matches}"
    return matches[0]


class TestOverloadUids:
    """C++ permits two definitions of one name in one scope, so a name alone
    cannot be a uid (ADR-0032). An overloaded name takes its signature into the
    qualified name; a name declared once keeps the uid it always had, which is
    what bounds the churn to what was already ambiguous.
    """

    def test_no_two_definitions_share_a_uid(self):
        """The load-bearing assertion, and it has to be a negative one: a
        positive check that some expected uid is present cannot notice that a
        second definition claimed it too and merged into the same graph node.
        """
        uids = _callable_uids(_parse(_OVERLOAD_SOURCE, path="src/example.cpp"))
        assert len(uids) == len(set(uids)), sorted(u for u in uids if uids.count(u) > 1)

    def test_a_name_declared_once_keeps_its_plain_uid(self):
        """The churn bound. Suffixing every callable would rewrite the uid of
        every unambiguous function in every C++ file for no gain, so a name
        declared once has to come out exactly as it did before.
        """
        uids = _callable_uids(_parse(_OVERLOAD_SOURCE, path="src/example.cpp"))
        assert f"{PROJECT}:src.example.detail.only_once" in uids
        assert f"{PROJECT}:src.example.Widget.draw" in uids

    def test_overloaded_constructors_get_one_uid_each(self):
        """The largest group in real C++ — 9 `basic_scan_arg` constructors in
        fmt's scan.h alone. They are named for their class, so they collide with
        each other rather than with anything else.
        """
        uids = _callable_uids(_parse(_OVERLOAD_SOURCE, path="src/example.cpp"))
        assert f"{PROJECT}:src.example.Widget.Widget()" in uids
        assert f"{PROJECT}:src.example.Widget.Widget(int)" in uids
        assert f"{PROJECT}:src.example.Widget.Widget(constWidget&)" in uids
        assert f"{PROJECT}:src.example.Widget.Widget(Widget&&)" in uids

    def test_overload_scope_is_not_only_the_class_body(self):
        """`detail::read` and the file-scope `pack` are namespace- and
        translation-unit-level overload sets. A rule that only looked at class
        members would leave both merged.
        """
        uids = _callable_uids(_parse(_OVERLOAD_SOURCE, path="src/example.cpp"))
        assert f"{PROJECT}:src.example.detail.read(int)" in uids
        assert f"{PROJECT}:src.example.detail.read(string&)" in uids
        assert f"{PROJECT}:src.example.pack(int)" in uids

    def test_an_ifdef_split_overload_set_is_still_one_set(self):
        """Both arms are walked because the build configuration is unknown, so
        the two `probe` definitions are siblings for naming purposes even though
        the grammar nests each under its own `preproc_ifdef`.
        """
        uids = _callable_uids(_parse(_OVERLOAD_SOURCE, path="src/example.cpp"))
        assert f"{PROJECT}:src.example.probe(int)" in uids
        assert f"{PROJECT}:src.example.probe(double)" in uids

    def test_template_parameters_separate_identical_signatures(self):
        """fmt's base-test.cc is two zero-argument `test_value` templates told
        apart solely by an `enable_if_t` in the template header. Parameters
        alone would leave them merged.
        """
        parsed = _parse(
            """
template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
auto test_value() -> T { return T(); }
template <typename T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
auto test_value() -> T { return T(); }
""",
            path="src/example.cpp",
        )
        uids = _callable_uids(parsed)
        assert len(uids) == len(set(uids)), uids
        assert any("is_integral" in u for u in uids)
        assert any("is_floating_point" in u for u in uids)

    def test_a_parameter_qualifier_is_part_of_the_suffix(self):
        """`f(int)` and `f(const int&)` are different overloads, and neither the
        `const` nor the `&` is inside the parameter's `type` field — one is a
        sibling qualifier, the other lives in the declarator.
        """
        parsed = _parse("void f(int v) {}\nvoid f(const int& v) {}\nvoid f(int* v) {}\n", path="src/example.cpp")
        assert sorted(_callable_uids(parsed)) == sorted(
            [
                f"{PROJECT}:src.example.f(int)",
                f"{PROJECT}:src.example.f(constint&)",
                f"{PROJECT}:src.example.f(int*)",
            ]
        )

    def test_trailing_cv_and_ref_qualifiers_separate_overloads(self):
        """fmt's ranges-test.cc declares `value() &`, `value() const&`,
        `value() &&` and `value() const&&` on one type — four definitions with
        identical parameter lists.
        """
        uids = _callable_uids(_parse(_OVERLOAD_SOURCE, path="src/example.cpp"))
        assert f"{PROJECT}:src.example.Widget.value()&" in uids
        assert f"{PROJECT}:src.example.Widget.value()const&" in uids

    def test_the_suffix_carries_no_dot(self):
        """A dot separates scope segments in a qualified name, so one inside the
        suffix would manufacture a scope that does not exist. A namespace
        qualifier and a parameter pack are the two things that would leave one.
        """
        uids = _callable_uids(_parse(_OVERLOAD_SOURCE, path="src/example.cpp"))
        for uid in uids:
            assert "." not in uid.partition("(")[2], uid
        assert f"{PROJECT}:src.example.pack<typename[]T>(T&[])" in uids

    def test_a_default_argument_does_not_reach_the_suffix(self):
        """A declaration spells the default and its out-of-line definition does
        not, so keeping it would put two different suffixes on one function.
        """
        uids = _callable_uids(_parse(_OUT_OF_LINE_SOURCE, path="src/example.cpp"))
        assert f"{PROJECT}:src.example.Widget.resize(int,int)" in uids
        assert f"{PROJECT}:src.example.Widget.resize(int)" in uids
        assert not any("10" in u for u in uids)

    def test_out_of_line_definitions_are_weighed_against_their_own_scope(self):
        """A known limit, pinned rather than claimed correct.

        The overload set is per scope, and a class body and the file scope
        holding an out-of-line definition are two of them. ``Widget::resize``
        defined once outside the class therefore keeps its plain uid even though
        the class declares two, so it no longer merges with its own declaration.
        Nothing is lost — both nodes exist and carry their own edges, and the
        three-way merge this replaced was worse — but a declaration and its
        definition are two nodes where they used to be one.

        Closing it needs the set keyed per *file* rather than per scope, which
        means walking every function body twice. Two out-of-line definitions of
        one name in one file are still weighed against each other, which is the
        shape that actually occurs (``file::dup2`` in fmt's src/os.cc).
        """
        uids = _callable_uids(_parse(_OUT_OF_LINE_SOURCE, path="src/example.cpp"))
        assert f"{PROJECT}:src.example.Widget.resize" in uids
        assert f"{PROJECT}:src.example.Widget.dup2(int)" in uids
        assert f"{PROJECT}:src.example.Widget.dup2(int,error_code&)" in uids

    def test_defines_points_at_the_suffixed_uid(self):
        """The entity and its DEFINES edge have to agree, or the parent's edge
        dangles at a uid nothing ever emitted.
        """
        parsed = _parse(_OVERLOAD_SOURCE, path="src/example.cpp")
        defined = {r.to_name for r in parsed.relationships if r.rel_type == RelType.DEFINES}
        assert _uid_ending(parsed, "Widget.Widget(int)") in defined
        assert _uid_ending(parsed, "detail.read(int)") in defined

    def test_the_display_name_stays_unsuffixed(self):
        """The suffix disambiguates the uid, not what a reader searches for."""
        parsed = _parse(_OVERLOAD_SOURCE, path="src/example.cpp")
        reads = [e for e in parsed.entities if e.qualified_name.rsplit(".", 1)[-1].startswith("read(")]
        assert len(reads) == 2
        assert {e.name for e in reads} == {"read"}


class TestMacroShim:
    """ATL-143. tree-sitter has no preprocessor, so a bare macro standing where a
    keyword belongs collapses a file into one ERROR node."""

    def test_a_macro_hidden_namespace_yields_its_contents(self):
        parsed = _parse(
            "FMT_BEGIN_NAMESPACE\n"
            "struct color_type {\n"
            "  auto value() const -> int { return v_; }\n"
            "};\n"
            "FMT_END_NAMESPACE\n",
            path="src/c.cpp",
        )
        names = {e.name for e in parsed.entities}
        assert {"color_type", "value"} <= names

    def test_a_clean_file_is_not_shimmed(self):
        """A file the grammar handles must be byte-identical to before: the shim is a
        recovery path, not a preprocessing step."""
        source = "namespace ns {\nstruct S {\n  int m() { return f(); }\n};\n}\n"
        parsed = _parse(source, path="src/clean.cpp")
        assert {e.qualified_name for e in parsed.entities} == {
            "test_project:src.clean",
            "test_project:src.clean.ns.S",
            "test_project:src.clean.ns.S.m",
        }

    def test_a_macro_inside_a_comment_is_left_alone(self):
        """Blanking prose to help the parser trades a parse problem for a retrieval one."""
        parsed = _parse(
            "FMT_BEGIN_NAMESPACE\n"
            "/// Uses FMT_API for export.\n"
            "struct S { int m() { return 1; } };\n"
            "FMT_END_NAMESPACE\n",
            path="src/doc.cpp",
        )
        docs = " ".join(e.docstring or "" for e in parsed.entities)
        assert "FMT_API" in docs

    def test_a_string_literal_is_left_alone(self):
        parsed = _parse(
            'FMT_BEGIN_NAMESPACE\nstruct S { const char* m() { return "ERROR_CODE"; } };\nFMT_END_NAMESPACE\n',
            path="src/str.cpp",
        )
        sources = " ".join(e.source or "" for e in parsed.entities)
        assert "ERROR_CODE" in sources

    def test_recovered_source_is_the_original_not_the_blanked_text(self):
        """The shimmed tree is used for STRUCTURE only. An agent reading a recovered
        entity's source must not see gaps where the code says FMT_CONSTEXPR."""
        parsed = _parse(
            "FMT_BEGIN_NAMESPACE\nstruct S {\n  FMT_CONSTEXPR int m() { return 1; }\n};\nFMT_END_NAMESPACE\n",
            path="src/src.cpp",
        )
        method = next(e for e in parsed.entities if e.name == "m")
        assert "FMT_CONSTEXPR" in (method.source or "")

    def test_line_numbers_survive_the_shim(self):
        """Length-preserving is the whole trick: an entity found in the shimmed tree has
        to point at the real file."""
        parsed = _parse(
            "FMT_BEGIN_NAMESPACE\n\nstruct S {\n  int m() { return 1; }\n};\nFMT_END_NAMESPACE\n",
            path="src/lines.cpp",
        )
        struct = next(e for e in parsed.entities if e.name == "S")
        assert struct.line_start == 3
