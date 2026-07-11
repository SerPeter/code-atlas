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
        assert "<stdio.h>" in imported

    def test_local_include(self):
        parsed = _parse('#include "local.h"\n')
        imports = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
        imported = {r.to_name for r in imports}
        assert '"local.h"' in imported


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
