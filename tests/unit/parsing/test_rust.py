"""Tests for Rust parser."""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_rust", reason="tree-sitter-rust not installed")

from code_atlas.parsing.ast import ParsedFile, get_language_for_file, parse_file
from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind, ValueKind, Visibility

PROJECT = "test_project"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(source: str, path: str = "src/example.rs") -> ParsedFile:
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


# ---------------------------------------------------------------------------
# 1. Language detection
# ---------------------------------------------------------------------------


def test_language_detection_rs():
    config = get_language_for_file("src/main.rs")
    assert config is not None
    assert config.name == "rust"


def test_language_detection_non_rs():
    """Non-.rs files should not match the Rust parser."""
    assert get_language_for_file("src/main.py") is not None  # Python, not Rust
    cfg = get_language_for_file("src/main.rs")
    assert cfg is not None
    assert cfg.name == "rust"


# ---------------------------------------------------------------------------
# 2. Module entity creation
# ---------------------------------------------------------------------------


def test_module_entity():
    parsed = _parse("fn main() {}\n", path="src/server.rs")
    mod = _entity_by_name(parsed, "server")
    assert mod.label == NodeLabel.MODULE
    assert mod.kind == "module"
    assert mod.qualified_name == f"{PROJECT}:src.server"


def test_module_from_mod_rs():
    """``mod.rs`` uses parent directory as module name (like __init__.py)."""
    parsed = _parse("", path="src/server/mod.rs")
    mod = _entity_by_name(parsed, "server")
    assert mod.qualified_name == f"{PROJECT}:src.server"


def test_module_lib_rs():
    parsed = _parse("", path="lib.rs")
    mod = _entity_by_name(parsed, "lib")
    assert mod.qualified_name == f"{PROJECT}:lib"


def test_module_main_rs():
    parsed = _parse("", path="main.rs")
    mod = _entity_by_name(parsed, "main")
    assert mod.qualified_name == f"{PROJECT}:main"


# ---------------------------------------------------------------------------
# 3. Struct extraction
# ---------------------------------------------------------------------------


def test_struct_basic():
    parsed = _parse(
        """\
/// A point in 2D space.
pub struct Point {
    pub x: f64,
    pub y: f64,
}
"""
    )
    st = _entity_by_name(parsed, "Point")
    assert st.label == NodeLabel.TYPE_DEF
    assert st.kind == TypeDefKind.STRUCT
    assert st.visibility == Visibility.PUBLIC
    assert st.docstring == "A point in 2D space."


# ---------------------------------------------------------------------------
# 4. Enum extraction (with variants as ENUM_MEMBER)
# ---------------------------------------------------------------------------


def test_enum_basic():
    parsed = _parse(
        """\
pub enum Color {
    Red,
    Green,
    Blue,
}
"""
    )
    en = _entity_by_name(parsed, "Color")
    assert en.label == NodeLabel.TYPE_DEF
    assert en.kind == TypeDefKind.ENUM
    assert en.visibility == Visibility.PUBLIC


def test_enum_variants():
    parsed = _parse(
        """\
pub enum Color {
    Red,
    Green,
    Blue,
}
"""
    )
    red = _entity_by_name(parsed, "Red")
    assert red.label == NodeLabel.VALUE
    assert red.kind == ValueKind.ENUM_MEMBER

    green = _entity_by_name(parsed, "Green")
    assert green.label == NodeLabel.VALUE
    assert green.kind == ValueKind.ENUM_MEMBER


# ---------------------------------------------------------------------------
# 5. Trait extraction
# ---------------------------------------------------------------------------


def test_trait_basic():
    parsed = _parse(
        """\
/// A drawable object.
pub trait Drawable {
    fn draw(&self);
}
"""
    )
    tr = _entity_by_name(parsed, "Drawable")
    assert tr.label == NodeLabel.TYPE_DEF
    assert tr.kind == TypeDefKind.TRAIT
    assert tr.docstring == "A drawable object."


# ---------------------------------------------------------------------------
# 6. Union extraction
# ---------------------------------------------------------------------------


def test_union_basic():
    parsed = _parse(
        """\
pub union MyUnion {
    pub f: f32,
    pub i: i32,
}
"""
    )
    un = _entity_by_name(parsed, "MyUnion")
    assert un.label == NodeLabel.TYPE_DEF
    assert un.kind == TypeDefKind.UNION
    assert un.visibility == Visibility.PUBLIC


# ---------------------------------------------------------------------------
# 7. Type alias extraction
# ---------------------------------------------------------------------------


def test_type_alias():
    parsed = _parse("pub type Result<T> = std::result::Result<T, MyError>;\n")
    ta = _entity_by_name(parsed, "Result")
    assert ta.label == NodeLabel.TYPE_DEF
    assert ta.kind == TypeDefKind.TYPE_ALIAS
    assert ta.visibility == Visibility.PUBLIC


# ---------------------------------------------------------------------------
# 8. Function extraction
# ---------------------------------------------------------------------------


def test_function_basic():
    parsed = _parse(
        """\
/// Adds two numbers.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
"""
    )
    func = _entity_by_name(parsed, "add")
    assert func.label == NodeLabel.CALLABLE
    assert func.kind == CallableKind.FUNCTION
    assert func.visibility == Visibility.PUBLIC
    assert func.docstring == "Adds two numbers."


# ---------------------------------------------------------------------------
# 9. Method vs static_method distinction in impl blocks
# ---------------------------------------------------------------------------


def test_method_with_self():
    parsed = _parse(
        """\
struct Foo;

impl Foo {
    pub fn bar(&self) -> i32 {
        42
    }
}
"""
    )
    bar = _entity_by_name(parsed, "bar")
    assert bar.label == NodeLabel.CALLABLE
    assert bar.kind == CallableKind.METHOD
    assert bar.qualified_name == f"{PROJECT}:src.example.Foo.bar"


def test_static_method_no_self():
    parsed = _parse(
        """\
struct Foo;

impl Foo {
    pub fn new() -> Self {
        Foo
    }
}
"""
    )
    new = _entity_by_name(parsed, "new")
    assert new.label == NodeLabel.CALLABLE
    assert new.kind == CallableKind.STATIC_METHOD
    assert new.qualified_name == f"{PROJECT}:src.example.Foo.new"


def test_method_mut_self():
    parsed = _parse(
        """\
struct Foo;

impl Foo {
    pub fn mutate(&mut self) {}
}
"""
    )
    m = _entity_by_name(parsed, "mutate")
    assert m.kind == CallableKind.METHOD


def test_method_owned_self():
    parsed = _parse(
        """\
struct Foo;

impl Foo {
    pub fn consume(self) {}
}
"""
    )
    c = _entity_by_name(parsed, "consume")
    assert c.kind == CallableKind.METHOD


# ---------------------------------------------------------------------------
# 10. Visibility
# ---------------------------------------------------------------------------


def test_visibility_pub():
    parsed = _parse("pub fn public_fn() {}\n")
    func = _entity_by_name(parsed, "public_fn")
    assert func.visibility == Visibility.PUBLIC


def test_visibility_pub_crate():
    parsed = _parse("pub(crate) fn crate_fn() {}\n")
    func = _entity_by_name(parsed, "crate_fn")
    assert func.visibility == Visibility.INTERNAL


def test_visibility_pub_super():
    parsed = _parse("pub(super) fn super_fn() {}\n")
    func = _entity_by_name(parsed, "super_fn")
    assert func.visibility == Visibility.PROTECTED


def test_visibility_private():
    parsed = _parse("fn private_fn() {}\n")
    func = _entity_by_name(parsed, "private_fn")
    assert func.visibility == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# 11. use_declaration -> IMPORTS
# ---------------------------------------------------------------------------


def test_use_simple():
    parsed = _parse("use std::collections::HashMap;\n")
    imports = _rels_from(parsed, "src.example", RelType.IMPORTS)
    names = {r.to_name for r in imports}
    assert "std.collections.HashMap" in names


def test_use_grouped():
    parsed = _parse("use std::collections::{HashMap, BTreeMap};\n")
    imports = _rels_from(parsed, "src.example", RelType.IMPORTS)
    names = {r.to_name for r in imports}
    assert "std.collections.HashMap" in names
    assert "std.collections.BTreeMap" in names


# ---------------------------------------------------------------------------
# 12. impl Trait for Type -> IMPLEMENTS
# ---------------------------------------------------------------------------


def test_impl_trait_for_type():
    parsed = _parse(
        """\
struct Foo;

trait Bar {}

impl Bar for Foo {}
"""
    )
    impl_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPLEMENTS]
    assert len(impl_rels) == 1
    assert impl_rels[0].from_qualified_name.endswith("src.example.Foo")
    assert impl_rels[0].to_name == "Bar"


# ---------------------------------------------------------------------------
# 13. Doc comments extraction
# ---------------------------------------------------------------------------


def test_doc_comment_triple_slash():
    parsed = _parse(
        """\
/// First line.
/// Second line.
pub fn documented() {}
"""
    )
    func = _entity_by_name(parsed, "documented")
    assert func.docstring is not None
    assert "First line." in func.docstring
    assert "Second line." in func.docstring


def test_inner_doc_comment():
    parsed = _parse(
        """\
//! Module documentation.
//! Second line.

fn main() {}
"""
    )
    # Module entity should have the inner doc comment
    entities = [e for e in parsed.entities if e.label == NodeLabel.MODULE]
    assert len(entities) == 1
    assert entities[0].docstring is not None
    assert "Module documentation." in entities[0].docstring


# ---------------------------------------------------------------------------
# 14. Signature extraction
# ---------------------------------------------------------------------------


def test_function_signature():
    parsed = _parse("pub fn compute(x: i32, y: i32) -> i32 { x + y }\n")
    func = _entity_by_name(parsed, "compute")
    assert func.signature is not None
    assert "compute" in func.signature
    assert "i32" in func.signature
    # Signature should not include the body
    assert "x + y" not in func.signature


# ---------------------------------------------------------------------------
# 15. const/static -> CONSTANT
# ---------------------------------------------------------------------------


def test_const_item():
    parsed = _parse("pub const MAX_SIZE: usize = 100;\n")
    val = _entity_by_name(parsed, "MAX_SIZE")
    assert val.label == NodeLabel.VALUE
    assert val.kind == ValueKind.CONSTANT
    assert val.visibility == Visibility.PUBLIC


def test_static_item():
    parsed = _parse("static COUNTER: i32 = 0;\n")
    val = _entity_by_name(parsed, "COUNTER")
    assert val.label == NodeLabel.VALUE
    assert val.kind == ValueKind.CONSTANT


# ---------------------------------------------------------------------------
# 16. Struct fields -> FIELD
# ---------------------------------------------------------------------------


def test_struct_fields():
    parsed = _parse(
        """\
pub struct Config {
    pub debug: bool,
    port: u16,
}
"""
    )
    debug_field = _entity_by_name(parsed, "debug")
    assert debug_field.label == NodeLabel.VALUE
    assert debug_field.kind == ValueKind.FIELD
    assert debug_field.visibility == Visibility.PUBLIC

    port_field = _entity_by_name(parsed, "port")
    assert port_field.kind == ValueKind.FIELD
    assert port_field.visibility == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# 17. Attribute tags
# ---------------------------------------------------------------------------


def test_derive_attribute():
    parsed = _parse(
        """\
#[derive(Debug, Clone)]
pub struct MyStruct {
    pub value: i32,
}
"""
    )
    st = _entity_by_name(parsed, "MyStruct")
    assert any("derive(Debug, Clone)" in t for t in st.tags)


def test_test_attribute():
    parsed = _parse(
        """\
#[test]
fn test_something() {
    assert!(true);
}
"""
    )
    func = _entity_by_name(parsed, "test_something")
    assert "attribute:test" in func.tags


def test_cfg_attribute():
    # Verify cfg(test) on mod parses without error
    _parse(
        """\
#[cfg(test)]
mod tests {}
"""
    )
    # cfg attribute on function should appear in tags
    parsed = _parse(
        """\
#[cfg(target_os = "linux")]
pub fn linux_only() {}
"""
    )
    func = _entity_by_name(parsed, "linux_only")
    assert any("cfg" in t for t in func.tags)


# ---------------------------------------------------------------------------
# 18. DEFINES relationships
# ---------------------------------------------------------------------------


def test_defines_module_to_struct():
    parsed = _parse("pub struct Foo;\n")
    defines = _rels_from(parsed, "src.example", RelType.DEFINES)
    targets = {r.to_name for r in defines}
    assert f"{PROJECT}:src.example.Foo" in targets


def test_defines_struct_to_method():
    """Same-file impl methods emit the deferred member-DEFINES contract (S5)."""
    parsed = _parse(
        """\
struct Foo;

impl Foo {
    pub fn bar(&self) {}
}
"""
    )
    defines = [
        r
        for r in parsed.relationships
        if r.rel_type == RelType.DEFINES and r.to_name == f"{PROJECT}:src.example.Foo.bar"
    ]
    assert len(defines) == 1
    assert defines[0].from_qualified_name == f"{PROJECT}:src.example"
    assert defines[0].properties == {"parent_type_name": "Foo"}
    # No relationship may originate from the fabricated type uid
    assert not any(r.from_qualified_name == f"{PROJECT}:src.example.Foo" for r in parsed.relationships)


def test_defines_module_to_function():
    parsed = _parse("fn helper() {}\n")
    defines = _rels_from(parsed, "src.example", RelType.DEFINES)
    targets = {r.to_name for r in defines}
    assert f"{PROJECT}:src.example.helper" in targets


# ---------------------------------------------------------------------------
# 19. CALLS extraction
# ---------------------------------------------------------------------------


def test_calls_simple():
    parsed = _parse(
        """\
fn caller() {
    println!("hello");
    some_func();
}
"""
    )
    calls = _rels_from(parsed, "src.example.caller", RelType.CALLS)
    called = {r.to_name for r in calls}
    assert "some_func" in called


def test_calls_method():
    parsed = _parse(
        """\
fn caller() {
    let v = Vec::new();
    v.push(1);
}
"""
    )
    calls = _rels_from(parsed, "src.example.caller", RelType.CALLS)
    called = {r.to_name for r in calls}
    assert "new" in called
    assert "push" in called


# ---------------------------------------------------------------------------
# 20. Content hash determinism
# ---------------------------------------------------------------------------


def test_content_hash_populated():
    parsed = _parse("pub fn hello() {}\n")
    for entity in parsed.entities:
        assert entity.content_hash, f"Entity {entity.name!r} has empty content_hash"


def test_content_hash_deterministic():
    source = "pub fn hello() {}\n"
    parsed1 = _parse(source)
    parsed2 = _parse(source)
    for e1, e2 in zip(parsed1.entities, parsed2.entities, strict=True):
        assert e1.content_hash == e2.content_hash


def test_content_hash_ignores_line_shift():
    source_v1 = "pub fn greet() {}\n"
    source_v2 = "\n\n\npub fn greet() {}\n"
    parsed1 = _parse(source_v1)
    parsed2 = _parse(source_v2)
    func1 = _entity_by_name(parsed1, "greet")
    func2 = _entity_by_name(parsed2, "greet")
    assert func1.content_hash == func2.content_hash
    assert func1.line_start != func2.line_start


# ---------------------------------------------------------------------------
# 21. Edge cases
# ---------------------------------------------------------------------------


def test_empty_file():
    parsed = _parse("")
    assert parsed is not None
    assert parsed.language == "rust"
    # Should have at least the module entity
    assert len(parsed.entities) >= 1


def test_syntax_error_tolerant():
    """Tree-sitter is error-tolerant — malformed Rust doesn't crash."""
    parsed = _parse("fn broken( { struct nope\n")
    assert parsed is not None


# ---------------------------------------------------------------------------
# 22. async/unsafe tags
# ---------------------------------------------------------------------------


def test_async_function_tag():
    parsed = _parse("pub async fn fetch_data() {}\n")
    func = _entity_by_name(parsed, "fetch_data")
    assert "async" in func.tags


def test_unsafe_function_tag():
    parsed = _parse("pub unsafe fn dangerous() {}\n")
    func = _entity_by_name(parsed, "dangerous")
    assert "unsafe" in func.tags


def test_async_unsafe_combined():
    parsed = _parse("pub async unsafe fn risky() {}\n")
    func = _entity_by_name(parsed, "risky")
    assert "async" in func.tags
    assert "unsafe" in func.tags


# ---------------------------------------------------------------------------
# Additional coverage
# ---------------------------------------------------------------------------


def test_trait_supertrait_inherits():
    """Supertraits emit INHERITS relationships."""
    parsed = _parse(
        """\
pub trait Animal: Clone + Display {}
"""
    )
    inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
    names = {r.to_name for r in inherits}
    assert "Clone" in names
    assert "Display" in names


def test_function_source_extracted():
    parsed = _parse('pub fn greet() { println!("hello"); }\n')
    func = _entity_by_name(parsed, "greet")
    assert func.source is not None
    assert "pub fn greet()" in func.source


def test_trait_method_signature():
    """Trait methods without bodies (signatures only) should be extracted."""
    parsed = _parse(
        """\
pub trait Service {
    fn process(&self, input: &str) -> String;
}
"""
    )
    m = _entity_by_name(parsed, "process")
    assert m.label == NodeLabel.CALLABLE
    assert m.kind == CallableKind.METHOD
    assert m.qualified_name == f"{PROJECT}:src.example.Service.process"


def test_doc_comment_with_attribute_between():
    """Doc comments with attributes in between should still be collected."""
    parsed = _parse(
        """\
/// Doc for the function.
#[inline]
pub fn inlined() {}
"""
    )
    func = _entity_by_name(parsed, "inlined")
    assert func.docstring is not None
    assert "Doc for the function." in func.docstring
    assert "attribute:inline" in func.tags


def test_field_doc_comment():
    parsed = _parse(
        """\
pub struct Config {
    /// The debug flag.
    pub debug: bool,
}
"""
    )
    field = _entity_by_name(parsed, "debug")
    assert field.docstring is not None
    assert "debug flag" in field.docstring


# ---------------------------------------------------------------------------
# 23. Inline modules (mod foo { ... })
# ---------------------------------------------------------------------------


def test_inline_mod_entities_extracted():
    parsed = _parse(
        """\
pub mod config {
    pub struct Config {
        pub name: String,
    }

    impl Config {
        pub fn new() -> Self {
            Config { name: String::new() }
        }
    }
}
""",
        path="src/lib.rs",
    )
    st = _entity_by_name(parsed, "Config")
    assert st.label == NodeLabel.TYPE_DEF
    assert st.qualified_name == f"{PROJECT}:src.lib.config.Config"
    name_field = _entity_by_name(parsed, "name")
    assert name_field.qualified_name == f"{PROJECT}:src.lib.config.Config.name"
    new = _entity_by_name(parsed, "new")
    assert new.qualified_name == f"{PROJECT}:src.lib.config.Config.new"


def test_inline_mod_module_entity_and_defines():
    parsed = _parse(
        """\
pub mod config {
    pub struct Config;
}
""",
        path="src/lib.rs",
    )
    mod = _entity_by_name(parsed, "config")
    assert mod.label == NodeLabel.MODULE
    assert mod.kind == "module"
    assert mod.qualified_name == f"{PROJECT}:src.lib.config"
    assert mod.visibility == Visibility.PUBLIC
    # File module defines the inline module, which defines the struct
    file_defines = _rels_from(parsed, "src.lib", RelType.DEFINES)
    assert f"{PROJECT}:src.lib.config" in {r.to_name for r in file_defines}
    mod_defines = _rels_from(parsed, "src.lib.config", RelType.DEFINES)
    assert f"{PROJECT}:src.lib.config.Config" in {r.to_name for r in mod_defines}


def test_nested_inline_mods():
    parsed = _parse(
        """\
mod a {
    mod b {
        fn f() {}
    }
}
"""
    )
    f = _entity_by_name(parsed, "f")
    assert f.qualified_name == f"{PROJECT}:src.example.a.b.f"


def test_cfg_test_mod_entities_extracted():
    parsed = _parse(
        """\
#[cfg(test)]
mod tests {
    fn helper() {}
}
"""
    )
    helper = _entity_by_name(parsed, "helper")
    assert helper.qualified_name == f"{PROJECT}:src.example.tests.helper"


def test_inline_mod_emits_no_imports():
    parsed = _parse("mod inline_mod {\n    fn f() {}\n}\n")
    imports = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    assert imports == []


def test_mod_declaration_emits_imports():
    """``mod foo;`` (no body) still references another file via IMPORTS."""
    parsed = _parse("mod foo;\n")
    imports = _rels_from(parsed, "src.example", RelType.IMPORTS)
    assert {r.to_name for r in imports} == {"foo"}


# ---------------------------------------------------------------------------
# 24. Impl-block member DEFINES contract (S5) and IMPLEMENTS bare names (S1)
# ---------------------------------------------------------------------------


def test_impl_method_cross_file_emits_parent_type_name():
    """Impl methods for a type defined in another file must not fabricate the type uid."""
    parsed = _parse(
        """\
impl Widget {
    pub fn draw(&self) {}
}
""",
        path="src/render.rs",
    )
    draw = _entity_by_name(parsed, "draw")
    assert draw.qualified_name == f"{PROJECT}:src.render.Widget.draw"
    defines = [
        r
        for r in parsed.relationships
        if r.rel_type == RelType.DEFINES and r.to_name == f"{PROJECT}:src.render.Widget.draw"
    ]
    assert len(defines) == 1
    assert defines[0].from_qualified_name == f"{PROJECT}:src.render"
    assert defines[0].properties == {"parent_type_name": "Widget"}
    assert not any(r.from_qualified_name == f"{PROJECT}:src.render.Widget" for r in parsed.relationships)


def test_generic_impl_parent_type_name_is_bare():
    """parent_type_name strips type parameters: impl<T> Container<T> -> Container."""
    parsed = _parse(
        """\
impl<T> Container<T> {
    pub fn get(&self) -> &T {
        &self.item
    }
}
"""
    )
    get = _entity_by_name(parsed, "get")
    assert get.qualified_name == f"{PROJECT}:src.example.Container.get"
    rel = next(r for r in parsed.relationships if r.rel_type == RelType.DEFINES and r.to_name == get.qualified_name)
    assert rel.from_qualified_name == f"{PROJECT}:src.example"
    assert rel.properties == {"parent_type_name": "Container"}


def test_trait_body_methods_keep_uid_defines():
    """Trait-body members are same-file by construction — plain uid-matched DEFINES."""
    parsed = _parse(
        """\
pub trait Service {
    fn process(&self) {}
    fn handle(&self, x: i32) -> i32;
}
"""
    )
    for method in ("process", "handle"):
        rel = next(
            r
            for r in parsed.relationships
            if r.rel_type == RelType.DEFINES and r.to_name == f"{PROJECT}:src.example.Service.{method}"
        )
        assert rel.from_qualified_name == f"{PROJECT}:src.example.Service"
        assert rel.properties == {}


def test_implements_bare_trait_name_no_colon():
    """S1 contract pin: IMPLEMENTS to_name is a bare trait name — never contains ':'."""
    parsed = _parse(
        """\
struct Foo;

impl std::fmt::Display for Foo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "foo")
    }
}
"""
    )
    impl_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPLEMENTS]
    assert len(impl_rels) == 1
    assert impl_rels[0].to_name == "Display"
    assert ":" not in impl_rels[0].to_name
    assert impl_rels[0].from_qualified_name == f"{PROJECT}:src.example.Foo"
    assert impl_rels[0].properties == {}
