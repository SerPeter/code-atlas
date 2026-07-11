"""Tests for TypeScript/JavaScript parser."""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_typescript", reason="tree-sitter-typescript not installed")

from code_atlas.parsing.ast import ParsedFile, get_language_for_file, parse_file
from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind, ValueKind, Visibility

PROJECT = "test_project"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(source: str, path: str = "src/example.ts") -> ParsedFile:
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


def test_language_detection_ts():
    assert get_language_for_file("src/main.ts") is not None


def test_language_detection_tsx():
    assert get_language_for_file("src/App.tsx") is not None


def test_language_detection_js():
    cfg = get_language_for_file("src/util.js")
    if cfg is None:
        pytest.skip("tree-sitter-javascript not installed")
    assert cfg is not None


def test_language_detection_jsx():
    cfg = get_language_for_file("src/Component.jsx")
    if cfg is None:
        pytest.skip("tree-sitter-javascript not installed")
    assert cfg is not None


def test_language_detection_mjs():
    cfg = get_language_for_file("lib/index.mjs")
    if cfg is None:
        pytest.skip("tree-sitter-javascript not installed")
    assert cfg is not None


def test_language_detection_cjs():
    cfg = get_language_for_file("lib/index.cjs")
    if cfg is None:
        pytest.skip("tree-sitter-javascript not installed")
    assert cfg is not None


# ---------------------------------------------------------------------------
# 2. Module entity creation
# ---------------------------------------------------------------------------


def test_module_entity():
    parsed = _parse("const x = 1;\n", path="src/components/Button.ts")
    module = _entity_by_name(parsed, "Button")
    assert module.label == NodeLabel.MODULE
    assert module.kind == "module"
    assert module.qualified_name == f"{PROJECT}:src.components.Button"


def test_module_entity_index_file():
    """index.ts gets the parent directory name, like __init__.py."""
    parsed = _parse("export {};\n", path="src/components/index.ts")
    module = _entity_by_name(parsed, "components")
    assert module.label == NodeLabel.MODULE
    assert module.qualified_name == f"{PROJECT}:src.components"


# ---------------------------------------------------------------------------
# 3. Class extraction
# ---------------------------------------------------------------------------


def test_class_basic():
    parsed = _parse("""\
/** A simple class */
class MyClass {
  greet() { return "hello"; }
}
""")
    cls = _entity_by_name(parsed, "MyClass")
    assert cls.label == NodeLabel.TYPE_DEF
    assert cls.kind == TypeDefKind.CLASS
    assert cls.docstring == "A simple class"
    assert cls.visibility == Visibility.PUBLIC


def test_abstract_class():
    parsed = _parse("""\
abstract class Widget {
  abstract render(): void;
}
""")
    cls = _entity_by_name(parsed, "Widget")
    assert cls.label == NodeLabel.TYPE_DEF
    assert cls.kind == TypeDefKind.CLASS
    assert "abstract" in cls.tags


# ---------------------------------------------------------------------------
# 4. Interface extraction
# ---------------------------------------------------------------------------


def test_interface_basic():
    parsed = _parse("""\
/** User interface */
interface IUser {
  name: string;
  age: number;
}
""")
    iface = _entity_by_name(parsed, "IUser")
    assert iface.label == NodeLabel.TYPE_DEF
    assert iface.kind == TypeDefKind.INTERFACE
    assert iface.docstring == "User interface"


# ---------------------------------------------------------------------------
# 5. Enum extraction
# ---------------------------------------------------------------------------


def test_enum_basic():
    parsed = _parse("""\
enum Color {
  Red,
  Green = 2,
  Blue
}
""")
    enum = _entity_by_name(parsed, "Color")
    assert enum.label == NodeLabel.TYPE_DEF
    assert enum.kind == TypeDefKind.ENUM


# ---------------------------------------------------------------------------
# 6. Type alias extraction
# ---------------------------------------------------------------------------


def test_type_alias():
    parsed = _parse("type UserID = string | number;\n")
    ta = _entity_by_name(parsed, "UserID")
    assert ta.label == NodeLabel.TYPE_DEF
    assert ta.kind == TypeDefKind.TYPE_ALIAS


# ---------------------------------------------------------------------------
# 7. Function extraction
# ---------------------------------------------------------------------------


def test_function_basic():
    parsed = _parse("""\
/** Say hello */
function greet(name: string): string {
  return "Hello " + name;
}
""")
    func = _entity_by_name(parsed, "greet")
    assert func.label == NodeLabel.CALLABLE
    assert func.kind == CallableKind.FUNCTION
    assert func.docstring == "Say hello"


def test_async_function():
    parsed = _parse("""\
async function fetchData(): Promise<void> {
  return;
}
""")
    func = _entity_by_name(parsed, "fetchData")
    assert func.kind == CallableKind.FUNCTION
    assert "async" in func.tags


# ---------------------------------------------------------------------------
# 8. Method/constructor distinction
# ---------------------------------------------------------------------------


def test_method_vs_constructor():
    parsed = _parse("""\
class Foo {
  constructor() {}
  bar() { return 1; }
}
""")
    ctor = _entity_by_name(parsed, "constructor")
    assert ctor.kind == CallableKind.CONSTRUCTOR
    assert ctor.qualified_name == f"{PROJECT}:src.example.Foo.constructor"

    bar = _entity_by_name(parsed, "bar")
    assert bar.kind == CallableKind.METHOD
    assert bar.qualified_name == f"{PROJECT}:src.example.Foo.bar"


def test_static_method():
    parsed = _parse("""\
class Foo {
  static create() { return new Foo(); }
}
""")
    create = _entity_by_name(parsed, "create")
    assert create.kind == CallableKind.STATIC_METHOD


# ---------------------------------------------------------------------------
# 9. Visibility rules
# ---------------------------------------------------------------------------


def test_visibility_public():
    parsed = _parse("""\
class Foo {
  public name: string;
  public greet() {}
}
""")
    name_field = _entity_by_name(parsed, "name")
    assert name_field.visibility == Visibility.PUBLIC
    greet = _entity_by_name(parsed, "greet")
    assert greet.visibility == Visibility.PUBLIC


def test_visibility_private():
    parsed = _parse("""\
class Foo {
  private _count: number;
  private helper() {}
}
""")
    count = _entity_by_name(parsed, "_count")
    assert count.visibility == Visibility.PRIVATE
    helper = _entity_by_name(parsed, "helper")
    assert helper.visibility == Visibility.PRIVATE


def test_visibility_protected():
    parsed = _parse("""\
class Foo {
  protected data: string;
  protected process() {}
}
""")
    data = _entity_by_name(parsed, "data")
    assert data.visibility == Visibility.PROTECTED
    process = _entity_by_name(parsed, "process")
    assert process.visibility == Visibility.PROTECTED


def test_visibility_hash_private():
    parsed = _parse("""\
class Foo {
  #secret: boolean = true;
}
""")
    secret = _entity_by_name(parsed, "#secret")
    assert secret.visibility == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# 10. Import extraction -> IMPORTS
# ---------------------------------------------------------------------------


def test_import_named():
    parsed = _parse('import { foo, bar } from "./module";\n')
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    assert len(import_rels) == 1
    assert import_rels[0].to_name == "./module"


def test_import_default():
    parsed = _parse('import React from "react";\n')
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    assert len(import_rels) == 1
    assert import_rels[0].to_name == "react"


def test_import_namespace():
    parsed = _parse('import * as path from "path";\n')
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    assert len(import_rels) == 1
    assert import_rels[0].to_name == "path"


# ---------------------------------------------------------------------------
# 11. Inheritance -> INHERITS
# ---------------------------------------------------------------------------


def test_inherits():
    parsed = _parse("class Child extends Parent {}\n")
    inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
    assert len(inherits) == 1
    assert inherits[0].to_name == "Parent"


# ---------------------------------------------------------------------------
# 12. Implements -> IMPLEMENTS
# ---------------------------------------------------------------------------


def test_implements():
    parsed = _parse("class Foo implements IBar, IBaz {}\n")
    impl_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPLEMENTS]
    iface_names = {r.to_name for r in impl_rels}
    assert "IBar" in iface_names
    assert "IBaz" in iface_names


def test_extends_and_implements():
    parsed = _parse("class Foo extends Base implements IBar {}\n")
    inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
    impl_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPLEMENTS]
    assert len(inherits) == 1
    assert inherits[0].to_name == "Base"
    assert len(impl_rels) == 1
    assert impl_rels[0].to_name == "IBar"


def test_implements_bare_name_contract():
    """S1 contract: IMPLEMENTS is emitted with a bare interface name (never uid-shaped).

    GraphClient._create_relationships routes IMPLEMENTS by shape — ``:`` in to_name
    means uid (detector path), no ``:`` means bare name resolved like INHERITS.
    The parser must emit from_qualified_name as the full uid and to_name bare.
    """
    parsed = _parse("class FileLogger implements Logger {}\n", path="src/logger.ts")
    impl_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPLEMENTS]
    assert len(impl_rels) == 1
    rel = impl_rels[0]
    assert rel.from_qualified_name == f"{PROJECT}:src.logger.FileLogger"
    assert rel.to_name == "Logger"
    assert ":" not in rel.to_name
    assert not rel.properties


# ---------------------------------------------------------------------------
# 13. Docstring (JSDoc) extraction
# ---------------------------------------------------------------------------


def test_jsdoc_function():
    parsed = _parse("""\
/** Greets a user */
function greet(name: string) {
  return "Hello " + name;
}
""")
    func = _entity_by_name(parsed, "greet")
    assert func.docstring == "Greets a user"


def test_jsdoc_multiline():
    parsed = _parse("""\
/**
 * Process data.
 * Returns the result.
 */
function process() {}
""")
    func = _entity_by_name(parsed, "process")
    assert func.docstring is not None
    assert "Process data." in func.docstring
    assert "Returns the result." in func.docstring


def test_jsdoc_class():
    parsed = _parse("""\
/** A widget class */
class Widget {}
""")
    cls = _entity_by_name(parsed, "Widget")
    assert cls.docstring == "A widget class"


def test_regular_comment_not_jsdoc():
    parsed = _parse("""\
// Regular comment
function foo() {}
""")
    func = _entity_by_name(parsed, "foo")
    assert func.docstring is None


def test_block_comment_not_jsdoc():
    parsed = _parse("""\
/* Not a JSDoc comment */
function foo() {}
""")
    func = _entity_by_name(parsed, "foo")
    assert func.docstring is None


# ---------------------------------------------------------------------------
# 14. Signature extraction
# ---------------------------------------------------------------------------


def test_function_signature():
    parsed = _parse("function greet(name: string): string { return name; }\n")
    func = _entity_by_name(parsed, "greet")
    assert func.signature is not None
    assert "greet" in func.signature
    assert "name: string" in func.signature
    # Body should not be in signature
    assert "return" not in func.signature


def test_method_signature():
    parsed = _parse("""\
class Foo {
  async greet(name: string): Promise<void> {
    console.log(name);
  }
}
""")
    greet = _entity_by_name(parsed, "greet")
    assert greet.signature is not None
    assert "greet" in greet.signature
    assert "console" not in greet.signature


# ---------------------------------------------------------------------------
# 15. Values (const/let/var)
# ---------------------------------------------------------------------------


def test_const_value():
    parsed = _parse("const MAX_SIZE = 100;\n")
    val = _entity_by_name(parsed, "MAX_SIZE")
    assert val.label == NodeLabel.VALUE
    assert val.kind == ValueKind.CONSTANT


def test_let_value():
    parsed = _parse("let counter = 0;\n")
    val = _entity_by_name(parsed, "counter")
    assert val.label == NodeLabel.VALUE
    assert val.kind == ValueKind.VARIABLE


def test_var_value():
    parsed = _parse("var legacy = true;\n")
    val = _entity_by_name(parsed, "legacy")
    assert val.label == NodeLabel.VALUE
    assert val.kind == ValueKind.VARIABLE


# ---------------------------------------------------------------------------
# 16. Enum members
# ---------------------------------------------------------------------------


def test_enum_members():
    parsed = _parse("""\
enum Color {
  Red,
  Green = 2,
  Blue
}
""")
    red = _entity_by_name(parsed, "Red")
    assert red.label == NodeLabel.VALUE
    assert red.kind == ValueKind.ENUM_MEMBER

    green = _entity_by_name(parsed, "Green")
    assert green.kind == ValueKind.ENUM_MEMBER

    blue = _entity_by_name(parsed, "Blue")
    assert blue.kind == ValueKind.ENUM_MEMBER


# ---------------------------------------------------------------------------
# 17. Class fields
# ---------------------------------------------------------------------------


def test_class_fields():
    parsed = _parse("""\
class Foo {
  public name: string;
  private _count: number;
}
""")
    name_field = _entity_by_name(parsed, "name")
    assert name_field.label == NodeLabel.VALUE
    assert name_field.kind == ValueKind.FIELD
    assert name_field.visibility == Visibility.PUBLIC

    count_field = _entity_by_name(parsed, "_count")
    assert count_field.kind == ValueKind.FIELD
    assert count_field.visibility == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# 18. DEFINES relationships
# ---------------------------------------------------------------------------


def test_defines_relationships():
    parsed = _parse("""\
class Foo {
  bar() { return 1; }
}

function baz() {}
""")
    # Module DEFINES Foo
    mod_defines = _rels_from(parsed, "src.example", RelType.DEFINES)
    targets = {r.to_name for r in mod_defines}
    assert f"{PROJECT}:src.example.Foo" in targets
    assert f"{PROJECT}:src.example.baz" in targets

    # Foo DEFINES bar
    foo_defines = _rels_from(parsed, "src.example.Foo", RelType.DEFINES)
    assert any(r.to_name == f"{PROJECT}:src.example.Foo.bar" for r in foo_defines)


def test_enum_defines_members():
    parsed = _parse("""\
enum Color {
  Red,
  Green
}
""")
    enum_defines = _rels_from(parsed, "src.example.Color", RelType.DEFINES)
    targets = {r.to_name for r in enum_defines}
    assert f"{PROJECT}:src.example.Color.Red" in targets
    assert f"{PROJECT}:src.example.Color.Green" in targets


# ---------------------------------------------------------------------------
# 19. CALLS extraction
# ---------------------------------------------------------------------------


def test_calls_in_function():
    parsed = _parse("""\
function caller() {
  console.log("hello");
  someFunc();
}
""")
    calls = _rels_from(parsed, "src.example.caller", RelType.CALLS)
    called = {r.to_name for r in calls}
    assert "log" in called
    assert "someFunc" in called


def test_calls_in_method():
    parsed = _parse("""\
class Foo {
  bar() {
    this.helper();
    doSomething();
  }
}
""")
    calls = _rels_from(parsed, "src.example.Foo.bar", RelType.CALLS)
    called = {r.to_name for r in calls}
    assert "helper" in called
    assert "doSomething" in called


# ---------------------------------------------------------------------------
# 20. Export handling
# ---------------------------------------------------------------------------


def test_export_function():
    parsed = _parse("export function greet() { return 1; }\n")
    func = _entity_by_name(parsed, "greet")
    assert "exported" in func.tags


def test_export_class():
    parsed = _parse("export class Foo {}\n")
    cls = _entity_by_name(parsed, "Foo")
    assert "exported" in cls.tags


def test_export_const():
    parsed = _parse("export const MAX = 100;\n")
    val = _entity_by_name(parsed, "MAX")
    assert "exported" in val.tags


def test_export_interface():
    parsed = _parse("export interface IFoo { bar(): void; }\n")
    iface = _entity_by_name(parsed, "IFoo")
    assert "exported" in iface.tags


def test_export_enum():
    parsed = _parse("export enum Status { Active, Inactive }\n")
    enum = _entity_by_name(parsed, "Status")
    assert "exported" in enum.tags


def test_export_type_alias():
    parsed = _parse("export type ID = string;\n")
    ta = _entity_by_name(parsed, "ID")
    assert "exported" in ta.tags


def test_export_default_class():
    parsed = _parse("export default class Foo {}\n")
    cls = _entity_by_name(parsed, "Foo")
    assert "exported" in cls.tags


# ---------------------------------------------------------------------------
# 21. Content hash determinism
# ---------------------------------------------------------------------------


def test_content_hash_populated():
    parsed = _parse("""\
class Foo {
  bar() { return 1; }
}
""")
    for entity in parsed.entities:
        assert entity.content_hash, f"Entity {entity.name!r} has empty content_hash"


def test_content_hash_deterministic():
    source = """\
function greet(name: string): string {
  return "Hello " + name;
}
"""
    parsed1 = _parse(source)
    parsed2 = _parse(source)
    for e1, e2 in zip(parsed1.entities, parsed2.entities, strict=True):
        assert e1.content_hash == e2.content_hash


def test_content_hash_ignores_line_shift():
    source_v1 = "function greet() { return 1; }\n"
    source_v2 = "\n\n\nfunction greet() { return 1; }\n"
    parsed1 = _parse(source_v1)
    parsed2 = _parse(source_v2)
    func1 = _entity_by_name(parsed1, "greet")
    func2 = _entity_by_name(parsed2, "greet")
    assert func1.content_hash == func2.content_hash
    assert func1.line_start != func2.line_start


# ---------------------------------------------------------------------------
# 22. Edge cases
# ---------------------------------------------------------------------------


def test_empty_file():
    parsed = _parse("")
    assert parsed is not None
    assert parsed.language == "typescript"
    # Should have at least the module entity
    assert len(parsed.entities) >= 1


def test_syntax_error_tolerant():
    """Tree-sitter is error-tolerant — malformed files don't crash."""
    parsed = _parse("function broken(\n    class nope\n")
    assert parsed is not None


def test_unsupported_extension():
    result = parse_file("data.csv", b"a,b,c", PROJECT)
    assert result is None


# ---------------------------------------------------------------------------
# 23. Arrow function as module-level const
# ---------------------------------------------------------------------------


def test_arrow_function_as_const():
    parsed = _parse("""\
const helper = (x: number): number => {
  return x * 2;
};
""")
    func = _entity_by_name(parsed, "helper")
    assert func.label == NodeLabel.CALLABLE
    assert func.kind == CallableKind.FUNCTION


def test_arrow_function_exported():
    parsed = _parse("export const handler = () => { doWork(); };\n")
    func = _entity_by_name(parsed, "handler")
    assert func.label == NodeLabel.CALLABLE
    assert func.kind == CallableKind.FUNCTION
    assert "exported" in func.tags


def test_arrow_function_calls():
    parsed = _parse("""\
const handler = () => {
  doWork();
  console.log("done");
};
""")
    calls = _rels_from(parsed, "src.example.handler", RelType.CALLS)
    called = {r.to_name for r in calls}
    assert "doWork" in called
    assert "log" in called


# ---------------------------------------------------------------------------
# 24. TSX (JSX) parsing
# ---------------------------------------------------------------------------


def test_tsx_jsx_function_component():
    """.tsx files must use the TSX grammar — JSX-returning components are extracted."""
    parsed = _parse(
        """\
export function App() {
  return <div className="app">{render()}</div>;
}
""",
        path="src/components/App.tsx",
    )
    funcs = [e for e in parsed.entities if e.name == "App" and e.label == NodeLabel.CALLABLE]
    assert len(funcs) == 1
    func = funcs[0]
    assert func.kind == CallableKind.FUNCTION
    assert "exported" in func.tags

    defines = [r for r in parsed.relationships if r.rel_type == RelType.DEFINES]
    assert any(r.to_name == f"{PROJECT}:src.components.App.App" for r in defines)

    calls = [r for r in parsed.relationships if r.rel_type == RelType.CALLS]
    assert any(r.to_name == "render" for r in calls)


def test_tsx_jsx_multiple_components():
    """Declarations following JSX are not swallowed into ERROR subtrees."""
    parsed = _parse(
        """\
function Toolbar() {
  return <div className="toolbar">{render()}</div>;
}

export const Button = () => <button onClick={handleClick}>Go</button>;

export function App() {
  return (
    <main>
      <Toolbar />
      <Button />
    </main>
  );
}
""",
        path="src/components/App.tsx",
    )
    names = {e.name for e in parsed.entities if e.label == NodeLabel.CALLABLE}
    assert names == {"Toolbar", "Button", "App"}

    calls = _rels_from(parsed, "src.components.App.Toolbar", RelType.CALLS)
    assert any(r.to_name == "render" for r in calls)


def test_ts_old_style_type_assertion():
    """.ts stays on the plain typescript grammar — old-style <T>expr assertions parse."""
    parsed = _parse("const x = <string>getValue();\n", path="src/legacy.ts")
    val = _entity_by_name(parsed, "x")
    assert val.label == NodeLabel.VALUE
    assert val.kind == ValueKind.CONSTANT


# ---------------------------------------------------------------------------
# JavaScript-specific tests
# ---------------------------------------------------------------------------

js_installed = pytest.importorskip("tree_sitter_javascript", reason="tree-sitter-javascript not installed")


def test_js_function():
    parsed = _parse("function hello() { return 1; }\n", path="src/util.js")
    func = _entity_by_name(parsed, "hello")
    assert func.label == NodeLabel.CALLABLE
    assert func.kind == CallableKind.FUNCTION
    assert parsed.language == "javascript"


def test_js_class():
    parsed = _parse(
        """\
class Animal {
  constructor(name) {
    this.name = name;
  }
  speak() {
    return this.name;
  }
}
""",
        path="src/animal.js",
    )
    cls = _entity_by_name(parsed, "Animal")
    assert cls.label == NodeLabel.TYPE_DEF
    assert cls.kind == TypeDefKind.CLASS

    ctor = _entity_by_name(parsed, "constructor")
    assert ctor.kind == CallableKind.CONSTRUCTOR

    speak = _entity_by_name(parsed, "speak")
    assert speak.kind == CallableKind.METHOD


def test_js_module_entity():
    parsed = _parse("const x = 1;\n", path="src/util.js")
    module = _entity_by_name(parsed, "util")
    assert module.label == NodeLabel.MODULE
    assert module.qualified_name == f"{PROJECT}:src.util"


# ---------------------------------------------------------------------------
# import type detection (type_only flag)
# ---------------------------------------------------------------------------


def test_import_type_marked_type_only():
    """TypeScript `import type` syntax gets type_only=True property."""
    parsed = _parse('import type { User } from "./models";\n')
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    assert len(import_rels) == 1
    assert import_rels[0].properties.get("type_only") is True


def test_regular_import_not_type_only():
    """Regular TS imports have no type_only property."""
    parsed = _parse('import { foo } from "./module";\n')
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    assert len(import_rels) == 1
    assert not import_rels[0].properties.get("type_only")


# ---------------------------------------------------------------------------
# USES_TYPE extraction (TypeScript)
# ---------------------------------------------------------------------------


def test_uses_type_from_ts_function():
    """TypeScript function type annotations emit USES_TYPE."""
    parsed = _parse("""\
function process(user: User, config: Config): Result {
  return {} as Result;
}
""")
    uses_type = [r for r in parsed.relationships if r.rel_type == RelType.USES_TYPE]
    type_names = {r.to_name for r in uses_type}
    assert "User" in type_names
    assert "Config" in type_names
    assert "Result" in type_names


def test_uses_type_skips_ts_builtins():
    """TS built-in types like string, number, boolean don't produce USES_TYPE."""
    parsed = _parse("""\
function add(x: number, y: string): boolean {
  return true;
}
""")
    uses_type = [r for r in parsed.relationships if r.rel_type == RelType.USES_TYPE]
    assert len(uses_type) == 0


def test_uses_type_from_ts_method():
    """TypeScript method type annotations emit USES_TYPE."""
    parsed = _parse("""\
class Service {
  handle(req: Request): Response {
    return {} as Response;
  }
}
""")
    uses_type = [r for r in parsed.relationships if r.rel_type == RelType.USES_TYPE]
    type_names = {r.to_name for r in uses_type}
    assert "Request" in type_names
    assert "Response" in type_names


def test_uses_type_from_arrow_function():
    """Arrow function type annotations emit USES_TYPE."""
    parsed = _parse("""\
const handler = (req: Request): Response => {
  return {} as Response;
};
""")
    uses_type = [r for r in parsed.relationships if r.rel_type == RelType.USES_TYPE]
    type_names = {r.to_name for r in uses_type}
    assert "Request" in type_names
    assert "Response" in type_names
