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


def test_interface_extends_inherits():
    """interface_declaration's extends_type_clause maps to INHERITS (finding typescript.py:514)."""
    parsed = _parse("interface AdminUser extends User {}\n")
    inherits = _rels_from(parsed, "src.example.AdminUser", RelType.INHERITS)
    assert len(inherits) == 1
    assert inherits[0].to_name == "User"


def test_interface_extends_multiple_inherits():
    parsed = _parse("interface AdminUser extends User, Named {}\n")
    inherits = _rels_from(parsed, "src.example.AdminUser", RelType.INHERITS)
    to_names = {r.to_name for r in inherits}
    assert to_names == {"User", "Named"}


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


def test_reexport_named_produces_imports():
    """'export { x, y } from './mod'' is a barrel re-export — must emit IMPORTS (typescript.py:1093)."""
    parsed = _parse("export { x, y } from './mod';\n")
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    assert len(import_rels) == 1
    assert import_rels[0].to_name == "./mod"


def test_reexport_star_produces_imports():
    parsed = _parse("export * from './mod';\n")
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    assert len(import_rels) == 1
    assert import_rels[0].to_name == "./mod"


def test_reexport_star_as_produces_imports():
    parsed = _parse("export * as ns from './mod';\n")
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    assert len(import_rels) == 1
    assert import_rels[0].to_name == "./mod"


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


def test_extends_qualified_member_expression():
    """'extends ns.Base' parses as member_expression — must still yield INHERITS (typescript.py:227)."""
    parsed = _parse("class Button extends React.Component {}\n")
    inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
    assert len(inherits) == 1
    assert inherits[0].to_name == "Component"


def test_implements_generic_type():
    """'implements IRepo<User>' parses as generic_type — must still yield IMPLEMENTS to 'IRepo'."""
    parsed = _parse("class Foo implements IRepo<User> {}\n")
    impl_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPLEMENTS]
    assert len(impl_rels) == 1
    assert impl_rels[0].to_name == "IRepo"


def test_implements_qualified_nested_type_identifier():
    """'implements ns.IFace' parses as nested_type_identifier — must still yield IMPLEMENTS to 'IFace'."""
    parsed = _parse("class Foo implements ns.IFace {}\n")
    impl_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPLEMENTS]
    assert len(impl_rels) == 1
    assert impl_rels[0].to_name == "IFace"


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


def test_calls_inside_arrow_callback():
    """Calls nested inside an arrow-function callback argument are attributed to the enclosing
    function, not dropped (typescript.py:184)."""
    parsed = _parse("""\
function caller() {
  items.forEach((item) => {
    doWork(item);
  });
}
""")
    calls = _rels_from(parsed, "src.example.caller", RelType.CALLS)
    called = {r.to_name for r in calls}
    assert "forEach" in called
    assert "doWork" in called


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


def test_decorator_on_unexported_class():
    parsed = _parse("@Injectable()\nclass Service {}\n")
    cls = _entity_by_name(parsed, "Service")
    assert any(t.startswith("decorator:Injectable") for t in cls.tags)


def test_decorator_on_exported_class():
    """Decorators on exported classes attach to export_statement, not the class (typescript.py:134)."""
    parsed = _parse("@Injectable()\nexport class Service {}\n")
    cls = _entity_by_name(parsed, "Service")
    assert any(t.startswith("decorator:Injectable") for t in cls.tags)
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


# ---------------------------------------------------------------------------
# 25. Rationale extraction (intent-bearing comments)
# ---------------------------------------------------------------------------


def test_rationale_line_comment_above_function():
    parsed = _parse("""\
// WHY: debounce avoids hammering the API on every keystroke
export function search(q: string) {
  return q;
}
""")
    assert _entity_by_name(parsed, "search").rationale == ("WHY: debounce avoids hammering the API on every keystroke")


def test_rationale_jsdoc_block_comment():
    """`/** ... */` leading asterisks are stripped before marker matching."""
    parsed = _parse("""\
function render() {
  /**
   * NOTE: an empty query short-circuits before the fetch.
   */
  return 1;
}
""")
    assert _entity_by_name(parsed, "render").rationale == "NOTE: an empty query short-circuits before the fetch."


def test_rationale_method_beats_enclosing_class():
    parsed = _parse("""\
class Widget {
  // HACK: the constructor runs twice under StrictMode
  render() {
    return 1;
  }
}
""")
    assert _entity_by_name(parsed, "render").rationale == "HACK: the constructor runs twice under StrictMode"
    assert _entity_by_name(parsed, "Widget").rationale is None


def test_rationale_citations_in_ts_comment():
    parsed = _parse("""\
// NOTE: content negotiation follows RFC-7231
export function negotiate() {
  return 1;
}
""")
    assert _entity_by_name(parsed, "negotiate").citations == ["RFC-7231"]


def test_rationale_todo_off_by_default_ts():
    parsed = _parse("""\
// TODO: migrate to the new client
export function old() {
  return 1;
}
""")
    assert _entity_by_name(parsed, "old").rationale is None


def test_rationale_absent_for_plain_ts_file():
    parsed = _parse("""\
export function plain() {
  return 1;
}
""")
    for entity in parsed.entities:
        assert entity.rationale is None
        assert entity.citations == []


# ---------------------------------------------------------------------------
# Salesforce Lightning Web Components
# ---------------------------------------------------------------------------

_LWC_PATH = "force-app/main/default/lwc/accountList/accountList.js"


def _lwc_imports(source: str) -> set[str]:
    parsed = _parse(source, path=_LWC_PATH)
    return {r.to_name for r in parsed.relationships if r.rel_type == RelType.IMPORTS}


def test_lwc_apex_import_targets_the_apex_qualified_name():
    """`apex.Class.method` is what parsing/languages/apex.py stores, so resolve_imports
    classifies the import as internal and wires the LWC module to the real Callable."""
    assert "apex.AccountService.getAccounts" in _lwc_imports(
        "import getAccounts from '@salesforce/apex/AccountService.getAccounts';\n"
    )


def test_lwc_apex_import_keeps_a_managed_package_namespace():
    assert "apex.ns.AccountService.getAccounts" in _lwc_imports(
        "import getAccounts from '@salesforce/apex/ns.AccountService.getAccounts';\n"
    )


def test_lwc_schema_import_targets_the_sobject_shared_with_apex():
    assert "sobject.Account" in _lwc_imports("import NAME from '@salesforce/schema/Account.Name';\n")


def test_lwc_schema_import_without_a_field():
    assert "sobject.Contact" in _lwc_imports("import CONTACT from '@salesforce/schema/Contact';\n")


def test_lwc_custom_object_field_reduces_to_the_object():
    assert "sobject.My_Object__c" in _lwc_imports("import F from '@salesforce/schema/My_Object__c.My_Field__c';\n")


def test_lwc_rewrite_replaces_the_raw_specifier():
    """Emitting both would leave a second, unjoinable ext/ stub beside the resolved target."""
    imports = _lwc_imports("import getAccounts from '@salesforce/apex/AccountService.getAccounts';\n")
    assert imports == {"apex.AccountService.getAccounts"}


def test_other_salesforce_pseudo_modules_are_left_alone():
    imports = _lwc_imports(
        "import userId from '@salesforce/user/Id';\nimport label from '@salesforce/label/c.Greeting';\n"
    )
    assert imports == {"@salesforce/user/Id", "@salesforce/label/c.Greeting"}


def test_ordinary_imports_are_unaffected():
    assert _lwc_imports("import { LightningElement } from 'lwc';\n") == {"lwc"}


# ---------------------------------------------------------------------------
# Scope walking: every call reaches a named owner (ADR-0031)
#
# Measured on sindresorhus/ky before this existed: 22.9% of the named function
# forms became entities and 8.5% of the call nodes became edges, because the
# walker only ever looked at the program's direct children.
# ---------------------------------------------------------------------------


def _calls_from(parsed: ParsedFile, from_qn_suffix: str) -> set[str]:
    return {r.to_name for r in _rels_from(parsed, from_qn_suffix, RelType.CALLS)}


def _callable_names(parsed: ParsedFile) -> set[str]:
    return {e.name for e in parsed.entities if e.label == NodeLabel.CALLABLE}


def test_module_scope_calls_belong_to_the_module():
    """Import-time work is nobody's function, so it used to be nobody's edge."""
    parsed = _parse("""\
setupGlobals();
const config = loadConfig();
""")
    assert _calls_from(parsed, "src.example") == {"setupGlobals", "loadConfig"}


def test_calls_in_a_module_level_callback_attribute_to_the_module():
    parsed = _parse("""\
items.forEach((item) => {
  doWork(item);
});
""")
    assert _calls_from(parsed, "src.example") == {"forEach", "doWork"}


def test_a_module_level_callback_gets_no_entity():
    """Category 3 is edges-only: an anonymous arrow must not inflate the node count."""
    parsed = _parse("""\
items.forEach((item) => {
  doWork(item);
});
""")
    assert _callable_names(parsed) == set()


def test_object_literal_method_is_an_entity():
    parsed = _parse("""\
const handlers = {
  async fetch(request) {
    return send(request);
  },
  get duplex() {
    return this.mode;
  },
};
""")
    fetch = _entity_by_name(parsed, "fetch")
    assert fetch.label == NodeLabel.CALLABLE
    assert fetch.kind == CallableKind.METHOD
    assert "async" in fetch.tags
    # Named through the binding, because that is how a developer reaches it.
    assert fetch.qualified_name == f"{PROJECT}:src.example.handlers.fetch"
    assert _entity_by_name(parsed, "duplex").label == NodeLabel.CALLABLE


def test_object_literal_method_is_named_through_a_nested_property_chain():
    parsed = _parse("""\
const config = {
  hooks: {
    beforeRequest(request) {
      return tag(request);
    },
  },
};
""")
    beforerequest = _entity_by_name(parsed, "beforeRequest")
    assert beforerequest.qualified_name == f"{PROJECT}:src.example.config.hooks.beforeRequest"


def test_object_literal_method_owns_its_own_calls():
    parsed = _parse("""\
const handlers = {
  async fetch(request) {
    return send(request);
  },
};
""")
    assert _calls_from(parsed, "src.example.handlers.fetch") == {"send"}
    # ...and the enclosing scope is not credited with them.
    assert "send" not in _calls_from(parsed, "src.example")


def test_object_literal_method_in_argument_position_gets_no_entity():
    """The shape ky uses 179 times. It is spelled `method_definition`, but the object it
    hangs off is an anonymous argument, so no name reaches it — it is a callback, and
    ADR-0031's test is the name, not the grammar node."""
    parsed = _parse("""\
test('retries', async t => {
  await ky('https://x.invalid', {
    async fetch(request) {
      return stub(request);
    },
  });
});
""")
    assert _callable_names(parsed) == set()
    assert _calls_from(parsed, "src.example") == {"test", "ky", "stub"}


def test_repeated_unbound_object_methods_do_not_collide_on_one_uid():
    """Two callbacks of the same shape must not merge into one node. A positive
    assertion cannot catch this — the entity 'exists' either way, and the second
    silently overwrites the first at upsert time."""
    parsed = _parse("""\
test('one', async t => {
  await ky(url, {async fetch(r) { return first(r); }});
});
test('two', async t => {
  await ky(url, {async fetch(r) { return second(r); }});
});
""")
    qns = [e.qualified_name for e in parsed.entities]
    assert len(qns) == len(set(qns)), f"duplicate uid: {sorted(qns)}"
    # Both bodies still reach the graph, attributed upward.
    assert {"first", "second"} <= _calls_from(parsed, "src.example")


def test_two_bound_objects_with_the_same_method_name_stay_distinct():
    parsed = _parse("""\
const alpha = {run() { one(); }};
const beta = {run() { two(); }};
""")
    qns = [e.qualified_name for e in parsed.entities]
    assert len(qns) == len(set(qns)), f"duplicate uid: {sorted(qns)}"
    assert _calls_from(parsed, "src.example.alpha.run") == {"one"}
    assert _calls_from(parsed, "src.example.beta.run") == {"two"}


def test_object_method_on_a_class_field_is_named_through_the_field():
    parsed = _parse("""\
class Widget {
  handlers = {
    refresh() {
      redraw();
    },
  };
}
""")
    refresh = _entity_by_name(parsed, "refresh")
    assert refresh.qualified_name == f"{PROJECT}:src.example.Widget.handlers.refresh"


def test_method_on_an_inline_returned_object_gets_no_entity():
    """`(request) => ({get headers() {...}})` — the object is a return value, not a
    binding, so `headers` is reachable only through whatever the caller does with it."""
    parsed = _parse("""\
const createRequestLike = (request) => ({
  get headers() {
    return request.headers;
  },
});
""")
    assert _callable_names(parsed) == {"createRequestLike"}


def test_nested_function_declaration_is_named_after_its_enclosing_function():
    parsed = _parse("""\
export function delay(ms) {
  return new Promise((resolve, reject) => {
    function abortHandler() {
      clearTimeout(timeoutId);
    }
    signal.addEventListener('abort', abortHandler);
  });
}
""")
    handler = _entity_by_name(parsed, "abortHandler")
    assert handler.label == NodeLabel.CALLABLE
    assert handler.qualified_name == f"{PROJECT}:src.example.delay.abortHandler"
    assert _calls_from(parsed, "src.example.delay.abortHandler") == {"clearTimeout"}
    # The nested body's calls are the nested function's, not delay's.
    assert "clearTimeout" not in _calls_from(parsed, "src.example.delay")


def test_arrow_bound_to_a_local_const_is_an_entity():
    parsed = _parse("""\
test('decodes', async t => {
  const customFetch = async () => {
    return build();
  };
  await use(customFetch);
});
""")
    fn = _entity_by_name(parsed, "customFetch")
    assert fn.label == NodeLabel.CALLABLE
    assert fn.kind == CallableKind.FUNCTION
    assert _calls_from(parsed, "src.example.customFetch") == {"build"}


def test_a_local_const_holding_a_plain_value_gets_no_entity():
    """Only module-level bindings are worth a node; a local would be one per line."""
    parsed = _parse("""\
function outer() {
  const scratch = 41;
  return scratch;
}
""")
    assert [e.name for e in parsed.entities if e.name == "scratch"] == []


def test_a_local_const_initialiser_still_reports_its_calls():
    parsed = _parse("""\
function outer() {
  const value = compute();
  return value;
}
""")
    assert _calls_from(parsed, "src.example.outer") == {"compute"}


def test_destructured_binding_gets_no_entity_but_keeps_its_calls():
    parsed = _parse("const {alpha, beta} = loadPair();\n")
    assert _callable_names(parsed) == set()
    assert _calls_from(parsed, "src.example") == {"loadPair"}


def test_iife_body_calls_belong_to_the_module():
    """`export const x = (() => {...})()` — the shape ky opens constants.ts with."""
    parsed = _parse("""\
export const supported = (() => {
  probeFeature();
  return true;
})();
""")
    assert _calls_from(parsed, "src.example") == {"probeFeature"}
    assert _entity_by_name(parsed, "supported").label == NodeLabel.VALUE


def test_new_expression_is_a_call():
    parsed = _parse("""\
function build() {
  return new Widget(1);
}
""")
    assert _calls_from(parsed, "src.example.build") == {"Widget"}


def test_new_expression_on_a_namespace_records_the_receiver():
    parsed = _parse("""\
function build() {
  return new globalThis.Request('https://x.invalid');
}
""")
    rels = _rels_from(parsed, "src.example.build", RelType.CALLS)
    assert [(r.to_name, r.properties.get("receiver")) for r in rels] == [("Request", "globalThis")]


def test_awaited_call_with_type_arguments_still_names_its_callee():
    """With explicit type arguments the grammar puts `await` *inside* the call, so the
    callee sits one hop further down than it does for `await ky.get(u).json()`."""
    parsed = _parse("""\
async function run() {
  return await ky.get(url).json<Payload>();
}
""")
    assert "json" in _calls_from(parsed, "src.example.run")


def test_non_null_assertion_before_a_call_still_names_its_callee():
    parsed = _parse("""\
function run() {
  return options.parseJson!(text);
}
""")
    assert "parseJson" in _calls_from(parsed, "src.example.run")


def test_generator_function_declaration_is_an_entity():
    parsed = _parse("""\
function* pages() {
  yield fetchPage();
}
""")
    gen = _entity_by_name(parsed, "pages")
    assert gen.label == NodeLabel.CALLABLE
    assert _calls_from(parsed, "src.example.pages") == {"fetchPage"}


def test_named_class_expression_is_an_entity_with_its_methods():
    """`globalThis.Headers = class Headers extends Base {...}` — a real ky test shape."""
    parsed = _parse("""\
globalThis.Headers = class Headers extends OriginalHeaders {
  constructor(init) {
    super(init);
    record(init);
  }
};
""")
    cls = _entity_by_name(parsed, "Headers")
    assert cls.label == NodeLabel.TYPE_DEF
    ctor = _entity_by_name(parsed, "constructor")
    assert ctor.qualified_name == f"{PROJECT}:src.example.Headers.constructor"
    assert "record" in _calls_from(parsed, "src.example.Headers.constructor")


def test_export_default_expression_keeps_its_calls():
    parsed = _parse("""\
export default () => {
  boot();
};
""")
    assert _calls_from(parsed, "src.example") == {"boot"}


def test_class_field_initialiser_calls_belong_to_the_class():
    parsed = _parse("""\
class Widget {
  private handler = () => {
    refresh();
  };
}
""")
    assert _calls_from(parsed, "src.example.Widget") == {"refresh"}


def test_deeply_nested_callbacks_attribute_to_the_nearest_named_scope():
    """ADR-0031 accepts losing the intermediate structure — but not the edge."""
    parsed = _parse("""\
function outer() {
  a(() => {
    b(() => {
      c(() => {
        deep();
      });
    });
  });
}
""")
    assert _calls_from(parsed, "src.example.outer") == {"a", "b", "c", "deep"}


def test_js_scope_walking():
    """.js/.mjs share this walker, and the JS grammar is a separate grammar object."""
    if get_language_for_file("src/example.js") is None:
        pytest.skip("tree-sitter-javascript not installed")
    parsed = _parse(
        """\
register(handler);
const boot = () => {
  start();
};
module.exports = {
  run() {
    go();
  },
};
""",
        path="src/example.js",
    )
    assert _calls_from(parsed, "src.example.boot") == {"start"}
    # `module.exports = {...}` is an assignment, not a binding this walker names,
    # so `run` is a callback: no entity, and `go` belongs to the module.
    assert _calls_from(parsed, "src.example") == {"register", "go"}
    assert _callable_names(parsed) == {"boot"}


def test_js_function_expression_bound_to_a_name_is_an_entity():
    """`const f = function () {}` is the pre-arrow spelling of the same binding."""
    if get_language_for_file("src/example.js") is None:
        pytest.skip("tree-sitter-javascript not installed")
    parsed = _parse("const legacy = function () {\n  work();\n};\n", path="src/example.js")
    assert _entity_by_name(parsed, "legacy").label == NodeLabel.CALLABLE
    assert _calls_from(parsed, "src.example.legacy") == {"work"}


def test_tsx_callback_inside_jsx_keeps_its_calls():
    parsed = _parse(
        """\
export function Panel() {
  return <button onClick={() => track('click')}>go</button>;
}
""",
        path="src/Panel.tsx",
    )
    assert "track" in _calls_from(parsed, "src.Panel.Panel")
