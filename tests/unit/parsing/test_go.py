"""Tests for Go parser."""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_go", reason="tree-sitter-go not installed")

from code_atlas.parsing.ast import ParsedFile, get_language_for_file, parse_file
from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind, ValueKind, Visibility

PROJECT = "test_project"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(source: str, path: str = "src/example.go") -> ParsedFile:
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


def test_language_detection_go():
    cfg = get_language_for_file("src/main.go")
    assert cfg is not None
    assert cfg.name == "go"


def test_language_detection_not_go():
    assert get_language_for_file("data.csv") is None
    assert get_language_for_file("readme.txt") is None


# ---------------------------------------------------------------------------
# 2. Module entity creation
# ---------------------------------------------------------------------------


def test_module_entity():
    parsed = _parse("package main\n", path="src/server.go")
    mod = _entity_by_name(parsed, "server")
    assert mod.label == NodeLabel.MODULE
    assert mod.kind == "module"
    assert mod.qualified_name == f"{PROJECT}:src.server"


def test_module_entity_nested_path():
    parsed = _parse("package handlers\n", path="cmd/api/handlers.go")
    mod = _entity_by_name(parsed, "handlers")
    assert mod.qualified_name == f"{PROJECT}:cmd.api.handlers"


def test_module_entity_test_file():
    parsed = _parse("package main\n", path="src/server_test.go")
    mod = _entity_by_name(parsed, "server_test")
    assert mod.qualified_name == f"{PROJECT}:src.server_test"


# ---------------------------------------------------------------------------
# 3. Struct extraction
# ---------------------------------------------------------------------------


def test_struct_basic():
    parsed = _parse("""\
package main

type Server struct {
    Name string
    port int
}
""")
    srv = _entity_by_name(parsed, "Server")
    assert srv.label == NodeLabel.TYPE_DEF
    assert srv.kind == TypeDefKind.STRUCT
    assert srv.visibility == Visibility.PUBLIC


# ---------------------------------------------------------------------------
# 4. Interface extraction
# ---------------------------------------------------------------------------


def test_interface_basic():
    parsed = _parse("""\
package main

type Reader interface {
    Read(p []byte) (n int, err error)
}
""")
    reader = _entity_by_name(parsed, "Reader")
    assert reader.label == NodeLabel.TYPE_DEF
    assert reader.kind == TypeDefKind.INTERFACE


def test_interface_method_spec():
    """Interface methods are extracted as Callable METHOD."""
    parsed = _parse("""\
package main

type Reader interface {
    Read(p []byte) (n int, err error)
}
""")
    read = _entity_by_name(parsed, "Read")
    assert read.label == NodeLabel.CALLABLE
    assert read.kind == CallableKind.METHOD
    assert read.signature is not None
    assert "Read" in read.signature


# ---------------------------------------------------------------------------
# 5. Type alias extraction
# ---------------------------------------------------------------------------


def test_type_alias():
    parsed = _parse("""\
package main

type Alias = int
""")
    alias = _entity_by_name(parsed, "Alias")
    assert alias.label == NodeLabel.TYPE_DEF
    assert alias.kind == TypeDefKind.TYPE_ALIAS


# ---------------------------------------------------------------------------
# 6. Function extraction
# ---------------------------------------------------------------------------


def test_function_basic():
    parsed = _parse("""\
package main

func hello(name string) string { return "Hello " + name }
""")
    func = _entity_by_name(parsed, "hello")
    assert func.label == NodeLabel.CALLABLE
    assert func.kind == CallableKind.FUNCTION
    assert func.qualified_name == f"{PROJECT}:src.example.hello"


# ---------------------------------------------------------------------------
# 7. Method (with receiver) extraction
# ---------------------------------------------------------------------------


def test_method_with_pointer_receiver():
    parsed = _parse("""\
package main

func (s *Server) Handle(req Request) { }
""")
    method = _entity_by_name(parsed, "Handle")
    assert method.label == NodeLabel.CALLABLE
    assert method.kind == CallableKind.METHOD
    assert method.qualified_name == f"{PROJECT}:src.example.Server.Handle"


def test_method_with_value_receiver():
    parsed = _parse("""\
package main

func (s Server) String() string { return s.Name }
""")
    method = _entity_by_name(parsed, "String")
    assert method.kind == CallableKind.METHOD
    assert method.qualified_name == f"{PROJECT}:src.example.Server.String"


# ---------------------------------------------------------------------------
# 8. Visibility (capitalization-based)
# ---------------------------------------------------------------------------


def test_visibility_public():
    parsed = _parse("""\
package main

func Hello() {}
""")
    func = _entity_by_name(parsed, "Hello")
    assert func.visibility == Visibility.PUBLIC


def test_visibility_internal():
    parsed = _parse("""\
package main

func hello() {}
""")
    func = _entity_by_name(parsed, "hello")
    assert func.visibility == Visibility.INTERNAL


def test_visibility_struct_fields():
    parsed = _parse("""\
package main

type Server struct {
    Name string
    port int
}
""")
    name_field = _entity_by_name(parsed, "Name")
    assert name_field.visibility == Visibility.PUBLIC

    port_field = _entity_by_name(parsed, "port")
    assert port_field.visibility == Visibility.INTERNAL


# ---------------------------------------------------------------------------
# 9. Import extraction -> IMPORTS
# ---------------------------------------------------------------------------


def test_import_single():
    parsed = _parse("""\
package main

import "fmt"
""")
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    assert len(import_rels) == 1
    assert import_rels[0].to_name == "fmt"


def test_import_grouped():
    parsed = _parse("""\
package main

import (
    "fmt"
    "net/http"
)
""")
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    imported = {r.to_name for r in import_rels}
    assert "fmt" in imported
    assert "net/http" in imported


# ---------------------------------------------------------------------------
# 10. Struct embedding -> INHERITS
# ---------------------------------------------------------------------------


def test_struct_embedding():
    parsed = _parse("""\
package main

type Base struct {}

type Child struct {
    Base
    Name string
}
""")
    inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
    assert len(inherits) == 1
    assert inherits[0].to_name == "Base"
    assert inherits[0].from_qualified_name.endswith("Child")


# ---------------------------------------------------------------------------
# 11. Doc comments extraction
# ---------------------------------------------------------------------------


def test_doc_comment_function():
    parsed = _parse("""\
package main

// Hello greets someone.
// It returns a greeting string.
func Hello(name string) string { return "Hello " + name }
""")
    func = _entity_by_name(parsed, "Hello")
    assert func.docstring is not None
    assert "Hello greets someone." in func.docstring
    assert "It returns a greeting string." in func.docstring


def test_doc_comment_struct():
    parsed = _parse("""\
package main

// Server handles HTTP requests.
type Server struct {
    Name string
}
""")
    srv = _entity_by_name(parsed, "Server")
    assert srv.docstring is not None
    assert "Server handles HTTP requests." in srv.docstring


# ---------------------------------------------------------------------------
# 12. Signature extraction
# ---------------------------------------------------------------------------


def test_function_signature():
    parsed = _parse("""\
package main

func hello(name string) string { return "Hello " + name }
""")
    func = _entity_by_name(parsed, "hello")
    assert func.signature is not None
    assert "func hello(name string) string" in func.signature
    # Body should not be in signature
    assert "return" not in func.signature


def test_method_signature():
    parsed = _parse("""\
package main

func (s *Server) Handle(req Request) error { return nil }
""")
    method = _entity_by_name(parsed, "Handle")
    assert method.signature is not None
    assert "(s *Server)" in method.signature
    assert "Handle(req Request) error" in method.signature
    assert "return" not in method.signature


# ---------------------------------------------------------------------------
# 13. Constants and variables
# ---------------------------------------------------------------------------


def test_const_single():
    parsed = _parse("""\
package main

const MaxRetries = 3
""")
    c = _entity_by_name(parsed, "MaxRetries")
    assert c.label == NodeLabel.VALUE
    assert c.kind == ValueKind.CONSTANT


def test_var_single():
    parsed = _parse("""\
package main

var globalCount int
""")
    v = _entity_by_name(parsed, "globalCount")
    assert v.label == NodeLabel.VALUE
    assert v.kind == ValueKind.VARIABLE


def test_const_grouped():
    parsed = _parse("""\
package main

const (
    A = 1
    B = 2
)
""")
    a = _entity_by_name(parsed, "A")
    assert a.kind == ValueKind.CONSTANT
    b = _entity_by_name(parsed, "B")
    assert b.kind == ValueKind.CONSTANT


def test_var_grouped():
    parsed = _parse("""\
package main

var (
    c = 3
    d int
)
""")
    c = _entity_by_name(parsed, "c")
    assert c.kind == ValueKind.VARIABLE
    d = _entity_by_name(parsed, "d")
    assert d.kind == ValueKind.VARIABLE


# ---------------------------------------------------------------------------
# 14. Struct fields
# ---------------------------------------------------------------------------


def test_struct_fields():
    parsed = _parse("""\
package main

type Server struct {
    Name string
    port int
}
""")
    name_field = _entity_by_name(parsed, "Name")
    assert name_field.label == NodeLabel.VALUE
    assert name_field.kind == ValueKind.FIELD
    assert name_field.qualified_name == f"{PROJECT}:src.example.Server.Name"

    port_field = _entity_by_name(parsed, "port")
    assert port_field.label == NodeLabel.VALUE
    assert port_field.kind == ValueKind.FIELD
    assert port_field.qualified_name == f"{PROJECT}:src.example.Server.port"


# ---------------------------------------------------------------------------
# 15. DEFINES relationships
# ---------------------------------------------------------------------------


def test_defines_module_to_function():
    parsed = _parse("""\
package main

func hello() {}
""")
    defines = _rels_from(parsed, "src.example", RelType.DEFINES)
    targets = {r.to_name for r in defines}
    assert f"{PROJECT}:src.example.hello" in targets


def test_defines_module_to_struct():
    parsed = _parse("""\
package main

type Server struct {}
""")
    defines = _rels_from(parsed, "src.example", RelType.DEFINES)
    targets = {r.to_name for r in defines}
    assert f"{PROJECT}:src.example.Server" in targets


def test_defines_struct_to_method():
    parsed = _parse("""\
package main

func (s *Server) Handle() {}
""")
    defines = _rels_from(parsed, "src.example.Server", RelType.DEFINES)
    targets = {r.to_name for r in defines}
    assert f"{PROJECT}:src.example.Server.Handle" in targets


def test_defines_struct_to_field():
    parsed = _parse("""\
package main

type Server struct {
    Name string
}
""")
    defines = _rels_from(parsed, "src.example.Server", RelType.DEFINES)
    targets = {r.to_name for r in defines}
    assert f"{PROJECT}:src.example.Server.Name" in targets


# ---------------------------------------------------------------------------
# 16. CALLS extraction
# ---------------------------------------------------------------------------


def test_calls_simple():
    parsed = _parse("""\
package main

func main() { hello("world") }
""")
    calls = _rels_from(parsed, "src.example.main", RelType.CALLS)
    called = {r.to_name for r in calls}
    assert "hello" in called


def test_calls_selector():
    parsed = _parse("""\
package main

import "fmt"

func main() { fmt.Println("hello") }
""")
    calls = _rels_from(parsed, "src.example.main", RelType.CALLS)
    called = {r.to_name for r in calls}
    assert "Println" in called


def test_calls_from_method():
    parsed = _parse("""\
package main

import "fmt"

func (s *Server) Handle() { fmt.Println(s.Name) }
""")
    calls = _rels_from(parsed, "src.example.Server.Handle", RelType.CALLS)
    called = {r.to_name for r in calls}
    assert "Println" in called


# ---------------------------------------------------------------------------
# 17. Special tags (init, main, test)
# ---------------------------------------------------------------------------


def test_tag_init():
    parsed = _parse("""\
package main

func init() {}
""")
    func = _entity_by_name(parsed, "init")
    assert "init_func" in func.tags


def test_tag_main():
    parsed = _parse("""\
package main

func main() {}
""")
    func = _entity_by_name(parsed, "main")
    assert "entry_point" in func.tags


def test_tag_test():
    parsed = _parse("""\
package main

func TestFoo(t *testing.T) {}
""")
    func = _entity_by_name(parsed, "TestFoo")
    assert "test" in func.tags


def test_tag_benchmark():
    parsed = _parse("""\
package main

func BenchmarkFoo(b *testing.B) {}
""")
    func = _entity_by_name(parsed, "BenchmarkFoo")
    assert "benchmark" in func.tags


# ---------------------------------------------------------------------------
# 18. Content hash determinism
# ---------------------------------------------------------------------------


def test_content_hash_populated():
    parsed = _parse("""\
package main

func hello() {}
""")
    for entity in parsed.entities:
        assert entity.content_hash, f"Entity {entity.name!r} has empty content_hash"


def test_content_hash_deterministic():
    source = """\
package main

func hello(name string) string { return "Hello " + name }
"""
    parsed1 = _parse(source)
    parsed2 = _parse(source)
    for e1, e2 in zip(parsed1.entities, parsed2.entities, strict=True):
        assert e1.content_hash == e2.content_hash


def test_content_hash_ignores_line_shift():
    source_v1 = """\
package main

func greet() {}
"""
    source_v2 = """\
package main



func greet() {}
"""
    parsed1 = _parse(source_v1)
    parsed2 = _parse(source_v2)
    func1 = _entity_by_name(parsed1, "greet")
    func2 = _entity_by_name(parsed2, "greet")
    assert func1.content_hash == func2.content_hash
    assert func1.line_start != func2.line_start


# ---------------------------------------------------------------------------
# 19. Edge cases (empty file, syntax errors)
# ---------------------------------------------------------------------------


def test_empty_file():
    parsed = _parse("")
    assert parsed is not None
    assert parsed.language == "go"
    # Should have at least the module entity
    assert len(parsed.entities) >= 1


def test_syntax_error_tolerant():
    """Tree-sitter is error-tolerant -- malformed files don't crash."""
    parsed = _parse("package main\nfunc broken(\ntype nope\n")
    assert parsed is not None


def test_unsupported_extension():
    result = parse_file("data.csv", b"a,b,c", PROJECT)
    assert result is None


# ---------------------------------------------------------------------------
# 20. Method receiver in qualified name
# ---------------------------------------------------------------------------


def test_method_receiver_pointer_qn():
    parsed = _parse("""\
package main

func (s *Server) Handle() {}
""")
    method = _entity_by_name(parsed, "Handle")
    assert "Server.Handle" in method.qualified_name
    assert f"{PROJECT}:src.example.Server.Handle" == method.qualified_name


def test_method_receiver_value_qn():
    parsed = _parse("""\
package main

func (s Server) String() string { return "" }
""")
    method = _entity_by_name(parsed, "String")
    assert "Server.String" in method.qualified_name
    assert f"{PROJECT}:src.example.Server.String" == method.qualified_name


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_source_extraction():
    """Function entities have source containing full function text."""
    parsed = _parse("""\
package main

func greet(name string) string { return "Hello " + name }
""")
    func = _entity_by_name(parsed, "greet")
    assert func.source is not None
    assert "func greet(name string)" in func.source


def test_const_var_defines():
    """Constants and variables have DEFINES from module."""
    parsed = _parse("""\
package main

const MaxRetries = 3
var globalCount int
""")
    defines = _rels_from(parsed, "src.example", RelType.DEFINES)
    targets = {r.to_name for r in defines}
    assert f"{PROJECT}:src.example.MaxRetries" in targets
    assert f"{PROJECT}:src.example.globalCount" in targets


def test_interface_embedding():
    """Interfaces that embed other interfaces produce INHERITS."""
    parsed = _parse("""\
package main

type ReadWriter interface {
    Reader
    Writer
}
""")
    inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
    embedded_names = {r.to_name for r in inherits}
    assert "Reader" in embedded_names
    assert "Writer" in embedded_names
