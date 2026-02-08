"""Unit tests for the tree-sitter parser module."""

from __future__ import annotations

from code_atlas.parser import ParsedFile, get_language_for_file, parse_file
from code_atlas.schema import (
    CallableKind,
    NodeLabel,
    RelType,
    TypeDefKind,
    ValueKind,
    Visibility,
)

PROJECT = "test_project"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(source: str, path: str = "src/example.py") -> ParsedFile:
    result = parse_file(path, source.encode("utf-8"), PROJECT)
    assert result is not None
    return result


def _entity_by_name(parsed: ParsedFile, name: str):
    matches = [e for e in parsed.entities if e.name == name]
    names = [e.name for e in parsed.entities]
    assert len(matches) == 1, f"Expected 1 entity named {name!r}, got {len(matches)}: {names}"
    return matches[0]


def _rels_from(parsed: ParsedFile, from_qn_suffix: str, rel_type: RelType):
    return [
        r for r in parsed.relationships if r.from_qualified_name.endswith(from_qn_suffix) and r.rel_type == rel_type
    ]


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def test_language_detection_python():
    assert get_language_for_file("src/main.py") is not None
    assert get_language_for_file("src/stubs.pyi") is not None


def test_language_detection_unsupported():
    assert get_language_for_file("src/main.rs") is None
    assert get_language_for_file("data.csv") is None


# ---------------------------------------------------------------------------
# Module / Package
# ---------------------------------------------------------------------------


def test_module_entity():
    parsed = _parse("x = 1\n", path="src/code_atlas/parser.py")
    module = _entity_by_name(parsed, "parser")
    assert module.label == NodeLabel.MODULE
    assert module.kind == "module"
    assert module.qualified_name == f"{PROJECT}:src.code_atlas.parser"


def test_package_entity():
    parsed = _parse("", path="src/code_atlas/__init__.py")
    pkg = _entity_by_name(parsed, "code_atlas")
    assert pkg.label == NodeLabel.PACKAGE
    assert pkg.kind == "package"
    assert pkg.qualified_name == f"{PROJECT}:src.code_atlas"


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


def test_class_basic():
    parsed = _parse(
        '''\
class MyClass:
    """A docstring."""
    pass
'''
    )
    cls = _entity_by_name(parsed, "MyClass")
    assert cls.label == NodeLabel.TYPE_DEF
    assert cls.kind == TypeDefKind.CLASS
    assert cls.docstring == "A docstring."
    assert cls.visibility == Visibility.PUBLIC


def test_class_inheritance():
    parsed = _parse("class Child(Parent, Mixin):\n    pass\n")
    inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
    base_names = {r.to_name for r in inherits}
    assert "Parent" in base_names
    assert "Mixin" in base_names


def test_private_class():
    parsed = _parse("class _PrivateClass:\n    pass\n")
    cls = _entity_by_name(parsed, "_PrivateClass")
    assert cls.visibility == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# Functions and methods
# ---------------------------------------------------------------------------


def test_function():
    parsed = _parse("def my_func(x, y):\n    return x + y\n")
    func = _entity_by_name(parsed, "my_func")
    assert func.label == NodeLabel.CALLABLE
    assert func.kind == CallableKind.FUNCTION
    assert func.signature is not None
    assert "my_func" in func.signature


def test_method_vs_function():
    parsed = _parse(
        """\
class Foo:
    def bar(self):
        pass

def baz():
    pass
"""
    )
    bar = _entity_by_name(parsed, "bar")
    assert bar.kind == CallableKind.METHOD
    assert bar.qualified_name == f"{PROJECT}:src.example.Foo.bar"

    baz = _entity_by_name(parsed, "baz")
    assert baz.kind == CallableKind.FUNCTION
    assert baz.qualified_name == f"{PROJECT}:src.example.baz"


def test_constructor():
    parsed = _parse("class Foo:\n    def __init__(self):\n        pass\n")
    init = _entity_by_name(parsed, "__init__")
    assert init.kind == CallableKind.CONSTRUCTOR


def test_static_method():
    parsed = _parse("class Foo:\n    @staticmethod\n    def bar():\n        pass\n")
    bar = _entity_by_name(parsed, "bar")
    assert bar.kind == CallableKind.STATIC_METHOD


def test_class_method():
    parsed = _parse("class Foo:\n    @classmethod\n    def bar(cls):\n        pass\n")
    bar = _entity_by_name(parsed, "bar")
    assert bar.kind == CallableKind.CLASS_METHOD


def test_property():
    parsed = _parse("class Foo:\n    @property\n    def name(self):\n        return self._name\n")
    name = _entity_by_name(parsed, "name")
    assert name.kind == CallableKind.PROPERTY


def test_function_docstring():
    parsed = _parse(
        '''\
def greet(name):
    """Say hello."""
    print(f"Hello {name}")
'''
    )
    func = _entity_by_name(parsed, "greet")
    assert func.docstring == "Say hello."


def test_private_function():
    parsed = _parse("def _private():\n    pass\n")
    func = _entity_by_name(parsed, "_private")
    assert func.visibility == Visibility.PRIVATE


def test_dunder_function_public():
    parsed = _parse("class Foo:\n    def __repr__(self):\n        pass\n")
    func = _entity_by_name(parsed, "__repr__")
    assert func.visibility == Visibility.PUBLIC


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


def test_import_statement():
    parsed = _parse("import os\nimport sys\n")
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    imported = {r.to_name for r in import_rels}
    assert "os" in imported
    assert "sys" in imported


def test_import_from():
    parsed = _parse("from os.path import join, exists\n")
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    imported = {r.to_name for r in import_rels}
    assert "os.path.join" in imported
    assert "os.path.exists" in imported


# ---------------------------------------------------------------------------
# Module-level assignments (Values)
# ---------------------------------------------------------------------------


def test_variable():
    parsed = _parse("my_var = 42\n")
    var = _entity_by_name(parsed, "my_var")
    assert var.label == NodeLabel.VALUE
    assert var.kind == ValueKind.VARIABLE


def test_constant():
    parsed = _parse("MAX_SIZE = 100\n")
    const = _entity_by_name(parsed, "MAX_SIZE")
    assert const.label == NodeLabel.VALUE
    assert const.kind == ValueKind.CONSTANT


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


def test_defines_relationships():
    parsed = _parse(
        """\
class Foo:
    def bar(self):
        pass

def baz():
    pass
"""
    )
    # Module DEFINES Foo
    mod_defines = _rels_from(parsed, "src.example", RelType.DEFINES)
    targets = {r.to_name for r in mod_defines}
    assert f"{PROJECT}:src.example.Foo" in targets
    assert f"{PROJECT}:src.example.baz" in targets

    # Foo DEFINES bar
    foo_defines = _rels_from(parsed, "src.example.Foo", RelType.DEFINES)
    assert any(r.to_name == f"{PROJECT}:src.example.Foo.bar" for r in foo_defines)


def test_calls_relationship():
    parsed = _parse(
        """\
def caller():
    print("hello")
    some_func()
"""
    )
    calls = _rels_from(parsed, "src.example.caller", RelType.CALLS)
    called = {r.to_name for r in calls}
    assert "print" in called
    assert "some_func" in called


# ---------------------------------------------------------------------------
# Edge cases (competitor insight P0)
# ---------------------------------------------------------------------------


def test_empty_file():
    parsed = _parse("")
    assert parsed is not None
    assert parsed.language == "python"
    # Should have at least the module entity
    assert len(parsed.entities) >= 1


def test_syntax_error_tolerant():
    """Tree-sitter is error-tolerant — malformed files don't crash."""
    parsed = _parse("def broken(\n    class nope\n")
    assert parsed is not None


def test_unsupported_extension():
    result = parse_file("main.rs", b"fn main() {}", PROJECT)
    assert result is None


def test_binary_content():
    """Binary content shouldn't crash the parser."""
    parsed = parse_file("data.py", b"\x00\x01\x02\xff\xfe", PROJECT)
    assert parsed is not None


# ---------------------------------------------------------------------------
# Decorators as tags
# ---------------------------------------------------------------------------


def test_decorator_tags():
    parsed = _parse(
        """\
class Foo:
    @staticmethod
    def bar():
        pass
"""
    )
    bar = _entity_by_name(parsed, "bar")
    assert any("staticmethod" in t for t in bar.tags)


# ---------------------------------------------------------------------------
# Qualified names
# ---------------------------------------------------------------------------


def test_nested_class_qualified_name():
    parsed = _parse(
        """\
class Outer:
    class Inner:
        pass
"""
    )
    inner = _entity_by_name(parsed, "Inner")
    assert "Outer.Inner" in inner.qualified_name


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------


def _parse_md(source: str, path: str = "docs/test.md") -> ParsedFile:
    result = parse_file(path, source.encode("utf-8"), PROJECT)
    assert result is not None
    return result


def _sections(parsed: ParsedFile) -> list:
    return [e for e in parsed.entities if e.label == NodeLabel.DOC_SECTION]


def test_markdown_basic_sections():
    parsed = _parse_md(
        """\
# Introduction

Intro text.

## Details

Detail text.
"""
    )
    doc_file = [e for e in parsed.entities if e.label == NodeLabel.DOC_FILE]
    assert len(doc_file) == 1
    assert doc_file[0].name == "test.md"

    sections = _sections(parsed)
    assert len(sections) == 2

    intro = _entity_by_name(parsed, "Introduction")
    assert intro.label == NodeLabel.DOC_SECTION
    assert intro.header_level == 1
    assert intro.header_path == "Introduction"
    assert intro.line_start == 1

    details = _entity_by_name(parsed, "Details")
    assert details.header_level == 2
    assert details.header_path == "Introduction > Details"


def test_markdown_nested_headers():
    parsed = _parse_md(
        """\
# Top

## Middle

### Deep

Deepest content.
"""
    )
    sections = _sections(parsed)
    assert len(sections) == 3

    deep = _entity_by_name(parsed, "Deep")
    assert deep.header_level == 3
    assert deep.header_path == "Top > Middle > Deep"


def test_markdown_header_path_disambiguation():
    parsed = _parse_md(
        """\
# Parent A

## Overview

Content A.

# Parent B

## Overview

Content B.
"""
    )
    sections = _sections(parsed)
    overview_sections = [s for s in sections if s.name == "Overview"]
    assert len(overview_sections) == 2
    qns = {s.qualified_name for s in overview_sections}
    assert f"{PROJECT}:docs/test.md > Parent A > Overview" in qns
    assert f"{PROJECT}:docs/test.md > Parent B > Overview" in qns


def test_markdown_code_blocks():
    parsed = _parse_md(
        """\
# Code Section

```python
print("hello")
```

```bash
echo hi
```
"""
    )
    section = _entity_by_name(parsed, "Code Section")
    assert "lang:python" in section.tags
    assert "lang:bash" in section.tags


def test_markdown_preamble():
    parsed = _parse_md(
        """\
This is preamble text.

More preamble.

# First Heading

Content.
"""
    )
    sections = _sections(parsed)
    preamble = [s for s in sections if s.header_level == 0]
    assert len(preamble) == 1
    assert preamble[0].name == "test.md"
    assert preamble[0].docstring is not None
    assert "preamble" in preamble[0].docstring.lower()


def test_markdown_setext_headings():
    parsed = _parse_md(
        """\
Title
=====

Some text.

Subtitle
--------

More text.
"""
    )
    title = _entity_by_name(parsed, "Title")
    assert title.header_level == 1

    subtitle = _entity_by_name(parsed, "Subtitle")
    assert subtitle.header_level == 2
    assert subtitle.header_path == "Title > Subtitle"


def test_markdown_empty_file():
    parsed = _parse_md("")
    doc_files = [e for e in parsed.entities if e.label == NodeLabel.DOC_FILE]
    assert len(doc_files) == 1
    assert doc_files[0].name == "test.md"
    assert _sections(parsed) == []


def test_markdown_contains_relationships():
    parsed = _parse_md(
        """\
# One

## Two

## Three
"""
    )
    contains_rels = [r for r in parsed.relationships if r.rel_type == RelType.CONTAINS]
    assert len(contains_rels) == 3
    for rel in contains_rels:
        assert rel.from_qualified_name == f"{PROJECT}:docs/test.md"


def test_markdown_language_detection():
    assert get_language_for_file("docs/readme.md") is not None
    assert get_language_for_file("notes.txt") is None
    assert get_language_for_file("readme.rst") is None


def test_markdown_content_extraction():
    parsed = _parse_md(
        """\
# Section

The quick brown fox.

Another paragraph.
"""
    )
    section = _entity_by_name(parsed, "Section")
    assert section.docstring is not None
    assert "quick brown fox" in section.docstring
    assert "Another paragraph" in section.docstring
