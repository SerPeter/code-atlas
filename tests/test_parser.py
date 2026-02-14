"""Unit tests for the tree-sitter parser module."""

from __future__ import annotations

from code_atlas.parsing.ast import ParsedFile, get_language_for_file, parse_file
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


def test_decorator_tags_with_args():
    """Decorator with arguments preserves full text including args."""
    parsed = _parse(
        """\
@app.get("/users/{id}")
def get_user(id: int):
    pass
"""
    )
    get_user = _entity_by_name(parsed, "get_user")
    assert 'decorator:app.get("/users/{id}")' in get_user.tags


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


# ---------------------------------------------------------------------------
# Markdown doc-code linking
# ---------------------------------------------------------------------------


def _doc_rels(parsed: ParsedFile) -> list:
    return [r for r in parsed.relationships if r.rel_type == RelType.DOCUMENTS]


def test_md_header_as_symbol():
    """CamelCase heading at H2+ emits an explicit doc-code link."""
    parsed = _parse_md(
        """\
# Docs

## UserService

Describes the user service.
"""
    )
    rels = _doc_rels(parsed)
    assert len(rels) == 1
    assert rels[0].to_name == "UserService"
    assert rels[0].properties["link_type"] == "explicit"
    assert rels[0].properties["confidence"] == 0.9


def test_md_header_snake_case():
    """snake_case heading at H2+ emits an explicit doc-code link."""
    parsed = _parse_md(
        """\
# API

## validate_token

Validates the token.
"""
    )
    rels = _doc_rels(parsed)
    assert any(r.to_name == "validate_token" and r.properties["link_type"] == "explicit" for r in rels)


def test_md_backtick_symbols():
    """Backtick mentions in content emit symbol_mention links."""
    parsed = _parse_md(
        """\
# Overview

Use `validate_token()` and `UserService` for authentication.
"""
    )
    rels = _doc_rels(parsed)
    names = {r.to_name for r in rels}
    assert "validate_token" in names
    assert "UserService" in names
    for rel in rels:
        assert rel.properties["link_type"] == "symbol_mention"
        assert rel.properties["confidence"] == 0.8


def test_md_file_path_refs():
    """File path patterns in content emit file_ref links."""
    parsed = _parse_md(
        """\
# Architecture

The auth module lives in `src/auth/service.py`.
"""
    )
    rels = _doc_rels(parsed)
    file_rels = [r for r in rels if r.properties.get("is_file_ref")]
    assert len(file_rels) == 1
    assert file_rels[0].to_name == "src/auth/service.py"
    assert file_rels[0].properties["link_type"] == "file_ref"
    assert file_rels[0].properties["confidence"] == 0.85


def test_md_dedup_highest_confidence():
    """Same symbol in heading and body keeps heading's higher confidence."""
    parsed = _parse_md(
        """\
# Docs

## UserService

The `UserService` handles users.
"""
    )
    rels = _doc_rels(parsed)
    user_rels = [r for r in rels if r.to_name == "UserService"]
    assert len(user_rels) == 1
    assert user_rels[0].properties["confidence"] == 0.9
    assert user_rels[0].properties["link_type"] == "explicit"


def test_md_short_names_filtered():
    """Names shorter than 3 chars are excluded."""
    parsed = _parse_md(
        """\
# Notes

Use `os` and `io` modules.
"""
    )
    rels = _doc_rels(parsed)
    assert len(rels) == 0


def test_md_h1_not_explicit():
    """H1 headings are doc titles, not code references even if CamelCase."""
    parsed = _parse_md(
        """\
# UserService

Some content.
"""
    )
    rels = _doc_rels(parsed)
    explicit_rels = [r for r in rels if r.properties.get("link_type") == "explicit"]
    assert len(explicit_rels) == 0


def test_md_no_refs_plain_heading():
    """Multi-word headings don't match identifier pattern."""
    parsed = _parse_md(
        """\
# Getting Started

## How to install

Just run the installer.
"""
    )
    rels = _doc_rels(parsed)
    explicit_rels = [r for r in rels if r.properties.get("link_type") == "explicit"]
    assert len(explicit_rels) == 0


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------


def test_content_hash_populated():
    """Every entity produced by parse_file has a non-empty content_hash."""
    parsed = _parse(
        """\
class Foo:
    def bar(self):
        pass
"""
    )
    for entity in parsed.entities:
        assert entity.content_hash, f"Entity {entity.name!r} has empty content_hash"


def test_content_hash_deterministic():
    """Parsing the same source twice produces identical content_hashes."""
    source = """\
def greet(name):
    \"\"\"Say hello.\"\"\"
    print(f"Hello {name}")
"""
    parsed1 = _parse(source)
    parsed2 = _parse(source)
    for e1, e2 in zip(parsed1.entities, parsed2.entities, strict=True):
        assert e1.content_hash == e2.content_hash


def test_content_hash_ignores_line_shift():
    """Inserting blank lines above an entity doesn't change its content_hash."""
    source_v1 = "def greet():\n    pass\n"
    source_v2 = "\n\n\ndef greet():\n    pass\n"
    parsed1 = _parse(source_v1)
    parsed2 = _parse(source_v2)
    func1 = _entity_by_name(parsed1, "greet")
    func2 = _entity_by_name(parsed2, "greet")
    assert func1.content_hash == func2.content_hash
    # But line_start differs
    assert func1.line_start != func2.line_start


def test_content_hash_changes_on_signature():
    """Different function parameters produce different content_hashes."""
    parsed1 = _parse("def work(x):\n    pass\n")
    parsed2 = _parse("def work(x, y):\n    pass\n")
    func1 = _entity_by_name(parsed1, "work")
    func2 = _entity_by_name(parsed2, "work")
    assert func1.content_hash != func2.content_hash


def test_content_hash_changes_on_docstring():
    """Different docstrings produce different content_hashes."""
    parsed1 = _parse('def work():\n    """Version 1."""\n    pass\n')
    parsed2 = _parse('def work():\n    """Version 2."""\n    pass\n')
    func1 = _entity_by_name(parsed1, "work")
    func2 = _entity_by_name(parsed2, "work")
    assert func1.content_hash != func2.content_hash


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------


def test_function_source_extracted():
    """Function entities have source containing full function text."""
    parsed = _parse("def greet(name):\n    return f'Hello {name}'\n")
    func = _entity_by_name(parsed, "greet")
    assert func.source is not None
    assert "def greet(name):" in func.source
    assert "return f'Hello {name}'" in func.source


def test_assignment_source_extracted():
    """Assignment entities have source containing the assignment text."""
    parsed = _parse("MAX_SIZE = 100\n")
    val = _entity_by_name(parsed, "MAX_SIZE")
    assert val.source is not None
    assert "MAX_SIZE = 100" in val.source


def test_class_source_is_none():
    """TypeDef (class) entities have source=None — children carry the source."""
    parsed = _parse("class Foo:\n    pass\n")
    cls = _entity_by_name(parsed, "Foo")
    assert cls.source is None


def test_module_source_is_none():
    """Module entities have source=None — module source is the entire file."""
    parsed = _parse("x = 1\n", path="src/mod.py")
    mod = _entity_by_name(parsed, "mod")
    assert mod.source is None


def test_source_truncated():
    """Source longer than 2000 chars is truncated by default."""
    body = "    x = 1\n" * 300  # ~3000 chars
    source_code = f"def big():\n{body}"
    parsed = _parse(source_code)
    func = _entity_by_name(parsed, "big")
    assert func.source is not None
    assert len(func.source) == 2000


def test_source_truncated_custom():
    """parse_file(max_source_chars=50) truncates at 50."""
    source_code = "def big():\n" + "    x = 1\n" * 20
    result = parse_file("src/example.py", source_code.encode("utf-8"), PROJECT, max_source_chars=50)
    assert result is not None
    func = _entity_by_name(result, "big")
    assert func.source is not None
    assert len(func.source) == 50


def test_source_not_in_content_hash():
    """Same signature/name but different bodies produce the same content_hash."""
    parsed1 = _parse("def work():\n    return 1\n")
    parsed2 = _parse("def work():\n    return 2\n")
    func1 = _entity_by_name(parsed1, "work")
    func2 = _entity_by_name(parsed2, "work")
    assert func1.content_hash == func2.content_hash
    # But source differs
    assert func1.source != func2.source
