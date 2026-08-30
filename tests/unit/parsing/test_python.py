"""Unit tests for the tree-sitter parser module."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from code_atlas.parsing.ast import (
    DEFAULT_MAX_SOURCE_CHARS,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    get_language_for_file,
    parse_file,
)
from code_atlas.parsing.languages.python import (
    ClassOverridesDetector,
    ModuleExportsDetector,
    module_qualified_name,
)
from code_atlas.schema import (
    CallableKind,
    NodeLabel,
    RelType,
    TypeDefKind,
    ValueKind,
    Visibility,
)
from code_atlas.settings import RationaleSettings

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
    assert get_language_for_file("data.csv") is None


# ---------------------------------------------------------------------------
# Module / Package
# ---------------------------------------------------------------------------


def test_module_entity():
    parsed = _parse("x = 1\n", path="src/code_atlas/parser.py")
    module = _entity_by_name(parsed, "parser")
    assert module.label == NodeLabel.MODULE
    assert module.kind == "module"
    assert module.qualified_name == f"{PROJECT}:code_atlas.parser"


def test_package_entity():
    parsed = _parse("", path="src/code_atlas/__init__.py")
    pkg = _entity_by_name(parsed, "code_atlas")
    assert pkg.label == NodeLabel.PACKAGE
    assert pkg.kind == "package"
    assert pkg.qualified_name == f"{PROJECT}:code_atlas"


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
    assert bar.qualified_name == f"{PROJECT}:example.Foo.bar"

    baz = _entity_by_name(parsed, "baz")
    assert baz.kind == CallableKind.FUNCTION
    assert baz.qualified_name == f"{PROJECT}:example.baz"


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


def test_async_function_tagged():
    """A top-level 'async def' function gets the 'async' tag."""
    parsed = _parse("async def fetch():\n    pass\n")
    fetch = _entity_by_name(parsed, "fetch")
    assert "async" in fetch.tags


def test_async_method_tagged():
    """An 'async def' method inside a class gets the 'async' tag."""
    parsed = _parse("class Foo:\n    async def bar(self):\n        pass\n")
    bar = _entity_by_name(parsed, "bar")
    assert "async" in bar.tags


def test_async_decorated_method_tagged():
    """A decorated 'async def' method still gets the 'async' tag."""
    parsed = _parse(
        """\
class Foo:
    @staticmethod
    async def bar():
        pass
"""
    )
    bar = _entity_by_name(parsed, "bar")
    assert "async" in bar.tags


def test_sync_function_not_tagged_async():
    """A regular 'def' function does NOT get the 'async' tag."""
    parsed = _parse("def fetch():\n    pass\n")
    fetch = _entity_by_name(parsed, "fetch")
    assert "async" not in fetch.tags


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


def test_raw_string_docstring_prefix_stripped():
    """A raw-string docstring (r\"\"\"...\"\"\") has its prefix and quotes stripped."""
    parsed = _parse(
        '''\
def greet(name):
    r"""Raw docstring."""
    print(name)
'''
    )
    func = _entity_by_name(parsed, "greet")
    assert func.docstring == "Raw docstring."


def test_bytes_string_docstring_prefix_stripped():
    """A bytes-string-prefixed docstring (b\"\"\"...\"\"\") has its prefix and quotes stripped."""
    parsed = _parse(
        '''\
def greet(name):
    b"""Bytes docstring."""
    print(name)
'''
    )
    func = _entity_by_name(parsed, "greet")
    assert func.docstring == "Bytes docstring."


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
# Source-root stripping and relative import resolution (S2 contract)
# ---------------------------------------------------------------------------


def _imports(parsed: ParsedFile) -> set[str]:
    return {r.to_name for r in parsed.relationships if r.rel_type == RelType.IMPORTS}


@pytest.mark.parametrize(
    ("file_path", "expected"),
    [
        ("src/code_atlas/events.py", "code_atlas.events"),
        ("src/code_atlas/__init__.py", "code_atlas"),
        ("code_atlas/util.py", "code_atlas.util"),
        ("src/__init__.py", "src"),
        ("src/main.py", "main"),
        ("src/code_atlas/parser.py", "code_atlas.parser"),
    ],
)
def test_module_qualified_name_contract_examples(file_path: str, expected: str):
    """S2 namespace contract: import-system names with the source root stripped."""
    assert module_qualified_name(file_path) == expected


def test_module_qualified_name_strips_source_root():
    """Source-root dirs like src/ are stripped so module qns match the import system."""
    parsed = _parse("x = 1\n", path="src/code_atlas/parser.py")
    module = _entity_by_name(parsed, "parser")
    assert module.qualified_name == f"{PROJECT}:code_atlas.parser"

    parsed = _parse("", path="src/code_atlas/__init__.py")
    pkg = _entity_by_name(parsed, "code_atlas")
    assert pkg.label == NodeLabel.PACKAGE
    assert pkg.qualified_name == f"{PROJECT}:code_atlas"

    # Flat layout unchanged
    parsed = _parse("x = 1\n", path="code_atlas/util.py")
    module = _entity_by_name(parsed, "util")
    assert module.qualified_name == f"{PROJECT}:code_atlas.util"

    # A source root that is itself a package is kept
    parsed = _parse("", path="src/__init__.py")
    pkg = _entity_by_name(parsed, "src")
    assert pkg.qualified_name == f"{PROJECT}:src"


def test_relative_import_resolved_from_module():
    """Relative imports resolve against the module's parent package at parse time."""
    parsed = _parse(
        "from .other import thing\nfrom . import sibling\nfrom ..top import x\n",
        path="pkg/sub/mod.py",
    )
    assert _imports(parsed) == {"pkg.sub.other.thing", "pkg.sub.sibling", "pkg.top.x"}


def test_relative_import_resolved_from_package_init():
    """__init__.py relative imports resolve against the package itself, not its parent."""
    parsed = _parse("from .mod import foo\nfrom . import bar\n", path="pkg/sub/__init__.py")
    assert _imports(parsed) == {"pkg.sub.mod.foo", "pkg.sub.bar"}


def test_relative_import_beyond_top_level_dropped():
    """Relative imports whose dots escape the top-level package emit no relationship."""
    parsed = _parse("from . import x\n", path="mod.py")
    assert _imports(parsed) == set()

    parsed = _parse("from ...deep import y\n", path="pkg/mod.py")
    assert _imports(parsed) == set()


def test_relative_import_multilevel_with_alias():
    """'..pkg.sub'-style dotted suffixes and aliased names resolve correctly."""
    parsed = _parse("from ..util.text import slug as s\n", path="app/core/handlers/mod.py")
    assert _imports(parsed) == {"app.core.util.text.slug"}


def test_relative_import_src_layout():
    """Relative import in a src-layout __init__ resolves into the stripped namespace."""
    parsed = _parse("from .ast import parse_file\n", path="src/code_atlas/parsing/__init__.py")
    assert _imports(parsed) == {"code_atlas.parsing.ast.parse_file"}


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
    mod_defines = _rels_from(parsed, "example", RelType.DEFINES)
    targets = {r.to_name for r in mod_defines}
    assert f"{PROJECT}:example.Foo" in targets
    assert f"{PROJECT}:example.baz" in targets

    # Foo DEFINES bar
    foo_defines = _rels_from(parsed, "example.Foo", RelType.DEFINES)
    assert any(r.to_name == f"{PROJECT}:example.Foo.bar" for r in foo_defines)


def test_calls_relationship():
    parsed = _parse(
        """\
def caller():
    print("hello")
    some_func()
"""
    )
    calls = _rels_from(parsed, "example.caller", RelType.CALLS)
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
    result = parse_file("data.csv", b"a,b,c", PROJECT)
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


def test_deeply_nested_class_qualified_name():
    """A 3-level nesting must retain the full outer chain, not just the innermost class."""
    parsed = _parse(
        """\
class Outer:
    class Middle:
        class Inner:
            def method(self):
                pass
"""
    )
    inner = _entity_by_name(parsed, "Inner")
    assert inner.qualified_name == f"{PROJECT}:example.Outer.Middle.Inner"

    method = _entity_by_name(parsed, "method")
    assert method.qualified_name == f"{PROJECT}:example.Outer.Middle.Inner.method"


def test_nested_classes_same_name_no_uid_collision():
    """Same-named nested classes under different outer classes must not collide."""
    parsed = _parse(
        """\
class A:
    class B:
        class Leaf:
            pass

class C:
    class D:
        class Leaf:
            pass
"""
    )
    leaves = [e for e in parsed.entities if e.name == "Leaf"]
    assert len(leaves) == 2
    qns = {e.qualified_name for e in leaves}
    assert qns == {
        f"{PROJECT}:example.A.B.Leaf",
        f"{PROJECT}:example.C.D.Leaf",
    }


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
    """Source longer than the default cap is truncated to it.

    Sized off DEFAULT_MAX_SOURCE_CHARS rather than a literal: this asserted 2000 while
    the default moved to 48,000, and a stale literal here is how the cap could shrink
    back under a model's input limit — making EmbedChunk unreachable again — without
    anything failing.
    """
    line = "    x = 1\n"
    body = line * (DEFAULT_MAX_SOURCE_CHARS // len(line) + 100)
    source_code = f"def big():\n{body}"
    parsed = _parse(source_code)
    func = _entity_by_name(parsed, "big")
    assert func.source is not None
    assert len(func.source) == DEFAULT_MAX_SOURCE_CHARS


def test_source_truncated_custom():
    """parse_file(max_source_chars=50) truncates at 50."""
    source_code = "def big():\n" + "    x = 1\n" * 20
    result = parse_file("src/example.py", source_code.encode("utf-8"), PROJECT, max_source_chars=50)
    assert result is not None
    func = _entity_by_name(result, "big")
    assert func.source is not None
    assert len(func.source) == 50


def test_content_hash_changes_on_body():
    """Same signature/name but different bodies produce different content_hashes."""
    parsed1 = _parse("def work():\n    return 1\n")
    parsed2 = _parse("def work():\n    return 2\n")
    func1 = _entity_by_name(parsed1, "work")
    func2 = _entity_by_name(parsed2, "work")
    assert func1.content_hash != func2.content_hash


def test_content_hash_covers_source_beyond_truncation():
    """Edits past the max_source_chars cap still change the hash (hash runs pre-truncation)."""
    line = "    x = 1\n"
    filler = line * (DEFAULT_MAX_SOURCE_CHARS // len(line) + 100)  # past the default cap
    p1 = _parse(f"def big():\n{filler}    return 1\n")
    p2 = _parse(f"def big():\n{filler}    return 2\n")
    f1, f2 = _entity_by_name(p1, "big"), _entity_by_name(p2, "big")
    assert f1.content_hash != f2.content_hash
    assert f1.source == f2.source  # truncated prefix identical


# ---------------------------------------------------------------------------
# Enum detection
# ---------------------------------------------------------------------------


def test_enum_class_kind():
    """Enum subclass gets kind=enum instead of class."""
    parsed = _parse(
        """\
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
"""
    )
    cls = _entity_by_name(parsed, "Color")
    assert cls.label == NodeLabel.TYPE_DEF
    assert cls.kind == TypeDefKind.ENUM


def test_enum_member_kind():
    """Assignments inside an Enum class get kind=enum_member."""
    parsed = _parse(
        """\
from enum import StrEnum

class Status(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
"""
    )
    active = _entity_by_name(parsed, "ACTIVE")
    assert active.label == NodeLabel.VALUE
    assert active.kind == ValueKind.ENUM_MEMBER

    inactive = _entity_by_name(parsed, "INACTIVE")
    assert inactive.kind == ValueKind.ENUM_MEMBER


def test_non_enum_class_unchanged():
    """Regular class fields still get kind=field."""
    parsed = _parse(
        """\
class Config:
    debug = True
"""
    )
    cls = _entity_by_name(parsed, "Config")
    assert cls.kind == TypeDefKind.CLASS

    debug = _entity_by_name(parsed, "debug")
    assert debug.kind == ValueKind.FIELD


def test_nested_enum_member_kind():
    """Assignments inside an Enum class nested in another class still get kind=enum_member."""
    parsed = _parse(
        """\
from enum import Enum

class Outer:
    class Color(Enum):
        RED = 1
        GREEN = 2
"""
    )
    color = _entity_by_name(parsed, "Color")
    assert color.kind == TypeDefKind.ENUM

    red = _entity_by_name(parsed, "RED")
    assert red.kind == ValueKind.ENUM_MEMBER

    green = _entity_by_name(parsed, "GREEN")
    assert green.kind == ValueKind.ENUM_MEMBER


def test_int_flag_detected():
    """IntFlag is recognized as an enum base."""
    parsed = _parse(
        """\
from enum import IntFlag

class Perms(IntFlag):
    READ = 4
    WRITE = 2
"""
    )
    cls = _entity_by_name(parsed, "Perms")
    assert cls.kind == TypeDefKind.ENUM

    read = _entity_by_name(parsed, "READ")
    assert read.kind == ValueKind.ENUM_MEMBER


# ---------------------------------------------------------------------------
# Conditional definitions
# ---------------------------------------------------------------------------


def test_conditional_definitions():
    """Duplicate qualified_name entities get a 'conditional' tag."""
    parsed = _parse(
        """\
import sys

if sys.platform == "win32":
    def get_path():
        return "C:\\\\"

if sys.platform == "linux":
    def get_path():
        return "/tmp"
"""
    )
    # Both should exist
    get_paths = [e for e in parsed.entities if e.name == "get_path"]
    assert len(get_paths) == 2

    # First occurrence: no conditional tag
    assert "conditional" not in get_paths[0].tags

    # Second occurrence: has conditional tag
    assert "conditional" in get_paths[1].tags


def test_no_conditional_tag_for_unique_defs():
    """Unique definitions don't get a conditional tag."""
    parsed = _parse(
        """\
def foo():
    pass

def bar():
    pass
"""
    )
    foo = _entity_by_name(parsed, "foo")
    assert "conditional" not in foo.tags
    bar = _entity_by_name(parsed, "bar")
    assert "conditional" not in bar.tags


# ---------------------------------------------------------------------------
# TYPE_CHECKING imports (type_only flag)
# ---------------------------------------------------------------------------


def test_type_checking_imports_marked_type_only():
    """Imports inside `if TYPE_CHECKING:` blocks get type_only=True property."""
    parsed = _parse(
        """\
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os.path import join
    import sys
"""
    )
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    type_only_rels = [r for r in import_rels if r.properties.get("type_only")]
    non_type_only_rels = [r for r in import_rels if not r.properties.get("type_only")]

    type_only_names = {r.to_name for r in type_only_rels}
    assert "os.path.join" in type_only_names
    assert "sys" in type_only_names

    # typing import should NOT be type_only
    non_type_only_names = {r.to_name for r in non_type_only_rels}
    assert "typing.TYPE_CHECKING" in non_type_only_names


def test_regular_imports_not_type_only():
    """Normal imports have no type_only property."""
    parsed = _parse("import os\nfrom pathlib import Path\n")
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    for rel in import_rels:
        assert not rel.properties.get("type_only"), f"Expected no type_only for {rel.to_name}"


def test_type_checking_attribute_form_marked_type_only():
    """`if typing.TYPE_CHECKING:` (attribute form) is also detected, not just the bare identifier."""
    parsed = _parse(
        """\
import typing

if typing.TYPE_CHECKING:
    from os.path import join
"""
    )
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    type_only_names = {r.to_name for r in import_rels if r.properties.get("type_only")}
    assert "os.path.join" in type_only_names


# ---------------------------------------------------------------------------
# USES_TYPE extraction
# ---------------------------------------------------------------------------


def test_uses_type_from_function_params():
    """Function with typed params emits USES_TYPE relationships."""
    parsed = _parse(
        """\
def process(user: User, config: Config) -> None:
    pass
"""
    )
    uses_type = [r for r in parsed.relationships if r.rel_type == RelType.USES_TYPE]
    type_names = {r.to_name for r in uses_type}
    assert "User" in type_names
    assert "Config" in type_names


def test_uses_type_from_return_type():
    """Function with return type emits USES_TYPE."""
    parsed = _parse(
        """\
def get_user() -> UserModel:
    pass
"""
    )
    uses_type = [r for r in parsed.relationships if r.rel_type == RelType.USES_TYPE]
    type_names = {r.to_name for r in uses_type}
    assert "UserModel" in type_names


def test_uses_type_skips_builtins():
    """Built-in types like int, str, None don't produce USES_TYPE."""
    parsed = _parse(
        """\
def add(x: int, y: str) -> bool:
    pass
"""
    )
    uses_type = [r for r in parsed.relationships if r.rel_type == RelType.USES_TYPE]
    assert len(uses_type) == 0


def test_uses_type_subscript_types():
    """Optional[Foo] extracts Foo, list[int] is ignored."""
    parsed = _parse(
        """\
def process(user: Optional[UserModel], items: list[int]) -> dict[str, Result]:
    pass
"""
    )
    uses_type = [r for r in parsed.relationships if r.rel_type == RelType.USES_TYPE]
    type_names = {r.to_name for r in uses_type}
    assert "UserModel" in type_names
    assert "Result" in type_names
    # Builtins should not appear
    assert "int" not in type_names
    assert "str" not in type_names
    assert "list" not in type_names
    assert "dict" not in type_names


def test_uses_type_no_duplicates():
    """Same type in params and return should only appear once per function."""
    parsed = _parse(
        """\
def identity(x: Foo) -> Foo:
    pass
"""
    )
    uses_type = [r for r in parsed.relationships if r.rel_type == RelType.USES_TYPE]
    assert len(uses_type) == 1
    assert uses_type[0].to_name == "Foo"


def test_uses_type_method():
    """Methods inside classes also emit USES_TYPE."""
    parsed = _parse(
        """\
class Service:
    def handle(self, req: Request) -> Response:
        pass
"""
    )
    uses_type = [r for r in parsed.relationships if r.rel_type == RelType.USES_TYPE]
    type_names = {r.to_name for r in uses_type}
    assert "Request" in type_names
    assert "Response" in type_names


def test_type_checking_else_branch_indexed():
    """Runtime code in the else branch of `if TYPE_CHECKING:` is still indexed."""
    parsed = _parse(
        """\
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import User
else:
    from stubs import UserStub

def process(u: User) -> None:
    pass
"""
    )
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    import_names = {r.to_name for r in import_rels}
    # Both the TYPE_CHECKING import and the runtime else import should be captured
    assert "models.User" in import_names
    assert "stubs.UserStub" in import_names
    # Only the TYPE_CHECKING import should be type_only
    type_only = {r.to_name for r in import_rels if r.properties.get("type_only")}
    non_type_only = {r.to_name for r in import_rels if not r.properties.get("type_only")}
    assert "models.User" in type_only
    assert "stubs.UserStub" in non_type_only


def test_uses_type_typing_attribute_containers():
    """typing.Optional[Foo] extracts Foo, not Optional."""
    parsed = _parse(
        """\
import typing

def process(user: typing.Optional[UserModel], items: typing.List[int]) -> typing.Dict[str, Result]:
    pass
"""
    )
    uses_type = [r for r in parsed.relationships if r.rel_type == RelType.USES_TYPE]
    type_names = {r.to_name for r in uses_type}
    assert "UserModel" in type_names
    assert "Result" in type_names
    # Container names should not appear
    assert "Optional" not in type_names
    assert "List" not in type_names
    assert "Dict" not in type_names


# ---------------------------------------------------------------------------
# Detector project scoping / shadowing regressions
# ---------------------------------------------------------------------------


def _make_entity(
    name: str,
    qn: str,
    label: NodeLabel,
    kind: str,
) -> ParsedEntity:
    return ParsedEntity(
        name=name,
        qualified_name=qn,
        label=label,
        kind=kind,
        line_start=1,
        line_end=5,
        file_path="src/app.py",
    )


async def test_class_overrides_query_filters_by_project_name():
    """ClassOverridesDetector's graph query must scope the base-class lookup to project_name
    to avoid creating cross-project OVERRIDES/IMPLEMENTS edges."""
    method = _make_entity(
        name="save",
        qn="proj:src.app.Child.save",
        label=NodeLabel.CALLABLE,
        kind=CallableKind.METHOD,
    )
    inherits_rel = ParsedRelationship(
        from_qualified_name="proj:src.app.Child",
        rel_type=RelType.INHERITS,
        to_name="Base",
    )
    parsed = ParsedFile(
        file_path="src/app.py",
        language="python",
        entities=[method],
        relationships=[inherits_rel],
    )

    graph = AsyncMock()
    graph.find_overridden_method = AsyncMock(return_value=("proj:src.base.Base.save", []))

    det = ClassOverridesDetector()
    await det.detect(parsed, "proj", graph)

    # The base-class lookup must be scoped to the entity's own project_name
    # (GraphBackend.find_overridden_method's job — see graph/client.py and
    # backends/sqlite_graph.py — to avoid cross-project OVERRIDES/IMPLEMENTS edges).
    graph.find_overridden_method.assert_awaited_once_with("proj", ["Base"], "save")


async def test_module_exports_no_shadowing_by_module_entity():
    """A real exported symbol with the same name as the module must win over
    the module/package entity itself in the EXPORTS resolution lookup."""
    module = _make_entity(
        name="app",
        qn="proj:src.app",
        label=NodeLabel.MODULE,
        kind="module",
    )
    # A symbol literally named "app" (e.g. `app = Flask(__name__)`), same
    # name as the module — a common Flask idiom.
    app_symbol = _make_entity(
        name="app",
        qn="proj:src.app.app",
        label=NodeLabel.VALUE,
        kind="variable",
    )
    all_val = ParsedEntity(
        name="__all__",
        qualified_name="proj:src.app.__all__",
        label=NodeLabel.VALUE,
        kind="variable",
        line_start=1,
        line_end=1,
        file_path="src/app.py",
        source="__all__ = ['app']",
    )
    parsed = ParsedFile(
        file_path="src/app.py",
        language="python",
        entities=[module, app_symbol, all_val],
        relationships=[],
    )

    det = ModuleExportsDetector()
    result = await det.detect(parsed, "proj", None)  # ty: ignore[invalid-argument-type]

    assert len(result.relationships) == 1
    assert result.relationships[0].to_name == "proj:src.app.app"


# ---------------------------------------------------------------------------
# Rationale extraction (intent-bearing comments -> entity.rationale/.citations)
# ---------------------------------------------------------------------------


def test_rationale_from_comment_inside_body():
    parsed = _parse("""\
def render():
    # HACK: upstream returns 204 with a body
    return 1
""")
    assert _entity_by_name(parsed, "render").rationale == "HACK: upstream returns 204 with a body"


def test_rationale_preceding_declaration_beats_enclosing_class():
    """A comment above a method belongs to the method, not the class that encloses it."""
    parsed = _parse("""\
class Widget:
    # WHY: rendering is lazy so the template cache stays warm
    def render(self):
        return 1
""")
    assert _entity_by_name(parsed, "render").rationale == "WHY: rendering is lazy so the template cache stays warm"
    assert _entity_by_name(parsed, "Widget").rationale is None


def test_rationale_skips_decorator_lines():
    """Decorators sit between the comment and ``def`` — they must not break attribution."""
    parsed = _parse("""\
import functools


# NOTE: cached because the lookup hits the network
@functools.cache
def lookup():
    return 1
""")
    assert _entity_by_name(parsed, "lookup").rationale == "NOTE: cached because the lookup hits the network"


def test_rationale_falls_back_to_module_when_code_intervenes():
    """Real code between the comment and the next declaration means it annotates the file."""
    parsed = _parse("""\
# NOTE: this module is legacy
import os

CONST = os.sep


def helper():
    return 1
""")
    assert _entity_by_name(parsed, "example").rationale == "NOTE: this module is legacy"
    assert _entity_by_name(parsed, "helper").rationale is None


def test_rationale_trailing_comment_stays_with_enclosing_function():
    """A note at the tail of a body must not be claimed by the next top-level function."""
    parsed = _parse("""\
def first():
    x = 1
    # NOTE: the caller relies on x being returned unwrapped
    return x


def second():
    return 2
""")
    assert _entity_by_name(parsed, "first").rationale == "NOTE: the caller relies on x being returned unwrapped"
    assert _entity_by_name(parsed, "second").rationale is None


def test_rationale_folds_wrapped_lines_into_one_entry():
    parsed = _parse("""\
def render():
    # WHY: the cache is per-instance because sharing it
    # across threads corrupted the counters
    return 1
""")
    expected = "WHY: the cache is per-instance because sharing it across threads corrupted the counters"
    assert _entity_by_name(parsed, "render").rationale == expected


def test_rationale_multiple_markers_are_newline_joined():
    parsed = _parse("""\
def render():
    # WHY: first reason
    x = 1
    # HACK: second reason
    return x
""")
    assert _entity_by_name(parsed, "render").rationale == "WHY: first reason\nHACK: second reason"


def test_rationale_ignores_lowercase_prose():
    """Uppercase-only matching keeps ordinary prose (``Note:``) out of the graph."""
    parsed = _parse("""\
def render():
    # Note: this is just prose
    # why: also prose
    return 1
""")
    assert _entity_by_name(parsed, "render").rationale is None


def test_rationale_ignores_marker_inside_string_literal():
    """Only comment nodes are scanned — a marker in a string is not a comment."""
    parsed = _parse("""\
def render():
    return "NOTE: not a comment"
""")
    assert _entity_by_name(parsed, "render").rationale is None


def test_rationale_todo_off_by_default():
    parsed = _parse("""\
def render():
    # TODO: rip this out
    return 1
""")
    assert _entity_by_name(parsed, "render").rationale is None


def test_rationale_tasks_opt_in():
    source = "def render():\n    # TODO: rip this out\n    return 1\n"
    result = parse_file("src/example.py", source.encode("utf-8"), PROJECT, rationale=RationaleSettings(tasks=True))
    assert result is not None
    assert _entity_by_name(result, "render").rationale == "TODO: rip this out"


def test_rationale_task_marker_does_not_get_absorbed_into_a_note():
    """A disabled TODO directly under a NOTE must terminate the note, not continue it."""
    parsed = _parse("""\
def render():
    # NOTE: real reason
    # TODO: unrelated chore
    return 1
""")
    assert _entity_by_name(parsed, "render").rationale == "NOTE: real reason"


def test_rationale_disabled_by_settings():
    source = "def render():\n    # NOTE: reason\n    return 1\n"
    result = parse_file("src/example.py", source.encode("utf-8"), PROJECT, rationale=RationaleSettings(enabled=False))
    assert result is not None
    entity = _entity_by_name(result, "render")
    assert entity.rationale is None
    assert entity.citations == []


def test_rationale_custom_marker_set():
    source = "def render():\n    # GOTCHA: watch out\n    # NOTE: ignored now\n    return 1\n"
    settings = RationaleSettings(markers=["GOTCHA"])
    result = parse_file("src/example.py", source.encode("utf-8"), PROJECT, rationale=settings)
    assert result is not None
    assert _entity_by_name(result, "render").rationale == "GOTCHA: watch out"


def test_rationale_marker_with_owner_parenthetical():
    parsed = _parse("""\
def render():
    # NOTE(peter): owner annotations are tolerated
    return 1
""")
    assert _entity_by_name(parsed, "render").rationale == "NOTE: owner annotations are tolerated"


def test_citations_recorded_as_normalized_strings():
    parsed = _parse("""\
def render():
    # WHY: see ADR 14 and RFC-7231
    return 1
""")
    assert _entity_by_name(parsed, "render").citations == ["ADR-14", "RFC-7231"]


def test_citations_recorded_from_unmarked_comments():
    """A bare reference is worth recording even without a NOTE/WHY marker."""
    parsed = _parse("""\
def render():
    # implements ADR-0014
    return 1
""")
    entity = _entity_by_name(parsed, "render")
    assert entity.citations == ["ADR-0014"]
    assert entity.rationale is None


def test_citations_deduplicated_and_sorted():
    parsed = _parse("""\
def render():
    # NOTE: see RFC 7231
    x = 1
    # HACK: still RFC 7231, and ADR-0014
    return x
""")
    assert _entity_by_name(parsed, "render").citations == ["ADR-0014", "RFC-7231"]


def test_citations_disabled_by_settings():
    source = "def render():\n    # NOTE: see ADR-0014\n    return 1\n"
    settings = RationaleSettings(citations=False)
    result = parse_file("src/example.py", source.encode("utf-8"), PROJECT, rationale=settings)
    assert result is not None
    entity = _entity_by_name(result, "render")
    assert entity.rationale == "NOTE: see ADR-0014"
    assert entity.citations == []


def test_rationale_removal_restores_original_content_hash():
    """Deleting the comment returns the entity to its pre-rationale hash.

    The property is rewritten wholesale on every upsert, so it self-heals —
    unlike detector ``PropertyEnrichment``, which is ``SET n += props`` and
    never clears.
    """
    without = _parse("def render():\n    return 1\n")
    with_note = _parse("def render():\n    # NOTE: reason\n    return 1\n")
    readded = _parse("def render():\n    return 1\n")

    assert _entity_by_name(with_note, "render").content_hash != _entity_by_name(without, "render").content_hash
    assert _entity_by_name(readded, "render").content_hash == _entity_by_name(without, "render").content_hash
    assert _entity_by_name(readded, "render").rationale is None


def test_rationale_absent_leaves_defaults_on_every_entity():
    parsed = _parse("""\
class Widget:
    def render(self):
        return 1
""")
    for entity in parsed.entities:
        assert entity.rationale is None
        assert entity.citations == []


# ---------------------------------------------------------------------------
# Runtime config surface — READS_ENV / REFERENCES_FILE
# ---------------------------------------------------------------------------


def _config_refs(parsed: ParsedFile, rel_type: RelType) -> set[tuple[str, str]]:
    """``{(from_qualified_name, to_name)}`` for one config relationship type."""
    return {(r.from_qualified_name, r.to_name) for r in parsed.relationships if r.rel_type == rel_type}


def _env_names(parsed: ParsedFile) -> set[str]:
    return {name for _, name in _config_refs(parsed, RelType.READS_ENV)}


def _file_paths(parsed: ParsedFile) -> set[str]:
    return {path for _, path in _config_refs(parsed, RelType.REFERENCES_FILE)}


def test_env_var_read_forms():
    """All five supported spellings produce a READS_ENV with the bare name."""
    parsed = _parse("""\
import os
from os import environ, getenv


def load():
    a = os.getenv("A")
    b = os.getenv("B", "fallback")
    c = os.environ["C"]
    d = os.environ.get("D")
    e = os.environ.get("E", "fallback")
    f = getenv("F")
    g = environ["G"]
    h = environ.get("H")
    return a, b, c, d, e, f, g, h
""")
    assert _env_names(parsed) == {"A", "B", "C", "D", "E", "F", "G", "H"}


def test_env_var_bare_forms_require_a_real_os_import():
    """A project's own ``getenv()`` helper or an unrelated ``environ`` dict must
    not mint EnvVar nodes — the bare spellings are gated on ``from os import``.
    """
    parsed = _parse("""\
from mylib import getenv

environ = {"NOT_AN_ENV_VAR": 1}


def load():
    return getenv("NOT_AN_ENV_VAR"), environ["ALSO_NOT"]
""")
    assert _env_names(parsed) == set()


def test_env_var_non_literal_name_is_ignored():
    parsed = _parse("""\
import os

KEY = "DYNAMIC"


def load():
    return os.getenv(KEY), os.environ[KEY], os.getenv(f"PREFIX_{KEY}")
""")
    assert _env_names(parsed) == set()


def test_env_var_attributed_to_innermost_entity():
    """Module-level reads hang off the Value they define; in-function reads off
    the function; method reads off the method.
    """
    parsed = _parse("""\
import os

DATABASE_URL = os.getenv("DATABASE_URL")


def load():
    return os.getenv("PORT")


class Service:
    def start(self):
        return os.getenv("SERVICE_HOST")
""")
    assert _config_refs(parsed, RelType.READS_ENV) == {
        (f"{PROJECT}:example.DATABASE_URL", "DATABASE_URL"),
        (f"{PROJECT}:example.load", "PORT"),
        (f"{PROJECT}:example.Service.start", "SERVICE_HOST"),
    }


def test_env_var_outside_any_definition_falls_back_to_the_module():
    parsed = _parse("""\
import os

if os.getenv("FEATURE_FLAG"):
    pass
""")
    assert _config_refs(parsed, RelType.READS_ENV) == {(f"{PROJECT}:example", "FEATURE_FLAG")}


def test_env_var_repeated_reads_collapse_to_one_relationship():
    """The graph stores no call-site multiplicity — two reads in one function
    are one edge (mirrors _plan_config_refs' own dedup).
    """
    parsed = _parse("""\
import os


def load():
    return os.getenv("PORT"), os.environ["PORT"], os.environ.get("PORT")
""")
    reads = [r for r in parsed.relationships if r.rel_type == RelType.READS_ENV]
    assert len(reads) == 1
    assert reads[0].to_name == "PORT"


def test_no_config_refs_when_the_file_has_none():
    parsed = _parse("""\
def add(a, b):
    return a + b
""")
    assert _env_names(parsed) == set()
    assert _file_paths(parsed) == set()


# ---------------------------------------------------------------------------
# SECURITY: env var NAMES only — never a value, never a default
#
# os.getenv("API_KEY", "sk-live-abc123") puts a live secret in the second
# argument. The extraction reads the FIRST positional argument and returns, so
# no later argument node is visited at all.
# ---------------------------------------------------------------------------

_SECRET = "sk-live-abc123-DO-NOT-PERSIST"


def test_getenv_default_secret_never_reaches_a_relationship():
    parsed = _parse(f'''\
import os


def load():
    """Load credentials."""
    return os.getenv("API_KEY", "{_SECRET}")
''')

    reads = [r for r in parsed.relationships if r.rel_type == RelType.READS_ENV]
    assert len(reads) == 1
    assert reads[0].to_name == "API_KEY"
    # No property channel exists for a default to ride out on.
    assert reads[0].properties == {}

    for rel in parsed.relationships:
        assert _SECRET not in rel.to_name
        assert _SECRET not in repr(rel.properties)


def test_getenv_default_secret_reaches_no_new_entity_field():
    """The only field that carries the secret is ``source`` — the verbatim entity
    body, which predates this extraction and is unchanged by it. Every other
    field, and every field of the EnvVar-bearing relationship, must be clean.

    (``source`` IS shipped to the embedding provider for Callables. That is a
    pre-existing exposure of *any* hardcoded literal, not something this change
    introduces, and it is out of scope here — EnvVar/ResourceFile themselves are
    deliberately non-embeddable.)
    """
    parsed = _parse(f"""\
import os

API_KEY = os.getenv("API_KEY", "{_SECRET}")


def load():
    return os.getenv("API_KEY", "{_SECRET}")
""")

    for entity in parsed.entities:
        leaking = {
            field: value for field, value in vars(entity).items() if field != "source" and _SECRET in repr(value)
        }
        assert leaking == {}, f"{entity.qualified_name} leaked the default via {sorted(leaking)}"


def test_env_var_names_are_shell_identifier_shaped():
    """A first argument that is a string but not a plausible variable name is a
    misparse, not an env var.
    """
    parsed = _parse("""\
import os


def load():
    return os.getenv("has spaces"), os.getenv("dotted.name"), os.getenv(""), os.getenv("9LEADING")
""")
    assert _env_names(parsed) == set()


# ---------------------------------------------------------------------------
# REFERENCES_FILE — conservative path literals only
# ---------------------------------------------------------------------------


def test_file_reference_openers():
    parsed = _parse("""\
from pathlib import Path
import pathlib


def load():
    a = open("data/fixtures.json")
    b = Path("config/schema.yaml")
    c = Path("wiki/notes.md").read_text()
    d = Path("assets/logo.bin").read_bytes()
    e = pathlib.Path("etc/defaults.toml")
    return a, b, c, d, e
""")
    assert _file_paths(parsed) == {
        "data/fixtures.json",
        "config/schema.yaml",
        "wiki/notes.md",
        "assets/logo.bin",
        "etc/defaults.toml",
    }


def test_file_reference_mode_argument_is_never_a_path():
    """``open(path, "rb")`` — only the first positional argument is inspected."""
    parsed = _parse("""\
def load():
    return open("data/fixtures.json", "rb")
""")
    assert _file_paths(parsed) == {"data/fixtures.json"}


def test_file_reference_requires_a_plain_literal():
    """f-strings, concatenation, variables and escapes are all rejected — a
    non-literal path would mint a node for a file that does not exist.
    """
    parsed = _parse(r"""
name = "x"


def load():
    a = open(f"data/{name}.json")
    b = open("data/" + name + ".json")
    c = open(name)
    d = open("data/" "fixtures.json")
    e = open("data\tfixtures.json")
    f = open(rb"data/fixtures.json")
    return a, b, c, d, e, f
""")
    assert _file_paths(parsed) == set()


def test_file_reference_rejects_non_path_literals():
    parsed = _parse("""\
from pathlib import Path


def load():
    return open("rb"), Path("."), Path("data"), open("/etc/passwd"), open("https://x.dev/a.json")
""")
    assert _file_paths(parsed) == set()


def test_file_reference_attributed_to_innermost_entity():
    parsed = _parse("""\
from pathlib import Path

SCHEMA = Path("config/schema.yaml")


def load():
    return open("data/fixtures.json")
""")
    assert _config_refs(parsed, RelType.REFERENCES_FILE) == {
        (f"{PROJECT}:example.SCHEMA", "config/schema.yaml"),
        (f"{PROJECT}:example.load", "data/fixtures.json"),
    }


def test_arbitrary_attribute_open_is_not_a_file_reference():
    """``zipfile.open("entry.txt")`` / ``self.open(...)`` name archive members and
    mock objects as often as real files, so only the bare builtin counts.
    """
    parsed = _parse("""\
def load(archive, session):
    return archive.open("member/entry.txt"), session.open("data/x.json")
""")
    assert _file_paths(parsed) == set()


# ---------------------------------------------------------------------------
# SECURITY: a sensitive file is recorded as a PATH and never read
# ---------------------------------------------------------------------------


def test_sensitive_file_references_are_recorded_as_paths():
    """Recording that code reads ``.env`` is the point of REFERENCES_FILE."""
    parsed = _parse("""\
from pathlib import Path


def load():
    a = open(".env")
    b = Path("certs/server.pem").read_text()
    c = open("secrets/credentials.json")
    d = Path(".ssh/id_rsa").read_bytes()
    return a, b, c, d
""")
    assert _file_paths(parsed) == {
        ".env",
        "certs/server.pem",
        "secrets/credentials.json",
        ".ssh/id_rsa",
    }


def test_parser_never_opens_a_referenced_file(monkeypatch, tmp_path):
    """The path is data, not an instruction to read. Parsing a module that
    references a REAL secret file on disk must not touch it — the parser only
    ever sees the source bytes it was handed.
    """
    import builtins
    import pathlib

    # Assembled rather than written literally: pre-commit's detect-private-key hook
    # scans for this exact marker and would flag the file. There is no key material
    # here — the marker alone is the bait, and the runtime bytes are identical.
    # Do NOT "tidy" this back into one string, and do NOT exclude this file from the
    # hook: that would blind it to a real key landing here later.
    pem_marker = "-----BEGIN " + "PRIVATE KEY-----\n"

    secret_file = tmp_path / "server.pem"
    secret_file.write_text(pem_marker)

    source = f"""
from pathlib import Path


def load():
    return open("{secret_file.name}"), Path("certs/server.pem").read_text(), open(".env")
""".encode()

    # Warm the memoized language-plugin discovery before the filesystem is sealed.
    parse_file("src/warmup.py", b"x = 1\n", PROJECT)

    def _forbidden(*args, **kwargs):
        raise AssertionError("the parser touched the filesystem")

    monkeypatch.setattr(builtins, "open", _forbidden)
    for attr in ("open", "read_text", "read_bytes", "exists", "stat", "resolve", "is_file"):
        monkeypatch.setattr(pathlib.Path, attr, _forbidden)

    parsed = parse_file("src/example.py", source, PROJECT)

    assert parsed is not None
    assert _file_paths(parsed) == {secret_file.name, "certs/server.pem", ".env"}
    assert not any(pem_marker.strip() in repr(r) for r in parsed.relationships)


# ---------------------------------------------------------------------------
# HASH SAFETY
#
# Config references live entirely in ParsedFile.relationships. Nothing about
# them reaches ParsedEntity, so content_hash must still be the eight-part
# formula — otherwise adding this feature reindexes every project.
# ---------------------------------------------------------------------------


def _eight_part_hash(entity: ParsedEntity) -> str:
    """The pre-change content_hash formula, recomputed independently."""
    import hashlib

    parts = [
        entity.name,
        entity.kind,
        entity.visibility,
        entity.signature or "",
        entity.docstring or "",
        ",".join(sorted(entity.tags)),
        entity.source or "",
        "",  # extra_properties empty
    ]
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


def test_plain_function_still_hashes_to_the_pre_change_formula():
    parsed = _parse("""\
def add(a, b):
    return a + b
""")
    for entity in parsed.entities:
        assert entity.content_hash == _eight_part_hash(entity), entity.qualified_name


def test_config_references_are_not_folded_into_content_hash():
    """An entity that DOES read env vars and files still hashes by the same
    eight-part formula — the references are edges, not entity state.
    """
    parsed = _parse("""\
import os
from pathlib import Path


def load():
    return os.getenv("PORT"), Path("config/schema.yaml").read_text()
""")
    assert _env_names(parsed) == {"PORT"}
    assert _file_paths(parsed) == {"config/schema.yaml"}
    for entity in parsed.entities:
        assert entity.content_hash == _eight_part_hash(entity), entity.qualified_name


def test_signature_excludes_lint_pragmas_but_keeps_hashes_in_defaults():
    """A signature was a raw byte slice, so a lint-suppression comment inside a
    multi-line signature rode into it — and from there into the outline, where the hash
    character already has two other meanings.

    Excising the ranges the grammar labelled as comments is correct where a regex could
    not be: the hash in a string default belongs to a `string` node and must survive.
    """
    parsed = _parse(
        "def wide(  # noqa: PLR0913\n"
        "    alpha: int,\n"
        "    sep: str = '#tag',  # trailing note\n"
        ") -> str:\n"
        "    return sep\n"
    )

    sig = _entity_by_name(parsed, "wide").signature
    assert sig is not None
    assert "noqa" not in sig
    assert "trailing note" not in sig
    assert "'#tag'" in sig  # a hash inside a string literal is not a comment
    assert "alpha: int" in sig
    assert "-> str" in sig


def test_signature_without_comments_is_unchanged():
    parsed = _parse("def plain(a: int) -> bool:\n    return True\n")

    assert _entity_by_name(parsed, "plain").signature == "def plain(a: int) -> bool"


def test_docstring_containing_a_hash_round_trips():
    """A hash in prose is not a marker and not a comment. The signature extractor elides
    only nodes the grammar labelled `comment`, so docstring text is untouched.
    """
    parsed = _parse('def rank() -> int:\n    """Return the #1 match."""\n    return 1\n')

    assert _entity_by_name(parsed, "rank").docstring == "Return the #1 match."


def test_protocol_and_abc_bases_mark_a_class_abstract():
    """A Protocol/ABC declaration's methods are `...` stubs that can never execute, so a
    call resolved to one is resolved to nothing.

    The parser already knew this and threw it away: it emits INHERITS -> Protocol and both
    graph write paths drop that edge, because `Protocol` is not an in-project TypeDef.
    """
    parsed = _parse(
        "from typing import Protocol\n"
        "import abc\n"
        "class Iface(Protocol):\n"
        "    def go(self) -> None: ...\n"
        "class Dotted(typing.Protocol):\n"
        "    def go(self) -> None: ...\n"
        "class Based(abc.ABC):\n"
        "    def go(self) -> None: ...\n"
        "class Real:\n"
        "    def go(self) -> None:\n"
        "        return None\n"
    )

    def abstract(name: str) -> bool:
        return bool(_entity_by_name(parsed, name).extra_properties.get("is_abstract"))

    assert abstract("Iface")
    # The dotted form is an `attribute` node, which the old identifier-only guard skipped.
    assert abstract("Dotted")
    assert abstract("Based")
    assert not abstract("Real")


def test_receiver_type_is_recovered_from_annotations_and_local_construction():
    """Most of what a name-only resolver treats as polymorphism is monomorphic: measured,
    772 of 915 fanned-out sites call exactly one concrete class. The receiver's declared
    type is what distinguishes them, and it is available at parse time.
    """
    parsed = _parse(
        "def handler(store: Store, untyped, maybe: Store | None) -> None:\n"
        "    store.save()\n"
        "    untyped.save()\n"
        "    maybe.save()\n"
        "    local = Store()\n"
        "    local.save()\n"
        "    helper = make_it()\n"
        "    helper.save()\n"
    )
    props = {
        (r.properties.get("receiver") or ""): r.properties
        for r in parsed.relationships
        if r.rel_type == RelType.CALLS and r.to_name == "save"
    }

    assert props["store"]["receiver_type"] == "Store"  # parameter annotation
    assert props["local"]["receiver_type"] == "Store"  # one-step construction
    # Declining to guess is the point — a wrong type sends the call to the wrong
    # implementation with full confidence, which is the failure being removed.
    assert "receiver_type" not in props["untyped"]
    assert "receiver_type" not in props["maybe"]  # union, not a bare class name
    assert "receiver_type" not in props["helper"]  # lowercase callee, not a constructor


def test_only_ellipsis_and_abstractmethod_count_as_stubs():
    """Per-method, not per-class. An ABC is the standard base for ONE abstractmethod plus
    a dozen concrete ones — TierConsumer here has 1 of 16 — and treating its real methods
    as stubs deleted true callees, mis-resolving even `await super().run()`.

    `pass` and docstring-only bodies are real no-op implementations that run and can
    legitimately be the callee, so they are not stubs.
    """
    parsed = _parse(
        "from typing import Protocol\n"
        "from abc import ABC, abstractmethod\n"
        "class Iface(Protocol):\n"
        "    def go(self) -> None: ...\n"
        "class Base(ABC):\n"
        "    @abstractmethod\n"
        "    def must(self) -> None: ...\n"
        "    def real(self) -> int:\n"
        "        return 1\n"
        "    def hook(self) -> None:\n"
        "        pass\n"
        "    def doc_only(self) -> None:\n"
        '        """Docstring."""\n'
    )

    def stub(qn_suffix: str) -> bool:
        e = next(x for x in parsed.entities if x.qualified_name.endswith(qn_suffix))
        return bool(e.extra_properties.get("is_stub"))

    assert stub("Iface.go")
    assert stub("Base.must")
    assert not stub("Base.real")
    assert not stub("Base.hook")
    assert not stub("Base.doc_only")


# ---------------------------------------------------------------------------
# Constant reads — REFERENCES via="const" onto module-level Values
# ---------------------------------------------------------------------------


def _const_refs(parsed: ParsedFile) -> set[tuple[str, str]]:
    return {
        (r.from_qualified_name, r.to_name)
        for r in parsed.relationships
        if r.rel_type == RelType.REFERENCES and r.properties.get("via") == "const"
    }


def test_a_function_reading_a_module_constant_gets_a_references_edge():
    """`_match_brace` reading `_OPEN_BRACE` was invisible: one REFERENCES edge landed
    on a Value in the whole graph, so every constant looked unused."""
    parsed = _parse(
        """
_OPEN_BRACE = "{"

def _match_brace(text):
    return text.startswith(_OPEN_BRACE)
"""
    )

    assert (f"{PROJECT}:example._match_brace", "_OPEN_BRACE") in _const_refs(parsed)


def test_a_local_binding_shadows_the_module_constant():
    """A bare name bound in the function is a local — Python's own scoping rule, and
    the exact false edge ADR-0022 exists to refuse."""
    parsed = _parse(
        """
LIMIT = 10

def shadowed():
    LIMIT = 3
    return LIMIT

def parameter(LIMIT):
    return LIMIT
"""
    )

    assert _const_refs(parsed) == set()


def test_a_global_declaration_unshadows():
    """`global NAME` declares the name module-scoped, so touching it really does
    touch the Value."""
    parsed = _parse(
        """
COUNTER = 0

def bump():
    global COUNTER
    COUNTER = COUNTER + 1
"""
    )

    assert (f"{PROJECT}:example.bump", "COUNTER") in _const_refs(parsed)


def test_module_level_reads_attribute_to_the_module():
    parsed = _parse(
        """
BASE = ("a",)
DERIVED = tuple(BASE)
"""
    )

    assert (f"{PROJECT}:example", "BASE") in _const_refs(parsed)


def test_one_edge_per_reader_not_per_mention():
    parsed = _parse(
        """
SEP = ","

def join(parts):
    return SEP + SEP.join(parts) + SEP
"""
    )

    refs = [
        r
        for r in parsed.relationships
        if r.rel_type == RelType.REFERENCES and r.properties.get("via") == "const" and r.to_name == "SEP"
    ]
    assert len(refs) == 1


# ---------------------------------------------------------------------------
# Long string literals as their own nodes
# ---------------------------------------------------------------------------

_LONG = "word " * 150  # 750 chars, comfortably over _MIN_TEXT_BLOCK_CHARS


def _text_blocks(parsed: ParsedFile) -> list:
    return [e for e in parsed.entities if e.kind == ValueKind.TEXT_BLOCK]


def test_a_long_literal_in_a_function_body_becomes_a_node():
    """Only module- and class-level assignments produce a Value, so this reached the
    graph as nothing but a slice of its function's capped source."""
    parsed = _parse(f'def run():\n    query = """{_LONG}"""\n    return query\n')

    blocks = _text_blocks(parsed)
    assert [b.qualified_name for b in blocks] == [f"{PROJECT}:example.run.query"]
    assert blocks[0].docstring == _LONG


def test_an_unassigned_literal_is_named_for_its_line():
    parsed = _parse(f'def run(conn):\n    conn.execute("""{_LONG}""")\n')

    blocks = _text_blocks(parsed)
    assert [b.name for b in blocks] == ["text_L2"]
    assert blocks[0].line_start == 2


def test_a_short_literal_is_left_alone():
    parsed = _parse('def run():\n    return "tiny"\n')
    assert _text_blocks(parsed) == []


def test_a_docstring_is_not_a_text_block():
    """It is already carried on the entity it documents; twice under two uids is worse."""
    parsed = _parse(f'def run():\n    """{_LONG}"""\n    return 1\n')
    assert _text_blocks(parsed) == []


def test_a_module_docstring_is_not_a_text_block():
    parsed = _parse(f'"""{_LONG}"""\n\nX = 1\n')
    assert _text_blocks(parsed) == []


def test_a_method_literal_is_owned_by_the_method():
    parsed = _parse(f'class C:\n    def m(self):\n        t = """{_LONG}"""\n        return t\n')
    assert [b.qualified_name for b in _text_blocks(parsed)] == [f"{PROJECT}:example.C.m.t"]


def test_a_nested_def_claims_its_own_literal_once():
    source = f'def outer():\n    def inner():\n        t = """{_LONG}"""\n        return t\n    return inner\n'
    blocks = _text_blocks(_parse(source))
    assert [b.qualified_name for b in blocks] == [f"{PROJECT}:example.outer.inner.t"]


def test_the_owner_defines_the_block():
    parsed = _parse(f'def run():\n    query = """{_LONG}"""\n    return query\n')
    defines = {(r.from_qualified_name, r.to_name) for r in parsed.relationships if r.rel_type == RelType.DEFINES}
    assert (f"{PROJECT}:example.run", f"{PROJECT}:example.run.query") in defines


def test_a_module_constant_keeps_its_node_and_gains_the_content():
    """Its source is capped at index.max_source_chars, which this is exactly the thing
    to exceed — so give the existing node the text rather than making a rival."""
    parsed = _parse(f'SQL = """{_LONG}"""\n')

    values = [e for e in parsed.entities if e.label == NodeLabel.VALUE]
    assert [v.qualified_name for v in values] == [f"{PROJECT}:example.SQL"]
    assert values[0].kind == ValueKind.CONSTANT
    assert values[0].docstring == _LONG


def test_string_prefixes_and_quotes_are_stripped():
    parsed = _parse(f"def run():\n    t = r'''{_LONG}'''\n    return t\n")
    assert _text_blocks(parsed)[0].docstring == _LONG


# ---------------------------------------------------------------------------
# Module docstrings
# ---------------------------------------------------------------------------


def test_module_docstring_lands_on_the_module_node():
    """It was indexed nowhere: _extract_docstring reads a `body` field, and a `module`
    node has none, so it declined every module silently."""
    parsed = _parse('"""What this module is for."""\n\nX = 1\n')
    module = next(e for e in parsed.entities if e.label == NodeLabel.MODULE)
    assert module.docstring == "What this module is for."


def test_module_docstring_reaches_the_embed_text():
    """The `if docstring` branch in _build_code_entity_text's Module arm was
    unreachable until the parser started populating it."""
    from code_atlas.search.embeddings import build_embed_text

    parsed = _parse('"""Why this exists."""\n')
    module = next(e for e in parsed.entities if e.label == NodeLabel.MODULE)
    text = build_embed_text(
        {
            "_label": "Module",
            "qualified_name": "example",
            "kind": "module",
            "signature": "",
            "docstring": module.docstring or "",
            "source": "",
        }
    )
    assert "Why this exists." in text


def test_a_package_docstring_lands_too():
    parsed = _parse('"""Package rationale."""\n', path="src/pkg/__init__.py")
    package = next(e for e in parsed.entities if e.label == NodeLabel.PACKAGE)
    assert package.docstring == "Package rationale."


def test_a_module_without_a_docstring_gets_none():
    parsed = _parse("import os\n\nX = 1\n")
    module = next(e for e in parsed.entities if e.label == NodeLabel.MODULE)
    assert module.docstring is None


def test_a_leading_string_that_is_not_first_is_not_a_docstring():
    parsed = _parse('import os\n\n"""Not a docstring."""\n')
    module = next(e for e in parsed.entities if e.label == NodeLabel.MODULE)
    assert module.docstring is None


def test_the_module_docstring_is_not_also_a_text_block():
    """It is carried on the module; a second node would embed the same prose twice."""
    long_doc = "word " * 200
    parsed = _parse(f'"""{long_doc}"""\n')
    blocks = [e for e in parsed.entities if e.kind == ValueKind.TEXT_BLOCK]
    assert blocks == []


# ---------------------------------------------------------------------------
# Attribute docstrings (PEP 258)
# ---------------------------------------------------------------------------


def _value(parsed: ParsedFile, name: str):
    return next(e for e in parsed.entities if e.label == NodeLabel.VALUE and e.name == name)


def test_a_constants_docstring_lands_on_its_value_node():
    """The string is not an entity of its own, so this prose reached the graph nowhere."""
    parsed = _parse('X = 3\n"""Why X is three."""\n')
    assert _value(parsed, "X").docstring == "Why X is three."


def test_an_annotated_constant_gets_it_too():
    parsed = _parse('X: int = 3\n"""Annotated, still documented."""\n')
    assert _value(parsed, "X").docstring == "Annotated, still documented."


def test_a_class_attribute_docstring_lands_on_the_field():
    parsed = _parse('class C:\n    x = 1\n    """The field rationale."""\n')
    assert _value(parsed, "x").docstring == "The field rationale."


def test_a_constant_with_no_following_string_has_no_docstring():
    parsed = _parse("X = 3\nY = 4\n")
    assert _value(parsed, "X").docstring is None


def test_a_string_two_statements_later_is_not_the_docstring():
    parsed = _parse('X = 3\nY = 4\n"""Belongs to Y, if anyone."""\n')
    assert _value(parsed, "X").docstring is None


def test_a_long_attribute_docstring_is_not_also_a_text_block():
    """It is carried on the Value it documents; a second node would embed it twice."""
    long_doc = "word " * 200
    parsed = _parse(f'X = 3\n"""{long_doc}"""\n')
    assert _value(parsed, "X").docstring == long_doc.strip()
    assert [e for e in parsed.entities if e.kind == ValueKind.TEXT_BLOCK] == []


def test_a_long_string_assigned_to_a_name_is_still_a_text_block_not_a_docstring():
    """Guards the guard: the exclusion must not swallow assigned literals."""
    long_doc = "word " * 200
    parsed = _parse(f'def f():\n    q = """{long_doc}"""\n    return q\n')
    blocks = [e for e in parsed.entities if e.kind == ValueKind.TEXT_BLOCK]
    assert [b.name for b in blocks] == ["q"]


# ---------------------------------------------------------------------------
# Source de-duplication
# ---------------------------------------------------------------------------


def _src(parsed, name: str) -> str:
    return next(e.source or "" for e in parsed.entities if e.name == name)


def test_a_docstring_is_not_repeated_inside_its_own_source():
    """It reaches the index as the docstring field; carrying it again in source indexed
    the same bytes under one entity twice."""
    parsed = _parse('def f():\n    """Why f exists."""\n    return 1\n')

    entity = next(e for e in parsed.entities if e.name == "f")
    assert entity.docstring == "Why f exists."
    assert "Why f exists." not in (entity.source or "")
    assert '"""..."""' in (entity.source or "")


def test_a_multi_line_signature_does_not_defeat_the_docstring_elision():
    """The first attempt scanned until a line that was not a decorator or a def, which
    silently failed on every multi-line signature -- most of the long functions here."""
    parsed = _parse('def f(\n    a: int,\n    b: int,\n) -> int:\n    """Why f exists."""\n    return a + b\n')
    assert "Why f exists." not in _src(parsed, "f")


def test_a_nested_function_is_replaced_by_a_reference():
    """It is its own entity with its own source; the parent carried it whole as well."""
    parsed = _parse("def outer():\n    def inner():\n        return 1\n    return inner\n")

    source = _src(parsed, "outer")
    assert "return 1" not in source
    assert "outer.inner" in source


def test_an_extracted_text_block_is_replaced_by_a_reference():
    long_text = "word " * 200
    parsed = _parse('def f():\n    q = """' + long_text + '"""\n    return q\n')

    source = _src(parsed, "f")
    assert "word word" not in source
    assert "-> " in source


def test_the_reference_names_the_node_that_holds_the_text():
    """A deletion would lose the fact that something is there; a reference does not."""
    parsed = _parse("def outer():\n    def inner():\n        return 1\n    return inner\n")

    source = _src(parsed, "outer")
    assert "inner" in source
    assert "#" in source


def test_a_function_with_nothing_to_elide_is_untouched():
    original = "def f(a):\n    return a + 1\n"
    parsed = _parse(original)
    assert _src(parsed, "f") == original.rstrip(chr(10))


def test_elision_does_not_lose_code_after_the_nested_definition():
    """Replacing a span by index is order-sensitive; a later span must still line up."""
    parsed = _parse("def outer():\n    def inner():\n        return 1\n    marker_after = 2\n    return marker_after\n")
    source = _src(parsed, "outer")
    assert "marker_after" in source
    assert "return 1" not in source


def test_the_docstring_field_itself_is_untouched():
    """De-duplication must remove the copy, never the original."""
    parsed = _parse('def f():\n    """Why f exists."""\n    return 1\n')
    assert next(e for e in parsed.entities if e.name == "f").docstring == "Why f exists."


def test_a_grandchild_span_does_not_eat_the_parents_own_code():
    """Shipped broken and caught in review. _child_spans matched descendants at ANY
    depth, and replacements are applied highest-line-first so an earlier one cannot
    shift a later index -- an argument that holds only for spans that do not overlap.
    A grandchild replaced first shrank the line list, and the child's now-stale slice
    ate the parent's own code below the nested definition. Ten functions in this repo's
    src/ were silently losing their tails."""
    parsed = _parse(
        "def outer():\n"
        "    def inner():\n"
        "        def deep():\n"
        "            return 1\n"
        "        return deep\n"
        "    tail_one = 10\n"
        "    tail_two = 20\n"
        "    return inner(tail_two)\n"
    )
    source = _src(parsed, "outer")
    assert "tail_one = 10" in source
    assert "tail_two = 20" in source
    assert "return 1" not in source


def test_only_the_outermost_child_is_elided():
    """Eliding a grandchild is unnecessary as well as unsafe: its text goes when its own
    parent's span is replaced."""
    parsed = _parse(
        "def outer():\n"
        "    def inner():\n"
        "        def deep():\n"
        "            return 1\n"
        "        return deep\n"
        "    return inner\n"
    )
    source = _src(parsed, "outer")
    assert source.count("# ...") == 1
    assert "outer.inner" in source
    assert "deep" not in source
