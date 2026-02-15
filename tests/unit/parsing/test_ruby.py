"""Tests for Ruby parser."""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_ruby", reason="tree-sitter-ruby not installed")

from code_atlas.parsing.ast import ParsedFile, get_language_for_file, parse_file
from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind, ValueKind, Visibility

PROJECT = "test_project"


def _parse(source: str, path: str = "lib/example.rb") -> ParsedFile:
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


def test_language_detection_rb():
    assert get_language_for_file("app/models/user.rb") is not None


def test_language_detection_rake():
    assert get_language_for_file("lib/tasks/deploy.rake") is not None


def test_language_detection_gemspec():
    assert get_language_for_file("my_gem.gemspec") is not None


# ---------------------------------------------------------------------------
# 2. Module entity creation
# ---------------------------------------------------------------------------


def test_module_entity():
    parsed = _parse("x = 1\n", path="lib/models/user.rb")
    module = _entity_by_name(parsed, "user")
    assert module.label == NodeLabel.MODULE
    assert module.kind == "module"
    assert module.qualified_name == f"{PROJECT}:lib.models.user"


def test_module_entity_rake():
    parsed = _parse("task :default\n", path="lib/tasks/deploy.rake")
    module = _entity_by_name(parsed, "deploy")
    assert module.label == NodeLabel.MODULE
    assert module.qualified_name == f"{PROJECT}:lib.tasks.deploy"


# ---------------------------------------------------------------------------
# 3. Class extraction
# ---------------------------------------------------------------------------


def test_class_basic():
    parsed = _parse("""\
# A user model.
class User
  def initialize(name)
    @name = name
  end
end
""")
    cls = _entity_by_name(parsed, "User")
    assert cls.label == NodeLabel.TYPE_DEF
    assert cls.kind == TypeDefKind.CLASS
    assert cls.docstring == "A user model."
    assert cls.visibility == Visibility.PUBLIC


def test_class_qualified_name():
    parsed = _parse("class MyClass\nend\n")
    cls = _entity_by_name(parsed, "MyClass")
    assert cls.qualified_name == f"{PROJECT}:lib.example.MyClass"


# ---------------------------------------------------------------------------
# 4. Ruby module extraction -> PROTOCOL kind
# ---------------------------------------------------------------------------


def test_ruby_module_as_protocol():
    parsed = _parse("""\
module Serializable
  def to_json
    # ...
  end
end
""")
    mod = _entity_by_name(parsed, "Serializable")
    assert mod.label == NodeLabel.TYPE_DEF
    assert mod.kind == TypeDefKind.PROTOCOL


# ---------------------------------------------------------------------------
# 5. Method extraction
# ---------------------------------------------------------------------------


def test_method_extraction():
    parsed = _parse("""\
class Greeter
  def greet(name)
    puts "Hello #{name}"
  end
end
""")
    method = _entity_by_name(parsed, "greet")
    assert method.label == NodeLabel.CALLABLE
    assert method.kind == CallableKind.METHOD
    assert method.qualified_name == f"{PROJECT}:lib.example.Greeter.greet"


def test_constructor_method():
    parsed = _parse("""\
class Foo
  def initialize(x)
    @x = x
  end
end
""")
    init = _entity_by_name(parsed, "initialize")
    assert init.kind == CallableKind.CONSTRUCTOR


def test_top_level_method():
    parsed = _parse("""\
def helper
  42
end
""")
    func = _entity_by_name(parsed, "helper")
    assert func.label == NodeLabel.CALLABLE
    assert func.kind == CallableKind.FUNCTION


# ---------------------------------------------------------------------------
# 6. Singleton method -> STATIC_METHOD
# ---------------------------------------------------------------------------


def test_singleton_method():
    parsed = _parse("""\
class Config
  def self.load(path)
    # ...
  end
end
""")
    method = _entity_by_name(parsed, "load")
    assert method.label == NodeLabel.CALLABLE
    assert method.kind == CallableKind.STATIC_METHOD


# ---------------------------------------------------------------------------
# 7. Visibility tracking (private/protected/public)
# ---------------------------------------------------------------------------


def test_visibility_private_block():
    parsed = _parse("""\
class Account
  def public_method
  end

  private

  def secret_method
  end
end
""")
    pub = _entity_by_name(parsed, "public_method")
    assert pub.visibility == Visibility.PUBLIC

    priv = _entity_by_name(parsed, "secret_method")
    assert priv.visibility == Visibility.PRIVATE


def test_visibility_protected():
    parsed = _parse("""\
class Base
  protected

  def compare(other)
  end
end
""")
    method = _entity_by_name(parsed, "compare")
    assert method.visibility == Visibility.PROTECTED


def test_visibility_restore_public():
    parsed = _parse("""\
class Example
  private

  def hidden
  end

  public

  def visible
  end
end
""")
    hidden = _entity_by_name(parsed, "hidden")
    assert hidden.visibility == Visibility.PRIVATE

    visible = _entity_by_name(parsed, "visible")
    assert visible.visibility == Visibility.PUBLIC


# ---------------------------------------------------------------------------
# 8. require / require_relative -> IMPORTS
# ---------------------------------------------------------------------------


def test_require_imports():
    parsed = _parse("""\
require 'json'
require "yaml"
""")
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    imported = {r.to_name for r in import_rels}
    assert "json" in imported
    assert "yaml" in imported


def test_require_relative_imports():
    parsed = _parse("""\
require_relative 'models/user'
""")
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    assert any(r.to_name == "models/user" for r in import_rels)


# ---------------------------------------------------------------------------
# 9. Class inheritance (< Base) -> INHERITS
# ---------------------------------------------------------------------------


def test_class_inheritance():
    parsed = _parse("""\
class Admin < User
end
""")
    inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
    base_names = {r.to_name for r in inherits}
    assert "User" in base_names


def test_class_inheritance_scoped():
    parsed = _parse("""\
class MyController < ApplicationController
end
""")
    inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
    assert any(r.to_name == "ApplicationController" for r in inherits)


# ---------------------------------------------------------------------------
# 10. include / extend / prepend -> INHERITS
# ---------------------------------------------------------------------------


def test_include_inherits():
    parsed = _parse("""\
class User
  include Comparable
  include Serializable
end
""")
    inherits = _rels_from(parsed, "lib.example.User", RelType.INHERITS)
    mixin_names = {r.to_name for r in inherits}
    assert "Comparable" in mixin_names
    assert "Serializable" in mixin_names


def test_extend_inherits():
    parsed = _parse("""\
class Config
  extend Forwardable
end
""")
    inherits = _rels_from(parsed, "lib.example.Config", RelType.INHERITS)
    assert any(r.to_name == "Forwardable" for r in inherits)


def test_prepend_inherits():
    parsed = _parse("""\
class Logger
  prepend Buffering
end
""")
    inherits = _rels_from(parsed, "lib.example.Logger", RelType.INHERITS)
    assert any(r.to_name == "Buffering" for r in inherits)


# ---------------------------------------------------------------------------
# 11. Doc comment extraction
# ---------------------------------------------------------------------------


def test_doc_comment_extraction():
    parsed = _parse("""\
# Calculate the sum of two numbers.
# Returns an integer.
def add(a, b)
  a + b
end
""")
    func = _entity_by_name(parsed, "add")
    assert func.docstring is not None
    assert "Calculate the sum" in func.docstring
    assert "Returns an integer" in func.docstring


def test_no_docstring_when_no_comment():
    parsed = _parse("""\
def bare
end
""")
    func = _entity_by_name(parsed, "bare")
    assert func.docstring is None


# ---------------------------------------------------------------------------
# 12. Signature extraction
# ---------------------------------------------------------------------------


def test_method_signature():
    parsed = _parse("""\
class Foo
  def bar(x, y)
  end
end
""")
    method = _entity_by_name(parsed, "bar")
    assert method.signature is not None
    assert "bar" in method.signature
    assert "x" in method.signature
    assert "y" in method.signature


def test_singleton_method_signature():
    parsed = _parse("""\
class Foo
  def self.create(attrs)
  end
end
""")
    method = _entity_by_name(parsed, "create")
    assert method.signature is not None
    assert "create" in method.signature


# ---------------------------------------------------------------------------
# 13. Constants (UPPER_CASE)
# ---------------------------------------------------------------------------


def test_constant_extraction():
    parsed = _parse("""\
MAX_SIZE = 100
""")
    const = _entity_by_name(parsed, "MAX_SIZE")
    assert const.label == NodeLabel.VALUE
    assert const.kind == ValueKind.CONSTANT


def test_constant_in_class():
    parsed = _parse("""\
class Config
  DEFAULT_PORT = 3000
end
""")
    const = _entity_by_name(parsed, "DEFAULT_PORT")
    assert const.label == NodeLabel.VALUE
    assert const.kind == ValueKind.CONSTANT


# ---------------------------------------------------------------------------
# 14. DEFINES relationships
# ---------------------------------------------------------------------------


def test_defines_class_from_module():
    parsed = _parse("""\
class Foo
  def bar
  end
end

def baz
end
""")
    # Module DEFINES Foo
    mod_defines = _rels_from(parsed, "lib.example", RelType.DEFINES)
    targets = {r.to_name for r in mod_defines}
    assert f"{PROJECT}:lib.example.Foo" in targets
    assert f"{PROJECT}:lib.example.baz" in targets

    # Foo DEFINES bar
    foo_defines = _rels_from(parsed, "lib.example.Foo", RelType.DEFINES)
    assert any(r.to_name == f"{PROJECT}:lib.example.Foo.bar" for r in foo_defines)


# ---------------------------------------------------------------------------
# 15. CALLS extraction
# ---------------------------------------------------------------------------


def test_calls_extraction():
    parsed = _parse("""\
class Worker
  def perform
    validate
    process_data
  end
end
""")
    calls = _rels_from(parsed, "lib.example.Worker.perform", RelType.CALLS)
    called = {r.to_name for r in calls}
    assert "validate" in called
    assert "process_data" in called


# ---------------------------------------------------------------------------
# 16. Content hash determinism
# ---------------------------------------------------------------------------


def test_content_hash_populated():
    parsed = _parse("""\
class Foo
  def bar
  end
end
""")
    for entity in parsed.entities:
        assert entity.content_hash, f"Entity {entity.name!r} has empty content_hash"


def test_content_hash_deterministic():
    source = """\
def greet(name)
  puts "Hello #{name}"
end
"""
    parsed1 = _parse(source)
    parsed2 = _parse(source)
    for e1, e2 in zip(parsed1.entities, parsed2.entities, strict=True):
        assert e1.content_hash == e2.content_hash


# ---------------------------------------------------------------------------
# 17. Edge cases (empty file, syntax errors)
# ---------------------------------------------------------------------------


def test_empty_file():
    parsed = _parse("")
    assert parsed is not None
    assert parsed.language == "ruby"
    # Should have at least the module entity
    assert len(parsed.entities) >= 1


def test_syntax_error_tolerant():
    """Tree-sitter is error-tolerant — malformed files don't crash."""
    parsed = _parse("def broken(\n  class nope\n")
    assert parsed is not None


def test_binary_content():
    """Binary content shouldn't crash the parser."""
    parsed = parse_file("data.rb", b"\x00\x01\x02\xff\xfe", PROJECT)
    assert parsed is not None


# ---------------------------------------------------------------------------
# 18. Nested classes / modules
# ---------------------------------------------------------------------------


def test_nested_class():
    parsed = _parse("""\
class Outer
  class Inner
    def work
    end
  end
end
""")
    inner = _entity_by_name(parsed, "Inner")
    assert "Outer.Inner" in inner.qualified_name
    assert inner.kind == TypeDefKind.CLASS

    work = _entity_by_name(parsed, "work")
    assert "Outer.Inner.work" in work.qualified_name


def test_nested_module_and_class():
    parsed = _parse("""\
module MyApp
  class Server
    def start
    end
  end
end
""")
    mod = _entity_by_name(parsed, "MyApp")
    assert mod.kind == TypeDefKind.PROTOCOL

    server = _entity_by_name(parsed, "Server")
    assert "MyApp.Server" in server.qualified_name

    start = _entity_by_name(parsed, "start")
    assert "MyApp.Server.start" in start.qualified_name


# ---------------------------------------------------------------------------
# Additional: attr_reader / attr_writer / attr_accessor tags
# ---------------------------------------------------------------------------


def test_attr_accessor_tags():
    parsed = _parse("""\
class Person
  attr_accessor :name, :age
end
""")
    name_entity = _entity_by_name(parsed, "name")
    assert name_entity.label == NodeLabel.VALUE
    assert "synthesized:attr_accessor" in name_entity.tags

    age_entity = _entity_by_name(parsed, "age")
    assert "synthesized:attr_accessor" in age_entity.tags


# ---------------------------------------------------------------------------
# Additional: method source extraction
# ---------------------------------------------------------------------------


def test_method_source_extracted():
    parsed = _parse("""\
class Foo
  def bar
    42
  end
end
""")
    method = _entity_by_name(parsed, "bar")
    assert method.source is not None
    assert "def bar" in method.source
    assert "42" in method.source
