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


def test_inline_private_def():
    parsed = _parse("""\
class Account
  def visible
  end

  private def helper
    42
  end

  def after_inline
  end
end
""")
    helper = _entity_by_name(parsed, "helper")
    assert helper.label == NodeLabel.CALLABLE
    assert helper.kind == CallableKind.METHOD
    assert helper.qualified_name == f"{PROJECT}:lib.example.Account.helper"
    assert helper.visibility == Visibility.PRIVATE

    # Inline modifier applies only to the wrapped method, not subsequent ones
    assert _entity_by_name(parsed, "after_inline").visibility == Visibility.PUBLIC


def test_inline_protected_def():
    parsed = _parse("""\
class Base
  protected def compare(other)
  end
end
""")
    method = _entity_by_name(parsed, "compare")
    assert method.visibility == Visibility.PROTECTED


def test_inline_public_def():
    parsed = _parse("""\
class Example
  private

  def hidden
  end

  public def shown
  end

  def still_hidden
  end
end
""")
    shown = _entity_by_name(parsed, "shown")
    assert shown.visibility == Visibility.PUBLIC

    # Inline `public def` does not end the surrounding private section
    assert _entity_by_name(parsed, "hidden").visibility == Visibility.PRIVATE
    assert _entity_by_name(parsed, "still_hidden").visibility == Visibility.PRIVATE


def test_inline_private_def_defines_and_calls():
    parsed = _parse("""\
class Worker
  def perform
    helper
  end

  private def helper
    validate
  end
end
""")
    defines = _rels_from(parsed, "lib.example.Worker", RelType.DEFINES)
    assert any(r.to_name == f"{PROJECT}:lib.example.Worker.helper" for r in defines)

    calls = _rels_from(parsed, "lib.example.Worker.helper", RelType.CALLS)
    assert any(r.to_name == "validate" for r in calls)


def test_inline_visibility_mixed_with_bare_sections():
    parsed = _parse("""\
class Mixed
  private def early_helper
  end

  private

  def section_private
  end
end
""")
    assert _entity_by_name(parsed, "early_helper").visibility == Visibility.PRIVATE
    assert _entity_by_name(parsed, "section_private").visibility == Visibility.PRIVATE


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


def test_compact_class_path_name_is_bare():
    """`class Admin::User` should store bare name 'User', matching the nested-form spelling."""
    parsed = _parse("""\
class Admin::User
end
""")
    cls = _entity_by_name(parsed, "User")
    assert cls.label == NodeLabel.TYPE_DEF
    assert cls.kind == TypeDefKind.CLASS
    assert cls.qualified_name == f"{PROJECT}:lib.example.Admin.User"


def test_compact_module_path_name_is_bare():
    """`module Admin::Helpers` should store bare name 'Helpers'."""
    parsed = _parse("""\
module Admin::Helpers
end
""")
    mod = _entity_by_name(parsed, "Helpers")
    assert mod.label == NodeLabel.TYPE_DEF
    assert mod.kind == TypeDefKind.PROTOCOL
    assert mod.qualified_name == f"{PROJECT}:lib.example.Admin.Helpers"


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


# ---------------------------------------------------------------------------
# 19. Blocks are not scopes (ADR-0031)
# ---------------------------------------------------------------------------


def test_calls_inside_a_block_attribute_to_the_enclosing_method():
    parsed = _parse("""\
class Worker
  def perform
    items.each do |item|
      transform(item)
    end
  end
end
""")
    called = {r.to_name for r in _rels_from(parsed, "lib.example.Worker.perform", RelType.CALLS)}
    assert "transform" in called


def test_calls_in_a_block_pyramid_all_reach_the_enclosing_method():
    """A call several callbacks deep still belongs to the def that passed them."""
    parsed = _parse("""\
class Worker
  def perform
    rows.each do |row|
      row.cells.map { |cell| normalize(cell) }
    end
  end
end
""")
    called = {r.to_name for r in _rels_from(parsed, "lib.example.Worker.perform", RelType.CALLS)}
    assert {"each", "cells", "map", "normalize"} <= called


def test_calls_in_a_dsl_block_at_class_scope_attribute_to_the_module():
    """`get '/' do ... end` has no enclosing def, so its calls belong to the module."""
    parsed = _parse("""\
class App
  get '/' do
    render_index
  end
end
""")
    called = {r.to_name for r in _rels_from(parsed, ":lib.example", RelType.CALLS)}
    assert "get" in called
    assert "render_index" in called


def test_class_body_call_attributes_to_the_module():
    parsed = _parse("""\
class App
  set :views, 'views'
end
""")
    called = {r.to_name for r in _rels_from(parsed, ":lib.example", RelType.CALLS)}
    assert "set" in called


def test_module_scope_call_attributes_to_the_module():
    parsed = _parse("""\
configure_app
Widget.new
""")
    called = {r.to_name for r in _rels_from(parsed, ":lib.example", RelType.CALLS)}
    assert "configure_app" in called
    assert "new" in called


def test_block_produces_no_callable_entity():
    """ADR-0031: an anonymous form has no name to be looked up by, so it gets no node."""
    parsed = _parse("""\
class App
  get '/' do
    helper
  end

  configure { setup }

  handler = lambda { |x| x }
end
""")
    callables = [e for e in parsed.entities if e.label == NodeLabel.CALLABLE]
    assert callables == [], f"a block became a Callable: {[e.name for e in callables]}"


def test_assignment_inside_a_block_is_not_a_value_entity():
    """A block's assignments bind locals; only a class body declares fields."""
    parsed = _parse("""\
class App
  LIMIT = 10

  get '/' do
    scratch = 1
  end
end
""")
    names = {e.name for e in parsed.entities if e.label == NodeLabel.VALUE}
    assert "LIMIT" in names
    assert "scratch" not in names, "a block local became a Value entity"


# ---------------------------------------------------------------------------
# 20. Named definitions the walker used to step over
# ---------------------------------------------------------------------------


def test_singleton_class_methods_are_static():
    parsed = _parse("""\
class Config
  class << self
    def load(path)
    end
  end
end
""")
    method = _entity_by_name(parsed, "load")
    assert method.label == NodeLabel.CALLABLE
    assert method.kind == CallableKind.STATIC_METHOD
    assert method.qualified_name == f"{PROJECT}:lib.example.Config.self.load"


def test_singleton_class_body_defines_from_the_reopened_class():
    parsed = _parse("""\
class Config
  class << self
    def load(path)
    end
  end
end
""")
    defines = _rels_from(parsed, "lib.example.Config", RelType.DEFINES)
    assert any(r.to_name == f"{PROJECT}:lib.example.Config.self.load" for r in defines)


def test_def_in_a_conditional_branch_is_an_entity():
    parsed = _parse("""\
module Sinatra
  class Cookies
    if RUBY_VERSION >= '3.0'
      def modern_each
      end
    end
  end
end
""")
    method = _entity_by_name(parsed, "modern_each")
    assert method.kind == CallableKind.METHOD
    assert method.qualified_name == f"{PROJECT}:lib.example.Sinatra.Cookies.modern_each"


def test_def_behind_an_if_modifier_is_an_entity():
    parsed = _parse("""\
class IndifferentHash
  def except(*keys)
  end if RUBY_VERSION < '3.0'
end
""")
    method = _entity_by_name(parsed, "except")
    assert method.kind == CallableKind.METHOD
    assert method.qualified_name == f"{PROJECT}:lib.example.IndifferentHash.except"


def test_require_inside_a_rescue_guard_is_an_import():
    parsed = _parse("""\
begin
  require 'yajl'
rescue LoadError
  require 'json'
end
""")
    imported = {r.to_name for r in parsed.relationships if r.rel_type == RelType.IMPORTS}
    assert imported == {"yajl", "json"}


# ---------------------------------------------------------------------------
# 21. Edges the walker must NOT invent
# ---------------------------------------------------------------------------


def test_bare_local_variable_read_is_not_a_call():
    """`result` alone reads a local; `content_type` alone calls a method."""
    parsed = _parse("""\
class Worker
  def perform
    result = compute
    content_type
    result
  end
end
""")
    called = {r.to_name for r in _rels_from(parsed, "lib.example.Worker.perform", RelType.CALLS)}
    assert "content_type" in called
    assert "result" not in called, "a local variable read became a CALLS edge"


def test_block_parameter_read_is_not_a_call():
    parsed = _parse("""\
class Worker
  def perform
    items.each do |item|
      item
    end
  end
end
""")
    called = {r.to_name for r in _rels_from(parsed, "lib.example.Worker.perform", RelType.CALLS)}
    assert "item" not in called, "a block parameter read became a CALLS edge"


def test_mixin_with_a_receiver_is_not_an_inherits():
    """`base.include Helpers` names a receiver the walker cannot resolve to a type."""
    parsed = _parse("""\
class Installer
  def self.installed(base)
    base.include Helpers
  end
end
""")
    inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
    assert inherits == [], f"invented INHERITS from a call with a receiver: {[r.to_name for r in inherits]}"


# ---------------------------------------------------------------------------
# 22. A uid must identify exactly one definition (ADR-0032)
# ---------------------------------------------------------------------------


def _callable_uids(parsed: ParsedFile) -> list[str]:
    return [e.qualified_name for e in parsed.entities if e.label == NodeLabel.CALLABLE]


def _duplicate_uids(parsed: ParsedFile) -> list[str]:
    uids = _callable_uids(parsed)
    return sorted({u for u in uids if uids.count(u) > 1})


def test_singleton_and_instance_method_of_one_name_get_distinct_uids():
    """sinatra's `Base.settings` / `Base#settings` — two definitions, one uid."""
    parsed = _parse("""\
class Base
  def self.settings
    self
  end

  def settings
    self.class.settings
  end
end
""")
    assert _duplicate_uids(parsed) == []
    uids = set(_callable_uids(parsed))
    assert uids == {f"{PROJECT}:lib.example.Base.self.settings", f"{PROJECT}:lib.example.Base.settings"}


def test_class_self_and_instance_method_of_one_name_get_distinct_uids():
    """The other spelling: `def foo` inside `class << self` is still a singleton method."""
    parsed = _parse("""\
class Base
  class << self
    def call(env)
    end
  end

  def call(env)
  end
end
""")
    assert _duplicate_uids(parsed) == []
    uids = set(_callable_uids(parsed))
    assert uids == {f"{PROJECT}:lib.example.Base.self.call", f"{PROJECT}:lib.example.Base.call"}


def test_no_two_definitions_in_one_file_share_a_uid():
    """The negative form: a positive assertion cannot catch a merge it did not name."""
    parsed = _parse("""\
module Sinatra
  class Base
    def self.settings
    end

    def settings
    end

    class << self
      def force_encoding(data)
      end
    end

    def force_encoding(data)
    end
  end

  class IndifferentHash
    def self.[](*args)
    end

    def [](key)
    end
  end
end

shared_examples_for 'protection' do
  def call(env)
  end
end

shared_examples_for 'other' do
  def call(env)
  end
end
""")
    assert _duplicate_uids(parsed) == []


def test_instance_method_in_a_plain_class_keeps_its_uid():
    """The churn bound: only the rarer singleton form moves."""
    parsed = _parse("""\
module Sinatra
  class Base
    def initialize(app = nil)
    end

    def call(env)
    end
  end
end
""")
    uids = set(_callable_uids(parsed))
    assert uids == {
        f"{PROJECT}:lib.example.Sinatra.Base.initialize",
        f"{PROJECT}:lib.example.Sinatra.Base.call",
    }


def test_singleton_method_on_a_non_self_receiver_takes_no_self_segment():
    """`def enc.generate` defines on a runtime object, not on the enclosing class.

    A `self` segment would claim the class owns it. This pins the current
    behaviour so that changing it stays a decision rather than a side effect.
    """
    parsed = _parse("""\
class Encoder
  def enc.generate(obj)
  end
end
""")
    assert _callable_uids(parsed) == [f"{PROJECT}:lib.example.Encoder.generate"]


# --- category C: an entity needs every enclosing scope to be named -----------


def test_def_inside_a_block_produces_no_entity():
    """Reverses ATL-096. A block names nothing, so no scope path reaches the def."""
    parsed = _parse("""\
describe 'Delegator' do
  def delegation_agent
    Object.new
  end
end
""")
    callables = [e for e in parsed.entities if e.label == NodeLabel.CALLABLE]
    assert callables == [], f"a block-nested def became a Callable: {[e.name for e in callables]}"


def test_singleton_def_inside_a_block_produces_no_entity():
    parsed = _parse("""\
describe JsonCsrf do
  def self.env_for(url)
  end
end
""")
    callables = [e for e in parsed.entities if e.label == NodeLabel.CALLABLE]
    assert callables == [], f"a block-nested singleton def became a Callable: {[e.name for e in callables]}"


def test_def_inside_a_class_level_block_produces_no_entity():
    """`superclass.class_eval do def call ... end` attaches to the receiver, not the class."""
    parsed = _parse("""\
class Base
  class_eval do
    def call(env)
    end
  end

  def call(env)
  end
end
""")
    assert _callable_uids(parsed) == [f"{PROJECT}:lib.example.Base.call"]
    defines = _rels_from(parsed, "lib.example.Base", RelType.DEFINES)
    assert len(defines) == 1, f"a declined def still got a DEFINES edge: {[r.to_name for r in defines]}"


def test_sibling_block_defs_of_one_name_produce_no_entity():
    """rack-protection's shared_examples.rb: three `def call`s, one uid between them."""
    parsed = _parse("""\
shared_examples_for 'protection' do
  def call(env)
    a
  end

  it 'x' do
    def call(env)
      b
    end
  end

  def call(env)
    c
  end
end
""")
    callables = [e for e in parsed.entities if e.label == NodeLabel.CALLABLE]
    assert callables == [], f"sibling block defs became Callables: {[e.name for e in callables]}"


def test_declining_a_def_keeps_its_calls():
    """Declining relocates a call to the enclosing scope; it must not drop it."""
    parsed = _parse("""\
describe 'Delegator' do
  def delegation_agent
    Object.new
    normalize(1)
  end
end
""")
    called = {r.to_name for r in _rels_from(parsed, ":lib.example", RelType.CALLS)}
    assert {"new", "normalize"} <= called


def test_declining_a_def_attributes_its_calls_to_the_enclosing_method():
    parsed = _parse("""\
class Worker
  def perform
    items.each do
      def handler
        transform
      end
    end
  end
end
""")
    called = {r.to_name for r in _rels_from(parsed, "lib.example.Worker.perform", RelType.CALLS)}
    assert "transform" in called


def test_declined_def_parameters_are_still_local_reads():
    """The declined body keeps its own bindings, so a parameter is not an implicit call."""
    parsed = _parse("""\
describe 'x' do
  def handler(payload)
    payload
    content_type
  end
end
""")
    called = {r.to_name for r in _rels_from(parsed, ":lib.example", RelType.CALLS)}
    assert "content_type" in called
    assert "payload" not in called, "a declined def's parameter became a CALLS edge"


def test_def_nested_in_a_named_method_is_still_an_entity():
    """Only an anonymous link in the scope chain disqualifies."""
    parsed = _parse("""\
class Worker
  def perform
    def helper
    end
  end
end
""")
    uids = set(_callable_uids(parsed))
    assert f"{PROJECT}:lib.example.Worker.perform.helper" in uids


def test_def_in_a_named_class_inside_a_block_is_an_entity():
    """A `class` re-anchors the chain: Ruby resolves its constant lexically, past the block."""
    parsed = _parse("""\
class BaseTest
  describe 'subclasses' do
    class TestApp < Sinatra::Base
      def initialize(argument:)
      end
    end
  end
end
""")
    uids = set(_callable_uids(parsed))
    assert uids == {f"{PROJECT}:lib.example.BaseTest.TestApp.initialize"}


def test_class_self_inside_a_block_produces_no_entity():
    """`class << self` in a block reopens the singleton of the block's receiver."""
    parsed = _parse("""\
mock_app do
  class << self
    def configure!
    end
  end
end
""")
    callables = [e for e in parsed.entities if e.label == NodeLabel.CALLABLE]
    assert callables == [], f"a block-nested `class << self` def became a Callable: {[e.name for e in callables]}"
