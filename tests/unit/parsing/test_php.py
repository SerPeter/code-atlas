"""Tests for PHP parser."""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_php", reason="tree-sitter-php not installed")

from code_atlas.parsing.ast import ParsedFile, get_language_for_file, parse_file
from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind, ValueKind, Visibility

PROJECT = "test_project"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(source: str, path: str = "src/Example.php") -> ParsedFile:
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
# 1. Language detection (.php)
# ---------------------------------------------------------------------------


def test_language_detection_php():
    cfg = get_language_for_file("src/App.php")
    assert cfg is not None
    assert cfg.name == "php"


def test_language_detection_not_php():
    assert get_language_for_file("src/main.py") is not None  # Python, not PHP
    assert get_language_for_file("data.csv") is None


# ---------------------------------------------------------------------------
# 2. Module entity creation
# ---------------------------------------------------------------------------


def test_module_entity():
    parsed = _parse("<?php\n", path="src/Models/User.php")
    mod = _entity_by_name(parsed, "User")
    assert mod.label == NodeLabel.MODULE
    assert mod.kind == "module"
    assert mod.qualified_name == f"{PROJECT}:src.Models.User"


def test_module_entity_simple_path():
    parsed = _parse("<?php\n", path="App.php")
    mod = _entity_by_name(parsed, "App")
    assert mod.qualified_name == f"{PROJECT}:App"


# ---------------------------------------------------------------------------
# 3. Class extraction
# ---------------------------------------------------------------------------


def test_class_basic():
    parsed = _parse("""\
<?php
/**
 * A user model.
 */
class User {
}
""")
    cls = _entity_by_name(parsed, "User")
    assert cls.label == NodeLabel.TYPE_DEF
    assert cls.kind == TypeDefKind.CLASS
    assert cls.docstring is not None
    assert "A user model." in cls.docstring
    assert cls.visibility == Visibility.PUBLIC


# ---------------------------------------------------------------------------
# 4. Interface extraction
# ---------------------------------------------------------------------------


def test_interface_basic():
    parsed = _parse("""\
<?php
interface Cacheable {
    public function cacheKey(): string;
}
""")
    iface = _entity_by_name(parsed, "Cacheable")
    assert iface.label == NodeLabel.TYPE_DEF
    assert iface.kind == TypeDefKind.INTERFACE


# ---------------------------------------------------------------------------
# 5. Trait extraction
# ---------------------------------------------------------------------------


def test_trait_basic():
    parsed = _parse("""\
<?php
trait HasTimestamps {
    public function createdAt(): string {
        return "now";
    }
}
""")
    trait = _entity_by_name(parsed, "HasTimestamps")
    assert trait.label == NodeLabel.TYPE_DEF
    assert trait.kind == TypeDefKind.TRAIT


# ---------------------------------------------------------------------------
# 6. Enum extraction (PHP 8.1+)
# ---------------------------------------------------------------------------


def test_enum_basic():
    parsed = _parse("""\
<?php
enum Color: string {
    case Red = "red";
    case Blue = "blue";
}
""")
    enum = _entity_by_name(parsed, "Color")
    assert enum.label == NodeLabel.TYPE_DEF
    assert enum.kind == TypeDefKind.ENUM


# ---------------------------------------------------------------------------
# 7. Function extraction
# ---------------------------------------------------------------------------


def test_function_basic():
    parsed = _parse("""\
<?php
function helper(): void {
    echo "hello";
}
""")
    func = _entity_by_name(parsed, "helper")
    assert func.label == NodeLabel.CALLABLE
    assert func.kind == CallableKind.FUNCTION
    assert func.qualified_name == f"{PROJECT}:src.Example.helper"


# ---------------------------------------------------------------------------
# 8. Method extraction
# ---------------------------------------------------------------------------


def test_method_basic():
    parsed = _parse("""\
<?php
class User {
    public function getName(): string {
        return "name";
    }
}
""")
    method = _entity_by_name(parsed, "getName")
    assert method.label == NodeLabel.CALLABLE
    assert method.kind == CallableKind.METHOD
    assert method.qualified_name == f"{PROJECT}:src.Example.User.getName"


# ---------------------------------------------------------------------------
# 9. Constructor and Destructor
# ---------------------------------------------------------------------------


def test_constructor():
    parsed = _parse("""\
<?php
class User {
    public function __construct(string $name) {}
}
""")
    ctor = _entity_by_name(parsed, "__construct")
    assert ctor.kind == CallableKind.CONSTRUCTOR


def test_destructor():
    parsed = _parse("""\
<?php
class Resource {
    public function __destruct() {}
}
""")
    dtor = _entity_by_name(parsed, "__destruct")
    assert dtor.kind == CallableKind.DESTRUCTOR


# ---------------------------------------------------------------------------
# 10. Visibility (public/private/protected)
# ---------------------------------------------------------------------------


def test_visibility_public():
    parsed = _parse("""\
<?php
class User {
    public function getName(): string { return ""; }
}
""")
    method = _entity_by_name(parsed, "getName")
    assert method.visibility == Visibility.PUBLIC


def test_visibility_private():
    parsed = _parse("""\
<?php
class User {
    private function secret(): void {}
}
""")
    method = _entity_by_name(parsed, "secret")
    assert method.visibility == Visibility.PRIVATE


def test_visibility_protected():
    parsed = _parse("""\
<?php
class User {
    protected function internalMethod(): void {}
}
""")
    method = _entity_by_name(parsed, "internalMethod")
    assert method.visibility == Visibility.PROTECTED


def test_property_visibility():
    parsed = _parse("""\
<?php
class User {
    private string $name;
    protected int $age;
    public string $email;
}
""")
    name = _entity_by_name(parsed, "name")
    assert name.visibility == Visibility.PRIVATE

    age = _entity_by_name(parsed, "age")
    assert age.visibility == Visibility.PROTECTED

    email = _entity_by_name(parsed, "email")
    assert email.visibility == Visibility.PUBLIC


# ---------------------------------------------------------------------------
# 11. namespace_use_declaration -> IMPORTS
# ---------------------------------------------------------------------------


def test_namespace_use_import():
    parsed = _parse("""\
<?php
use App\\Models\\User;
use App\\Traits\\HasTimestamps;
""")
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    imported = {r.to_name for r in import_rels}
    assert any("User" in name for name in imported)
    assert any("HasTimestamps" in name for name in imported)


def test_namespace_use_aliased():
    parsed = _parse("""\
<?php
use App\\Models\\User as UserModel;
""")
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    # Should import the original name, not the alias
    assert len(import_rels) >= 1
    assert any("User" in r.to_name for r in import_rels)
    # Should not contain the alias
    assert not any("UserModel" in r.to_name for r in import_rels)


def test_namespace_use_grouped():
    parsed = _parse("""\
<?php
use App\\Models\\{User, Post};
""")
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    imported = {r.to_name for r in import_rels}
    assert any("User" in name for name in imported)
    assert any("Post" in name for name in imported)


# ---------------------------------------------------------------------------
# 12. require/include -> IMPORTS
# ---------------------------------------------------------------------------


def test_require_import():
    parsed = _parse("""\
<?php
require "vendor/autoload.php";
require_once "config/app.php";
include "helpers.php";
include_once "utils.php";
""")
    import_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    imported = {r.to_name for r in import_rels}
    assert "vendor/autoload.php" in imported
    assert "config/app.php" in imported
    assert "helpers.php" in imported
    assert "utils.php" in imported


# ---------------------------------------------------------------------------
# 13. extends -> INHERITS
# ---------------------------------------------------------------------------


def test_extends_inherits():
    parsed = _parse("""\
<?php
class Child extends Parent {
}
""")
    inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
    assert len(inherits) >= 1
    base_names = {r.to_name for r in inherits}
    assert "Parent" in base_names


# ---------------------------------------------------------------------------
# 14. implements -> IMPLEMENTS
# ---------------------------------------------------------------------------


def test_implements():
    parsed = _parse("""\
<?php
class User implements Cacheable, Serializable {
}
""")
    impl_rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPLEMENTS]
    iface_names = {r.to_name for r in impl_rels}
    assert "Cacheable" in iface_names
    assert "Serializable" in iface_names


# ---------------------------------------------------------------------------
# 15. use TraitName -> INHERITS (trait usage)
# ---------------------------------------------------------------------------


def test_use_trait_inherits():
    parsed = _parse("""\
<?php
class User {
    use HasTimestamps;
}
""")
    inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
    assert len(inherits) >= 1
    trait_names = {r.to_name for r in inherits}
    assert "HasTimestamps" in trait_names
    # Should come from the class, not the module
    assert any(r.from_qualified_name.endswith("User") for r in inherits)


# ---------------------------------------------------------------------------
# 16. PHPDoc docstring extraction
# ---------------------------------------------------------------------------


def test_phpdoc_class():
    parsed = _parse("""\
<?php
/**
 * Represents a user in the system.
 *
 * @package App
 */
class User {
}
""")
    cls = _entity_by_name(parsed, "User")
    assert cls.docstring is not None
    assert "Represents a user in the system." in cls.docstring


def test_phpdoc_method():
    parsed = _parse("""\
<?php
class User {
    /**
     * Get the user's full name.
     * @return string
     */
    public function getFullName(): string {
        return "";
    }
}
""")
    method = _entity_by_name(parsed, "getFullName")
    assert method.docstring is not None
    assert "Get the user's full name." in method.docstring


def test_phpdoc_function():
    parsed = _parse("""\
<?php
/**
 * A helper function.
 */
function helper(): void {}
""")
    func = _entity_by_name(parsed, "helper")
    assert func.docstring is not None
    assert "A helper function." in func.docstring


def test_no_docstring_regular_comment():
    """Regular comments (non-PHPDoc) should not be extracted as docstrings."""
    parsed = _parse("""\
<?php
// This is a regular comment.
class User {
}
""")
    cls = _entity_by_name(parsed, "User")
    assert cls.docstring is None


# ---------------------------------------------------------------------------
# 17. Signature extraction
# ---------------------------------------------------------------------------


def test_function_signature():
    parsed = _parse("""\
<?php
function greet(string $name): string {
    return "Hello " . $name;
}
""")
    func = _entity_by_name(parsed, "greet")
    assert func.signature is not None
    assert "function greet" in func.signature
    assert "string $name" in func.signature
    # Body should not be in signature
    assert "return" not in func.signature


def test_method_signature():
    parsed = _parse("""\
<?php
class User {
    public function getName(): string {
        return "";
    }
}
""")
    method = _entity_by_name(parsed, "getName")
    assert method.signature is not None
    assert "function getName" in method.signature
    assert "return" not in method.signature


def test_abstract_method_signature():
    parsed = _parse("""\
<?php
abstract class Base {
    abstract public function process(): void;
}
""")
    method = _entity_by_name(parsed, "process")
    assert method.signature is not None
    assert "function process" in method.signature


# ---------------------------------------------------------------------------
# 18. Properties -> FIELD
# ---------------------------------------------------------------------------


def test_property_field():
    parsed = _parse("""\
<?php
class User {
    private string $name;
    protected int $age;
}
""")
    name = _entity_by_name(parsed, "name")
    assert name.label == NodeLabel.VALUE
    assert name.kind == ValueKind.FIELD
    assert name.qualified_name == f"{PROJECT}:src.Example.User.name"

    age = _entity_by_name(parsed, "age")
    assert age.label == NodeLabel.VALUE
    assert age.kind == ValueKind.FIELD


# ---------------------------------------------------------------------------
# 19. Constants -> CONSTANT
# ---------------------------------------------------------------------------


def test_class_constant():
    parsed = _parse("""\
<?php
class User {
    public const STATUS_ACTIVE = 1;
    public const STATUS_INACTIVE = 0;
}
""")
    active = _entity_by_name(parsed, "STATUS_ACTIVE")
    assert active.label == NodeLabel.VALUE
    assert active.kind == ValueKind.CONSTANT
    assert active.qualified_name == f"{PROJECT}:src.Example.User.STATUS_ACTIVE"

    inactive = _entity_by_name(parsed, "STATUS_INACTIVE")
    assert inactive.kind == ValueKind.CONSTANT


# ---------------------------------------------------------------------------
# 20. Enum cases -> ENUM_MEMBER
# ---------------------------------------------------------------------------


def test_enum_cases():
    parsed = _parse("""\
<?php
enum Color: string {
    case Red = "red";
    case Blue = "blue";
}
""")
    red = _entity_by_name(parsed, "Red")
    assert red.label == NodeLabel.VALUE
    assert red.kind == ValueKind.ENUM_MEMBER
    assert red.qualified_name == f"{PROJECT}:src.Example.Color.Red"

    blue = _entity_by_name(parsed, "Blue")
    assert blue.kind == ValueKind.ENUM_MEMBER


# ---------------------------------------------------------------------------
# 21. PHP 8 attribute tags
# ---------------------------------------------------------------------------


def test_attribute_tags_on_method():
    parsed = _parse("""\
<?php
class Controller {
    #[Route("/api/users")]
    public function index(): void {}
}
""")
    method = _entity_by_name(parsed, "index")
    assert any("attribute:" in tag for tag in method.tags)
    assert any("Route" in tag for tag in method.tags)


def test_attribute_tags_on_class():
    parsed = _parse("""\
<?php
#[Deprecated]
class OldService {
}
""")
    cls = _entity_by_name(parsed, "OldService")
    assert any("attribute:Deprecated" in tag for tag in cls.tags)


def test_attribute_tags_on_function():
    parsed = _parse("""\
<?php
#[Pure]
function compute(): int {
    return 42;
}
""")
    func = _entity_by_name(parsed, "compute")
    assert any("attribute:Pure" in tag for tag in func.tags)


# ---------------------------------------------------------------------------
# 22. Abstract/static/final tags
# ---------------------------------------------------------------------------


def test_abstract_tag():
    parsed = _parse("""\
<?php
abstract class Base {
    abstract public function process(): void;
}
""")
    cls = _entity_by_name(parsed, "Base")
    assert "abstract" in cls.tags

    method = _entity_by_name(parsed, "process")
    assert "abstract" in method.tags


def test_static_tag():
    parsed = _parse("""\
<?php
class Factory {
    public static function create(): self {
        return new self();
    }
}
""")
    method = _entity_by_name(parsed, "create")
    assert "static" in method.tags
    assert method.kind == CallableKind.STATIC_METHOD


def test_final_tag():
    parsed = _parse("""\
<?php
class User {
    final public function getId(): int {
        return 1;
    }
}
""")
    method = _entity_by_name(parsed, "getId")
    assert "final" in method.tags


def test_readonly_tag():
    parsed = _parse("""\
<?php
class Config {
    public readonly string $name;
}
""")
    name = _entity_by_name(parsed, "name")
    assert "readonly" in name.tags


# ---------------------------------------------------------------------------
# 23. DEFINES relationships
# ---------------------------------------------------------------------------


def test_defines_module_to_class():
    parsed = _parse("""\
<?php
class User {
}

function helper(): void {}
""")
    mod_defines = _rels_from(parsed, "src.Example", RelType.DEFINES)
    targets = {r.to_name for r in mod_defines}
    assert f"{PROJECT}:src.Example.User" in targets
    assert f"{PROJECT}:src.Example.helper" in targets


def test_defines_class_to_method():
    parsed = _parse("""\
<?php
class User {
    public function getName(): string { return ""; }
}
""")
    class_defines = _rels_from(parsed, "src.Example.User", RelType.DEFINES)
    targets = {r.to_name for r in class_defines}
    assert f"{PROJECT}:src.Example.User.getName" in targets


def test_defines_class_to_property():
    parsed = _parse("""\
<?php
class User {
    private string $name;
}
""")
    class_defines = _rels_from(parsed, "src.Example.User", RelType.DEFINES)
    targets = {r.to_name for r in class_defines}
    assert f"{PROJECT}:src.Example.User.name" in targets


def test_defines_class_to_constant():
    parsed = _parse("""\
<?php
class User {
    public const MAX = 100;
}
""")
    class_defines = _rels_from(parsed, "src.Example.User", RelType.DEFINES)
    targets = {r.to_name for r in class_defines}
    assert f"{PROJECT}:src.Example.User.MAX" in targets


def test_defines_enum_to_case():
    parsed = _parse("""\
<?php
enum Color {
    case Red;
}
""")
    enum_defines = _rels_from(parsed, "src.Example.Color", RelType.DEFINES)
    targets = {r.to_name for r in enum_defines}
    assert f"{PROJECT}:src.Example.Color.Red" in targets


# ---------------------------------------------------------------------------
# 24. CALLS extraction
# ---------------------------------------------------------------------------


def test_calls_function_call():
    parsed = _parse("""\
<?php
function doWork(): void {
    helper();
    process();
}
""")
    calls = _rels_from(parsed, "src.Example.doWork", RelType.CALLS)
    called = {r.to_name for r in calls}
    assert "helper" in called
    assert "process" in called


def test_calls_method_call():
    parsed = _parse("""\
<?php
class User {
    public function save(): void {
        $this->validate();
    }
}
""")
    calls = _rels_from(parsed, "src.Example.User.save", RelType.CALLS)
    called = {r.to_name for r in calls}
    assert "validate" in called


def test_calls_static_call():
    parsed = _parse("""\
<?php
class Factory {
    public static function create(): void {
        Logger::info("created");
    }
}
""")
    calls = _rels_from(parsed, "src.Example.Factory.create", RelType.CALLS)
    called = {r.to_name for r in calls}
    assert "info" in called


# ---------------------------------------------------------------------------
# 25. Content hash determinism
# ---------------------------------------------------------------------------


def test_content_hash_populated():
    parsed = _parse("""\
<?php
class User {
    public function getName(): string { return ""; }
}
""")
    for entity in parsed.entities:
        assert entity.content_hash, f"Entity {entity.name!r} has empty content_hash"


def test_content_hash_deterministic():
    source = """\
<?php
function greet(string $name): string {
    return "Hello " . $name;
}
"""
    parsed1 = _parse(source)
    parsed2 = _parse(source)
    for e1, e2 in zip(parsed1.entities, parsed2.entities, strict=True):
        assert e1.content_hash == e2.content_hash


def test_content_hash_ignores_line_shift():
    source_v1 = """\
<?php
function greet(): void {}
"""
    source_v2 = """\
<?php


function greet(): void {}
"""
    parsed1 = _parse(source_v1)
    parsed2 = _parse(source_v2)
    func1 = _entity_by_name(parsed1, "greet")
    func2 = _entity_by_name(parsed2, "greet")
    assert func1.content_hash == func2.content_hash
    assert func1.line_start != func2.line_start


# ---------------------------------------------------------------------------
# 26. Edge cases (empty file, syntax errors)
# ---------------------------------------------------------------------------


def test_empty_file():
    parsed = _parse("<?php\n")
    assert parsed is not None
    assert parsed.language == "php"
    # Should have at least the module entity
    assert len(parsed.entities) >= 1


def test_truly_empty_file():
    parsed = _parse("")
    assert parsed is not None
    assert parsed.language == "php"
    assert len(parsed.entities) >= 1


def test_syntax_error_tolerant():
    """Tree-sitter is error-tolerant -- malformed files don't crash."""
    parsed = _parse("<?php\nfunction broken(\nclass nope\n")
    assert parsed is not None


def test_binary_content():
    """Binary content shouldn't crash the parser."""
    parsed = parse_file("data.php", b"\x00\x01\x02\xff\xfe", PROJECT)
    assert parsed is not None


# ---------------------------------------------------------------------------
# Comprehensive PHP file
# ---------------------------------------------------------------------------


def test_comprehensive_php_file():
    """End-to-end test with a realistic PHP file using many features."""
    parsed = _parse("""\
<?php
namespace App\\Models;

use App\\Interfaces\\Cacheable;
use App\\Traits\\HasTimestamps;

/**
 * User model for the application.
 */
abstract class User extends BaseModel implements Cacheable {
    use HasTimestamps;

    public const STATUS_ACTIVE = 1;
    private string $name;

    #[Route("/api/users")]
    public function __construct(string $name) {
        $this->name = $name;
    }

    abstract public function getRole(): string;

    public static function create(string $name): self {
        return new self($name);
    }
}

interface Loggable {
    public function log(): void;
}

trait Auditable {
    public function audit(): void {}
}

enum Status: string {
    case Active = "active";
    case Inactive = "inactive";
}

function helper(): void {
    echo "hello";
}

require_once "vendor/autoload.php";
""")
    # Check class
    user = _entity_by_name(parsed, "User")
    assert user.kind == TypeDefKind.CLASS
    assert user.docstring is not None
    assert "abstract" in user.tags

    # Check interface
    loggable = _entity_by_name(parsed, "Loggable")
    assert loggable.kind == TypeDefKind.INTERFACE

    # Check trait
    auditable = _entity_by_name(parsed, "Auditable")
    assert auditable.kind == TypeDefKind.TRAIT

    # Check enum
    status = _entity_by_name(parsed, "Status")
    assert status.kind == TypeDefKind.ENUM

    # Check enum members
    active = _entity_by_name(parsed, "Active")
    assert active.kind == ValueKind.ENUM_MEMBER

    # Check function
    helper = _entity_by_name(parsed, "helper")
    assert helper.kind == CallableKind.FUNCTION

    # Check constructor
    ctor = _entity_by_name(parsed, "__construct")
    assert ctor.kind == CallableKind.CONSTRUCTOR

    # Check constant
    const = _entity_by_name(parsed, "STATUS_ACTIVE")
    assert const.kind == ValueKind.CONSTANT

    # Check property
    name = _entity_by_name(parsed, "name")
    assert name.kind == ValueKind.FIELD

    # Check INHERITS (extends + use trait)
    inherits = [r for r in parsed.relationships if r.rel_type == RelType.INHERITS]
    inherits_names = {r.to_name for r in inherits}
    assert "BaseModel" in inherits_names
    assert "HasTimestamps" in inherits_names

    # Check IMPLEMENTS
    impl = [r for r in parsed.relationships if r.rel_type == RelType.IMPLEMENTS]
    impl_names = {r.to_name for r in impl}
    assert "Cacheable" in impl_names

    # Check IMPORTS (namespace use + require)
    imports = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    import_names = {r.to_name for r in imports}
    assert any("Cacheable" in n for n in import_names)
    assert "vendor/autoload.php" in import_names
