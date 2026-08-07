"""Tests for language plugin discovery (code_atlas.parsing.languages).

Two concerns live here, and only these two:

* ``discover_plugins`` itself — the import loop, its failure isolation, and the
  module list it walks.
* The *registry contract* every language module must satisfy once discovery has
  run: it registers, it round-trips a real file of its format, it survives an
  empty one, and the uid it mints for a file is unique to that file.

Per-language extraction detail (which blocks, which edges, which kinds) belongs
in ``test_<language>.py``, not here. What this file asserts is the thin
cross-language contract that a *new* module is most likely to get wrong.
"""

from __future__ import annotations

import pkgutil
from typing import NamedTuple

import pytest

from code_atlas.parsing import ast as parsing_ast
from code_atlas.parsing import languages as languages_init
from code_atlas.parsing.ast import get_language_for_file, parse_file
from code_atlas.schema import NodeLabel

PROJECT = "test_project"

_FILE_ROOT_LABELS = frozenset({NodeLabel.MODULE, NodeLabel.DOC_FILE})
"""Labels a language may use for the entity that stands for the file itself.

Every module emits exactly one of these; ``markdown`` is the only one that picks
``DocFile`` over ``Module``. It is the entity whose ``qualified_name`` is the
file's uid, which is what the collision tests below compare.
"""


# ---------------------------------------------------------------------------
# discover_plugins
# ---------------------------------------------------------------------------


def test_discover_plugins_continues_after_one_module_fails(monkeypatch):
    """A failing built-in language module must not prevent later modules in the
    list from being imported, and must not permanently latch discovery as done
    before all modules have been attempted."""
    monkeypatch.setattr(languages_init, "_discovered", False)
    monkeypatch.setattr(
        languages_init,
        "_BUILTIN_LANGUAGE_MODULES",
        ("fake.bad.module", "fake.good.module_a", "fake.good.module_b"),
    )

    calls: list[str] = []

    def fake_import_module(name: str):
        calls.append(name)
        if name == "fake.bad.module":
            msg = "simulated import failure"
            raise ImportError(msg)

    monkeypatch.setattr(languages_init.importlib, "import_module", fake_import_module)

    languages_init.discover_plugins()

    # All three entries were attempted, including the two after the failure.
    assert calls == ["fake.bad.module", "fake.good.module_a", "fake.good.module_b"]
    # Discovery is marked complete only after every module was attempted.
    assert languages_init._discovered is True


def test_discover_plugins_is_noop_after_first_call(monkeypatch):
    """Safe to call multiple times — a second call must not re-import anything."""
    monkeypatch.setattr(languages_init, "_discovered", False)
    monkeypatch.setattr(languages_init, "_BUILTIN_LANGUAGE_MODULES", ("fake.mod_a", "fake.mod_b"))

    calls: list[str] = []

    def record_import(name: str) -> None:
        calls.append(name)

    monkeypatch.setattr(languages_init.importlib, "import_module", record_import)

    languages_init.discover_plugins()
    assert calls == ["fake.mod_a", "fake.mod_b"]

    calls.clear()
    languages_init.discover_plugins()
    assert calls == []


def test_discover_plugins_all_modules_failing_still_completes(monkeypatch):
    """Even if every built-in module fails, discover_plugins must not raise and
    must still mark discovery as complete."""
    monkeypatch.setattr(languages_init, "_discovered", False)
    monkeypatch.setattr(languages_init, "_BUILTIN_LANGUAGE_MODULES", ("fake.bad_a", "fake.bad_b"))

    def always_fail(name: str):
        msg = f"simulated failure for {name}"
        raise ImportError(msg)

    monkeypatch.setattr(languages_init.importlib, "import_module", always_fail)

    languages_init.discover_plugins()  # must not raise

    assert languages_init._discovered is True


def test_builtin_module_list_is_sorted_and_unique():
    """Alphabetical order is the convention; duplicates would double-import."""
    modules = languages_init._BUILTIN_LANGUAGE_MODULES
    assert list(modules) == sorted(modules)
    assert len(set(modules)) == len(modules)


def test_builtin_module_list_matches_the_package_contents():
    """Every module file in the package must be listed, and nothing else.

    A module absent from the tuple is never imported, so it never calls
    ``register_language`` and its language simply does not exist — with no error
    anywhere, because nothing ever asked for it. Comparing against the directory
    rather than a hand-maintained name list is what makes adding a file enough
    to trip this.
    """
    on_disk = sorted(
        f"{languages_init.__name__}.{info.name}"
        for info in pkgutil.iter_modules(languages_init.__path__)
        if not info.ispkg
    )
    assert on_disk == sorted(languages_init._BUILTIN_LANGUAGE_MODULES)


# ---------------------------------------------------------------------------
# The registry contract
#
# One representative real file per registered language. Grammars ship as
# optional extras, so every case below skips on its own ``importorskip`` rather
# than the module skipping as a whole — a single missing wheel must not hide the
# other formats.
#
# Deliberately no expected `kind` column: kinds are dialect-dependent (one YAML
# file is a `k8s_manifest`, the next a `github_workflow`, the next a plain
# `config_file`) and pinning them here duplicates the per-language test files
# while adding a second place to update. What is pinned is the contract those
# kinds all have to satisfy.
# ---------------------------------------------------------------------------


class _Sample(NamedTuple):
    language: str
    """Registry key — what ``get_language_for_file`` must resolve *path* to."""
    grammar: str
    """Importable grammar module; absent means the extra is not installed."""
    path: str
    source: str


_SAMPLES = [
    _Sample(
        # Apex has no grammar of its own — it reuses tree-sitter-java behind a
        # length-preserving shim (see parsing/languages/apex.py).
        "apex",
        "tree_sitter_java",
        "force-app/main/default/classes/AccountService.cls",
        "public with sharing class AccountService {\n"
        "    public static List<Account> getAccounts() {\n"
        "        return [SELECT Id FROM Account];\n"
        "    }\n"
        "}\n",
    ),
    _Sample(
        "c",
        "tree_sitter_c",
        "src/util.c",
        "#include <stdio.h>\n\nint add(int a, int b) {\n    return a + b;\n}\n",
    ),
    _Sample(
        "containerfile",
        "tree_sitter_containerfile",
        "Dockerfile",
        "FROM alpine:3 AS base\nRUN apk add --no-cache curl\nCOPY . /app\n",
    ),
    _Sample(
        "cpp",
        "tree_sitter_cpp",
        "src/widget.cpp",
        "namespace ui {\n\nclass Widget {\npublic:\n    int width() const { return width_; }\n\n"
        "private:\n    int width_ = 0;\n};\n\n}  // namespace ui\n",
    ),
    _Sample(
        "csharp",
        "tree_sitter_c_sharp",
        "src/Service.cs",
        "namespace Acme;\n\npublic class Service\n{\n    public int Add(int a, int b) => a + b;\n}\n",
    ),
    _Sample(
        "go",
        "tree_sitter_go",
        "cmd/server/main.go",
        'package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println("listening")\n}\n',
    ),
    _Sample(
        "hcl",
        "tree_sitter_hcl",
        "infra/main.tf",
        'resource "aws_s3_bucket" "logs" {\n  bucket = "acme-logs"\n}\n',
    ),
    _Sample(
        "java",
        "tree_sitter_java",
        "src/main/java/com/acme/App.java",
        "package com.acme;\n\npublic class App {\n    public int add(int a, int b) {\n"
        "        return a + b;\n    }\n}\n",
    ),
    _Sample(
        "javascript",
        "tree_sitter_javascript",
        "web/greet.js",
        "export function greet(name) {\n  return `hi ${name}`;\n}\n",
    ),
    _Sample(
        "json",
        "tree_sitter_json",
        "package.json",
        '{\n  "name": "acme",\n  "version": "1.0.0",\n  "scripts": {\n    "build": "tsc"\n  }\n}\n',
    ),
    # Plain documentation, not a vault note: `kind: note` frontmatter routes the
    # same module down the Note branch, which `test_markdown.py` owns.
    _Sample(
        "markdown",
        "tree_sitter_markdown",
        "docs/architecture.md",
        "# Architecture\n\nThe indexer writes to Memgraph.\n\n## Pipeline\n\nWatcher, then AST, then embed.\n",
    ),
    _Sample(
        "php",
        "tree_sitter_php",
        "src/Controller.php",
        "<?php\n\nnamespace Acme;\n\nclass Controller\n{\n    public function index(): string\n"
        "    {\n        return 'ok';\n    }\n}\n",
    ),
    _Sample(
        "python",
        "tree_sitter_python",
        "src/acme/service.py",
        'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b\n',
    ),
    _Sample("ruby", "tree_sitter_ruby", "lib/acme.rb", "module Acme\n  def self.add(a, b)\n    a + b\n  end\nend\n"),
    _Sample("rust", "tree_sitter_rust", "src/lib.rs", "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n"),
    _Sample(
        "shell",
        "tree_sitter_bash",
        "scripts/deploy.sh",
        '#!/usr/bin/env bash\nset -euo pipefail\n\ndeploy() {\n  echo "deploying"\n}\n\ndeploy\n',
    ),
    _Sample(
        "sql",
        "tree_sitter_sql",
        "db/schema.sql",
        "CREATE TABLE users (\n  id INT PRIMARY KEY,\n  email TEXT NOT NULL\n);\n",
    ),
    _Sample("toml", "tree_sitter_toml", "pyproject.toml", '[project]\nname = "acme"\nversion = "1.0.0"\n'),
    _Sample("tsx", "tree_sitter_typescript", "web/App.tsx", "export function App() {\n  return <div>hi</div>;\n}\n"),
    _Sample(
        "typescript",
        "tree_sitter_typescript",
        "web/api.ts",
        "export interface User {\n  id: number;\n}\n\nexport function get(): User {\n  return { id: 1 };\n}\n",
    ),
    _Sample(
        "xml",
        "tree_sitter_xml",
        "pom.xml",
        "<project>\n  <modelVersion>4.0.0</modelVersion>\n  <artifactId>acme</artifactId>\n</project>\n",
    ),
    _Sample(
        "yaml",
        "tree_sitter_yaml",
        "deploy/service.yaml",
        "apiVersion: v1\nkind: Service\nmetadata:\n  name: web\n",
    ),
]

_SAMPLE_IDS = [sample.language for sample in _SAMPLES]
_SAMPLE_BY_LANGUAGE = {sample.language: sample for sample in _SAMPLES}


def test_every_registered_language_has_a_sample():
    """A language with no sample above is exercised by nothing in this file.

    This is what keeps the table below honest as languages are added: register a
    grammar without adding a row and the round-trip, empty-file and uid
    contracts silently stop covering it.
    """
    languages_init.discover_plugins()
    unsampled = sorted(set(parsing_ast._LANGUAGES) - set(_SAMPLE_BY_LANGUAGE))
    assert not unsampled, f"registered languages with no sample in _SAMPLES: {unsampled}"


@pytest.mark.parametrize("sample", _SAMPLES, ids=_SAMPLE_IDS)
def test_installed_grammar_registers_its_language(sample: _Sample):
    """Guards the failure mode the modules' ``except ImportError`` cannot catch.

    A grammar that imports but whose ``Language(...)`` construction fails — ABI
    mismatch, or the tree-sitter-xml ``language()``-vs-``language_xml()`` trap —
    raises something *other* than ImportError. That escapes the module's own
    guard and is then swallowed by ``discover_plugins``' broad except, so the
    language vanishes leaving nothing but a warning in the log. Skipping on the
    wheel and asserting on the registration is what tells the two apart.
    """
    pytest.importorskip(sample.grammar, reason=f"{sample.grammar} not installed")

    config = get_language_for_file(sample.path)
    assert config is not None, (
        f"{sample.grammar} is installed but {sample.path} resolved to no language — registration failed silently"
    )
    assert config.name == sample.language


@pytest.mark.parametrize("sample", _SAMPLES, ids=_SAMPLE_IDS)
def test_registered_language_round_trips_a_real_file(sample: _Sample):
    """A representative file of the format parses into a usable ParsedFile.

    ``None`` here means "no language claimed this file", which for an installed
    grammar is a registration bug; empty entities means the module registered an
    extension it then extracts nothing from (the state ``.toml`` was in before
    the generic fallback existed) — indistinguishable from unsupported at query
    time.
    """
    pytest.importorskip(sample.grammar, reason=f"{sample.grammar} not installed")

    parsed = parse_file(sample.path, sample.source.encode("utf-8"), PROJECT)

    assert parsed is not None, f"{sample.path} produced no ParsedFile"
    assert parsed.file_path == sample.path
    # `tsx` and `typescript` are separate registrations of one handler, so the
    # reported language need not be the registry key — but it must be a real one.
    assert parsed.language in parsing_ast._LANGUAGES
    assert parsed.entities, f"{sample.path} parsed to zero entities"

    qualified_names = [entity.qualified_name for entity in parsed.entities]
    assert len(set(qualified_names)) == len(qualified_names), f"duplicate uids within one file: {qualified_names}"

    for entity in parsed.entities:
        assert entity.file_path == sample.path
        assert entity.qualified_name.startswith(f"{PROJECT}:"), entity.qualified_name
        assert entity.kind, f"{entity.qualified_name} has no kind"
        assert entity.content_hash, f"{entity.qualified_name} has no content_hash"
        assert entity.line_start >= 1
        assert entity.line_end >= entity.line_start

    roots = [entity for entity in parsed.entities if entity.label in _FILE_ROOT_LABELS]
    assert len(roots) == 1, f"expected exactly one file-level entity, got {[r.qualified_name for r in roots]}"
    assert roots[0].line_start == 1


@pytest.mark.parametrize("sample", _SAMPLES, ids=_SAMPLE_IDS)
def test_empty_file_yields_at_most_a_file_entity(sample: _Sample):
    """A whitespace-only file must parse, and must not mint child entities.

    Every module has to survive this — an empty ``__init__.py``, a stubbed-out
    Dockerfile and a placeholder values.yaml are all real. The floor asserted
    here is cross-language: no crash, no ``None`` (which would drop the file out
    of the hash gate and force a re-parse every pass), no relationships, and
    nothing beyond the single file-level entity.
    """
    pytest.importorskip(sample.grammar, reason=f"{sample.grammar} not installed")

    parsed = parse_file(sample.path, b"   \n\n", PROJECT)

    assert parsed is not None
    assert parsed.relationships == []
    assert len(parsed.entities) <= 1
    assert all(entity.label in _FILE_ROOT_LABELS for entity in parsed.entities)


# The config/infra family goes further than the floor above: no entity at all.
# A code module keeps its stub because an empty `__init__.py` is still a package
# member other files import; an empty config file names nothing, so a Module
# node for it is an unsearchable stub that still costs an embedding.
_NO_STUB_LANGUAGES = ["containerfile", "hcl", "json", "shell", "sql", "toml", "xml", "yaml"]


@pytest.mark.parametrize("language", _NO_STUB_LANGUAGES)
def test_empty_config_file_produces_no_entities_at_all(language: str):
    sample = _SAMPLE_BY_LANGUAGE[language]
    pytest.importorskip(sample.grammar, reason=f"{sample.grammar} not installed")

    parsed = parse_file(sample.path, b"   \n\n", PROJECT)

    assert parsed is not None
    assert parsed.entities == []
    assert parsed.relationships == []


def test_sql_tolerates_dialects_the_grammar_cannot_parse():
    """T-SQL bracket quoting produces ERROR nodes. The file must still parse.

    Gating on ``root.has_error`` would silently drop every T-SQL and PL/SQL
    schema in a repo, so this is a hard requirement on the SQL parse function.
    """
    pytest.importorskip("tree_sitter_sql", reason="tree-sitter-sql not installed")

    tsql = "SELECT [Id], [Name] FROM [dbo].[Users] WHERE [Id] = 1;\n"
    parsed = parse_file("db/tsql.sql", tsql.encode("utf-8"), PROJECT)

    assert parsed is not None
    roots = [entity for entity in parsed.entities if entity.label in _FILE_ROOT_LABELS]
    assert len(roots) == 1, "a file the grammar only partly understands still gets its file entity"
    assert roots[0].qualified_name == f"{PROJECT}:db.tsql_sql"


# ---------------------------------------------------------------------------
# uid collisions
#
# `qualified_name` IS the graph uid. Two files that mint the same one do not
# error — the later upsert silently overwrites the earlier node — so these are
# the assertions with no runtime signal behind them.
# ---------------------------------------------------------------------------

_STEM_PAIRS = [
    (
        ["tree_sitter_json", "tree_sitter_yaml"],
        ("api/openapi.json", '{"openapi": "3.0.0", "info": {"title": "acme"}}\n'),
        ("api/openapi.yaml", 'openapi: "3.0.0"\ninfo:\n  title: acme\n'),
    ),
    (
        ["tree_sitter_bash", "tree_sitter_python"],
        ("build.sh", "echo building\n"),
        ("build.py", "print('building')\n"),
    ),
    (
        ["tree_sitter_hcl"],
        ("infra/main.tf", 'resource "aws_s3_bucket" "logs" {\n  bucket = "acme-logs"\n}\n'),
        ("infra/main.tfvars", 'region = "eu-west-1"\n'),
    ),
    (
        ["tree_sitter_toml", "tree_sitter_xml"],
        ("conf/app.toml", '[server]\nhost = "localhost"\n'),
        ("conf/app.xml", "<server>\n  <host>localhost</host>\n</server>\n"),
    ),
]


@pytest.mark.parametrize(
    ("grammars", "first", "second"),
    _STEM_PAIRS,
    ids=["openapi.json-vs-yaml", "build.sh-vs-build.py", "main.tf-vs-main.tfvars", "app.toml-vs-app.xml"],
)
def test_files_sharing_a_stem_get_distinct_uids(grammars, first, second):
    """One stem under two formats is routine — openapi.json beside openapi.yaml,
    build.sh beside build.py. Stripping the extension the way the code-language
    modules do would make both files claim one node."""
    for grammar in grammars:
        pytest.importorskip(grammar, reason=f"{grammar} not installed")

    parsed_first = parse_file(first[0], first[1].encode("utf-8"), PROJECT)
    parsed_second = parse_file(second[0], second[1].encode("utf-8"), PROJECT)
    assert parsed_first is not None
    assert parsed_second is not None

    uids_first = {entity.qualified_name for entity in parsed_first.entities}
    uids_second = {entity.qualified_name for entity in parsed_second.entities}
    assert uids_first, first[0]
    assert uids_second, second[0]
    shared = uids_first & uids_second
    assert not shared, f"uid collision between {first[0]} and {second[0]}: {shared}"


_DOT_DIR_CASES = [
    ("hcl", "main.tf", 'resource "aws_s3_bucket" "logs" {\n  bucket = "acme-logs"\n}\n'),
    ("json", "conf.json", '{"server": {"host": "localhost"}}\n'),
    ("shell", "run.sh", "echo running\n"),
    ("sql", "schema.sql", "CREATE TABLE t (id INT);\n"),
    ("toml", "conf.toml", '[server]\nhost = "localhost"\n'),
    ("xml", "conf.xml", "<server>\n  <host>localhost</host>\n</server>\n"),
    ("yaml", "conf.yaml", "server:\n  host: localhost\n"),
    pytest.param(
        "containerfile",
        "Dockerfile",
        "FROM alpine:3\n",
        marks=pytest.mark.xfail(
            strict=True,
            reason=(
                "containerfile._module_qualified_name folds dots in the basename only, so "
                "a.b/Dockerfile and a/b/Dockerfile both render {project}:a.b.Dockerfile. "
                "Drop this marker once it folds every path segment, as hcl/shell/sql/config now do."
            ),
        ),
    ),
]


@pytest.mark.parametrize(("language", "basename", "source"), _DOT_DIR_CASES)
def test_dot_in_a_directory_name_does_not_collide_with_real_nesting(language: str, basename: str, source: str):
    """``a.b/X`` and ``a/b/X`` are different files and need different uids.

    ``.`` is the qualified name's own separator, so a dot left inside a
    *directory* name fakes a nesting level. Directories with dots in them are
    everywhere — ``.github``, ``app.v2``, ``com.acme`` — which makes this a
    collision that actually happens rather than a theoretical one.
    """
    sample = _SAMPLE_BY_LANGUAGE[language]
    pytest.importorskip(sample.grammar, reason=f"{sample.grammar} not installed")

    encoded = source.encode("utf-8")
    dotted = parse_file(f"a.b/{basename}", encoded, PROJECT)
    nested = parse_file(f"a/b/{basename}", encoded, PROJECT)
    assert dotted is not None
    assert nested is not None

    def root_uid(parsed) -> str:
        roots = [entity for entity in parsed.entities if entity.label in _FILE_ROOT_LABELS]
        assert len(roots) == 1, f"expected one file-level entity, got {[r.qualified_name for r in roots]}"
        return roots[0].qualified_name

    assert root_uid(dotted) != root_uid(nested), f"uid collision: {root_uid(dotted)}"


class TestMissingGrammarReporting:
    """A grammar behind an uninstalled extra must be reported, never silently skipped.

    A default install ships only the Python and Markdown grammars. Every language module
    swallows its own ImportError so one absent wheel cannot take the others down,
    `_DEFAULT_INCLUDE` still lists the extensions, and `parse_file` returns None with no
    log at any level — so a TypeScript repo produced `Done - 4823 files, 0 entities` and
    exit 0 (ATL-110).
    """

    def test_an_unregistered_extension_maps_to_the_extra_that_provides_it(self, monkeypatch):
        from code_atlas.parsing import languages as lang_mod

        # Simulate a default install: nothing is registered.
        monkeypatch.setattr("code_atlas.parsing.ast.get_language_for_file", lambda _p: None)
        assert lang_mod.missing_grammar_extras({".ts"}) == {".ts": "typescript"}
        assert lang_mod.missing_grammar_extras({".go", ".rs"}) == {".go": "go", ".rs": "rust"}

    def test_a_registered_extension_is_not_reported(self):
        from code_atlas.parsing.languages import discover_plugins, missing_grammar_extras

        discover_plugins()
        # Python ships in the base dependencies, so it is never an install problem.
        assert missing_grammar_extras({".py"}) == {}

    def test_an_unsupported_extension_is_omitted_rather_than_guessed_at(self, monkeypatch):
        """Telling a user to install something that would not help is worse than silence."""
        from code_atlas.parsing import languages as lang_mod

        monkeypatch.setattr("code_atlas.parsing.ast.get_language_for_file", lambda _p: None)
        assert lang_mod.missing_grammar_extras({".zzz", ".kt"}) == {}

    def test_the_install_hint_collapses_once_it_is_most_of_them(self):
        from code_atlas.parsing.languages import install_hint

        assert install_hint(["typescript"]) == "typescript"
        assert install_hint(["go", "typescript"]) == "go,typescript"
        # Four or more is where naming each one stops helping.
        assert install_hint(["typescript", "go", "rust", "java"]) == "all-languages"
