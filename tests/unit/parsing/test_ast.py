"""Unit tests for parsing.ast — content hash formula (v4) contract, dispatch, parse_func contract."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from code_atlas.parsing.ast import ParsedEntity, ParsedFile, _compute_content_hash
from code_atlas.schema import NodeLabel


def _entity(**overrides: Any) -> ParsedEntity:
    defaults: dict[str, Any] = {
        "name": "work",
        "qualified_name": "proj:mod.work",
        "label": NodeLabel.CALLABLE,
        "kind": "function",
        "line_start": 1,
        "line_end": 3,
        "file_path": "mod.py",
        "docstring": "Doc.",
        "signature": "def work(x)",
        "tags": ["async"],
        "source": "def work(x):\n    return x\n",
    }
    defaults.update(overrides)
    return ParsedEntity(**defaults)


def test_content_hash_formula_v4():
    """content_hash = sha256 over name/kind/visibility/signature/docstring/sorted tags/source/extra_properties."""
    entity = _entity()
    parts = [
        entity.name,
        entity.kind,
        entity.visibility,
        entity.signature or "",
        entity.docstring or "",
        ",".join(sorted(entity.tags)),
        entity.source or "",
        "",  # extra_properties empty -> not serialized (see docstring)
    ]
    expected = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]
    assert _compute_content_hash(entity) == expected


def test_content_hash_source_none_equals_empty():
    """Entities without source (Module, TypeDef, DocSection) hash "" for the source element."""
    assert _compute_content_hash(_entity(source=None)) == _compute_content_hash(_entity(source=""))


def test_content_hash_ignores_positional_fields():
    """line_start/line_end/file_path do not affect the hash."""
    a = _entity()
    b = _entity(line_start=42, line_end=44, file_path="other/mod.py")
    assert _compute_content_hash(a) == _compute_content_hash(b)


def test_content_hash_extra_properties_changes_hash():
    """A Note's frontmatter (extra_properties) is folded into the hash when non-empty."""
    a = _entity(extra_properties={})
    b = _entity(extra_properties={"tags": ["x"]})
    assert _compute_content_hash(a) != _compute_content_hash(b)


def test_content_hash_extra_properties_order_independent():
    """extra_properties is JSON-serialized with sort_keys — dict insertion order doesn't affect the hash."""
    a = _entity(extra_properties={"a": 1, "b": 2})
    b = _entity(extra_properties={"b": 2, "a": 1})
    assert _compute_content_hash(a) == _compute_content_hash(b)


def test_content_hash_extra_properties_matches_json_dumps():
    """The extra_properties hash element is exactly json.dumps(..., sort_keys=True, default=str)."""
    entity = _entity(extra_properties={"id": "foo", "kind": "note"})
    parts = [
        entity.name,
        entity.kind,
        entity.visibility,
        entity.signature or "",
        entity.docstring or "",
        ",".join(sorted(entity.tags)),
        entity.source or "",
        json.dumps(entity.extra_properties, sort_keys=True, default=str),
    ]
    expected = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]
    assert _compute_content_hash(entity) == expected


# ---------------------------------------------------------------------------
# Rationale / citations — hash contract
#
# The invariant that matters operationally: an entity with no intent-bearing
# comment must hash EXACTLY as it did before the fields existed, otherwise
# adding the feature reindexes every project.
# ---------------------------------------------------------------------------


def test_content_hash_without_rationale_matches_eight_part_formula():
    """No rationale/citations -> the parts list stays at eight elements, byte-identical."""
    entity = _entity()
    assert entity.rationale is None
    assert entity.citations == []
    parts = [
        entity.name,
        entity.kind,
        entity.visibility,
        entity.signature or "",
        entity.docstring or "",
        ",".join(sorted(entity.tags)),
        entity.source or "",
        "",
    ]
    expected = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]
    assert _compute_content_hash(entity) == expected


def test_content_hash_golden_digest_is_frozen():
    """A literal pin on the formula's output, not just its shape.

    Every parser addition that touches what entities carry (rationale, citations,
    and now the env-var/referenced-file extraction) has to leave this digest
    alone — a change here means every indexed project re-diffs and re-embeds on
    upgrade. Recomputing the parts list in the tests above cannot catch a change
    to the parts list itself; this can.
    """
    assert _compute_content_hash(_entity()) == "ae12f7ed54977d9b"


def test_content_hash_rationale_changes_hash():
    assert _compute_content_hash(_entity()) != _compute_content_hash(_entity(rationale="WHY: because"))


def test_content_hash_citations_change_hash():
    assert _compute_content_hash(_entity()) != _compute_content_hash(_entity(citations=["ADR-0014"]))


def test_content_hash_citations_order_independent():
    a = _entity(citations=["ADR-0014", "RFC-7231"])
    b = _entity(citations=["RFC-7231", "ADR-0014"])
    assert _compute_content_hash(a) == _compute_content_hash(b)


def test_content_hash_rationale_and_citations_are_distinguishable():
    """The same payload in either field must not collide (each element is key-prefixed)."""
    a = _entity(rationale="ADR-0014")
    b = _entity(citations=["ADR-0014"])
    assert _compute_content_hash(a) != _compute_content_hash(b)


def test_rationale_settings_defaults_match_parser_defaults():
    """settings.py and ast.py each carry the marker defaults — pin them together."""
    from code_atlas.parsing.ast import (
        DEFAULT_CITATION_SCHEMES,
        DEFAULT_RATIONALE_MARKERS,
        DEFAULT_TASK_MARKERS,
    )
    from code_atlas.settings import RationaleSettings

    settings = RationaleSettings()
    assert settings.markers == list(DEFAULT_RATIONALE_MARKERS)
    assert settings.task_markers == list(DEFAULT_TASK_MARKERS)
    assert settings.citation_schemes == list(DEFAULT_CITATION_SCHEMES)
    assert settings.enabled is True
    assert settings.tasks is False


# ---------------------------------------------------------------------------
# looks_like_resource_path — the shared REFERENCES_FILE gate
#
# Biased hard towards rejection: a false positive mints a ResourceFile node for
# a path that does not exist, which then shows up in search and dependency
# views as if the repo really read it. A miss just costs an edge.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "literal",
    [
        "data/fixtures.json",
        "config/y.yaml",
        ".env",
        "certs/server.pem",
        "../shared/config.yaml",
        "./data/x.json",
        "schema.sql",
        "a/b/c/d.txt",
        # Extensionless, but the separator is evidence enough — this is the
        # `.ssh/id_rsa` shape, which must be recordable as a path.
        ".ssh/id_rsa",
        "etc/hosts",
    ],
)
def test_resource_path_accepts_relative_file_paths(literal: str):
    from code_atlas.parsing.ast import looks_like_resource_path

    assert looks_like_resource_path(literal) is True


@pytest.mark.parametrize(
    ("literal", "why"),
    [
        ("", "empty"),
        ("rb", "an open() mode string"),
        ("w+", "an open() mode string"),
        (".", "the cwd"),
        ("..", "the parent dir"),
        ("data", "a bare directory name"),
        ("data/", "a trailing separator means a directory"),
        ("logs/..", "resolves to a directory"),
        ("/etc/passwd", "absolute"),
        ("C:/Users/x.json", "a Windows drive letter (the ':' rule)"),
        ("s3://bucket/key.json", "a URL scheme"),
        ("https://example.com/a.json", "a URL"),
        ("~/.config/app.yaml", "home-relative, not project-relative"),
        ("data/{name}.json", "a format template"),
        ("data/%s.json", "a printf template"),
        ("data/*.json", "a glob"),
        ("data/?.json", "a glob"),
        ("my file.json", "whitespace"),
        ("-v.txt", "looks like a CLI flag"),
        ("x" * 201 + ".json", "longer than MAX_RESOURCE_PATH_CHARS"),
    ],
)
def test_resource_path_rejects_non_paths(literal: str, why: str):
    from code_atlas.parsing.ast import looks_like_resource_path

    assert looks_like_resource_path(literal) is False, f"wrongly accepted {literal!r} ({why})"


def test_resource_path_never_touches_the_filesystem(monkeypatch):
    """The gate is a pure string predicate. It must reach a verdict on
    ``certs/server.pem`` without stat-ing, opening or resolving anything —
    that is what makes recording a path to a secret safe.
    """
    import builtins
    import pathlib

    def _forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("looks_like_resource_path touched the filesystem")

    from code_atlas.parsing.ast import looks_like_resource_path

    monkeypatch.setattr(builtins, "open", _forbidden)
    for attr in ("open", "read_text", "read_bytes", "exists", "stat", "resolve", "is_file"):
        monkeypatch.setattr(pathlib.Path, attr, _forbidden)

    assert looks_like_resource_path("certs/server.pem") is True
    assert looks_like_resource_path(".env") is True
    assert looks_like_resource_path("nonexistent/nowhere.json") is True


# ---------------------------------------------------------------------------
# Filename dispatch
#
# get_language_for_file matched on suffix only, which made extensionless
# formats unreachable: PurePosixPath("Dockerfile").suffix == "". Registering ""
# as an extension is not an option — it would hijack LICENSE, Makefile and
# every other extensionless file — so basenames get their own registry.
# ---------------------------------------------------------------------------


def test_language_config_filenames_defaults_to_empty():
    """The field must default: LanguageConfig is frozen and all 9 pre-existing
    modules construct it without a `filenames` argument."""
    import tree_sitter_python as ts_python
    from tree_sitter import Language, Query

    from code_atlas.parsing.ast import LanguageConfig

    lang = Language(ts_python.language())
    config = LanguageConfig(
        name="probe",
        extensions=frozenset({".probe"}),
        language=lang,
        query=Query(lang, "(module) @root"),
        parse_func=lambda path, source, root, project: ParsedFile(path, "probe", [], []),
    )
    assert config.filenames == frozenset()


def test_filename_dispatch_routes_extensionless_file():
    """Dockerfile has no suffix at all — it must still resolve, in any casing."""
    pytest.importorskip("tree_sitter_containerfile", reason="tree-sitter-containerfile not installed")
    from code_atlas.parsing.ast import get_language_for_file

    for path in ("Dockerfile", "dockerfile", "build/Dockerfile", "deploy/prod/Containerfile"):
        config = get_language_for_file(path)
        assert config is not None, f"{path} did not resolve to a language"
        assert config.name == "containerfile"


def test_filename_dispatch_is_whole_basename_not_prefix():
    """`dockerfile.txt` must NOT route to the container language.

    Its basename is "dockerfile.txt" (absent from the filename registry) and its
    suffix is ".txt" (absent from the extension registry), so it is unsupported.
    """
    from code_atlas.parsing.ast import get_language_for_file

    assert get_language_for_file("dockerfile.txt") is None
    assert get_language_for_file("Dockerfile.bak") is None
    assert get_language_for_file("my-dockerfile") is None


def test_filename_dispatch_does_not_capture_other_extensionless_files():
    """The regression the empty-string extension would have caused."""
    from code_atlas.parsing.ast import get_language_for_file

    for path in ("LICENSE", "Makefile", "CODEOWNERS", ".gitignore", "src/Jenkinsfile"):
        assert get_language_for_file(path) is None, f"{path} must not resolve"


def test_filename_map_takes_precedence_over_extension_map(monkeypatch):
    """Basename is checked before suffix, so a filename registration wins."""
    from code_atlas.parsing import ast as ast_module
    from code_atlas.parsing.ast import get_language_for_file

    monkeypatch.setitem(ast_module._EXTENSION_MAP, ".conf", "ext-lang")
    monkeypatch.setitem(ast_module._FILENAME_MAP, "special.conf", "name-lang")
    monkeypatch.setitem(ast_module._LANGUAGES, "ext-lang", "EXT-SENTINEL")
    monkeypatch.setitem(ast_module._LANGUAGES, "name-lang", "NAME-SENTINEL")

    assert get_language_for_file("special.conf") == "NAME-SENTINEL"
    assert get_language_for_file("other.conf") == "EXT-SENTINEL"


def test_register_language_populates_both_maps():
    """register_language must fill _FILENAME_MAP alongside _EXTENSION_MAP."""
    import tree_sitter_python as ts_python
    from tree_sitter import Language, Query

    from code_atlas.parsing import ast as ast_module
    from code_atlas.parsing.ast import LanguageConfig, register_language

    lang = Language(ts_python.language())
    config = LanguageConfig(
        name="_probe_lang",
        extensions=frozenset({".probeext"}),
        language=lang,
        query=Query(lang, "(module) @root"),
        parse_func=lambda path, source, root, project: ParsedFile(path, "_probe_lang", [], []),
        filenames=frozenset({"probefile"}),
    )
    try:
        register_language(config)
        assert ast_module._EXTENSION_MAP[".probeext"] == "_probe_lang"
        assert ast_module._FILENAME_MAP["probefile"] == "_probe_lang"
    finally:
        ast_module._EXTENSION_MAP.pop(".probeext", None)
        ast_module._FILENAME_MAP.pop("probefile", None)
        ast_module._LANGUAGES.pop("_probe_lang", None)


# ---------------------------------------------------------------------------
# parse_func declining a file
#
# A handler returning None means "I have no dialect for this file", NOT
# "unsupported language". The distinction is load-bearing: the AST consumer only
# records a file hash for paths that produced a ParsedFile, so a None leaking
# out of parse_file would keep the file outside the hash gate and re-parse it on
# every indexing pass forever.
# ---------------------------------------------------------------------------


def test_parse_func_returning_none_yields_empty_parsed_file(monkeypatch):
    import tree_sitter_python as ts_python
    from tree_sitter import Language, Query

    from code_atlas.parsing import ast as ast_module
    from code_atlas.parsing.ast import LanguageConfig, parse_file

    lang = Language(ts_python.language())
    config = LanguageConfig(
        name="_declining",
        extensions=frozenset({".declines"}),
        language=lang,
        query=Query(lang, "(module) @root"),
        parse_func=lambda path, source, root, project: None,
    )
    monkeypatch.setitem(ast_module._LANGUAGES, "_declining", config)
    monkeypatch.setitem(ast_module._EXTENSION_MAP, ".declines", "_declining")

    parsed = parse_file("some/file.declines", b"whatever = 1\n", "test_project")

    assert parsed is not None, "declining a file must not look like an unsupported language"
    assert parsed.entities == []
    assert parsed.relationships == []
    assert parsed.file_path == "some/file.declines"
    assert parsed.language == "_declining"


def test_parse_file_still_returns_none_for_unregistered_extension():
    """The None return stays reserved for 'no language registered'."""
    from code_atlas.parsing.ast import parse_file

    assert parse_file("data.csv", b"a,b\n1,2\n", "test_project") is None


# ---------------------------------------------------------------------------
# Pre-parse safety guard (_parse_hazard / _block_depth)
#
# Some grammars die NATIVELY on deeply-nested input — a scanner-buffer overflow
# inside Parser.parse() that kills the process (Windows 0xC0000005, POSIX
# SIGSEGV) with no Python exception to catch. Anything that could trigger that
# must be exercised in a SUBPROCESS: an in-process assertion would take the
# test runner down with it.
# ---------------------------------------------------------------------------

_CRASH_PROBE = """
import sys
from code_atlas.parsing.ast import parse_file
src = sys.argv[1].encode() * int(sys.argv[2])
parse_file(sys.argv[3], src + b"\n", "probe")
print("SURVIVED")
"""


def _parse_in_subprocess(unit: str, repeat: int, path: str) -> int:
    """Return the child's exit code. 0/1 are survivable; anything else is a native kill."""
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-c", _CRASH_PROBE, unit, str(repeat), path],
        capture_output=True,
        check=False,
    ).returncode


@pytest.mark.parametrize(
    ("unit", "path"),
    [
        (">", "deep.md"),  # blockquote, marker-only line
        ("> ", "deep.md"),  # blockquote with padding
        ("- ", "deep.yaml"),  # block sequence, marker-only line
        ("- ", "deep.md"),  # nested list, marker-only line
    ],
)
def test_marker_only_lines_cannot_kill_the_process(unit: str, path: str):
    """Regression: `_prefix_shape` folds marker bytes into its `columns` return,
    so a line of PURE markers satisfies `columns == len(line)` exactly like a
    blank line does. Skipping those scored them depth 0 and handed the very
    inputs the guard's own measurements list as native kills straight through
    to Parser.parse(). Reproduced as exit 3221225477 before the fix.
    """
    assert _parse_in_subprocess(unit, 400, path) in (0, 1), (
        f"{unit!r} x400 in {path} killed the interpreter — the pre-parse guard was bypassed"
    )


def test_deeply_indented_block_nesting_is_refused_not_crashed():
    from code_atlas.parsing.ast import MAX_BLOCK_DEPTH, _block_depth, _parse_hazard

    src = b"\n".join(b" " * i + b"k:" for i in range(MAX_BLOCK_DEPTH + 10)) + b"\n"
    assert _block_depth(src) >= MAX_BLOCK_DEPTH
    assert _parse_hazard(src, 0) is not None


def test_guard_leaves_ordinary_files_alone():
    """The guard must not be implemented by refusing everything."""
    from code_atlas.parsing.ast import _parse_hazard

    ordinary = [
        b"apiVersion: apps/v1\nkind: Deployment\nspec:\n  template:\n    spec:\n      containers:\n        - name: c\n",
        b"[project]\nname = 'x'\n[tool.ruff.lint.per-file-ignores]\n'a.py' = ['E1']\n",
        b'{"a": {"b": {"c": {"d": 1}}}}\n',
        b"# Title\n\n- one\n  - two\n    - three\n\n> quoted\n",
    ]
    for src in ordinary:
        assert _parse_hazard(src, 0) is None, f"guard wrongly refused: {src[:40]!r}"


def test_size_ceiling_refuses_oversized_input():
    from code_atlas.parsing.ast import _parse_hazard

    assert _parse_hazard(b"x" * 2048, 1024) is not None
    assert _parse_hazard(b"x" * 512, 1024) is None
    assert _parse_hazard(b"x" * 99_999, 0) is None, "0 must disable the ceiling"


def test_max_parse_bytes_is_actually_threaded_from_settings():
    """A guard limit that is configurable but never passed to the call site is
    no guard at all. `[rationale]` shipped exactly that bug earlier in this same
    release — settings block defined, never handed to parse_file.
    """
    import inspect

    from code_atlas.indexing.consumers import ASTConsumer

    src = inspect.getsource(ASTConsumer._parse_file)
    assert "max_parse_bytes=" in src, "IndexSettings.max_parse_bytes is defined but never reaches parse_file"
