"""One extension, several formats that declare themselves (ATL-167).

`.json` is owned by the generic structural config handler, which is the right floor and
the wrong ceiling: a Power BI PBIR `visual.json` says what it is in its own first bytes,
and so do a great many other application formats. `register_dialect` lets such a format
claim its files without editing the router.

Three properties carry the whole thing, and each is silent when broken:

* **The floor stays reachable.** A file no sniff matches must produce exactly what it
  produced before the route existed. A route that swallows unmatched files does not fail
  loudly; it quietly replaces every config entity in the graph.
* **The order is a rule, not an accident.** "First match wins" means nothing unless the
  order is stated, and two overlapping sniffs is the normal case, not the pathological one.
* **A bad plugin cannot take JSON down with it.** The sniff runs before any grammar does,
  on every `.json` in the repo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from code_atlas.parsing import ast as ast_mod
from code_atlas.parsing.ast import (
    DIALECT_SNIFF_BYTES,
    LanguageConfig,
    get_language_for_file,
    parse_file,
    register_dialect,
    register_language,
)

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("tree_sitter_json")

_BLOB = b'{"service": {"name": "api", "port": 8080}}'
_CLAIMED = b'{"$schema": "https://example.invalid/thing/1.0.0", "displayName": "Sales"}'


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is module-global, so a test that registers must not leak into the
    next one -- and `_LANGUAGES` too, since a dialect has to be a registered language."""
    # Explicitly, before snapshotting: `_LANGUAGES` fills lazily on the first lookup, so a
    # snapshot taken ahead of discovery restores an EMPTY registry and takes every
    # language in the process down with it.
    from code_atlas.parsing.languages import discover_plugins

    discover_plugins()
    dialects = {k: list(v) for k, v in ast_mod._DIALECTS.items()}
    languages = dict(ast_mod._LANGUAGES)
    broken = set(ast_mod._BROKEN_SNIFFS)
    try:
        yield
    finally:
        ast_mod._DIALECTS.clear()
        ast_mod._DIALECTS.update({k: list(v) for k, v in dialects.items()})
        ast_mod._LANGUAGES.clear()
        ast_mod._LANGUAGES.update(languages)
        ast_mod._BROKEN_SNIFFS.clear()
        ast_mod._BROKEN_SNIFFS.update(broken)


def _register_fake(name: str, sniff, *, marker: str = "claimed") -> list[str]:
    """Register *name* as a JSON dialect. Returns the list its handler appends to."""
    seen: list[str] = []

    def _parse(path: str, source: bytes, root: Any, project_name: str) -> None:
        # Declines every file. The framework turns that into an EMPTY ParsedFile, not a
        # None -- so a dialect that claims a file and then declines it still costs that
        # file its entities, which is what makes the fallback tests below meaningful.
        seen.append(f"{marker}:{path}")

    json_config = ast_mod._LANGUAGES["json"]
    register_language(
        LanguageConfig(
            name=name,
            extensions=frozenset(),  # reachable only through the dialect route
            language=json_config.language,
            query=json_config.query,
            parse_func=_parse,
        )
    )
    register_dialect(".json", name, sniff)
    return seen


def _lang(path: str, source: bytes | None = None, **kwargs) -> str:
    """The name of the language that would parse *path*. Fails loudly on no match --
    a `None` here means the extension is unregistered, which is a different bug from
    the one every test below is about."""
    config = get_language_for_file(path, source, **kwargs)
    assert config is not None, f"no language registered for {path}"
    return config.name


class TestClaiming:
    def test_a_matching_sniff_claims_the_file(self):
        _register_fake("fake_a", lambda head: b"$schema" in head)
        assert _lang("x.json", _CLAIMED) == "fake_a"

    def test_an_unmatched_file_keeps_the_generic_handler(self):
        _register_fake("fake_a", lambda head: b"$schema" in head)
        assert _lang("x.json", _BLOB) == "json"

    def test_the_claiming_handler_is_the_one_that_runs(self):
        seen = _register_fake("fake_a", lambda head: b"$schema" in head)
        parse_file("x.json", _CLAIMED, "proj")
        assert seen == ["claimed:x.json"]

    def test_the_generic_handler_runs_for_everything_else(self):
        seen = _register_fake("fake_a", lambda head: b"$schema" in head)
        result = parse_file("x.json", _BLOB, "proj")
        assert seen == []
        assert result is not None
        assert {e.kind for e in result.entities} == {"config_file", "config_section", "config_setting"}


class TestTheFloorStaysReachable:
    def test_unmatched_json_is_byte_for_byte_what_it_was(self):
        """The property that matters most and fails most quietly. Compared against the
        same parse with no dialect registered at all, so it holds whatever the config
        handler happens to produce today."""
        before = parse_file("conf/app.json", _BLOB, "proj")
        _register_fake("fake_a", lambda head: b"$schema" in head)
        after = parse_file("conf/app.json", _BLOB, "proj")

        assert before is not None
        assert after is not None
        assert [(e.qualified_name, e.kind, e.name, e.line_start) for e in before.entities] == [
            (e.qualified_name, e.kind, e.name, e.line_start) for e in after.entities
        ]

    def test_a_dialect_naming_an_unregistered_language_falls_back(self):
        """`_LANGUAGES.get(name, config)` is the guard. A dialect whose language failed to
        register -- a missing grammar wheel, say -- must leave the file exactly where it
        was rather than making it unparseable."""
        register_dialect(".json", "never_registered", lambda head: True)
        assert _lang("x.json", _CLAIMED) == "json"

    def test_other_extensions_are_untouched(self):
        _register_fake("fake_a", lambda head: True)
        assert _lang("x.yaml", _BLOB) == "yaml"
        assert _lang("x.toml", b"a = 1") == "toml"


class TestOrdering:
    def test_the_first_registered_matching_sniff_wins(self):
        _register_fake("fake_a", lambda head: True)
        _register_fake("fake_b", lambda head: True)
        assert _lang("x.json", _CLAIMED) == "fake_a"

    def test_a_later_dialect_still_gets_files_the_earlier_one_declines(self):
        """Non-vacuity for the test above: "first wins" must not mean "first only"."""
        _register_fake("fake_a", lambda head: b"displayName" in head)
        _register_fake("fake_b", lambda head: b"service" in head)
        assert _lang("x.json", _CLAIMED) == "fake_a"
        assert _lang("y.json", _BLOB) == "fake_b"


class TestContainment:
    def test_a_raising_sniff_does_not_break_json(self):
        def _boom(head: bytes) -> bool:
            raise RuntimeError("bad plugin")

        _register_fake("fake_boom", _boom)
        assert _lang("x.json", _CLAIMED) == "json"

    def test_a_raising_sniff_does_not_stop_the_next_one(self):
        def _boom(head: bytes) -> bool:
            raise RuntimeError("bad plugin")

        _register_fake("fake_boom", _boom)
        _register_fake("fake_b", lambda head: True)
        assert _lang("x.json", _CLAIMED) == "fake_b"

    def test_a_raising_sniff_is_logged_once(self):
        calls: list[str] = []

        def _boom(head: bytes) -> bool:
            raise RuntimeError("bad plugin")

        _register_fake("fake_boom", _boom)
        from loguru import logger

        sink = logger.add(lambda msg: calls.append(str(msg)), level="WARNING", format="{message}")
        try:
            for i in range(5):
                get_language_for_file(f"x{i}.json", _CLAIMED)
        finally:
            logger.remove(sink)
        assert len([c for c in calls if "fake_boom" in c]) == 1, calls


class TestTheSniffIsBounded:
    def test_a_sniff_sees_a_bounded_prefix(self):
        seen: list[int] = []
        _register_fake("fake_a", lambda head: seen.append(len(head)) or False)
        get_language_for_file("x.json", b"{" + b"x" * (DIALECT_SNIFF_BYTES * 3) + b"}")
        assert seen == [DIALECT_SNIFF_BYTES]


class TestTheScanGateReadsNothing:
    """`get_language_for_file` reads the file from disk when a caller passes no source.
    `FileScope.scan` calls it per file to ask whether the file is indexable at all, and
    every dialect of a suffix answers that identically -- so the read buys nothing and
    costs one syscall per file. It was already happening for every `.h` in the repo."""

    def test_resolve_content_false_never_reads(self, monkeypatch):
        def _never(path: str):
            raise AssertionError(f"read {path} to answer a question that did not need it")

        monkeypatch.setattr(ast_mod, "_read_for_dialect", _never)
        _register_fake("fake_a", lambda head: True)
        assert _lang("x.json", resolve_content=False) == "json"

    def test_the_scan_gate_uses_it(self, tmp_path: Path, monkeypatch):
        from code_atlas.indexing.orchestrator import FileScope
        from code_atlas.settings import AtlasSettings

        (tmp_path / "a.json").write_bytes(_CLAIMED)
        (tmp_path / "b.h").write_bytes(b"#pragma once\n")

        def _never(path: str):
            raise AssertionError(f"FileScope.scan read {path} to decide language support")

        monkeypatch.setattr(ast_mod, "_read_for_dialect", _never)
        _register_fake("fake_a", lambda head: True)

        scanned = FileScope(tmp_path, AtlasSettings(project_root=tmp_path)).scan()
        assert "a.json" in scanned

    def test_a_real_lookup_still_resolves_by_content(self):
        """The mirror: `resolve_content` defaults to True, and turning it off for the scan
        gate must not have turned it off for the parse path."""
        _register_fake("fake_a", lambda head: True)
        assert _lang("x.json", _CLAIMED) == "fake_a"
