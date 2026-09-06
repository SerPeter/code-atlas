"""A language may parse text itself, with no tree-sitter grammar (ATL-168).

Every parser here is tree-sitter, and for good reasons -- but TMDL is line-oriented and
indentation-scoped, which tree-sitter reaches only with a hand-written external scanner,
and Microsoft ships no public grammar. Vendoring one would be a standing maintenance
commitment for a single file type.

So `language=None` plus `text_parse_func` is an escape hatch. What these tests pin is
that it is *only* a hatch around the grammar:

* Everything else on the path still applies. `_parse_hazard`, the `RecursionError` catch,
  the empty-ParsedFile-on-decline rule and the content hashing are what keep one
  pathological file from taking the indexer down, and a text parser is no less capable of
  meeting one than a grammar is.
* Grammar languages pay nothing. Two fields rather than widening `parse_func`'s node
  parameter to `Node | None`, because that widening would have made all sixteen existing
  handlers declare a `None` they can never receive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from code_atlas.parsing import ast as ast_mod
from code_atlas.parsing.ast import (
    DEFAULT_MAX_PARSE_BYTES,
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    parse_file,
    register_language,
)
from code_atlas.schema import NodeLabel, Visibility

if TYPE_CHECKING:
    from tree_sitter import Node


@pytest.fixture(autouse=True)
def _clean_registry():
    from code_atlas.parsing.languages import discover_plugins

    discover_plugins()  # before the snapshot: the registry fills lazily on first lookup
    languages = dict(ast_mod._LANGUAGES)
    extensions = dict(ast_mod._EXTENSION_MAP)
    try:
        yield
    finally:
        ast_mod._LANGUAGES.clear()
        ast_mod._LANGUAGES.update(languages)
        ast_mod._EXTENSION_MAP.clear()
        ast_mod._EXTENSION_MAP.update(extensions)


def _entity(name: str, path: str) -> ParsedEntity:
    return ParsedEntity(
        name=name,
        qualified_name=f"proj:{name}",
        label=NodeLabel.CALLABLE,
        kind="function",
        line_start=1,
        line_end=1,
        file_path=path,
        visibility=Visibility.PUBLIC,
    )


def _register_text(ext: str = ".fake", *, handler=None) -> list[tuple[str, bytes, str]]:
    """Register a text language on *ext*. Returns the list its handler records into."""
    seen: list[tuple[str, bytes, str]] = []

    def _default(path: str, source: bytes, project_name: str) -> ParsedFile:
        seen.append((path, source, project_name))
        return ParsedFile(
            file_path=path,
            language="faketext",
            entities=[_entity("thing", path)],
            relationships=[],
        )

    register_language(
        LanguageConfig(
            name="faketext",
            extensions=frozenset({ext}),
            language=None,
            query=None,
            text_parse_func=handler or _default,
        )
    )
    return seen


class TestTheHatch:
    def test_a_text_language_parses_with_no_grammar(self):
        _register_text()
        result = parse_file("a.fake", b"anything at all", "proj")
        assert result is not None
        assert [e.name for e in result.entities] == ["thing"]

    def test_the_handler_gets_the_raw_bytes_and_nothing_else(self):
        seen = _register_text()
        parse_file("dir/a.fake", b"raw bytes", "proj")
        assert seen == [("dir/a.fake", b"raw bytes", "proj")]

    def test_a_grammar_language_still_gets_a_root_node(self):
        """The other half of the hatch: registered languages must be untouched."""
        result = parse_file("m.py", b"def f():\n    return 1\n", "proj")
        assert result is not None
        assert {e.name for e in result.entities} >= {"f"}

    def test_content_hashing_still_runs(self):
        """A text language is a language, not a bypass. Without a content_hash the delta
        classifier cannot tell an unchanged file from a new one, so every parse would
        re-upsert every entity."""
        _register_text()
        result = parse_file("a.fake", b"x", "proj")
        assert result is not None
        assert result.entities[0].content_hash


class TestTheSafetyRailsStillApply:
    def test_the_pre_parse_guard_still_refuses_an_oversized_file(self):
        """`_parse_hazard` runs before the language branch, so a text handler cannot be
        handed a file the guard would have refused."""
        seen = _register_text()
        oversized = b"x" * (DEFAULT_MAX_PARSE_BYTES + 1)
        assert parse_file("a.fake", oversized, "proj") is None
        assert seen == [], "the handler was called for a file the guard refused"

    def test_a_recursing_handler_is_caught_not_propagated(self):
        """Left uncaught this is a poison pill: the AST consumer logs "batch failed, will
        retry" and retries the same file forever."""

        def _recurse(path: str, source: bytes, project_name: str) -> ParsedFile:
            raise RecursionError

        _register_text(handler=_recurse)
        assert parse_file("a.fake", b"x", "proj") is None

    def test_declining_yields_an_empty_parsed_file_not_none(self):
        """None would mean "unsupported language", and the AST consumer only records a
        file hash for files that produced a ParsedFile -- so the file would fall outside
        the hash gate and be re-read on every pass, forever."""

        def _decline(path: str, source: bytes, project_name: str) -> None:
            return None

        _register_text(handler=_decline)
        result = parse_file("a.fake", b"x", "proj")
        assert result is not None
        assert result.entities == []

    def test_rationale_extraction_is_skipped_rather_than_crashing(self):
        """It walks the tree, so a text language cannot have it. Opting out by leaving
        `comment_node_types` empty is the intended route; this is the second lock, so
        setting the field by mistake produces no rationale rather than an AttributeError
        on every file of that type."""
        register_language(
            LanguageConfig(
                name="faketext",
                extensions=frozenset({".fake"}),
                language=None,
                query=None,
                text_parse_func=lambda path, source, project: ParsedFile(
                    file_path=path, language="faketext", entities=[_entity("thing", path)], relationships=[]
                ),
                comment_node_types=frozenset({"comment"}),  # wrong, and must not be fatal
            )
        )
        result = parse_file("a.fake", b"# WHY: because\nx\n", "proj")
        assert result is not None
        assert [e.name for e in result.entities] == ["thing"]


class TestTheModesAreExclusive:
    """Checked at registration, because the alternative is a `TypeError` deep inside
    `parse_file`, once per file, on a language somebody added months ago."""

    def _grammar(self):
        return ast_mod._LANGUAGES["python"].language

    def _node_handler(self, path: str, source: bytes, root: Node, project_name: str) -> None:
        return None

    def test_neither_handler_is_refused(self):
        with pytest.raises(ValueError, match="exactly one of parse_func"):
            LanguageConfig(name="x", extensions=frozenset(), language=None, query=None)

    def test_both_handlers_are_refused(self):
        with pytest.raises(ValueError, match="exactly one of parse_func"):
            LanguageConfig(
                name="x",
                extensions=frozenset(),
                language=self._grammar(),
                query=None,
                parse_func=self._node_handler,
                text_parse_func=lambda p, s, n: None,
            )

    def test_a_grammar_with_a_text_handler_is_refused(self):
        with pytest.raises(ValueError, match="go together"):
            LanguageConfig(
                name="x",
                extensions=frozenset(),
                language=self._grammar(),
                query=None,
                text_parse_func=lambda p, s, n: None,
            )

    def test_no_grammar_with_a_node_handler_is_refused(self):
        """The one that would otherwise reach production: it registers cleanly and then
        hands `parse_func` a `None` root on the first file of that type."""
        with pytest.raises(ValueError, match="go together"):
            LanguageConfig(
                name="x",
                extensions=frozenset(),
                language=None,
                query=None,
                parse_func=self._node_handler,
            )
