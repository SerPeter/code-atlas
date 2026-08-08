"""Call-candidate narrowing before resolution (ATL-113, ADR-0030).

Placement is the whole point: this runs *before* any name-matching strategy reads the
list, so the surviving count becomes ``candidate_count``, which drives every edge's
weight (ADR-0014). A candidate left in the pool does not merely add a wrong edge — it
halves the weight of the right one.

The two filters deliberately differ in what they do when narrowing empties the pool, and
the asymmetry is what these tests pin.
"""

from __future__ import annotations

from code_atlas.graph.client import _namespace_group, _narrow_candidates

# (uid, file_path, qualified_name) — the shape resolve_calls passes through.
PY = ("u:py.render", "app/views.py", "app.views.render")
TS = ("u:ts.render", "web/views.ts", "web.views.render")
TSX = ("u:tsx.render", "web/Card.tsx", "web.Card.render")
JS = ("u:js.render", "web/legacy.js", "web.legacy.render")
CPP = ("u:cpp.render", "engine/draw.cpp", "engine.draw.render")
HPP = ("u:hpp.render", "engine/draw.hpp", "engine.draw.render")
JAVA = ("u:java.render", "svc/View.java", "svc.View.render")
CS = ("u:cs.render", "svc/View.cs", "svc.View.render")


class TestNamespaceGroup:
    def test_typescript_and_javascript_share_a_group(self):
        """One runtime, one module system — a .ts call really can reach a .js definition."""
        assert _namespace_group("a.ts") == _namespace_group("a.js") == _namespace_group("a.tsx")

    def test_c_and_cpp_share_a_group(self):
        """A header is included across the boundary; splitting them would drop real edges."""
        assert _namespace_group("a.c") == _namespace_group("a.hpp") == _namespace_group("a.cc")

    def test_java_and_csharp_do_not_share_a_group(self):
        """Both are 'JVM-family' to the parser registry, but neither can call the other."""
        assert _namespace_group("A.java") != _namespace_group("A.cs")

    def test_an_unknown_extension_is_not_a_group(self):
        """Empty means "do not partition on this" — never "matches nothing"."""
        assert _namespace_group("script.zzz") == ""
        assert _namespace_group("Makefile") == ""

    def test_windows_separators_and_case_are_handled(self):
        assert _namespace_group("app\\views.PY") == "python"


class TestLanguagePartitioning:
    """Strict: an empty result stays empty."""

    def test_a_python_call_cannot_reach_a_typescript_definition(self):
        survivors = _narrow_candidates([PY, TS], "u:caller", "app/main.py", frozenset())

        assert survivors == [PY]

    def test_the_pool_is_emptied_rather_than_falling_back(self):
        """The asymmetry that matters.

        With only the TypeScript definition present, falling back would leave one
        candidate — and `project_unique` would then report that cross-language match as
        confidence "resolved" rather than ambiguous. A confident wrong edge is worse than
        no edge; a Python function cannot be reached in-process from TypeScript at all.
        """
        survivors = _narrow_candidates([TS], "u:caller", "app/main.py", frozenset())

        assert survivors == []

    def test_candidate_count_drops_to_the_same_language_count(self):
        """The count is what weights the edge, so narrowing must reach it."""
        pool = [PY, TS, TSX, JS, CPP, JAVA, CS]

        assert len(_narrow_candidates(pool, "u:c", "app/main.py", frozenset())) == 1
        assert len(_narrow_candidates(pool, "u:c", "web/app.ts", frozenset())) == 3
        assert len(_narrow_candidates(pool, "u:c", "engine/main.cpp", frozenset())) == 1

    def test_a_javascript_caller_reaches_typescript_definitions(self):
        survivors = _narrow_candidates([PY, TS, JS], "u:caller", "web/legacy.js", frozenset())

        assert set(survivors) == {TS, JS}

    def test_a_cpp_caller_reaches_headers(self):
        survivors = _narrow_candidates([CPP, HPP, PY], "u:caller", "engine/main.cc", frozenset())

        assert set(survivors) == {CPP, HPP}

    def test_an_unmapped_caller_extension_filters_nothing(self):
        """Safe by construction: an unknown language must not silently drop every edge."""
        pool = [PY, TS, CPP]

        assert _narrow_candidates(pool, "u:caller", "build.zzz", frozenset()) == pool

    def test_single_language_resolution_is_unchanged(self):
        """The no-op case, which is every pure-Python repo — nothing may move."""
        pool = [PY, ("u:py.other", "app/other.py", "app.other.render")]

        assert _narrow_candidates(pool, "u:caller", "app/main.py", frozenset()) == pool


class TestTestProvenance:
    """Falls back: an empty result reverts to the unfiltered pool (ADR-0030)."""

    def test_production_code_does_not_resolve_onto_a_test_definition(self):
        prod = ("u:prod", "app/impl.py", "app.impl.helper")
        test = ("u:test", "tests/test_impl.py", "tests.test_impl.helper")

        survivors = _narrow_candidates([prod, test], "u:caller", "app/main.py", frozenset({"u:test"}))

        assert survivors == [prod]

    def test_a_test_only_name_still_resolves_rather_than_vanishing(self):
        """The other half of the asymmetry.

        Where every definition lives in test code, that genuinely is the best available
        answer — a production→test call is unusual, not impossible. Emptying the pool
        would trade a diluted-but-present edge for a silent absence, which is the failure
        ADR-0014 exists to prevent.
        """
        test = ("u:test", "tests/test_impl.py", "tests.test_impl.helper")

        survivors = _narrow_candidates([test], "u:caller", "app/main.py", frozenset({"u:test"}))

        assert survivors == [test]

    def test_a_test_caller_filters_nothing(self):
        """Calling the code under test is what a test is for."""
        prod = ("u:prod", "app/impl.py", "app.impl.helper")
        test = ("u:test", "tests/test_impl.py", "tests.test_impl.helper")

        survivors = _narrow_candidates([prod, test], "u:test", "tests/test_impl.py", frozenset({"u:test"}))

        assert set(survivors) == {prod, test}


class TestBothFiltersTogether:
    def test_language_narrowing_runs_before_test_provenance(self):
        """A cross-language test definition must not become the fallback answer."""
        ts_test = ("u:ts.test", "web/views.test.ts", "web.views.test.render")

        survivors = _narrow_candidates([TS, ts_test], "u:caller", "app/main.py", frozenset({"u:ts.test"}))

        assert survivors == [], "the language filter empties the pool, and it stays empty"
