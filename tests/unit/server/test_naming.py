"""Entity breadcrumbs (ATL-124).

A bare basename identifies nothing: `conftest` appears four times in this project's
module map and `test_client` twice.
"""

from __future__ import annotations

from code_atlas.server.web.naming import breadcrumb


class TestBreadcrumb:
    def test_a_method_shows_its_class(self):
        b = breadcrumb(
            qualified_name="code_atlas.graph.client.GraphClient.resolve_calls",
            file_path="src/code_atlas/graph/client.py",
        )

        assert b.path == "code_atlas/graph/client.py"
        assert b.owner == "GraphClient"
        assert b.symbol == "resolve_calls"

    def test_a_module_level_function_does_not_invent_a_class(self):
        """An owner slot filled with the module name would be a lie dressed as structure."""
        b = breadcrumb(qualified_name="code_atlas.cli.main", file_path="src/code_atlas/cli.py")

        assert b.owner == ""
        assert b.symbol == "main"

    def test_two_files_with_the_same_basename_are_distinguishable(self):
        """The whole point — four `conftest` modules must not read identically."""
        a = breadcrumb(
            qualified_name="tests.unit.graph.conftest", file_path="tests/unit/graph/conftest.py", label="Module"
        )
        b = breadcrumb(
            qualified_name="tests.integration.graph.conftest",
            file_path="tests/integration/graph/conftest.py",
            label="Module",
        )

        assert a.full != b.full

    def test_a_module_does_not_name_itself_twice(self):
        b = breadcrumb(
            qualified_name="code_atlas.graph.client", file_path="src/code_atlas/graph/client.py", label="Module"
        )

        assert b.short == "client.py", "`client.py > client` says one thing twice"

    def test_the_src_prefix_is_dropped(self):
        """Every path in the project starts with it, so it distinguishes nothing."""
        b = breadcrumb(qualified_name="a.b", file_path="src/a/b.py", label="Module")

        assert not b.path.startswith("src/")

    def test_windows_separators_are_normalised(self):
        b = breadcrumb(qualified_name="a.b", file_path="src\a\b.py", label="Module")

        assert "\\" not in b.path


class TestTruncation:
    def test_truncation_drops_leading_parts_and_keeps_the_symbol(self):
        b = breadcrumb(
            qualified_name="code_atlas.graph.client.GraphClient.resolve_calls",
            file_path="src/code_atlas/graph/client.py",
        )

        short = b.truncated(30)

        assert "resolve_calls" in short, "the symbol is the answer; it survives"
        assert len(short) <= 34

    def test_a_symbol_longer_than_the_limit_still_renders(self):
        """A label reading only an ellipsis identifies nothing at all."""
        b = breadcrumb(qualified_name="pkg.mod." + "x" * 60, file_path="pkg/mod.py")

        assert b.truncated(10).endswith("x" * 60)

    def test_a_short_name_is_left_alone(self):
        b = breadcrumb(qualified_name="a.b", file_path="a.py", label="Module")

        assert "…" not in b.truncated(80)
