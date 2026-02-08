"""Tests for the indexer module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from code_atlas.events import EventBus
from code_atlas.indexer import FileScope, index_project, scan_files
from code_atlas.schema import NodeLabel
from code_atlas.settings import AtlasSettings, ScopeSettings

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(root: Path, rel_path: str, content: str = "") -> Path:
    """Write a file at root/rel_path, creating parent dirs."""
    p = root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _make_settings(root: Path, **scope_kwargs) -> AtlasSettings:
    """Create AtlasSettings pointing at *root* with optional scope overrides."""
    return AtlasSettings(
        project_root=root,
        scope=ScopeSettings(**scope_kwargs),
    )


# ---------------------------------------------------------------------------
# scan_files — unit tests (no infrastructure needed)
# ---------------------------------------------------------------------------


class TestScanFiles:
    def test_discovers_py_files(self, tmp_path):
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "lib/utils.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert "app.py" in result
        assert "lib/utils.py" in result

    def test_ignores_non_python_files(self, tmp_path):
        _write(tmp_path, "readme.md", "# Hello")
        _write(tmp_path, "data.json", "{}")
        _write(tmp_path, "app.py", "x = 1")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.py"]

    def test_default_excludes(self, tmp_path):
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "__pycache__/module.pyc", "")
        _write(tmp_path, ".git/config", "")
        _write(tmp_path, "node_modules/pkg/index.py", "")
        _write(tmp_path, ".venv/lib/site.py", "")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.py"]

    def test_respects_gitignore(self, tmp_path):
        _write(tmp_path, ".gitignore", "generated/\n*.generated.py\n")
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "generated/out.py", "y = 2")
        _write(tmp_path, "foo.generated.py", "z = 3")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.py"]

    def test_respects_atlasignore(self, tmp_path):
        _write(tmp_path, ".atlasignore", "vendor/\n")
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "vendor/lib.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.py"]

    def test_scope_paths(self, tmp_path):
        _write(tmp_path, "src/app.py", "x = 1")
        _write(tmp_path, "tests/test_app.py", "y = 2")
        _write(tmp_path, "scripts/deploy.py", "z = 3")

        result = scan_files(tmp_path, _make_settings(tmp_path), scope_paths=["src"])

        assert result == ["src/app.py"]

    def test_include_paths(self, tmp_path):
        _write(tmp_path, "src/app.py", "x = 1")
        _write(tmp_path, "tests/test_app.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path, include_paths=["src"]))

        assert result == ["src/app.py"]

    def test_exclude_before_include(self, tmp_path):
        """Excluded files are NOT rescued by include paths."""
        _write(tmp_path, ".atlasignore", "src/generated/\n")
        _write(tmp_path, "src/app.py", "x = 1")
        _write(tmp_path, "src/generated/out.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path, include_paths=["src"]))

        assert result == ["src/app.py"]

    def test_exclude_patterns_from_settings(self, tmp_path):
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "tmp/scratch.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path, exclude_patterns=["tmp/"]))

        assert result == ["app.py"]

    def test_returns_sorted_posix_paths(self, tmp_path):
        _write(tmp_path, "z.py", "")
        _write(tmp_path, "a.py", "")
        _write(tmp_path, "m/b.py", "")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["a.py", "m/b.py", "z.py"]

    def test_empty_directory(self, tmp_path):
        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == []

    def test_pyi_files_included(self, tmp_path):
        _write(tmp_path, "app.pyi", "x: int")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.pyi"]

    def test_nested_gitignore(self, tmp_path):
        """A .gitignore in a subdirectory excludes files below it."""
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "lib/core.py", "y = 2")
        _write(tmp_path, "lib/.gitignore", "generated/\n")
        _write(tmp_path, "lib/generated/out.py", "z = 3")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert "app.py" in result
        assert "lib/core.py" in result
        assert "lib/generated/out.py" not in result

    def test_nested_gitignore_does_not_leak_up(self, tmp_path):
        """Patterns in a nested .gitignore don't affect sibling dirs."""
        _write(tmp_path, "a/skip.py", "x = 1")
        _write(tmp_path, "b/.gitignore", "skip.py\n")
        _write(tmp_path, "b/skip.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert "a/skip.py" in result
        assert "b/skip.py" not in result

    def test_default_excludes_vendor_and_target(self, tmp_path):
        """vendor/ and target/ are excluded by default."""
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "vendor/lib.py", "y = 2")
        _write(tmp_path, "target/out.py", "z = 3")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.py"]

    def test_file_scope_reuse(self, tmp_path):
        """FileScope can be constructed once and queried multiple times."""
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "lib/utils.py", "y = 2")
        _write(tmp_path, ".gitignore", "ignored/\n")
        _write(tmp_path, "ignored/secret.py", "z = 3")

        scope = FileScope(tmp_path, _make_settings(tmp_path))
        # Must call scan() first to populate nested specs (and verify it works)
        files = scope.scan()
        assert "app.py" in files
        assert "lib/utils.py" in files

        # is_included() works without re-scanning
        assert scope.is_included("app.py") is True
        assert scope.is_included("lib/utils.py") is True
        assert scope.is_included("ignored/secret.py") is False


# ---------------------------------------------------------------------------
# Integration tests (require Memgraph + Valkey)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIndexProjectIntegration:
    @pytest.fixture
    def project_dir(self, tmp_path):
        """Create a minimal Python project for indexing."""
        _write(tmp_path, "src/__init__.py", "")
        _write(tmp_path, "src/app.py", 'def hello():\n    """Say hello."""\n    return "hello"\n')
        _write(tmp_path, "src/utils.py", "MAGIC = 42\n\ndef add(a, b):\n    return a + b\n")
        return tmp_path

    @pytest.fixture
    async def bus(self, settings):
        """EventBus fixture — skips if Valkey is unreachable."""
        bus = EventBus(settings.redis)
        try:
            await bus.ping()
        except Exception:
            pytest.skip("Valkey not available")
        yield bus
        await bus.close()

    async def test_index_project_creates_graph(self, project_dir, graph_client, bus):
        settings = AtlasSettings(project_root=project_dir)
        await graph_client.ensure_schema()

        result = await index_project(settings, graph_client, bus)

        assert result.files_scanned >= 2
        assert result.entities_total > 0
        assert result.duration_s > 0

    async def test_index_project_creates_hierarchy(self, project_dir, graph_client, bus):
        settings = AtlasSettings(project_root=project_dir)
        await graph_client.ensure_schema()

        await index_project(settings, graph_client, bus)

        # Project node exists
        projects = await graph_client.execute(f"MATCH (p:{NodeLabel.PROJECT}) RETURN p.name AS name")
        assert len(projects) >= 1

        # Package node for src/ exists
        packages = await graph_client.execute(
            f"MATCH (p:{NodeLabel.PACKAGE} {{project_name: $pn}}) RETURN p.qualified_name AS qn",
            {"pn": project_dir.name},
        )
        assert any(p["qn"] == "src" for p in packages)

        # CONTAINS edge exists: Project → Package
        contains = await graph_client.execute("MATCH (a)-[:CONTAINS]->(b) RETURN a.uid AS from_uid, b.uid AS to_uid")
        assert len(contains) > 0

    async def test_index_project_full_reindex(self, project_dir, graph_client, bus):
        settings = AtlasSettings(project_root=project_dir)
        await graph_client.ensure_schema()

        # First index
        r1 = await index_project(settings, graph_client, bus)
        assert r1.entities_total > 0

        # Full reindex — should work cleanly
        r2 = await index_project(settings, graph_client, bus, full_reindex=True)
        assert r2.entities_total > 0

    async def test_index_project_error_resilience(self, tmp_path, graph_client, bus):
        """One unparseable file shouldn't abort the whole indexing run."""
        _write(tmp_path, "good.py", "x = 1\n")
        _write(tmp_path, "bad.py", "def (\n")  # syntax error — tree-sitter handles gracefully

        settings = AtlasSettings(project_root=tmp_path)
        await graph_client.ensure_schema()

        result = await index_project(settings, graph_client, bus)

        # Should still complete and index the good file
        assert result.files_scanned >= 2
        assert result.entities_total > 0
