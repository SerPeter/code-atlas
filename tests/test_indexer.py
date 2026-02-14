"""Tests for the indexer module."""

from __future__ import annotations

import os
import subprocess
import time
from typing import TYPE_CHECKING

import pytest

from code_atlas.events import EventBus
from code_atlas.indexing.orchestrator import (
    FileScope,
    StalenessChecker,
    _git_changed_files,
    _read_git_head,
    index_project,
    scan_files,
)
from code_atlas.schema import NodeLabel
from code_atlas.settings import AtlasSettings, IndexSettings, ScopeSettings

if TYPE_CHECKING:
    from pathlib import Path

# Skip decorator for tests that require symlink support (needs admin/dev mode on Windows)
needs_symlinks = pytest.mark.skipif(
    not os.environ.get("CI") and os.name == "nt",
    reason="Symlinks require admin or Developer Mode on Windows",
)


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

    def test_ignores_unsupported_files(self, tmp_path):
        _write(tmp_path, "data.json", "{}")
        _write(tmp_path, "image.png", "")
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "readme.md", "# Hello")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.py", "readme.md"]

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

    def test_bom_in_gitignore(self, tmp_path):
        """A .gitignore saved with UTF-8 BOM should still work."""
        bom = b"\xef\xbb\xbf"
        gi = tmp_path / ".gitignore"
        gi.write_bytes(bom + b"secret/\n")
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "secret/key.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.py"]

    @needs_symlinks
    def test_symlinked_dir_not_followed(self, tmp_path):
        """Symlinked directories are skipped (matches git default behavior)."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        _write(tmp_path, "real/module.py", "x = 1")
        _write(tmp_path, "app.py", "y = 2")

        link = tmp_path / "linked"
        link.symlink_to(real_dir, target_is_directory=True)

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert "app.py" in result
        assert "real/module.py" in result
        # Symlinked directory content should NOT appear
        assert "linked/module.py" not in result

    @needs_symlinks
    def test_broken_symlink_skipped(self, tmp_path):
        """Broken file symlinks don't crash the scanner."""
        _write(tmp_path, "app.py", "x = 1")
        broken = tmp_path / "broken.py"
        broken.symlink_to(tmp_path / "nonexistent.py")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.py"]

    def test_non_ascii_paths(self, tmp_path):
        """Non-ASCII characters in paths don't crash the scanner."""
        _write(tmp_path, "über/app.py", "x = 1")
        _write(tmp_path, "日本語/lib.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert "über/app.py" in result
        assert "日本語/lib.py" in result


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


# ---------------------------------------------------------------------------
# _git_changed_files — unit tests (requires git CLI, no infrastructure)
# ---------------------------------------------------------------------------


def _git(cwd, *args):
    """Run a git command in cwd."""
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


def _init_git_repo(tmp_path):
    """Initialise a git repo with an initial commit."""
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@test.com")
    _git(tmp_path, "config", "user.name", "Test")


def _get_head(tmp_path):
    """Return the full HEAD hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


class TestGitChangedFiles:
    def test_added_file(self, tmp_path):
        """New files since the base commit are detected as 'created'."""
        _init_git_repo(tmp_path)
        _write(tmp_path, "app.py", "x = 1")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")
        base_hash = _get_head(tmp_path)

        _write(tmp_path, "new.py", "y = 2")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "add new")

        changes = _git_changed_files(tmp_path, base_hash)

        assert changes is not None
        paths = {p for p, _ in changes}
        types = dict(changes)
        assert "new.py" in paths
        assert types["new.py"] == "created"

    def test_modified_file(self, tmp_path):
        """Modified files are detected as 'modified'."""
        _init_git_repo(tmp_path)
        _write(tmp_path, "app.py", "x = 1")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")
        base_hash = _get_head(tmp_path)

        _write(tmp_path, "app.py", "x = 2")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "modify")

        changes = _git_changed_files(tmp_path, base_hash)

        assert changes is not None
        types = dict(changes)
        assert types["app.py"] == "modified"

    def test_deleted_file(self, tmp_path):
        """Deleted files are detected as 'deleted'."""
        _init_git_repo(tmp_path)
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "old.py", "y = 2")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")
        base_hash = _get_head(tmp_path)

        (tmp_path / "old.py").unlink()
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "delete")

        changes = _git_changed_files(tmp_path, base_hash)

        assert changes is not None
        types = dict(changes)
        assert types["old.py"] == "deleted"

    def test_invalid_hash_returns_none(self, tmp_path):
        """An invalid base hash returns None (fallback to full mode)."""
        _init_git_repo(tmp_path)
        _write(tmp_path, "app.py", "x = 1")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")

        result = _git_changed_files(tmp_path, "0000000000000000000000000000000000000000")

        assert result is None

    def test_no_changes(self, tmp_path):
        """No changes returns empty list."""
        _init_git_repo(tmp_path)
        _write(tmp_path, "app.py", "x = 1")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")
        base_hash = _get_head(tmp_path)

        changes = _git_changed_files(tmp_path, base_hash)

        assert changes is not None
        assert changes == []


# ---------------------------------------------------------------------------
# Delta threshold — unit test
# ---------------------------------------------------------------------------


class TestDeltaThreshold:
    def test_threshold_setting_default(self):
        """Default delta threshold is 0.3."""
        settings = IndexSettings()
        assert settings.delta_threshold == 0.3

    def test_threshold_setting_custom(self):
        """Custom delta threshold is respected."""
        settings = IndexSettings(delta_threshold=0.5)
        assert settings.delta_threshold == 0.5


# ---------------------------------------------------------------------------
# Delta indexing — integration tests (require Memgraph + Valkey + git)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDeltaIndexIntegration:
    @pytest.fixture
    def git_project(self, tmp_path):
        """Create a git-tracked Python project with initial commit.

        Uses 5 files so that modifying 1 file stays under the 30% delta
        threshold (1/5 = 20%).
        """
        _init_git_repo(tmp_path)
        _write(tmp_path, "src/__init__.py", "")
        _write(tmp_path, "src/app.py", 'def hello():\n    """Say hello."""\n    return "hello"\n')
        _write(tmp_path, "src/utils.py", "MAGIC = 42\n\ndef add(a, b):\n    return a + b\n")
        _write(tmp_path, "src/config.py", "DEBUG = False\n\ndef get_config():\n    return {}\n")
        _write(tmp_path, "src/models.py", "class User:\n    name: str\n    email: str\n")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")
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

    async def test_delta_index_mode(self, git_project, graph_client, bus):
        """Re-indexing without changes uses delta mode."""
        settings = AtlasSettings(project_root=git_project)
        await graph_client.ensure_schema()

        # First index — full mode
        r1 = await index_project(settings, graph_client, bus)
        assert r1.mode == "full"
        assert r1.entities_total > 0

        # Re-index without changes — delta mode, 0 published
        r2 = await index_project(settings, graph_client, bus)
        assert r2.mode == "delta"
        assert r2.files_published == 0
        assert r2.entities_total == r1.entities_total

    async def test_delta_index_publishes_only_changed(self, git_project, graph_client, bus):
        """Modifying one file only publishes that file in delta mode."""
        settings = AtlasSettings(project_root=git_project)
        await graph_client.ensure_schema()

        # First index
        r1 = await index_project(settings, graph_client, bus)
        assert r1.mode == "full"

        # Modify one file and commit
        _write(git_project, "src/app.py", 'def hello():\n    """Say hello!""""\n    return "hello world"\n')
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "modify app")

        # Delta re-index
        r2 = await index_project(settings, graph_client, bus)
        assert r2.mode == "delta"
        assert r2.delta_stats is not None
        assert r2.delta_stats.files_modified >= 1
        assert r2.files_published >= 1

    async def test_delta_index_detects_new_files(self, git_project, graph_client, bus):
        """New files are picked up in delta mode."""
        settings = AtlasSettings(project_root=git_project)
        await graph_client.ensure_schema()

        # First index
        await index_project(settings, graph_client, bus)

        # Add new file and commit
        _write(git_project, "src/new_module.py", "NEW_CONST = 99\n")
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "add new module")

        # Delta re-index
        r2 = await index_project(settings, graph_client, bus)
        assert r2.mode == "delta"
        assert r2.delta_stats is not None
        assert r2.delta_stats.files_added >= 1

    async def test_delta_index_handles_deletion(self, git_project, graph_client, bus):
        """Deleted files' entities are removed from the graph."""
        settings = AtlasSettings(project_root=git_project)
        await graph_client.ensure_schema()

        # First index
        r1 = await index_project(settings, graph_client, bus)
        e1 = r1.entities_total

        # Delete a file and commit
        (git_project / "src" / "utils.py").unlink()
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "remove utils")

        # Delta re-index
        r2 = await index_project(settings, graph_client, bus)
        assert r2.mode == "delta"
        assert r2.delta_stats is not None
        assert r2.delta_stats.files_deleted >= 1
        # Entity count should have dropped
        assert r2.entities_total < e1

    async def test_delta_index_full_fallback_on_threshold(self, git_project, graph_client, bus):
        """Exceeding the threshold triggers full mode."""
        # Set a very low threshold so any change exceeds it
        settings = AtlasSettings(project_root=git_project, index=IndexSettings(delta_threshold=0.0))
        await graph_client.ensure_schema()

        # First index
        await index_project(settings, graph_client, bus)

        # Modify a file and commit
        _write(git_project, "src/app.py", 'def hello():\n    return "changed"\n')
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "change")

        # Re-index with threshold=0.0 — should fall back to full
        r2 = await index_project(settings, graph_client, bus)
        assert r2.mode == "full"

    async def test_delta_index_preserves_unchanged(self, git_project, graph_client, bus):
        """Unchanged entities keep their entity count after delta re-index."""
        settings = AtlasSettings(project_root=git_project)
        await graph_client.ensure_schema()

        # First index
        r1 = await index_project(settings, graph_client, bus)

        # Re-index without changes
        r2 = await index_project(settings, graph_client, bus)
        assert r2.mode == "delta"
        assert r2.entities_total == r1.entities_total

    async def test_full_reindex_flag_overrides_delta(self, git_project, graph_client, bus):
        """--full flag forces full mode even when delta is available."""
        settings = AtlasSettings(project_root=git_project)
        await graph_client.ensure_schema()

        await index_project(settings, graph_client, bus)

        r2 = await index_project(settings, graph_client, bus, full_reindex=True)
        assert r2.mode == "full"


# ---------------------------------------------------------------------------
# _read_git_head — unit tests (no infrastructure)
# ---------------------------------------------------------------------------


class TestReadGitHead:
    def test_reads_ref_head(self, tmp_path):
        """Reads HEAD via ref: pointer to a loose ref file."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        refs_dir = git_dir / "refs" / "heads"
        refs_dir.mkdir(parents=True)
        fake_hash = "a" * 40
        (refs_dir / "main").write_text(fake_hash + "\n", encoding="utf-8")

        assert _read_git_head(tmp_path) == fake_hash

    def test_reads_detached_head(self, tmp_path):
        """Detached HEAD with a raw 40-char hex hash."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        fake_hash = "b" * 40
        (git_dir / "HEAD").write_text(fake_hash + "\n", encoding="utf-8")

        assert _read_git_head(tmp_path) == fake_hash

    def test_reads_packed_ref(self, tmp_path):
        """Falls back to packed-refs when the loose ref file is missing."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        fake_hash = "c" * 40
        (git_dir / "packed-refs").write_text(
            f"# pack-refs with: peeled fully-peeled sorted\n{fake_hash} refs/heads/main\n",
            encoding="utf-8",
        )

        assert _read_git_head(tmp_path) == fake_hash

    def test_returns_none_no_git(self, tmp_path):
        """Returns None for directories without .git."""
        assert _read_git_head(tmp_path) is None


# ---------------------------------------------------------------------------
# StalenessChecker — unit tests (no infrastructure)
# ---------------------------------------------------------------------------


class TestStalenessChecker:
    def test_current_head_caches_by_mtime(self, tmp_path):
        """Same mtime returns cached value without re-reading the file."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        refs_dir = git_dir / "refs" / "heads"
        refs_dir.mkdir(parents=True)
        fake_hash = "d" * 40
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (refs_dir / "main").write_text(fake_hash + "\n", encoding="utf-8")

        checker = StalenessChecker(tmp_path)
        h1 = checker.current_head()
        assert h1 == fake_hash

        # Overwrite file content but keep the same text — mtime hasn't changed (fast enough)
        # The cache should still return the same hash
        h2 = checker.current_head()
        assert h2 == fake_hash

    def test_current_head_invalidates_on_mtime_change(self, tmp_path):
        """New ref content with changed mtime triggers re-read."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        refs_dir = git_dir / "refs" / "heads"
        refs_dir.mkdir(parents=True)
        hash1 = "d" * 40
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        ref_file = refs_dir / "main"
        ref_file.write_text(hash1 + "\n", encoding="utf-8")

        checker = StalenessChecker(tmp_path)
        assert checker.current_head() == hash1

        # Write a new hash and force mtime change
        time.sleep(0.05)
        hash2 = "e" * 40
        ref_file.write_text(hash2 + "\n", encoding="utf-8")

        assert checker.current_head() == hash2

    def test_current_head_non_git_dir(self, tmp_path):
        """Non-git directory returns None."""
        checker = StalenessChecker(tmp_path)
        assert checker.current_head() is None

    def test_project_name(self, tmp_path):
        """project_name is derived from project_root.name."""
        checker = StalenessChecker(tmp_path)
        assert checker.project_name == tmp_path.resolve().name


# ---------------------------------------------------------------------------
# stale_mode setting — unit test
# ---------------------------------------------------------------------------


class TestStaleMode:
    def test_stale_mode_default(self):
        """Default stale_mode is 'warn'."""
        settings = IndexSettings()
        assert settings.stale_mode == "warn"


# ---------------------------------------------------------------------------
# Staleness check — integration tests (require Memgraph + Valkey + git)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestStalenessCheckIntegration:
    @pytest.fixture
    def git_project(self, tmp_path):
        """Create a git-tracked Python project with initial commit."""
        _init_git_repo(tmp_path)
        _write(tmp_path, "src/__init__.py", "")
        _write(tmp_path, "src/app.py", 'def hello():\n    """Say hello."""\n    return "hello"\n')
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")
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

    async def test_not_stale_when_hashes_match(self, git_project, graph_client, bus):
        """After indexing, the checker reports not stale."""
        settings = AtlasSettings(project_root=git_project)
        await graph_client.ensure_schema()
        await index_project(settings, graph_client, bus)

        checker = StalenessChecker(git_project)
        info = await checker.check(graph_client)

        assert info.stale is False
        assert info.current_commit is not None
        assert info.last_indexed_commit is not None

    async def test_stale_when_new_commit(self, git_project, graph_client, bus):
        """A new commit after indexing makes the checker report stale."""
        settings = AtlasSettings(project_root=git_project)
        await graph_client.ensure_schema()
        await index_project(settings, graph_client, bus)

        # Make a new commit
        _write(git_project, "src/new.py", "x = 1\n")
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "new file")

        checker = StalenessChecker(git_project)
        info = await checker.check(graph_client)

        assert info.stale is True
        assert info.changed_files  # should list at least src/new.py

    async def test_not_stale_non_git_dir(self, tmp_path, graph_client):
        """Non-git directory is never stale."""
        checker = StalenessChecker(tmp_path)
        info = await checker.check(graph_client)

        assert info.stale is False

    async def test_stale_never_indexed(self, git_project, graph_client):
        """Git project with no stored hash is stale."""
        await graph_client.ensure_schema()

        checker = StalenessChecker(git_project)
        info = await checker.check(graph_client)

        assert info.stale is True
        assert info.last_indexed_commit is None
        assert info.current_commit is not None
