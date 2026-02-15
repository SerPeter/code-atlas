"""Unit tests for the indexer/orchestrator module (no infrastructure needed)."""

from __future__ import annotations

import os
import subprocess
import time
from typing import TYPE_CHECKING

import pytest

from code_atlas.indexing.orchestrator import (
    FileScope,
    StalenessChecker,
    _git_changed_files,
    _read_git_head,
    scan_files,
)
from code_atlas.settings import AtlasSettings, IndexSettings, ScopeSettings, derive_project_name, resolve_git_dir

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
# resolve_git_dir — unit tests (no infrastructure)
# ---------------------------------------------------------------------------


class TestResolveGitDir:
    def test_normal_repo(self, tmp_path):
        """Normal repo with .git directory returns it."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        assert resolve_git_dir(tmp_path) == git_dir

    def test_worktree(self, tmp_path):
        """.git file with gitdir: pointer follows to the real git dir."""
        real_git_dir = tmp_path / "main-repo" / ".git" / "worktrees" / "feature"
        real_git_dir.mkdir(parents=True)
        worktree = tmp_path / "feature"
        worktree.mkdir()
        (worktree / ".git").write_text(f"gitdir: {real_git_dir}\n", encoding="utf-8")

        result = resolve_git_dir(worktree)
        assert result == real_git_dir

    def test_relative_gitdir(self, tmp_path):
        """Relative gitdir: path is resolved against project_root."""
        real_git_dir = tmp_path / "main" / ".git" / "worktrees" / "wt"
        real_git_dir.mkdir(parents=True)
        worktree = tmp_path / "wt"
        worktree.mkdir()
        # Write a relative path from worktree dir to real git dir
        rel = os.path.relpath(real_git_dir, worktree).replace("\\", "/")
        (worktree / ".git").write_text(f"gitdir: {rel}\n", encoding="utf-8")

        result = resolve_git_dir(worktree)
        assert result is not None
        assert result.resolve() == real_git_dir.resolve()

    def test_no_git(self, tmp_path):
        """No .git file or directory returns None."""
        assert resolve_git_dir(tmp_path) is None


# ---------------------------------------------------------------------------
# derive_project_name — unit tests (no infrastructure)
# ---------------------------------------------------------------------------


class TestDeriveProjectName:
    def test_normal_repo(self, tmp_path):
        """Normal repo returns directory basename."""
        (tmp_path / ".git").mkdir()
        assert derive_project_name(tmp_path) == tmp_path.resolve().name

    def test_worktree_with_branch(self, tmp_path):
        """Worktree returns basename@branch."""
        real_git_dir = tmp_path / "main-repo" / ".git" / "worktrees" / "feat-login"
        real_git_dir.mkdir(parents=True)
        (real_git_dir / "HEAD").write_text("ref: refs/heads/feat/login\n", encoding="utf-8")

        worktree = tmp_path / "myapp"
        worktree.mkdir()
        (worktree / ".git").write_text(f"gitdir: {real_git_dir}\n", encoding="utf-8")

        assert derive_project_name(worktree) == "myapp@feat/login"

    def test_detached_worktree_fallback(self, tmp_path):
        """Detached HEAD worktree uses git dir basename as fallback."""
        real_git_dir = tmp_path / "main-repo" / ".git" / "worktrees" / "hotfix-42"
        real_git_dir.mkdir(parents=True)
        detached_hash = "a" * 40
        (real_git_dir / "HEAD").write_text(detached_hash + "\n", encoding="utf-8")

        worktree = tmp_path / "myapp"
        worktree.mkdir()
        (worktree / ".git").write_text(f"gitdir: {real_git_dir}\n", encoding="utf-8")

        assert derive_project_name(worktree) == "myapp@hotfix-42"

    def test_non_git_directory(self, tmp_path):
        """Non-git directory returns plain basename."""
        assert derive_project_name(tmp_path) == tmp_path.resolve().name


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

    def test_reads_head_in_worktree(self, tmp_path):
        """_read_git_head follows .git file -> real git dir."""
        real_git_dir = tmp_path / "main-repo" / ".git" / "worktrees" / "wt"
        real_git_dir.mkdir(parents=True)
        fake_hash = "f" * 40
        (real_git_dir / "HEAD").write_text(fake_hash + "\n", encoding="utf-8")

        worktree = tmp_path / "wt"
        worktree.mkdir()
        (worktree / ".git").write_text(f"gitdir: {real_git_dir}\n", encoding="utf-8")

        assert _read_git_head(worktree) == fake_hash


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
        """project_name is derived via derive_project_name()."""
        checker = StalenessChecker(tmp_path)
        assert checker.project_name == derive_project_name(tmp_path)

    def test_project_name_worktree(self, tmp_path):
        """project_name includes worktree branch suffix."""
        real_git_dir = tmp_path / "main" / ".git" / "worktrees" / "dev"
        real_git_dir.mkdir(parents=True)
        (real_git_dir / "HEAD").write_text("ref: refs/heads/dev\n", encoding="utf-8")
        refs_dir = real_git_dir / "refs" / "heads"
        refs_dir.mkdir(parents=True)
        (refs_dir / "dev").write_text("a" * 40 + "\n", encoding="utf-8")

        worktree = tmp_path / "myapp"
        worktree.mkdir()
        (worktree / ".git").write_text(f"gitdir: {real_git_dir}\n", encoding="utf-8")

        checker = StalenessChecker(worktree)
        assert checker.project_name == "myapp@dev"


# ---------------------------------------------------------------------------
# stale_mode setting — unit test
# ---------------------------------------------------------------------------


class TestStaleMode:
    def test_stale_mode_default(self):
        """Default stale_mode is 'warn'."""
        settings = IndexSettings()
        assert settings.stale_mode == "warn"
