"""Unit tests for monorepo support — detection, classification, and scope expansion."""

from __future__ import annotations

from pathlib import Path

from code_atlas.indexing.orchestrator import (
    DetectedProject,
    classify_file_project,
    detect_sub_projects,
)
from code_atlas.search.engine import expand_scope
from code_atlas.settings import MonorepoSettings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(root: Path, rel_path: str, content: str = "") -> Path:
    """Write a file at root/rel_path, creating parent dirs."""
    p = root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# detect_sub_projects — unit tests (no infrastructure)
# ---------------------------------------------------------------------------


class TestDetectSubProjects:
    def test_detects_pyproject_markers(self, tmp_path):
        """Auto-detects sub-projects with pyproject.toml markers."""
        _write(tmp_path, "services/auth/pyproject.toml", "")
        _write(tmp_path, "services/auth/auth/__init__.py", "")
        _write(tmp_path, "libs/shared/pyproject.toml", "")
        _write(tmp_path, "libs/shared/shared/__init__.py", "")

        settings = MonorepoSettings()
        result = detect_sub_projects(tmp_path, settings)

        names = {sp.name for sp in result}
        assert "auth" in names
        assert "shared" in names
        assert len(result) == 2

    def test_skips_root_marker(self, tmp_path):
        """Root-level pyproject.toml does NOT create a sub-project."""
        _write(tmp_path, "pyproject.toml", "")
        _write(tmp_path, "services/auth/pyproject.toml", "")

        settings = MonorepoSettings()
        result = detect_sub_projects(tmp_path, settings)

        # Only the nested one, not root
        assert len(result) == 1
        assert result[0].name == "auth"

    def test_explicit_overrides_auto(self, tmp_path):
        """Explicit entries take precedence over auto-detected at the same path."""
        _write(tmp_path, "services/auth/pyproject.toml", "")

        settings = MonorepoSettings(
            projects=[{"path": "services/auth", "name": "custom-auth"}],
        )
        result = detect_sub_projects(tmp_path, settings)

        assert len(result) == 1
        assert result[0].name == "custom-auth"
        assert result[0].marker == "explicit"

    def test_naming_collision_uses_full_path(self, tmp_path):
        """Two sub-projects with the same basename get dash-separated names."""
        _write(tmp_path, "services/api/pyproject.toml", "")
        _write(tmp_path, "tools/api/pyproject.toml", "")

        settings = MonorepoSettings()
        result = detect_sub_projects(tmp_path, settings)

        names = {sp.name for sp in result}
        assert "services-api" in names
        assert "tools-api" in names

    def test_auto_detect_disabled(self, tmp_path):
        """auto_detect=False skips marker scanning."""
        _write(tmp_path, "services/auth/pyproject.toml", "")
        _write(tmp_path, "libs/shared/pyproject.toml", "")

        settings = MonorepoSettings(auto_detect=False)
        result = detect_sub_projects(tmp_path, settings)

        assert result == []

    def test_auto_detect_disabled_with_explicit(self, tmp_path):
        """auto_detect=False still includes explicit entries."""
        _write(tmp_path, "services/auth/pyproject.toml", "")

        settings = MonorepoSettings(
            auto_detect=False,
            projects=[{"path": "services/auth", "name": "auth"}],
        )
        result = detect_sub_projects(tmp_path, settings)

        assert len(result) == 1
        assert result[0].name == "auth"

    def test_sorted_by_depth(self, tmp_path):
        """Results are sorted by path depth (shallow first)."""
        _write(tmp_path, "deep/nested/sub/pyproject.toml", "")
        _write(tmp_path, "top/pyproject.toml", "")
        _write(tmp_path, "mid/sub/pyproject.toml", "")

        settings = MonorepoSettings()
        result = detect_sub_projects(tmp_path, settings)

        depths = [sp.path.count("/") for sp in result]
        assert depths == sorted(depths)

    def test_skips_default_excludes(self, tmp_path):
        """Default excluded dirs (node_modules, .venv, etc.) are not scanned."""
        _write(tmp_path, "node_modules/pkg/pyproject.toml", "")
        _write(tmp_path, ".venv/pyproject.toml", "")
        _write(tmp_path, "services/auth/pyproject.toml", "")

        settings = MonorepoSettings()
        result = detect_sub_projects(tmp_path, settings)

        assert len(result) == 1
        assert result[0].name == "auth"

    def test_multiple_marker_types(self, tmp_path):
        """Detects sub-projects with different marker types."""
        _write(tmp_path, "py-service/pyproject.toml", "")
        _write(tmp_path, "js-app/package.json", "{}")
        _write(tmp_path, "go-lib/go.mod", "")

        settings = MonorepoSettings()
        result = detect_sub_projects(tmp_path, settings)

        names = {sp.name for sp in result}
        assert names == {"py-service", "js-app", "go-lib"}

    def test_no_sub_projects(self, tmp_path):
        """Empty directory returns empty list."""
        settings = MonorepoSettings()
        result = detect_sub_projects(tmp_path, settings)

        assert result == []


# ---------------------------------------------------------------------------
# classify_file_project — unit tests
# ---------------------------------------------------------------------------


class TestClassifyFileProject:
    def test_matches_longest_prefix(self):
        """Returns the most specific (longest prefix) matching sub-project."""
        sub_projects = [
            DetectedProject(name="services", path="services", root=Path("x"), marker="explicit"),
            DetectedProject(name="auth", path="services/auth", root=Path("x"), marker="pyproject.toml"),
        ]

        result = classify_file_project("services/auth/main.py", sub_projects)
        assert result is not None
        assert result.name == "auth"
        result = classify_file_project("services/other.py", sub_projects)
        assert result is not None
        assert result.name == "services"

    def test_no_match_returns_none(self):
        """Files outside any sub-project return None."""
        sub_projects = [
            DetectedProject(name="auth", path="services/auth", root=Path("x"), marker="pyproject.toml"),
        ]

        assert classify_file_project("root_file.py", sub_projects) is None
        assert classify_file_project("other/file.py", sub_projects) is None

    def test_exact_path_match(self):
        """File path exactly matching a sub-project path is classified."""
        sub_projects = [
            DetectedProject(name="auth", path="services/auth", root=Path("x"), marker="pyproject.toml"),
        ]

        # This shouldn't normally happen (paths are to files), but test robustness
        result = classify_file_project("services/auth", sub_projects)
        assert result is not None
        assert result.name == "auth"

    def test_empty_sub_projects(self):
        """No sub-projects means all files are unclassified."""
        assert classify_file_project("any/file.py", []) is None


# ---------------------------------------------------------------------------
# expand_scope — unit tests
# ---------------------------------------------------------------------------


class TestExpandScope:
    def test_empty_scope_returns_none(self):
        """Empty scope -> None (no filter)."""
        assert expand_scope("", ["a", "b"]) is None

    def test_single_name(self):
        """Single project name -> list with that name."""
        result = expand_scope("auth", ["auth", "shared"])
        assert result == ["auth"]

    def test_single_name_with_always_include(self):
        """Single project name + always_include."""
        result = expand_scope("auth", ["auth", "shared"], always_include=["shared"])
        assert result == ["auth", "shared"]

    def test_glob_pattern(self):
        """Glob pattern expands to matching projects."""
        result = expand_scope("services-*", ["services-auth", "services-api", "libs-shared"])
        assert result is not None
        assert set(result) == {"services-auth", "services-api"}

    def test_comma_separated(self):
        """Comma-separated scope expands to multiple projects."""
        result = expand_scope("auth,shared", ["auth", "shared", "other"])
        assert result == ["auth", "shared"]

    def test_comma_separated_with_always_include(self):
        """Comma-separated + always_include."""
        result = expand_scope("auth", ["auth", "shared", "other"], always_include=["shared"])
        assert result == ["auth", "shared"]

    def test_deduplication(self):
        """Duplicate names are removed."""
        result = expand_scope("auth,auth", ["auth", "shared"])
        assert result == ["auth"]

    def test_always_include_dedup(self):
        """always_include doesn't duplicate existing entries."""
        result = expand_scope("shared", ["auth", "shared"], always_include=["shared"])
        assert result == ["shared"]

    def test_no_match_returns_none(self):
        """No matching projects -> None."""
        result = expand_scope("nonexistent", ["auth", "shared"])
        # Returns the requested name even if not in all_projects
        assert result == ["nonexistent"]
