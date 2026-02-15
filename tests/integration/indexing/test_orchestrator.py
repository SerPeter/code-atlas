"""Integration tests for the indexer/orchestrator module (require Memgraph + Valkey)."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from code_atlas.indexing.orchestrator import StalenessChecker, index_project
from code_atlas.schema import NodeLabel
from code_atlas.settings import AtlasSettings, IndexSettings
from tests.conftest import NO_EMBED, TEST_DRAIN_TIMEOUT_S

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(root: Path, rel_path: str, content: str = "") -> Path:
    """Write a file at root/rel_path, creating parent dirs."""
    p = root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


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


# ---------------------------------------------------------------------------
# Integration tests (require Memgraph + Valkey)
# ---------------------------------------------------------------------------


class TestIndexProjectIntegration:
    @pytest.fixture
    def project_dir(self, tmp_path):
        """Create a minimal Python project for indexing."""
        _write(tmp_path, "src/__init__.py", "")
        _write(tmp_path, "src/app.py", 'def hello():\n    """Say hello."""\n    return "hello"\n')
        _write(tmp_path, "src/utils.py", "MAGIC = 42\n\ndef add(a, b):\n    return a + b\n")
        return tmp_path

    async def test_index_project_creates_graph(self, project_dir, graph_client, event_bus):
        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        result = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert result.files_scanned >= 2
        assert result.entities_total > 0
        assert result.duration_s > 0

    async def test_index_project_creates_hierarchy(self, project_dir, graph_client, event_bus):
        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # Project node exists
        projects = await graph_client.execute(f"MATCH (p:{NodeLabel.PROJECT}) RETURN p.name AS name")
        assert len(projects) >= 1

        # Package node for src/ exists
        packages = await graph_client.execute(
            f"MATCH (p:{NodeLabel.PACKAGE} {{project_name: $pn}}) RETURN p.qualified_name AS qn",
            {"pn": project_dir.name},
        )
        assert any(p["qn"] == "src" for p in packages)

        # CONTAINS edge exists: Project -> Package
        contains = await graph_client.execute("MATCH (a)-[:CONTAINS]->(b) RETURN a.uid AS from_uid, b.uid AS to_uid")
        assert len(contains) > 0

    async def test_index_project_full_reindex(self, project_dir, graph_client, event_bus):
        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        # First index
        r1 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r1.entities_total > 0

        # Full reindex — should work cleanly
        r2 = await index_project(
            settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S
        )
        assert r2.entities_total > 0

    async def test_index_project_error_resilience(self, tmp_path, graph_client, event_bus):
        """One unparseable file shouldn't abort the whole indexing run."""
        _write(tmp_path, "good.py", "x = 1\n")
        _write(tmp_path, "bad.py", "def (\n")  # syntax error — tree-sitter handles gracefully

        settings = AtlasSettings(project_root=tmp_path, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        result = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # Should still complete and index the good file
        assert result.files_scanned >= 2
        assert result.entities_total > 0


# ---------------------------------------------------------------------------
# Delta indexing — integration tests (require Memgraph + Valkey + git)
# ---------------------------------------------------------------------------


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

    async def test_delta_index_mode(self, git_project, graph_client, event_bus):
        """Re-indexing without changes uses delta mode."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        # First index — full mode
        r1 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r1.mode == "full"
        assert r1.entities_total > 0

        # Re-index without changes — delta mode, 0 published
        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r2.mode == "delta"
        assert r2.files_published == 0
        assert r2.entities_total == r1.entities_total

    async def test_delta_index_publishes_only_changed(self, git_project, graph_client, event_bus):
        """Modifying one file only publishes that file in delta mode."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        # First index
        r1 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r1.mode == "full"

        # Modify one file and commit
        _write(git_project, "src/app.py", 'def hello():\n    """Say hello!""""\n    return "hello world"\n')
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "modify app")

        # Delta re-index
        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r2.mode == "delta"
        assert r2.delta_stats is not None
        assert r2.delta_stats.files_modified >= 1
        assert r2.files_published >= 1

    async def test_delta_index_detects_new_files(self, git_project, graph_client, event_bus):
        """New files are picked up in delta mode."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        # First index
        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # Add new file and commit
        _write(git_project, "src/new_module.py", "NEW_CONST = 99\n")
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "add new module")

        # Delta re-index
        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r2.mode == "delta"
        assert r2.delta_stats is not None
        assert r2.delta_stats.files_added >= 1

    async def test_delta_index_handles_deletion(self, git_project, graph_client, event_bus):
        """Deleted files' entities are removed from the graph."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        # First index
        r1 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        e1 = r1.entities_total

        # Delete a file and commit
        (git_project / "src" / "utils.py").unlink()
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "remove utils")

        # Delta re-index
        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r2.mode == "delta"
        assert r2.delta_stats is not None
        assert r2.delta_stats.files_deleted >= 1
        # Entity count should have dropped
        assert r2.entities_total < e1

    async def test_delta_index_full_fallback_on_threshold(self, git_project, graph_client, event_bus):
        """Exceeding the threshold triggers full mode."""
        # Set a very low threshold so any change exceeds it
        settings = AtlasSettings(
            project_root=git_project, index=IndexSettings(delta_threshold=0.0), embeddings=NO_EMBED
        )
        await graph_client.ensure_schema()

        # First index
        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # Modify a file and commit
        _write(git_project, "src/app.py", 'def hello():\n    return "changed"\n')
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "change")

        # Re-index with threshold=0.0 — should fall back to full
        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r2.mode == "full"

    async def test_delta_index_preserves_unchanged(self, git_project, graph_client, event_bus):
        """Unchanged entities keep their entity count after delta re-index."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        # First index
        r1 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # Re-index without changes
        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r2.mode == "delta"
        assert r2.entities_total == r1.entities_total

    async def test_full_reindex_flag_overrides_delta(self, git_project, graph_client, event_bus):
        """--full flag forces full mode even when delta is available."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        r2 = await index_project(
            settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S
        )
        assert r2.mode == "full"


# ---------------------------------------------------------------------------
# Staleness check — integration tests (require Memgraph + Valkey + git)
# ---------------------------------------------------------------------------


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

    async def test_not_stale_when_hashes_match(self, git_project, graph_client, event_bus):
        """After indexing, the checker reports not stale."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        checker = StalenessChecker(git_project)
        info = await checker.check(graph_client)

        assert info.stale is False
        assert info.current_commit is not None
        assert info.last_indexed_commit is not None

    async def test_stale_when_new_commit(self, git_project, graph_client, event_bus):
        """A new commit after indexing makes the checker report stale."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

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
