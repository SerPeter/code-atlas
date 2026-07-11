"""Integration tests for monorepo indexing (require Memgraph + Valkey)."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

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


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIndexMonorepoIntegration:
    @pytest.fixture
    def monorepo_dir(self, tmp_path):
        """Create a minimal monorepo with two sub-projects that import each other."""
        # services/auth
        _write(tmp_path, "services/auth/pyproject.toml", '[project]\nname = "auth"\n')
        _write(tmp_path, "services/auth/auth/__init__.py", "")
        _write(
            tmp_path,
            "services/auth/auth/service.py",
            "from shared import utils\n\ndef authenticate():\n    return utils.validate()\n",
        )

        # libs/shared
        _write(tmp_path, "libs/shared/pyproject.toml", '[project]\nname = "shared"\n')
        _write(tmp_path, "libs/shared/shared/__init__.py", "")
        _write(
            tmp_path,
            "libs/shared/shared/utils.py",
            "def validate():\n    return True\n",
        )

        # Root-level file (not in any sub-project)
        _write(tmp_path, "README.md", "# Monorepo\n")

        return tmp_path

    async def test_detect_and_index_monorepo(self, monorepo_dir, graph_client, event_bus):
        """Full monorepo index: detects sub-projects, creates Project nodes, indexes entities."""
        from code_atlas.indexing.orchestrator import index_monorepo
        from code_atlas.schema import NodeLabel
        from code_atlas.settings import AtlasSettings

        settings = AtlasSettings(project_root=monorepo_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        results = await index_monorepo(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # Should have indexed at least the two sub-projects
        assert len(results) >= 2

        # Check Project nodes exist with prefixed names
        root_name = monorepo_dir.resolve().name
        projects = await graph_client.execute(f"MATCH (p:{NodeLabel.PROJECT}) RETURN p.name AS name")
        project_names = {p["name"] for p in projects}
        assert f"{root_name}/auth" in project_names
        assert f"{root_name}/shared" in project_names

    async def test_scoped_monorepo_index(self, monorepo_dir, graph_client, event_bus):
        """Scoping to specific sub-projects only indexes those."""
        from code_atlas.indexing.orchestrator import index_monorepo
        from code_atlas.schema import NodeLabel
        from code_atlas.settings import AtlasSettings

        settings = AtlasSettings(project_root=monorepo_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        await index_monorepo(
            settings, graph_client, event_bus, scope_projects=["auth"], drain_timeout_s=TEST_DRAIN_TIMEOUT_S
        )

        # Should only have indexed auth (prefixed) + possibly root
        root_name = monorepo_dir.resolve().name
        projects = await graph_client.execute(f"MATCH (p:{NodeLabel.PROJECT}) RETURN p.name AS name")
        project_names = {p["name"] for p in projects}
        assert f"{root_name}/auth" in project_names

    async def test_delta_reindex_picks_up_modified_sub_project_file(self, monorepo_dir, graph_client, event_bus):
        """S4: a sub-project's delta run must see git-diff paths relative to ITS root.

        Before the ``git diff --relative`` fix, git printed monorepo-root-relative
        paths that never intersected the sub-project's file set: the delta run
        published 0 events (delta_files_modified == 0), git_hash still advanced,
        and the modified entity never reached the graph.
        """
        from code_atlas.indexing.orchestrator import index_monorepo
        from code_atlas.schema import NodeLabel
        from code_atlas.settings import AtlasSettings

        # Pad libs/shared so 1 modified file / 6 files stays under delta_threshold=0.3
        for i in range(4):
            _write(monorepo_dir, f"libs/shared/shared/pad_{i}.py", f"PAD_{i} = {i}\n")

        _git(monorepo_dir, "init")
        _git(monorepo_dir, "config", "user.email", "test@test.com")
        _git(monorepo_dir, "config", "user.name", "Test")
        _git(monorepo_dir, "add", ".")
        _git(monorepo_dir, "commit", "-m", "initial")

        settings = AtlasSettings(project_root=monorepo_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        await index_monorepo(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # Modify one file in the shared sub-project and commit
        _write(monorepo_dir, "libs/shared/shared/utils.py", "def validate_v2():\n    return False\n")
        _git(monorepo_dir, "add", ".")
        _git(monorepo_dir, "commit", "-m", "modify shared utils")

        await index_monorepo(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        root_name = monorepo_dir.resolve().name
        shared = f"{root_name}/shared"
        projects = await graph_client.execute(
            f"MATCH (p:{NodeLabel.PROJECT} {{name: $n}}) RETURN p.index_mode AS mode, p.delta_files_modified AS dfm",
            {"n": shared},
        )
        assert len(projects) == 1
        assert projects[0]["mode"] == "delta"
        assert projects[0]["dfm"] == 1

        callables = await graph_client.execute(
            f"MATCH (c:{NodeLabel.CALLABLE} {{project_name: $p, name: 'validate_v2'}}) RETURN c.uid AS uid",
            {"p": shared},
        )
        assert len(callables) == 1
        assert callables[0]["uid"].startswith(f"{shared}:")
