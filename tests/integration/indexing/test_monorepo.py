"""Integration tests for monorepo indexing (require Memgraph + Valkey)."""

from __future__ import annotations

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
