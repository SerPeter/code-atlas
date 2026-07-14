"""Tests for CLI output modes (--quiet, --json, --verbose, --no-color)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from code_atlas.cli import _output, app

runner = CliRunner()


def _reset_output() -> None:
    """Reset the global _output singleton to defaults between tests."""
    _output.quiet = False
    _output.json = False
    _output.verbose = 0
    _output.no_color = False


# ---------------------------------------------------------------------------
# Helpers to mock async health checks
# ---------------------------------------------------------------------------


def _mock_health_report(*, ok: bool = True):
    """Return a mock HealthReport with one check."""
    from code_atlas.server.health import CheckResult, CheckStatus, HealthReport

    return HealthReport(
        checks=[
            CheckResult(
                name="memgraph",
                status=CheckStatus.OK if ok else CheckStatus.FAIL,
                message="Connected" if ok else "Unreachable",
            )
        ],
        elapsed_ms=42.0,
    )


def _patch_health(report):
    """Patch run_health_checks where it's defined (code_atlas.health)."""
    return patch("code_atlas.server.health.run_health_checks", new_callable=AsyncMock, return_value=report)


def _patch_status(mock_graph):
    """Patch GraphClient and AtlasSettings at the modules where _run_status imports them."""
    return (
        patch("code_atlas.graph.client.GraphClient", return_value=mock_graph),
        patch("code_atlas.settings.AtlasSettings", return_value=AsyncMock()),
    )


# ---------------------------------------------------------------------------
# --json flag tests
# ---------------------------------------------------------------------------


class TestJsonHealth:
    def test_json_health_outputs_valid_json(self) -> None:
        _reset_output()
        report = _mock_health_report(ok=True)
        with _patch_health(report):
            result = runner.invoke(app, ["--json", "health"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["ok"] is True
        assert isinstance(payload["checks"], list)
        assert payload["checks"][0]["name"] == "memgraph"
        assert "elapsed_ms" in payload

    def test_json_health_fail_exit_code(self) -> None:
        _reset_output()
        report = _mock_health_report(ok=False)
        with _patch_health(report):
            result = runner.invoke(app, ["--json", "health"])

        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["ok"] is False

    def test_json_doctor_outputs_valid_json(self) -> None:
        _reset_output()
        report = _mock_health_report(ok=True)
        with _patch_health(report):
            result = runner.invoke(app, ["--json", "doctor"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["ok"] is True
        assert payload["checks"][0]["status"] == "ok"


class TestJsonStatus:
    def _make_mock_graph(self, projects, deps=None):
        mock_graph = AsyncMock()
        mock_graph.ping = AsyncMock()
        mock_graph.get_project_status = AsyncMock(return_value=projects)
        mock_graph.execute = AsyncMock(return_value=deps or [])
        mock_graph.close = AsyncMock()
        return mock_graph

    def test_json_status_outputs_valid_json(self) -> None:
        _reset_output()
        mock_projects = [
            {
                "n": {
                    "name": "myproject",
                    "last_indexed_at": 1700000000,
                    "file_count": 10,
                    "entity_count": 50,
                    "git_hash": "abc123",
                }
            }
        ]
        mock_graph = self._make_mock_graph(mock_projects)

        with (
            patch("code_atlas.graph.client.GraphClient", return_value=mock_graph),
            patch("code_atlas.settings.AtlasSettings"),
        ):
            result = runner.invoke(app, ["--json", "status"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "projects" in payload
        assert payload["projects"][0]["name"] == "myproject"
        assert payload["projects"][0]["file_count"] == 10

    def test_json_status_empty(self) -> None:
        _reset_output()
        mock_graph = self._make_mock_graph([])

        with (
            patch("code_atlas.graph.client.GraphClient", return_value=mock_graph),
            patch("code_atlas.settings.AtlasSettings"),
        ):
            result = runner.invoke(app, ["--json", "status"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["projects"] == []


# ---------------------------------------------------------------------------
# --quiet flag tests
# ---------------------------------------------------------------------------


class TestQuiet:
    def test_quiet_suppresses_info(self) -> None:
        _reset_output()
        report = _mock_health_report(ok=True)
        with _patch_health(report):
            result = runner.invoke(app, ["--quiet", "health"])

        assert result.exit_code == 0
        # In quiet mode, INFO-level loguru output is suppressed
        # stdout should be empty (no JSON output either)
        assert result.output.strip() == ""

    def test_quiet_via_env_var(self) -> None:
        _reset_output()
        report = _mock_health_report(ok=True)
        with _patch_health(report):
            result = runner.invoke(app, ["health"], env={"ATLAS_QUIET": "1"})

        assert result.exit_code == 0
        assert result.output.strip() == ""


# ---------------------------------------------------------------------------
# --verbose flag tests
# ---------------------------------------------------------------------------


class TestVerbose:
    def test_verbose_sets_output_mode(self) -> None:
        _reset_output()
        report = _mock_health_report(ok=True)
        with _patch_health(report):
            runner.invoke(app, ["-v", "health"])

        assert _output.verbose >= 1

    def test_double_verbose(self) -> None:
        _reset_output()
        report = _mock_health_report(ok=True)
        with _patch_health(report):
            runner.invoke(app, ["-v", "-v", "health"])

        assert _output.verbose >= 2


# ---------------------------------------------------------------------------
# --no-color flag tests
# ---------------------------------------------------------------------------


class TestMonorepoScopeDispatch:
    """--scope (or the auto-derived subdirectory scope) must not be silently
    discarded when monorepo mode kicks in — regression for the "indexes the
    entire monorepo" finding: monorepo mode used to ignore ``scope`` entirely
    and always index every detected sub-project.
    """

    async def _patch_common(self, monkeypatch, sub_projects) -> dict:
        from code_atlas import cli

        captured: dict = {}

        class FakeBus:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def ping(self) -> None:
                return None

            async def close(self) -> None:
                return None

        class FakeGraph:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def ping(self) -> None:
                return None

            async def ensure_schema(self) -> None:
                return None

            async def close(self) -> None:
                return None

        async def fake_monorepo_with_progress(settings, graph, bus, *, projects, full_reindex):
            captured["dispatch"] = "monorepo"
            captured["projects"] = projects
            return []

        async def fake_single_with_spinner(settings, graph, bus, *, scope, full_reindex):
            from code_atlas.indexing.orchestrator import IndexResult

            captured["dispatch"] = "single"
            captured["scope"] = scope
            return IndexResult(files_scanned=0, files_published=0, entities_total=0, duration_s=0.0)

        monkeypatch.setattr("code_atlas.events.EventBus", FakeBus)
        monkeypatch.setattr("code_atlas.graph.client.GraphClient", FakeGraph)
        monkeypatch.setattr("code_atlas.indexing.orchestrator.detect_sub_projects", lambda root, mono: sub_projects)
        monkeypatch.setattr(cli, "_index_monorepo_with_progress", fake_monorepo_with_progress)
        monkeypatch.setattr(cli, "_index_single_with_spinner", fake_single_with_spinner)
        return captured

    async def test_scope_matching_subproject_narrows_instead_of_full_repo(self, tmp_path, monkeypatch) -> None:
        from code_atlas import cli
        from code_atlas.indexing.orchestrator import DetectedProject

        _reset_output()
        sub_projects = [
            DetectedProject(name="foo", path="packages/foo", root=tmp_path / "packages" / "foo", marker="x"),
            DetectedProject(name="bar", path="packages/bar", root=tmp_path / "packages" / "bar", marker="x"),
        ]
        captured = await self._patch_common(monkeypatch, sub_projects)

        await cli._run_index(str(tmp_path), ["packages/foo"], False, no_embed=True, no_git_check=True)

        assert captured["dispatch"] == "monorepo"
        assert captured["projects"] == ["foo"]

    async def test_scope_outside_any_subproject_falls_back_to_single_project(self, tmp_path, monkeypatch) -> None:
        from code_atlas import cli
        from code_atlas.indexing.orchestrator import DetectedProject

        _reset_output()
        sub_projects = [
            DetectedProject(name="foo", path="packages/foo", root=tmp_path / "packages" / "foo", marker="x"),
        ]
        captured = await self._patch_common(monkeypatch, sub_projects)

        await cli._run_index(str(tmp_path), ["docs"], False, no_embed=True, no_git_check=True)

        assert captured["dispatch"] == "single"
        assert captured["scope"] == ["docs"]

    async def test_explicit_project_flag_without_scope_is_unaffected(self, tmp_path, monkeypatch) -> None:
        """Regression guard: --project alone (no --scope) keeps working as before."""
        from code_atlas import cli
        from code_atlas.indexing.orchestrator import DetectedProject

        _reset_output()
        sub_projects = [
            DetectedProject(name="foo", path="packages/foo", root=tmp_path / "packages" / "foo", marker="x"),
            DetectedProject(name="bar", path="packages/bar", root=tmp_path / "packages" / "bar", marker="x"),
        ]
        captured = await self._patch_common(monkeypatch, sub_projects)

        await cli._run_index(str(tmp_path), None, False, projects=["foo"], no_embed=True, no_git_check=True)

        assert captured["dispatch"] == "monorepo"
        assert captured["projects"] == ["foo"]


class TestNoColor:
    def test_no_color_sets_flag(self) -> None:
        _reset_output()
        report = _mock_health_report(ok=True)
        with _patch_health(report):
            runner.invoke(app, ["--no-color", "health"])

        assert _output.no_color is True

    def test_no_color_via_env_var(self) -> None:
        _reset_output()
        report = _mock_health_report(ok=True)
        with _patch_health(report):
            runner.invoke(app, ["health"], env={"NO_COLOR": "1"})

        assert _output.no_color is True


class TestDreamCommand:
    """`atlas dream` builds the report, writes HOME.md, and reports via the graph client."""

    async def test_dream_writes_home_and_closes_graph(self, tmp_path, monkeypatch) -> None:
        from code_atlas import cli
        from code_atlas.dream import DreamReport
        from code_atlas.settings import AtlasSettings

        _reset_output()
        settings = AtlasSettings(project_root=tmp_path)
        mock_graph = AsyncMock()
        empty_report = DreamReport(
            inbox_count=0,
            inbox_paths=[],
            orphan_notes=[],
            duplicate_ids=[],
            dangling_links=[],
            similar_pairs=[],
            promotion_candidates=[],
            memory_index_issues=[],
        )

        monkeypatch.setattr(cli, "_load_settings", lambda: settings)
        monkeypatch.setattr("code_atlas.graph.client.GraphClient", lambda s: mock_graph)
        monkeypatch.setattr("code_atlas.dream.build_dream_report", AsyncMock(return_value=empty_report))

        await cli._run_dream()

        home = tmp_path / "docs" / "HOME.md"
        assert home.is_file()
        assert "Knowledge Vault" in home.read_text(encoding="utf-8")
        mock_graph.ping.assert_awaited_once()
        mock_graph.close.assert_awaited_once()


class TestProjectRm:
    """`atlas project rm` deletes a project's graph data, with a confirmation gate."""

    def _mock_graph(self, *, found: bool = True):
        mock_graph = AsyncMock()
        mock_graph.ping = AsyncMock()
        mock_graph.get_project_status = AsyncMock(return_value=[{"n": {"name": "myproject"}}] if found else [])
        mock_graph.delete_project_data = AsyncMock()
        mock_graph.close = AsyncMock()
        return mock_graph

    async def test_yes_flag_deletes_without_prompt(self, tmp_path, monkeypatch) -> None:
        from code_atlas import cli
        from code_atlas.settings import AtlasSettings

        _reset_output()
        settings = AtlasSettings(project_root=tmp_path)
        mock_graph = self._mock_graph()

        monkeypatch.setattr(cli, "_load_settings", lambda: settings)
        monkeypatch.setattr("code_atlas.graph.client.GraphClient", lambda s: mock_graph)

        await cli._run_project_rm("myproject", skip_confirm=True)

        mock_graph.delete_project_data.assert_awaited_once_with("myproject")
        mock_graph.close.assert_awaited_once()

    async def test_missing_project_exits_with_error(self, tmp_path, monkeypatch) -> None:
        import pytest
        import typer

        from code_atlas import cli
        from code_atlas.settings import AtlasSettings

        _reset_output()
        settings = AtlasSettings(project_root=tmp_path)
        mock_graph = self._mock_graph(found=False)

        monkeypatch.setattr(cli, "_load_settings", lambda: settings)
        monkeypatch.setattr("code_atlas.graph.client.GraphClient", lambda s: mock_graph)

        with pytest.raises(typer.Exit):
            await cli._run_project_rm("ghost", skip_confirm=True)

        mock_graph.delete_project_data.assert_not_awaited()

    def test_confirmation_prompt_aborts_on_no(self, tmp_path, monkeypatch) -> None:
        from code_atlas import cli
        from code_atlas.settings import AtlasSettings

        _reset_output()
        settings = AtlasSettings(project_root=tmp_path)
        mock_graph = self._mock_graph()

        monkeypatch.setattr(cli, "_load_settings", lambda: settings)
        monkeypatch.setattr("code_atlas.graph.client.GraphClient", lambda s: mock_graph)

        result = runner.invoke(app, ["project", "rm", "myproject"], input="n\n")

        assert result.exit_code == 1
        mock_graph.delete_project_data.assert_not_awaited()

    def test_confirmation_prompt_deletes_on_yes(self, tmp_path, monkeypatch) -> None:
        from code_atlas import cli
        from code_atlas.settings import AtlasSettings

        _reset_output()
        settings = AtlasSettings(project_root=tmp_path)
        mock_graph = self._mock_graph()

        monkeypatch.setattr(cli, "_load_settings", lambda: settings)
        monkeypatch.setattr("code_atlas.graph.client.GraphClient", lambda s: mock_graph)

        result = runner.invoke(app, ["project", "rm", "myproject"], input="y\n")

        assert result.exit_code == 0
        mock_graph.delete_project_data.assert_awaited_once_with("myproject")
