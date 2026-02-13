"""Tests for CLI output modes (--quiet, --json, --verbose, --no-color)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from code_atlas.cli import OutputMode, _output, app

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
    from code_atlas.health import CheckResult, CheckStatus, HealthReport

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
    return patch("code_atlas.health.run_health_checks", new_callable=AsyncMock, return_value=report)


def _patch_status(mock_graph):
    """Patch GraphClient and AtlasSettings at the modules where _run_status imports them."""
    return (
        patch("code_atlas.graph.GraphClient", return_value=mock_graph),
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
            patch("code_atlas.graph.GraphClient", return_value=mock_graph),
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
            patch("code_atlas.graph.GraphClient", return_value=mock_graph),
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


# ---------------------------------------------------------------------------
# OutputMode dataclass
# ---------------------------------------------------------------------------


class TestOutputMode:
    def test_defaults(self) -> None:
        mode = OutputMode()
        assert mode.quiet is False
        assert mode.json is False
        assert mode.verbose == 0
        assert mode.no_color is False
