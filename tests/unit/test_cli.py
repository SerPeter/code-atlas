"""Tests for CLI output modes (--quiet, --json, --verbose, --no-color)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

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
        mock_graph.get_project_dependency_edges = AsyncMock(return_value=deps or [])
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
            patch("code_atlas.backends.GraphClient", return_value=mock_graph),
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
            patch("code_atlas.backends.GraphClient", return_value=mock_graph),
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

            # atlas index takes an exclusive lease so it cannot run alongside a daemon.
            async def acquire_indexer_lease(self, owner: str, ttl_ms: int) -> bool:
                return True

            async def renew_indexer_lease(self, owner: str, ttl_ms: int) -> bool:
                return True

            async def release_indexer_lease(self, owner: str) -> bool:
                return True

            async def read_indexer_lease(self) -> str | None:
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

        monkeypatch.setattr("code_atlas.backends.EventBus", FakeBus)
        monkeypatch.setattr("code_atlas.backends.GraphClient", FakeGraph)
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


class TestIndexWithGitSignals:
    """--with-git-signals reuses mine_git_signals/write_git_signals after indexing succeeds."""

    async def _patch_common(self, monkeypatch) -> dict:
        from code_atlas import cli
        from code_atlas.indexing.git_signals import GitSignalsResult
        from code_atlas.indexing.orchestrator import IndexResult

        captured: dict = {"order": []}

        class FakeBus:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def ping(self) -> None:
                return None

            async def close(self) -> None:
                return None

            # atlas index takes an exclusive lease so it cannot run alongside a daemon.
            async def acquire_indexer_lease(self, owner: str, ttl_ms: int) -> bool:
                return True

            async def renew_indexer_lease(self, owner: str, ttl_ms: int) -> bool:
                return True

            async def release_indexer_lease(self, owner: str) -> bool:
                return True

            async def read_indexer_lease(self) -> str | None:
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

        async def fake_single_with_spinner(settings, graph, bus, *, scope, full_reindex):
            captured["order"].append("index")
            return IndexResult(files_scanned=1, files_published=1, entities_total=2, duration_s=0.1)

        mine_mock = MagicMock(return_value=GitSignalsResult(file_signals=(), co_change_pairs=(), commits_scanned=5))

        async def fake_write_git_signals(graph, project_name, result):
            captured["order"].append("write")
            return {
                "commits_scanned": 5,
                "files_mined": 0,
                "files_matched": 0,
                "co_change_pairs_mined": 0,
                "co_change_edges": 0,
            }

        write_mock = AsyncMock(side_effect=fake_write_git_signals)

        monkeypatch.setattr("code_atlas.backends.EventBus", FakeBus)
        monkeypatch.setattr("code_atlas.backends.GraphClient", FakeGraph)
        monkeypatch.setattr("code_atlas.indexing.orchestrator.detect_sub_projects", lambda root, mono: [])
        monkeypatch.setattr(cli, "_index_single_with_spinner", fake_single_with_spinner)
        monkeypatch.setattr("code_atlas.indexing.git_signals.mine_git_signals", mine_mock)
        monkeypatch.setattr("code_atlas.indexing.git_signals.write_git_signals", write_mock)

        captured["mine_mock"] = mine_mock
        captured["write_mock"] = write_mock
        return captured

    async def test_flag_off_skips_git_signals_mining(self, tmp_path, monkeypatch) -> None:
        from code_atlas import cli

        _reset_output()
        captured = await self._patch_common(monkeypatch)

        await cli._run_index(str(tmp_path), None, False, no_embed=True, no_git_check=True)

        captured["mine_mock"].assert_not_called()
        captured["write_mock"].assert_not_awaited()

    async def test_flag_on_mines_and_writes_git_signals_after_indexing(self, tmp_path, monkeypatch) -> None:
        from code_atlas import cli

        _reset_output()
        captured = await self._patch_common(monkeypatch)

        await cli._run_index(
            str(tmp_path),
            None,
            False,
            no_embed=True,
            no_git_check=True,
            with_git_signals=True,
            co_change_threshold=5,
        )

        captured["mine_mock"].assert_called_once()
        mine_args, mine_kwargs = captured["mine_mock"].call_args
        assert mine_args[0] == tmp_path.resolve()
        assert mine_kwargs["co_change_threshold"] == 5

        captured["write_mock"].assert_awaited_once()

        # Indexing must complete before mining runs.
        assert captured["order"] == ["index", "write"]


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
        monkeypatch.setattr("code_atlas.backends.GraphClient", lambda s: mock_graph)
        monkeypatch.setattr("code_atlas.dream.build_dream_report", AsyncMock(return_value=empty_report))

        await cli._run_dream()

        home = tmp_path / settings.knowledge.vault_path / "HOME.md"
        assert home.is_file()
        assert "Knowledge Vault" in home.read_text(encoding="utf-8")
        # create_graph_client's "auto" resolution probes ping() once itself before
        # _run_dream's own explicit check pings again — awaited, not exactly-once.
        mock_graph.ping.assert_awaited()
        # Closed by the use_backends scope, so the assertion is on the exit it drives
        # rather than on close() -- which the command no longer calls by hand.
        mock_graph.__aexit__.assert_awaited_once()


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
        monkeypatch.setattr("code_atlas.backends.GraphClient", lambda s: mock_graph)

        await cli._run_project_rm("myproject", skip_confirm=True)

        mock_graph.delete_project_data.assert_awaited_once_with("myproject")
        # Closed by the use_backends scope, so the assertion is on the exit it drives
        # rather than on close() -- which the command no longer calls by hand.
        mock_graph.__aexit__.assert_awaited_once()

    async def test_missing_project_exits_with_error(self, tmp_path, monkeypatch) -> None:
        import pytest
        import typer

        from code_atlas import cli
        from code_atlas.settings import AtlasSettings

        _reset_output()
        settings = AtlasSettings(project_root=tmp_path)
        mock_graph = self._mock_graph(found=False)

        monkeypatch.setattr(cli, "_load_settings", lambda: settings)
        monkeypatch.setattr("code_atlas.backends.GraphClient", lambda s: mock_graph)

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
        monkeypatch.setattr("code_atlas.backends.GraphClient", lambda s: mock_graph)

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
        monkeypatch.setattr("code_atlas.backends.GraphClient", lambda s: mock_graph)

        result = runner.invoke(app, ["project", "rm", "myproject"], input="y\n")

        assert result.exit_code == 0
        mock_graph.delete_project_data.assert_awaited_once_with("myproject")

    async def test_json_without_yes_refuses_without_prompt(self, tmp_path, monkeypatch) -> None:
        """--json without --yes must refuse immediately — never delete, never prompt."""
        import pytest
        import typer

        from code_atlas import cli
        from code_atlas.settings import AtlasSettings

        _reset_output()
        _output.json = True
        settings = AtlasSettings(project_root=tmp_path)
        mock_graph = self._mock_graph()

        def _unexpected_confirm(*args, **kwargs):
            raise AssertionError("typer.confirm should not be called in --json mode")

        monkeypatch.setattr(cli, "_load_settings", lambda: settings)
        monkeypatch.setattr("code_atlas.backends.GraphClient", lambda s: mock_graph)
        monkeypatch.setattr(cli.typer, "confirm", _unexpected_confirm)

        with pytest.raises(typer.Exit):
            await cli._run_project_rm("myproject", skip_confirm=False)

        mock_graph.delete_project_data.assert_not_awaited()

    async def test_json_with_yes_deletes_without_prompt(self, tmp_path, monkeypatch) -> None:
        """--json with --yes still deletes, unchanged, and never prompts."""
        from code_atlas import cli
        from code_atlas.settings import AtlasSettings

        _reset_output()
        _output.json = True
        settings = AtlasSettings(project_root=tmp_path)
        mock_graph = self._mock_graph()

        def _unexpected_confirm(*args, **kwargs):
            raise AssertionError("typer.confirm should not be called when --yes is passed")

        monkeypatch.setattr(cli, "_load_settings", lambda: settings)
        monkeypatch.setattr("code_atlas.backends.GraphClient", lambda s: mock_graph)
        monkeypatch.setattr(cli.typer, "confirm", _unexpected_confirm)

        await cli._run_project_rm("myproject", skip_confirm=True)

        mock_graph.delete_project_data.assert_awaited_once_with("myproject")
        # Closed by the use_backends scope, so the assertion is on the exit it drives
        # rather than on close() -- which the command no longer calls by hand.
        mock_graph.__aexit__.assert_awaited_once()


class TestIndexExitCode:
    """An undrained pipeline means the graph does not match the working tree.

    Two incomplete indexes read as successes this session because `Done (full)` printed
    and the process exited 0 — the `drained` flag was surfaced only as a warning line.
    """

    @staticmethod
    def _patch_infra(monkeypatch, tmp_path, result):
        import contextlib

        from code_atlas import cli
        from code_atlas.settings import AtlasSettings

        _reset_output()
        settings = AtlasSettings(project_root=tmp_path)
        settings.embeddings.enabled = False
        monkeypatch.setattr(cli, "_load_settings", lambda: settings)
        monkeypatch.setattr("code_atlas.backends.create_event_bus", AsyncMock(return_value=AsyncMock()))
        monkeypatch.setattr("code_atlas.backends.create_graph_client", AsyncMock(return_value=AsyncMock()))
        monkeypatch.setattr("code_atlas.indexing.orchestrator.detect_sub_projects", lambda *a, **kw: [])
        monkeypatch.setattr(cli, "_index_single_with_spinner", AsyncMock(return_value=result))

        @contextlib.asynccontextmanager
        async def _lease(_bus, **_kwargs):
            yield "owner"

        monkeypatch.setattr("code_atlas.events.hold_indexer_lease", _lease)

    @staticmethod
    def _result(*, drained: bool):
        from code_atlas.indexing.orchestrator import IndexResult

        return IndexResult(
            files_scanned=3, files_published=3, entities_total=9, duration_s=1.0, mode="full", drained=drained
        )

    async def test_undrained_index_exits_nonzero(self, tmp_path, monkeypatch) -> None:
        import pytest
        import typer

        from code_atlas import cli

        self._patch_infra(monkeypatch, tmp_path, self._result(drained=False))
        with pytest.raises(typer.Exit) as excinfo:
            await cli._run_index(str(tmp_path), None, True, no_embed=True, no_git_check=True)
        assert excinfo.value.exit_code == 1

    async def test_drained_index_exits_zero(self, tmp_path, monkeypatch) -> None:
        from code_atlas import cli

        self._patch_infra(monkeypatch, tmp_path, self._result(drained=True))
        await cli._run_index(str(tmp_path), None, True, no_embed=True, no_git_check=True)


class TestProjectRootOutsideAGitRepo:
    """Read commands must work where the README says to run them (ATL-110).

    `_default_project_root` used to raise "Run from inside a git repo or pass an explicit
    path" — advice no user could follow, because `status` accepts no path. The commands
    that genuinely need a repo (`index`, `watch`, `mine-git-history`) enforce it
    themselves via `_resolve_project_root`, each with `--no-git-check`.
    """

    def test_falls_back_to_cwd_when_there_is_no_git_root(self, tmp_path, monkeypatch):
        from code_atlas import settings as settings_mod

        monkeypatch.setattr(settings_mod, "find_git_root", lambda *a, **k: None)
        monkeypatch.chdir(tmp_path)
        assert settings_mod._default_project_root() == tmp_path

    def test_prefers_the_git_root_when_there_is_one(self, tmp_path, monkeypatch):
        from code_atlas import settings as settings_mod

        monkeypatch.setattr(settings_mod, "find_git_root", lambda *a, **k: tmp_path / "repo")
        assert settings_mod._default_project_root() == tmp_path / "repo"


class TestConfigSectionsRejectUnknownKeys:
    """A typo inside a config section must fail loudly, not vanish (ATL-111).

    The root settings model has always been `extra="forbid"`, but a nested BaseModel
    defaults to `ignore` — so `[scope] include_paths = [...]` was accepted and dropped,
    and someone scoping indexing to three services silently indexed the whole monorepo.
    """

    def test_a_mistyped_scope_key_is_rejected(self):
        import pytest
        from pydantic import ValidationError

        from code_atlas.settings import ScopeSettings

        with pytest.raises(ValidationError, match="include_paths"):
            # Deliberately invalid. `extra="forbid"` also makes ty reject these
            # statically, which is a bonus of the change, not a problem with the test.
            ScopeSettings(include_paths=["a"], exclude_patterns=["b"])  # ty: ignore[unknown-argument]

    def test_the_real_keys_still_work(self):
        from code_atlas.settings import ScopeSettings

        s = ScopeSettings(paths=["svc/a"], extend_exclude=["*.tmp"])
        assert s.paths == ["svc/a"]
        assert s.extend_exclude == ["*.tmp"]

    def test_every_section_inherits_strictness(self):
        """Inherited rather than repeated, so a section added later is strict by default."""
        from code_atlas import settings as m

        sections = [
            v
            for v in vars(m).values()
            if isinstance(v, type) and issubclass(v, m.StrictSection) and v is not m.StrictSection
        ]
        assert len(sections) >= 15, "expected every atlas.toml section to inherit StrictSection"
        assert all(s.model_config.get("extra") == "forbid" for s in sections)


class TestMcpNoIndexFlag:
    """The flag is only useful if it survives the settings-precedence dance."""

    @staticmethod
    def _capture(monkeypatch):
        captured: dict = {}

        def fake_create(_settings, **kwargs):
            captured.update(kwargs)

            class _Server:
                def run(self, transport: str) -> None:
                    pass

            return _Server()

        monkeypatch.setattr("code_atlas.server.mcp.create_mcp_server", fake_create)
        return captured

    def test_no_index_disables_auto_index(self, monkeypatch):
        captured = self._capture(monkeypatch)
        result = runner.invoke(app, ["mcp", "--no-index"])
        assert result.exit_code == 0, result.output
        assert captured["auto_index"] is False

    def test_default_leaves_indexing_on(self, monkeypatch):
        captured = self._capture(monkeypatch)
        result = runner.invoke(app, ["mcp"])
        assert result.exit_code == 0, result.output
        assert captured["auto_index"] is True

    def test_the_configured_value_applies_when_no_flag_is_given(self, monkeypatch):
        import code_atlas.cli as cli_mod

        captured = self._capture(monkeypatch)
        base = cli_mod._load_settings()
        base.mcp.auto_index = False
        monkeypatch.setattr(cli_mod, "_load_settings", lambda: base)

        result = runner.invoke(app, ["mcp"])
        assert result.exit_code == 0, result.output
        assert captured["auto_index"] is False

    def test_an_explicit_flag_overrides_the_configured_value(self, monkeypatch):
        """--index wins over mcp.auto_index=false in atlas.toml or ATLAS_MCP__AUTO_INDEX.

        The first version of this made the flag one-way -- it could disable indexing but
        never re-enable it -- reasoning that a config which deliberately turned indexing
        off should not be overridden. That is backwards: typing a flag is the more
        explicit act of the two, and every other option on this command (--strict,
        --host, --port, --transport) already resolves that way. An override you cannot
        reach from the command line is not an override.
        """
        import code_atlas.cli as cli_mod

        captured = self._capture(monkeypatch)
        base = cli_mod._load_settings()
        base.mcp.auto_index = False
        monkeypatch.setattr(cli_mod, "_load_settings", lambda: base)

        result = runner.invoke(app, ["mcp", "--index"])
        assert result.exit_code == 0, result.output
        assert captured["auto_index"] is True
