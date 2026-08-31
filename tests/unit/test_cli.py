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

            # The CLI opens its backends through a scope now, so a double has to honour
            # the same protocol the real client does -- otherwise it fails at the
            # `async with`, which is a fake that stopped resembling the thing it stands in for.
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc) -> None:
                await self.close()

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

            # The CLI opens its backends through a scope now, so a double has to honour
            # the same protocol the real client does -- otherwise it fails at the
            # `async with`, which is a fake that stopped resembling the thing it stands in for.
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc) -> None:
                await self.close()

            async def ping(self) -> None:
                return None

            async def ensure_schema(self, *, force_drop_embeddings: bool = False) -> None:
                return None

            async def close(self) -> None:
                return None

        # ``**_reset_flags`` absorbs reset/reset_embeddings: these doubles are about
        # which dispatch a scope picks, and TestDestructiveIndexFlags below is where the
        # destructive axis is asserted. Named rather than ``**_kw`` so it is obvious
        # which flags are being ignored on purpose.
        async def fake_monorepo_with_progress(settings, graph, bus, *, projects, full_reindex, **_reset_flags):
            captured["dispatch"] = "monorepo"
            captured["projects"] = projects
            return []

        async def fake_single_with_spinner(settings, graph, bus, *, scope, full_reindex, **_reset_flags):
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

            # The CLI opens its backends through a scope now, so a double has to honour
            # the same protocol the real client does -- otherwise it fails at the
            # `async with`, which is a fake that stopped resembling the thing it stands in for.
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc) -> None:
                await self.close()

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

            # The CLI opens its backends through a scope now, so a double has to honour
            # the same protocol the real client does -- otherwise it fails at the
            # `async with`, which is a fake that stopped resembling the thing it stands in for.
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc) -> None:
                await self.close()

            async def ping(self) -> None:
                return None

            async def ensure_schema(self, *, force_drop_embeddings: bool = False) -> None:
                return None

            async def close(self) -> None:
                return None

        async def fake_single_with_spinner(settings, graph, bus, *, scope, full_reindex, **_reset_flags):
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


def _forbid_prompt(monkeypatch, why: str) -> None:
    """Fail loudly if a confirmation appears where none belongs.

    An exit code cannot tell a run that never asked from one that asked and got a yes
    out of a stray stdin, and ADR-0042 forbids the second on every path here.
    """
    from code_atlas import cli

    def _unexpected(*args, **kwargs):
        raise AssertionError(f"typer.confirm should not be called: {why}")

    monkeypatch.setattr(cli.typer, "confirm", _unexpected)


class TestDestructiveIndexFlags:
    """`atlas index` separates scope from destruction, and destruction is gated (ATL-148/149).

    ``--full`` used to delete the project's graph data before rebuilding it, which is
    not what anyone reads "full index" as meaning. Running it on the production graph to
    re-derive an extraction fix would have destroyed 35,104 embeddings and re-billed
    every one through a paid provider — nearly done by hand while investigating ADR-0040.
    Scope now lives on ``--full``, destruction on ``--reset`` / ``--reset-embeddings``,
    and the destructive pair has to state its blast radius and be told yes.

    Every refusal below asserts that the destructive graph calls did not happen, not
    merely that the exit code was non-zero -- an exit code says nothing about whether
    the data survived. The CLI can destroy by two routes: its own database-wide
    ``clear_embeddings`` for a dimension change, and the orchestrator behind
    ``reset``/``reset_embeddings``. A refusal has to close both, so both are asserted:
    the spies for the first, ``dispatch`` staying ``None`` for the second.
    """

    @staticmethod
    def _patch_common(monkeypatch, *, sub_projects=None, interactive: bool = True) -> dict:
        from code_atlas import cli

        captured: dict = {"counted": [], "deleted": [], "cleared": [], "dispatch": None}

        class FakeBus:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc) -> None:
                await self.close()

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

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc) -> None:
                await self.close()

            async def ping(self) -> None:
                return None

            async def close(self) -> None:
                return None

            async def ensure_schema(self, *, force_drop_embeddings: bool = False) -> None:
                # Recorded because this is DDL: a schema migration drops every vector
                # index, so a refused run that got this far already destroyed something.
                captured["ensure_schema"] = True

            async def count_project_data(self, project_name: str) -> list[dict]:
                captured["counted"].append(project_name)
                # Every project also answers with a `{name}/child` row, so a test can tell
                # whether the preflight printed the prefix set (what clear_embeddings
                # reaches) or only the exact name (what delete_project_data reaches).
                return [
                    {"name": project_name, "nodes": 12, "relationships": 30, "embedded_nodes": 9, "embed_chunks": 2},
                    {
                        "name": f"{project_name}/child",
                        "nodes": 4,
                        "relationships": 6,
                        "embedded_nodes": 3,
                        "embed_chunks": 1,
                    },
                ]

            async def delete_project_data(self, project_name: str) -> None:
                captured["deleted"].append(project_name)

            async def clear_embeddings(self, project_name: str | None) -> int:
                captured["cleared"].append(project_name)
                return 0

        async def fake_single_with_spinner(
            settings, graph, bus, *, scope, full_reindex, reset=False, reset_embeddings=False
        ):
            from code_atlas.indexing.orchestrator import IndexResult

            captured["dispatch"] = "single"
            captured["flags"] = (full_reindex, reset, reset_embeddings)
            return IndexResult(files_scanned=0, files_published=0, entities_total=0, duration_s=0.0)

        async def fake_monorepo_with_progress(
            settings, graph, bus, *, projects, full_reindex, reset=False, reset_embeddings=False
        ):
            captured["dispatch"] = "monorepo"
            captured["flags"] = (full_reindex, reset, reset_embeddings)
            captured["projects"] = projects
            return []

        monkeypatch.setattr("code_atlas.backends.EventBus", FakeBus)
        monkeypatch.setattr("code_atlas.backends.GraphClient", FakeGraph)
        monkeypatch.setattr(
            "code_atlas.indexing.orchestrator.detect_sub_projects", lambda root, mono: sub_projects or []
        )
        monkeypatch.setattr(cli, "_index_single_with_spinner", fake_single_with_spinner)
        monkeypatch.setattr(cli, "_index_monorepo_with_progress", fake_monorepo_with_progress)
        # Stated per test rather than inherited: CliRunner gives the command a stdin that
        # is not a TTY, so the non-TTY refusal would otherwise be the branch every test
        # here took by accident, including the ones named after the prompt.
        monkeypatch.setattr(cli, "_is_interactive", lambda: interactive)
        _reset_output()
        return captured

    def test_two_destructive_flags_together_is_a_usage_error(self, tmp_path, monkeypatch) -> None:
        captured = self._patch_common(monkeypatch)

        result = runner.invoke(app, ["index", str(tmp_path), "--reset", "--reset-embeddings", "--no-git-check"])

        assert result.exit_code == 2, result.output
        assert captured["counted"] == []
        assert captured["dispatch"] is None

    def test_full_combined_with_reset_is_a_usage_error(self, tmp_path, monkeypatch) -> None:
        """Not a precedence rule. Guessing which axis was meant is the original defect."""
        captured = self._patch_common(monkeypatch)

        result = runner.invoke(app, ["index", str(tmp_path), "--full", "--reset", "--no-git-check"])

        assert result.exit_code == 2, result.output
        assert captured["dispatch"] is None

    def test_a_plain_index_never_prompts_and_destroys_nothing(self, tmp_path, monkeypatch) -> None:
        captured = self._patch_common(monkeypatch)
        _forbid_prompt(monkeypatch, "a non-destructive index has nothing to confirm")

        result = runner.invoke(app, ["index", str(tmp_path), "--no-embed", "--no-git-check"])

        assert result.exit_code == 0, result.output
        assert captured["flags"] == (False, False, False)
        assert captured["counted"] == [], "a run that removes nothing has no blast radius to describe"
        assert captured["deleted"] == []
        assert captured["cleared"] == []

    def test_full_re_checks_every_file_and_destroys_nothing(self, tmp_path, monkeypatch) -> None:
        """The whole of ATL-148: --full is a scope decision now, not a destruction one."""
        captured = self._patch_common(monkeypatch)
        _forbid_prompt(monkeypatch, "--full is not destructive")

        result = runner.invoke(app, ["index", str(tmp_path), "--full", "--no-embed", "--no-git-check"])

        assert result.exit_code == 0, result.output
        assert captured["flags"] == (True, False, False)
        assert captured["counted"] == []
        assert captured["deleted"] == []
        assert captured["cleared"] == []

    def test_reset_without_a_tty_and_without_yes_removes_nothing(self, tmp_path, monkeypatch) -> None:
        """No prompt that default-accepts and no timeout that proceeds -- it refuses."""
        captured = self._patch_common(monkeypatch, interactive=False)
        _forbid_prompt(monkeypatch, "there is nobody present to answer")

        result = runner.invoke(app, ["index", str(tmp_path), "--reset", "--no-embed", "--no-git-check"])

        assert result.exit_code == 1, result.output
        assert captured["deleted"] == []
        assert captured["cleared"] == []
        assert captured["dispatch"] is None
        assert "ensure_schema" not in captured, "refused above the schema migration, not merely before the index"

    def test_reset_refused_at_the_prompt_removes_nothing(self, tmp_path, monkeypatch) -> None:
        captured = self._patch_common(monkeypatch)

        result = runner.invoke(app, ["index", str(tmp_path), "--reset", "--no-embed", "--no-git-check"], input="n\n")

        assert result.exit_code == 1, result.output
        # "Aborted." is echoed only by the prompt path, so this separates a genuine no
        # from the "nobody is there" refusal, which would also exit 1 with nothing removed.
        assert "Aborted." in result.output
        assert captured["deleted"] == []
        assert captured["cleared"] == []
        assert captured["dispatch"] is None
        assert "ensure_schema" not in captured

    def test_reset_with_yes_proceeds_and_names_only_what_the_delete_reaches(self, tmp_path, monkeypatch) -> None:
        captured = self._patch_common(monkeypatch)
        _forbid_prompt(monkeypatch, "--yes is the confirmation")

        result = runner.invoke(app, ["index", str(tmp_path), "--reset", "--yes", "--no-embed", "--no-git-check"])

        assert result.exit_code == 0, result.output
        # The delete itself belongs to index_project, which this double stands in for;
        # what the CLI owes is the authorisation arriving there and nowhere else.
        assert captured["flags"] == (False, True, False)
        assert captured["counted"] == [tmp_path.name]
        # delete_project_data is exact-match, so naming the prefix child would be the
        # over-report ADR-0042 forbids as firmly as an under-report.
        assert f"{tmp_path.name}/child" not in result.output
        assert "11 vector(s) to re-embed" in result.output

    def test_reset_embeddings_names_the_children_its_prefix_match_reaches(self, tmp_path, monkeypatch) -> None:
        """clear_embeddings matches name-or-prefix, so the sub-projects have to be named."""
        captured = self._patch_common(monkeypatch)
        _forbid_prompt(monkeypatch, "--yes is the confirmation")

        result = runner.invoke(
            app, ["index", str(tmp_path), "--reset-embeddings", "--yes", "--no-embed", "--no-git-check"]
        )

        assert result.exit_code == 0, result.output
        assert captured["flags"] == (False, False, True)
        assert f"{tmp_path.name}/child" in result.output
        assert "15 vector(s) to re-embed" in result.output

    def test_a_monorepo_reset_names_every_sub_project_the_run_will_visit(self, tmp_path, monkeypatch) -> None:
        from code_atlas.indexing.orchestrator import DetectedProject

        sub_projects = [
            DetectedProject(name="foo", path="packages/foo", root=tmp_path / "packages" / "foo", marker="x"),
            DetectedProject(name="bar", path="packages/bar", root=tmp_path / "packages" / "bar", marker="x"),
        ]
        captured = self._patch_common(monkeypatch, sub_projects=sub_projects)
        _forbid_prompt(monkeypatch, "--yes is the confirmation")

        result = runner.invoke(
            app, ["index", str(tmp_path), "--reset", "--yes", "-p", "foo", "--no-embed", "--no-git-check"]
        )

        assert result.exit_code == 0, result.output
        assert captured["dispatch"] == "monorepo"
        # One delete per sub-project the run visits, plus the root's own files -- and
        # not the sibling --project excluded, which this run never touches.
        assert sorted(captured["counted"]) == [tmp_path.name, f"{tmp_path.name}/foo"]
        assert f"{tmp_path.name}/bar" not in result.output

    def test_a_preflight_count_that_fails_aborts_with_nothing_removed(self, tmp_path, monkeypatch) -> None:
        """A destructive run that cannot describe its own blast radius aborts (ADR-0042).

        Proceeding on an estimate is not on offer: what is being estimated is
        unrecoverable and metered.
        """
        captured = self._patch_common(monkeypatch)

        async def boom(self, project_name: str) -> list[dict]:
            raise RuntimeError("count query timed out")

        monkeypatch.setattr("code_atlas.backends.GraphClient.count_project_data", boom)

        result = runner.invoke(app, ["index", str(tmp_path), "--reset", "--yes", "--no-embed", "--no-git-check"])

        assert result.exit_code == 1, result.output
        assert captured["deleted"] == []
        assert captured["cleared"] == []
        assert captured["dispatch"] is None
        assert "ensure_schema" not in captured


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
        # The confirmation gate reads the blast radius first (ATL-149), so a double
        # without this aborts the command before it reaches the question under test.
        mock_graph.count_project_data = AsyncMock(
            return_value=[
                {"name": "myproject", "nodes": 10, "relationships": 4, "embedded_nodes": 3, "embed_chunks": 1}
            ]
        )
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
        # CliRunner feeds stdin without making it a TTY, so without this the command
        # would take the "nobody is there to answer" branch and this test would pass
        # having never reached the prompt it is named after.
        monkeypatch.setattr(cli, "_is_interactive", lambda: True)

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
        monkeypatch.setattr(cli, "_is_interactive", lambda: True)

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

    def test_it_prints_what_it_will_remove_before_removing_it(self, tmp_path, monkeypatch) -> None:
        """The defect ATL-149 names: this command confirmed without saying what was at stake.

        Asserted with --yes so the counts are isolated from the prompt -- the preflight
        owes its numbers on every destructive path, not only the interactive one.
        """
        from code_atlas import cli
        from code_atlas.settings import AtlasSettings

        _reset_output()
        settings = AtlasSettings(project_root=tmp_path)
        mock_graph = self._mock_graph()

        monkeypatch.setattr(cli, "_load_settings", lambda: settings)
        monkeypatch.setattr("code_atlas.backends.GraphClient", lambda s: mock_graph)

        result = runner.invoke(app, ["project", "rm", "myproject", "--yes"])

        assert result.exit_code == 0, result.output
        assert "10 nodes" in result.output
        assert "4 relationships" in result.output
        assert "3 embedded nodes" in result.output
        assert "1 embed chunks" in result.output
        # Stated in the terms actually paid: those vectors have to be re-embedded, and
        # therefore re-billed.
        assert "4 vector(s) to re-embed" in result.output
        mock_graph.delete_project_data.assert_awaited_once_with("myproject")

    def test_without_a_tty_it_refuses_instead_of_prompting(self, tmp_path, monkeypatch) -> None:
        """A prompt nobody can answer is not a gate, so --yes is the only way through."""
        from code_atlas import cli
        from code_atlas.settings import AtlasSettings

        _reset_output()
        settings = AtlasSettings(project_root=tmp_path)
        mock_graph = self._mock_graph()

        monkeypatch.setattr(cli, "_load_settings", lambda: settings)
        monkeypatch.setattr("code_atlas.backends.GraphClient", lambda s: mock_graph)
        monkeypatch.setattr(cli, "_is_interactive", lambda: False)
        _forbid_prompt(monkeypatch, "there is nobody present to answer")

        result = runner.invoke(app, ["project", "rm", "myproject"], input="y\n")

        assert result.exit_code == 1, result.output
        mock_graph.delete_project_data.assert_not_awaited()


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
