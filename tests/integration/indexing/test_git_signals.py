"""Integration tests for git-signal mining (find_hotspots) against a real Memgraph instance.

Mines *this repo's own real git history* (read-only `git log`-equivalent
operations via GitPython — never mutates anything) and confirms the mined
signals land on seeded Module nodes, both via the pure write_git_signals path
and via the `atlas mine-git-history` CLI command end-to-end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from code_atlas import cli
from code_atlas.indexing.git_signals import mine_git_signals, write_git_signals
from code_atlas.schema import RelType
from code_atlas.settings import derive_project_name, find_git_root

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.graph.client import GraphClient

pytestmark = pytest.mark.integration

_found_root = find_git_root()
if _found_root is None:
    raise RuntimeError("tests must run inside the code-atlas git repo")
_REPO_ROOT: Path = _found_root
_PROJECT = derive_project_name(_REPO_ROOT)

# Two files with a long, well-established co-change history in this repo
# (cli.py and settings.py both change together across most CLI-facing commits).
_FILE_A = "src/code_atlas/cli.py"
_FILE_B = "src/code_atlas/settings.py"


async def _seed_modules(graph_client: GraphClient) -> None:
    await graph_client.merge_project_node(_PROJECT)
    for fp in (_FILE_A, _FILE_B):
        uid = f"{_PROJECT}:{fp}"
        await graph_client.execute_write(
            "CREATE (n:Module:Entity {uid: $uid, project_name: $p, name: $fp, qualified_name: $uid, "
            "file_path: $fp, kind: 'module', line_start: 1, line_end: 1})",
            {"uid": uid, "p": _PROJECT, "fp": fp},
        )


class TestWriteGitSignals:
    async def test_writes_per_file_signals_and_co_change_edge(self, graph_client):
        await _seed_modules(graph_client)

        result = mine_git_signals(_REPO_ROOT, co_change_threshold=3)
        stats = await write_git_signals(graph_client, _PROJECT, result)

        assert stats["commits_scanned"] > 0
        assert stats["files_matched"] == 2

        rows = await graph_client.execute(
            "MATCH (n:Module {project_name: $p}) WHERE n.git_commit_count IS NOT NULL "
            "RETURN n.file_path AS fp, n.git_commit_count AS cc, n.git_author_count AS ac, "
            "n.git_days_since_last_commit AS days ORDER BY fp",
            {"p": _PROJECT},
        )
        by_path = {r["fp"]: r for r in rows}
        assert set(by_path) == {_FILE_A, _FILE_B}
        assert by_path[_FILE_A]["cc"] > 0
        assert by_path[_FILE_A]["ac"] >= 1
        assert by_path[_FILE_A]["days"] >= 0

        edge_rows = await graph_client.execute(
            f"MATCH (a:Module {{project_name: $p, file_path: $fa}})"
            f"-[r:{RelType.CO_CHANGES_WITH}]->(b:Module {{project_name: $p, file_path: $fb}}) "
            "RETURN r.count AS count",
            {"p": _PROJECT, "fa": _FILE_A, "fb": _FILE_B},
        )
        assert edge_rows, "expected a CO_CHANGES_WITH edge between cli.py and settings.py"
        assert edge_rows[0]["count"] >= 3


class TestMineGitHistoryCliCommand:
    """`atlas mine-git-history` end-to-end: real git history in, graph writes out."""

    async def test_cli_command_mines_and_writes_signals(self, graph_client, monkeypatch):
        await _seed_modules(graph_client)

        # The CLI opens and closes its own client, as it does in production. It reaches
        # the same test Memgraph -- tests/conftest.py exports ATLAS_MEMGRAPH__* -- so the
        # assertions below still read what the command wrote.
        #
        # This used to patch code_atlas.graph.client.GraphClient to hand the CLI this
        # fixture's client, and mock GraphClient.close so the CLI could not close the
        # shared connection. Neither worked: backends/__init__.py binds GraphClient at
        # import, so the patch missed and a second real driver was built anyway -- and
        # the class-level close mock then leaked both it and this fixture's client. The
        # `finally: await graph.close()` the mock was guarding against no longer exists.
        await cli._run_mine_git_history(str(_REPO_ROOT), 3, no_git_check=False)

        rows = await graph_client.execute(
            "MATCH (n:Module {project_name: $p}) WHERE n.git_commit_count IS NOT NULL RETURN count(n) AS cnt",
            {"p": _PROJECT},
        )
        assert rows[0]["cnt"] == 2


class TestIndexCommandWithGitSignals:
    """`atlas index --with-git-signals` end-to-end: mining runs against real git
    history/Memgraph right after the (stubbed) indexing pass completes.

    The indexing dispatch itself (AST parse + embed pipeline) is stubbed out —
    that flow is already covered by the orchestrator/live-update integration
    tests — so this test isolates the new wiring: does `atlas index` actually
    invoke `mine_git_signals`/`write_git_signals` against real infra afterward.
    """

    async def test_cli_index_with_git_signals_mines_after_indexing(self, graph_client, monkeypatch):
        from code_atlas.indexing.orchestrator import IndexResult

        await _seed_modules(graph_client)

        calls: list[str] = []

        async def fake_single_with_spinner(settings, graph, bus, *, scope, full_reindex, **_reset_flags):
            calls.append("index")
            return IndexResult(files_scanned=0, files_published=0, entities_total=0, duration_s=0.0)

        # The CLI owns its own client here too -- see TestMineGitHistoryCliCommand for
        # why the old sharing patch was both ineffective and leaky.
        monkeypatch.setattr("code_atlas.indexing.orchestrator.detect_sub_projects", lambda root, mono: [])
        monkeypatch.setattr(cli, "_index_single_with_spinner", fake_single_with_spinner)

        await cli._run_index(
            str(_REPO_ROOT),
            None,
            False,
            no_embed=True,
            no_git_check=False,
            with_git_signals=True,
            co_change_threshold=3,
        )

        assert calls == ["index"]

        rows = await graph_client.execute(
            "MATCH (n:Module {project_name: $p}) WHERE n.git_commit_count IS NOT NULL RETURN count(n) AS cnt",
            {"p": _PROJECT},
        )
        assert rows[0]["cnt"] == 2

        edge_rows = await graph_client.execute(
            f"MATCH (a:Module {{project_name: $p, file_path: $fa}})"
            f"-[r:{RelType.CO_CHANGES_WITH}]->(b:Module {{project_name: $p, file_path: $fb}}) "
            "RETURN r.count AS count",
            {"p": _PROJECT, "fa": _FILE_A, "fb": _FILE_B},
        )
        assert edge_rows, "expected a CO_CHANGES_WITH edge between cli.py and settings.py"
