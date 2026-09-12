"""Integration tests for git-signal mining (find_hotspots) against a real Memgraph instance.

Mines a **throwaway git repo built per test**, not this checkout's own history, and
confirms the mined signals land on seeded Module nodes — both via the pure
`write_git_signals` path and via the `atlas mine-git-history` / `atlas index
--with-git-signals` CLI commands end-to-end.

It used to mine `_REPO_ROOT` and assert on `cli.py`/`settings.py`, which failed in two
ways once mining became bounded (ATL-188):

* **The assertions depended on this repo's recent commit pattern.** A
  `CO_CHANGES_WITH` edge with `count >= 3` needed three commits touching both files;
  across the last 25 there is one. The test was reading the repo's history as a fixture
  it did not control, so shipping the default window would have broken it for a reason
  that says nothing about the code.
* **It seeded a project named `code-atlas`** — `derive_project_name` of the real root —
  into the shared test Memgraph. The conftest wipe guard refuses any project not
  prefixed `test`/`bench`, so a run that died before cleanup wedged *every subsequent
  run* until someone cleared the instance by hand. That happened twice.

A synthetic repo fixes both: the history is exactly what the assertion needs, the
project name is test-prefixed, and the walk is a handful of commits rather than 544.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from git import Actor, Repo

from code_atlas import cli
from code_atlas.indexing.git_signals import mine_git_signals, write_git_signals
from code_atlas.schema import RelType
from code_atlas.settings import derive_project_name

if TYPE_CHECKING:
    from code_atlas.graph.client import GraphClient

pytestmark = pytest.mark.integration

_FILE_A = "src/app/cli.py"
_FILE_B = "src/app/settings.py"


def _commit(repo: Repo, files: dict[str, str], *, author: str) -> None:
    """Write/update *files* and commit them. Mirrors the unit suite's helper."""
    for name, content in files.items():
        path = Path(repo.working_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    repo.index.add(list(files.keys()))
    actor = Actor(author, f"{author}@example.com")
    repo.index.commit(f"update {', '.join(sorted(files))}", author=actor, committer=actor)


@pytest.fixture
def project(git_repo: Path) -> str:
    """The project name the CLI will derive for this repo, so seeded nodes and written
    signals agree without patching anything.

    pytest names `tmp_path` after the test, so every derived name here starts with
    `test_` — which is also what the conftest wipe guard requires. The old version of
    this file seeded `code-atlas`, the real root's name, and a run that died before
    cleanup then wedged every later run.
    """
    return derive_project_name(git_repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A repo whose history contains exactly what the assertions below need.

    Four commits touch both files, so a `CO_CHANGES_WITH` edge clears the threshold of
    3 with one to spare; two more touch one file each, so the per-file counts differ and
    a test cannot pass by writing the same number everywhere. Two authors, so
    `author_count` is not trivially 1.
    """
    repo = Repo.init(tmp_path)
    for i in range(4):
        _commit(repo, {_FILE_A: f"a{i}", _FILE_B: f"b{i}"}, author="alice" if i % 2 else "bob")
    _commit(repo, {_FILE_A: "a-solo"}, author="alice")
    _commit(repo, {_FILE_B: "b-solo"}, author="bob")
    return tmp_path


async def _seed_modules(graph_client: GraphClient, project: str) -> None:
    await graph_client.merge_project_node(project)
    for fp in (_FILE_A, _FILE_B):
        uid = f"{project}:{fp}"
        await graph_client.execute_write(
            "CREATE (n:Module:Entity {uid: $uid, project_name: $p, name: $fp, qualified_name: $uid, "
            "file_path: $fp, kind: 'module', line_start: 1, line_end: 1})",
            {"uid": uid, "p": project, "fp": fp},
        )


async def _signals_by_path(graph_client: GraphClient, project: str) -> dict[str, dict]:
    rows = await graph_client.execute(
        "MATCH (n:Module {project_name: $p}) WHERE n.git_commit_count IS NOT NULL "
        "RETURN n.file_path AS fp, n.git_commit_count AS cc, n.git_author_count AS ac, "
        "n.git_days_since_last_commit AS days ORDER BY fp",
        {"p": project},
    )
    return {r["fp"]: r for r in rows}


async def _co_change_count(graph_client: GraphClient, project: str) -> list[dict]:
    return await graph_client.execute(
        f"MATCH (a:Module {{project_name: $p, file_path: $fa}})"
        f"-[r:{RelType.CO_CHANGES_WITH}]->(b:Module {{project_name: $p, file_path: $fb}}) "
        "RETURN r.count AS count",
        {"p": project, "fa": _FILE_A, "fb": _FILE_B},
    )


class TestWriteGitSignals:
    async def test_writes_per_file_signals_and_co_change_edge(self, graph_client, git_repo, project):
        await _seed_modules(graph_client, project)

        result = mine_git_signals(git_repo, co_change_threshold=3)
        stats = await write_git_signals(graph_client, project, result)

        assert stats["commits_scanned"] == 6
        assert stats["files_matched"] == 2

        by_path = await _signals_by_path(graph_client, project)
        assert set(by_path) == {_FILE_A, _FILE_B}
        assert by_path[_FILE_A]["cc"] == 5, "4 shared commits plus one solo"
        assert by_path[_FILE_B]["cc"] == 5
        assert by_path[_FILE_A]["ac"] == 2
        assert by_path[_FILE_A]["days"] >= 0

        edge_rows = await _co_change_count(graph_client, project)
        assert edge_rows, f"expected a CO_CHANGES_WITH edge between {_FILE_A} and {_FILE_B}"
        assert edge_rows[0]["count"] == 4

    async def test_the_window_bounds_what_is_mined(self, graph_client, git_repo):
        """ATL-188. The default is the last 25 commits, not all of history, because each
        commit costs one `git diff` subprocess.

        Asserted on `commits_scanned` rather than on a clock: a timing assertion would be
        the flakiest test in the suite, and the count is what the bound actually controls.
        """
        assert mine_git_signals(git_repo, max_commits=0).commits_scanned == 6, "0 must mean all of history"
        assert mine_git_signals(git_repo, max_commits=2).commits_scanned == 2
        assert mine_git_signals(git_repo, max_commits=100).commits_scanned == 6, "a window past HEAD is not an error"

        # The window is the newest commits, not the oldest: only the two solo commits
        # remain, so each file has one commit and they no longer co-change at all.
        narrow = mine_git_signals(git_repo, co_change_threshold=1, max_commits=2)
        assert {s.file_path: s.commit_count for s in narrow.file_signals} == {_FILE_A: 1, _FILE_B: 1}
        assert narrow.co_change_pairs == ()


class TestMineGitHistoryCliCommand:
    """`atlas mine-git-history` end-to-end: real git history in, graph writes out."""

    async def test_cli_command_mines_and_writes_signals(self, graph_client, git_repo, project):
        await _seed_modules(graph_client, project)

        # The CLI opens and closes its own client, as it does in production. It reaches
        # the same test Memgraph -- tests/conftest.py exports ATLAS_BACKEND__GRAPH__MEMGRAPH__* --
        # so the assertions below still read what the command wrote.
        await cli._run_mine_git_history(str(git_repo), 3, 0, no_git_check=True)

        by_path = await _signals_by_path(graph_client, project)
        assert set(by_path) == {_FILE_A, _FILE_B}


class TestIndexCommandWithGitSignals:
    """`atlas index --with-git-signals` end-to-end: mining runs against real git
    history/Memgraph right after the (stubbed) indexing pass completes.

    The indexing dispatch itself (AST parse + embed pipeline) is stubbed out —
    that flow is already covered by the orchestrator/live-update integration
    tests — so this test isolates the new wiring: does `atlas index` actually
    invoke `mine_git_signals`/`write_git_signals` against real infra afterward.
    """

    async def test_cli_index_with_git_signals_mines_after_indexing(self, graph_client, git_repo, project, monkeypatch):
        from code_atlas.indexing.orchestrator import IndexResult

        await _seed_modules(graph_client, project)

        calls: list[str] = []

        async def fake_single_with_spinner(settings, graph, bus, *, scope, full_reindex, **_reset_flags):
            calls.append("index")
            return IndexResult(files_scanned=0, files_published=0, entities_total=0, duration_s=0.0)

        monkeypatch.setattr("code_atlas.indexing.orchestrator.detect_sub_projects", lambda root, mono: [])
        monkeypatch.setattr(cli, "_index_single_with_spinner", fake_single_with_spinner)

        await cli._run_index(
            str(git_repo),
            None,
            False,
            no_embed=True,
            no_git_check=True,
            with_git_signals=True,
            co_change_threshold=3,
            git_signals_max_commits=0,
        )

        assert calls == ["index"]

        by_path = await _signals_by_path(graph_client, project)
        assert set(by_path) == {_FILE_A, _FILE_B}

        edge_rows = await _co_change_count(graph_client, project)
        assert edge_rows, f"expected a CO_CHANGES_WITH edge between {_FILE_A} and {_FILE_B}"
