"""Git history mining for hotspot/bus-factor/co-change signals (ADR-0013 git_signals).

Mining (`mine_git_signals`) is pure Python over GitPython's structured commit
data — no graph-backend dependency, so it's testable against a throwaway git
repo with no mocking. Writing the mined signals into the graph
(`write_git_signals`) goes through ``GraphBackend.write_git_file_signals``/
``write_co_change_edges`` (graph/protocol.py) — backend-agnostic, works
against both ``GraphClient`` (Memgraph) and ``SqliteGraphClient``.

Not wired into the continuous file-watcher pipeline — full git-log history
mining doesn't map to a single file-changed event and is too expensive to run
per-save. Instead this is invoked by the one-shot ``atlas mine-git-history``
CLI command that a user/CI job re-runs periodically.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from git import Repo

from code_atlas.schema import NodeLabel

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.graph.protocol import GraphBackend

# Co-change pairs sharing fewer commits than this are dropped as graph noise.
DEFAULT_CO_CHANGE_THRESHOLD = 3


@dataclass(frozen=True)
class FileSignal:
    file_path: str
    commit_count: int
    author_count: int
    days_since_last_commit: float


@dataclass(frozen=True)
class CoChangePair:
    file_a: str
    file_b: str
    count: int


@dataclass(frozen=True)
class GitSignalsResult:
    file_signals: tuple[FileSignal, ...]
    co_change_pairs: tuple[CoChangePair, ...]
    commits_scanned: int


def mine_git_signals(repo_root: Path, *, co_change_threshold: int = DEFAULT_CO_CHANGE_THRESHOLD) -> GitSignalsResult:
    """Mine per-file commit/author/co-change signals from *repo_root*'s full git history.

    Uses GitPython's structured ``commit.stats.files`` (insertions/deletions per
    path) and ``commit.author`` instead of hand-parsing ``git log --numstat``
    text. Per file: total commit count (hotspot proxy), distinct author count
    (bus-factor proxy), and days since the most recent commit touching it.
    Co-change pairs are file pairs that appear together in the same commit at
    least *co_change_threshold* times.
    """
    repo = Repo(str(repo_root))

    commit_counts: dict[str, int] = {}
    authors: dict[str, set[str]] = {}
    last_commit_ts: dict[str, int] = {}
    co_change_counts: dict[tuple[str, str], int] = {}
    commits_scanned = 0

    try:
        commits = list(repo.iter_commits())
    except ValueError:
        # Unborn HEAD — a freshly `git init`'d repo with zero commits yet.
        commits = []

    for commit in commits:
        commits_scanned += 1
        files = sorted(commit.stats.files.keys())  # ty: ignore[invalid-argument-type]  # GitPython's commit.stats.files is untyped
        author = commit.author.email or commit.author.name or "unknown"
        committed_ts = commit.committed_date
        for f in files:
            commit_counts[f] = commit_counts.get(f, 0) + 1
            authors.setdefault(f, set()).add(author)
            if committed_ts > last_commit_ts.get(f, -1):
                last_commit_ts[f] = committed_ts
        for a, b in itertools.combinations(files, 2):
            co_change_counts[(a, b)] = co_change_counts.get((a, b), 0) + 1

    now = time.time()
    file_signals = tuple(
        FileSignal(
            file_path=f,
            commit_count=count,
            author_count=len(authors[f]),
            days_since_last_commit=round((now - last_commit_ts[f]) / 86400, 1),
        )
        for f, count in commit_counts.items()
    )
    co_change_pairs = tuple(
        CoChangePair(file_a=a, file_b=b, count=count)
        for (a, b), count in co_change_counts.items()
        if count >= co_change_threshold
    )
    return GitSignalsResult(file_signals=file_signals, co_change_pairs=co_change_pairs, commits_scanned=commits_scanned)


async def write_git_signals(graph: GraphBackend, project_name: str, result: GitSignalsResult) -> dict[str, int]:
    """Write mined per-file signals onto Module/DocFile nodes and CO_CHANGES_WITH edges.

    Matches nodes by ``(project_name, file_path)``. Files with no matching
    Module/DocFile node (deleted files, non-indexed extensions, etc.) are
    silently skipped — no error, just not written. Returns counts for CLI
    reporting: ``files_matched`` (signal properties written) and
    ``co_change_edges`` (CO_CHANGES_WITH edges created/updated).
    """
    files_matched = 0
    if result.file_signals:
        items = [
            {
                "fp": sig.file_path,
                "cc": sig.commit_count,
                "ac": sig.author_count,
                "days": sig.days_since_last_commit,
            }
            for sig in result.file_signals
        ]
        for label in (NodeLabel.MODULE, NodeLabel.DOC_FILE):
            files_matched += await graph.write_git_file_signals(project_name, label, items)

    co_change_edges = 0
    if result.co_change_pairs:
        # file_a < file_b always (mine_git_signals sorts before combinations) —
        # a single directed edge per pair is enough; readers treat it as
        # symmetric and don't care which side matched which.
        pairs = [{"a": p.file_a, "b": p.file_b, "cnt": p.count} for p in result.co_change_pairs]
        co_change_edges = await graph.write_co_change_edges(project_name, pairs)

    return {
        "commits_scanned": result.commits_scanned,
        "files_mined": len(result.file_signals),
        "files_matched": files_matched,
        "co_change_pairs_mined": len(result.co_change_pairs),
        "co_change_edges": co_change_edges,
    }
